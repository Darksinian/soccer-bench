"""Player controllers. Every agent maps an observation (team frame) to an action dict
{"action": "move"|"pass"|"stay", "x": float, "y": float} in the same frame, plus metadata.

Player specs (strings):
  random                 uniformly random moves and passes
  scripted               hand-written heuristic baseline
  <provider>/<model>     a model: an OpenRouter slug (backend "openrouter"), or
                         anthropic/<id> or openai/<id> called directly (backend "direct")
  <provider>/<model>@low append @<effort> to set the reasoning effort
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from .prompts import system_prompt, user_prompt
from .schema import CallMeta, Observation, Reply
from .sim import Config

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class Agent(Protocol):
    label: str

    async def act(self, obs: Observation) -> tuple[Reply, CallMeta]: ...


def make_agent(spec: str, cfg: Config, seed: int, backend: str = "openrouter") -> Agent:
    spec = spec.strip()
    if spec == "random":
        return RandomAgent(cfg, seed)
    if spec == "scripted":
        return ScriptedAgent(cfg)
    if "/" in spec and backend == "direct":
        from .direct import DirectAgent
        return DirectAgent(cfg, spec)
    if "/" in spec:
        model, _, effort = spec.partition("@")
        return OpenRouterAgent(cfg, model, effort or None)
    raise ValueError(f"unknown player spec {spec!r} (use random, scripted, or an OpenRouter model slug)")


def parse_action(text: str) -> Reply:
    """Pull the action JSON out of a model reply, tolerating code fences and extra prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    candidates += [text] + re.findall(r"\{[^{}]*\}", text, re.S)
    for c in candidates:
        try:
            data = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "action" in data:
            return {str(k): v for k, v in data.items()}
    raise ValueError(f"no action JSON in reply: {text[:200]!r}")


# ----- model-backed -----

class FatalAPIError(RuntimeError):
    """Auth or billing failure: every later call will fail too, so the match must stop."""


class OpenRouterAgent:
    max_retries = 3

    def __init__(self, cfg: Config, model: str, effort: str | None):
        self.cfg, self.model, self.effort = cfg, model, effort
        self.label = model + (f"@{effort}" if effort else "")
        self.system = system_prompt(cfg)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        req = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "X-Title": "soccer-eval",
            },
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data: dict[str, Any] = json.loads(resp.read())
                    return data
            except urllib.error.HTTPError as e:
                detail = f"OpenRouter HTTP {e.code} for {self.model}: {e.read()[:200]!r}"
                if e.code in (401, 402, 403):
                    raise FatalAPIError(detail) from e
                retryable = e.code == 429 or e.code >= 500
                if not retryable or attempt == self.max_retries:
                    raise RuntimeError(detail) from e
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.max_retries:
                    raise
            time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    async def act(self, obs: Observation) -> tuple[Reply, CallMeta]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": user_prompt(obs, self.cfg.roles)},
            ],
            "usage": {"include": True},
        }
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        data = await asyncio.to_thread(self._post, body)
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error']}")
        msg = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        meta: CallMeta = {
            "model": data.get("model", self.model),
            "in": usage.get("prompt_tokens", 0),
            "out": usage.get("completion_tokens", 0),
            "cost": usage.get("cost", 0.0) or 0.0,
            "raw": (msg.get("content") or "")[:500],
        }
        return parse_action(msg.get("content") or ""), meta


# ----- baselines -----

class RandomAgent:
    label = "random"

    def __init__(self, cfg: Config, seed: int):
        self.cfg, self.rng = cfg, random.Random(seed)

    async def act(self, obs: Observation) -> tuple[Reply, CallMeta]:
        me = next(p for p in obs["players"] if p["you"])
        x, y = self.rng.uniform(0, self.cfg.width), self.rng.uniform(0, self.cfg.height)
        kind = "pass" if me["has_ball"] and self.rng.random() < 0.3 else "move"
        return {"action": kind, "x": x, "y": y}, {}


class ScriptedAgent:
    """Chase the ball, dribble at goal, shoot when close, pass when blocked."""
    label = "scripted"
    shoot_range = 28.0

    def __init__(self, cfg: Config):
        self.cfg = cfg

    async def act(self, obs: Observation) -> tuple[Reply, CallMeta]:
        return self.decide(obs), {}

    def decide(self, obs: Observation) -> Reply:
        cfg = self.cfg
        W, H = cfg.width, cfg.height
        players = obs["players"]
        me = next(p for p in players if p["you"])
        mates = [p for p in players if p["ours"] and not p["you"]]
        opps = [p for p in players if not p["ours"]]
        ball = obs["ball"]
        bx, by = ball["pos"]
        mx, my = me["pos"]

        def move(x: float, y: float) -> Reply:
            return {"action": "move", "x": x, "y": y}

        def pass_to(x: float, y: float) -> Reply:
            return {"action": "pass", "x": x, "y": y}

        if me["has_ball"]:
            if me["keeper"]:
                outfield = [p for p in mates if not p["keeper"]]
                tgt = max(outfield, key=lambda p: p["pos"][0])
                return pass_to(*tgt["pos"])
            if math.dist(me["pos"], (W, H / 2)) < self.shoot_range:
                keeper = next(p for p in opps if p["keeper"])
                aim_y = cfg.goal_y0 + 1.5 if keeper["pos"][1] > H / 2 else cfg.goal_y1 - 1.5
                return pass_to(W, aim_y)
            blocked = any(0 < p["pos"][0] - mx < 8 and abs(p["pos"][1] - my) < 4 for p in opps)
            if blocked:
                ahead = [p for p in mates if not p["keeper"] and p["pos"][0] > mx - 5]
                if ahead:
                    return pass_to(*max(ahead, key=lambda p: p["pos"][0])["pos"])
                return move(mx + 4, my + (6 if my < H / 2 else -6))
            return move(W, H / 2)

        if me["keeper"]:
            if ball["holder"] is None and bx < cfg.box_depth and cfg.box_y0 <= by <= cfg.box_y1:
                return move(bx, by)
            return move(2.0, min(max(by, cfg.goal_y0), cfg.goal_y1))

        holder = next((p for p in players if p["id"] == ball["holder"]), None)
        if holder and holder["ours"]:
            side = 1 if int(me["id"][1:]) % 2 else -1
            return move(min(holder["pos"][0] + 15, W - 8), H / 2 + side * 12)
        if holder:
            # Predict where the ball will be next tick (it sits in front of the carrier).
            # The nearest defender attacks a point just ahead of it, so it arrives from the
            # front instead of bumping into the carrier's back; the rest block the line
            # between the carrier and our goal.
            hx, hy = holder["pos"][0] + holder["vel"][0], holder["pos"][1] + holder["vel"][1]
            fx, fy = holder["facing"]
            pbx, pby = bx + holder["vel"][0], by + holder["vel"][1]
            chaser = min((p for p in players if p["ours"] and not p["keeper"]),
                         key=lambda p: math.dist(p["pos"], ball["pos"]))
            if chaser["you"]:
                return move(pbx + fx * cfg.player_radius, pby + fy * cfg.player_radius)
            gx, gy = 0.0 - hx, H / 2 - hy
            d = math.hypot(gx, gy) or 1.0
            return move(hx + gx / d * 5, hy + gy / d * 5)

        # Free ball: the nearest outfield player intercepts, the rest hold shape.
        vx, vy = ball["vel"]
        nearest = min((p for p in players if p["ours"] and not p["keeper"]),
                      key=lambda p: math.dist(p["pos"], ball["pos"]))
        if nearest["you"]:
            for k in range(1, 8):
                px = min(max(bx + vx * k, 0), W)
                py = min(max(by + vy * k, 0), H)
                if math.dist((px, py), (mx, my)) <= cfg.player_speed * k:
                    return move(px, py)
            return move(bx, by)
        return move(max(bx - 10, 10.0), H / 2)
