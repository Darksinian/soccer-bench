"""Model players that call Anthropic or OpenAI directly, with ANTHROPIC_API_KEY /
OPENAI_API_KEY instead of an OpenRouter key (`run.py --backend direct`).

Spec: "anthropic/<model-id>" or "openai/<model-id>", optionally "@<effort>". The provider
prefix picks the API; the rest is passed through as the model ID.

Refusals are not retried on a fallback model: a fallback would put a different model on the
pitch, so a refused turn simply counts as "stay" (logged as an error).
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any

from .agents import FatalAPIError, parse_action
from .prompts import system_prompt, user_prompt
from .schema import CallMeta, Observation, Reply
from .sim import Config

# USD per million input / output tokens, for the cost column (list prices; cache discounts ignored).
PRICES = {
    "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.0, 50.0), "claude-fable-5-2": (8.0, 40.0), "claude-opus-4-8": (5.0, 25.0),
    "gpt-6.1-sol": (2.0, 10.0), "gpt-6-luna": (0.10, 0.50), "gpt-6-astra": (10.0, 50.0), "gpt-5.5": (5.0, 30.0),
    "gpt-5.6-sol": (4.40, 22.0),
}


def cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return (tokens_in * p_in + tokens_out * p_out) / 1e6


_clients: dict[str, Any] = {}
_lock = threading.Lock()


def _client(provider: str) -> Any:
    with _lock:
        if provider not in _clients:
            if provider == "anthropic":
                import anthropic
                _clients[provider] = anthropic.Anthropic(max_retries=4)
            else:
                import openai
                _clients[provider] = openai.OpenAI(max_retries=4)
        return _clients[provider]


class DirectAgent:
    def __init__(self, cfg: Config, spec: str):
        model, _, effort = spec.partition("@")
        self.provider, _, self.model = model.partition("/")
        if self.provider not in ("anthropic", "openai"):
            raise ValueError(f"direct backend supports anthropic/... and openai/... models, got {spec!r}")
        self.effort = effort or None
        self.label, self.roles = spec, cfg.roles
        self.system = system_prompt(cfg)

    async def act(self, obs: Observation) -> tuple[Reply, CallMeta]:
        call = self._anthropic if self.provider == "anthropic" else self._openai
        text, meta = await asyncio.to_thread(call, user_prompt(obs, self.roles))
        meta["raw"] = text[:500]
        meta["cost"] = cost(self.model, meta["in"], meta["out"])
        return parse_action(text), meta

    def _anthropic(self, user: str) -> tuple[str, CallMeta]:
        import anthropic
        kwargs: dict[str, Any] = {}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            # Streaming keeps long reasoning turns clear of HTTP timeouts.
            with _client("anthropic").messages.stream(
                model=self.model,
                max_tokens=64000,
                system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                **kwargs,
            ) as stream:
                msg = stream.get_final_message()
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
            raise FatalAPIError(f"Anthropic {type(e).__name__} for {self.model}: {e}") from e
        except anthropic.BadRequestError as e:
            if "credit" in str(e).lower():
                raise FatalAPIError(f"Anthropic billing error for {self.model}: {e}") from e
            raise
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"refusal ({getattr(msg.stop_details, 'category', None)})")
        text = "".join(b.text for b in msg.content if b.type == "text")
        u = msg.usage
        return text, {"model": msg.model, "in": u.input_tokens + (u.cache_read_input_tokens or 0)
                      + (u.cache_creation_input_tokens or 0), "out": u.output_tokens, "cost": 0.0}

    def _openai(self, user: str) -> tuple[str, CallMeta]:
        import openai
        kwargs: dict[str, Any] = {}
        if self.effort:
            kwargs["reasoning_effort"] = self.effort
        try:
            resp = _client("openai").chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                **kwargs,
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as e:
            raise FatalAPIError(f"OpenAI {type(e).__name__} for {self.model}: {e}") from e
        except openai.RateLimitError as e:
            # Out of credit comes back as a 429 too; retrying won't help, so stop the match.
            if any(k in str(e).lower() for k in ("insufficient_quota", "no credits", "credit", "billing", "quota")):
                raise FatalAPIError(f"OpenAI quota exhausted for {self.model}: {e}") from e
            raise
        text = resp.choices[0].message.content or ""
        u = resp.usage
        return text, {"model": resp.model, "in": u.prompt_tokens if u else 0,
                      "out": u.completion_tokens if u else 0, "cost": 0.0}
