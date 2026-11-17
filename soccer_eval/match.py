"""Runs one match: gathers every player's action each tick in parallel, then steps the sim."""
from __future__ import annotations

import asyncio
import dataclasses
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from .agents import FatalAPIError, make_agent
from .schema import ActionRecord, BySide, MatchResult, Reply, ShoutObs, TeamStats, TickRecord, empty_stats
from .sim import TEAM_NAMES, Action, Config, Sim, mirror


def expand_team(spec: str | list[str], n: int) -> list[str]:
    """A team is one spec for every player, or a comma-separated list with one per player
    (index 0 is the goalkeeper)."""
    specs = spec if isinstance(spec, list) else [s.strip() for s in spec.split(",")]
    if len(specs) == 1:
        specs = specs * n
    if len(specs) != n:
        raise ValueError(f"team spec has {len(specs)} players, expected 1 or {n}: {spec!r}")
    return specs


def clean_shout(cfg: Config, data: Reply) -> str:
    say = data.get("say")
    if not isinstance(say, str):
        return ""
    return " ".join(say.split())[:cfg.shout_max_chars]


def heard_by(sim: Sim, g: int, shouts: dict[int, str], modes: tuple[str, str]) -> list[ShoutObs]:
    """Last tick's shouts that player g can hear (its own included, as a reminder). A team's
    shouts reach its own players, and the opponents too when that team's mode is "all"."""
    me = sim.players[g]
    out: list[ShoutObs] = []
    for k, text in sorted(shouts.items()):
        p = sim.players[k]
        if p.team != me.team and modes[p.team] != "all":
            continue
        out.append({"id": p.name, "ours": p.team == me.team, "you": k == g, "text": text})
    return out


def _number(v: object) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise TypeError(f"not a number: {v!r}")
    return float(v)


def to_action(cfg: Config, team: int, data: Reply) -> Action:
    kind = data.get("action")
    if kind == "stay":
        return Action("stay")
    if kind != "move" and kind != "pass":
        raise ValueError(f"unknown action {kind!r}")
    x, y = _number(data["x"]), _number(data["y"])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("non-finite target")
    return Action(kind, mirror(cfg, team, (x, y)))


async def play_match(cfg: Config, team_specs: tuple[str | list[str], str | list[str]], seed: int = 0,
                     call_timeout: float = 300.0, verbose: bool = False, comms: tuple[str, str] | None = None,
                     roles: tuple[bool, bool] = (False, False), label: str = "", own_executor: bool = True,
                     backend: str = "openrouter",
                     on_tick: Callable[[TickRecord, Sim], None] | None = None) -> MatchResult:
    """`comms` sets each side's shouting mode ("none", "team" or "all"); it defaults to cfg.comms
    for both. `roles` turns on the formation prompt per side. `on_tick(record, sim)` is called
    after every tick (e.g. to save progress). Pass own_executor=False when the caller runs several
    matches on one event loop and has already sized the thread pool for all of them."""
    sim = Sim(cfg, seed)
    modes = comms or (cfg.comms, cfg.comms)
    team_cfgs = [dataclasses.replace(cfg, comms=modes[t], roles=roles[t]) for t in (0, 1)]
    specs = [expand_team(team_specs[t], cfg.n_per_team) for t in (0, 1)]
    agents = [make_agent(specs[p.team][p.idx], team_cfgs[p.team], seed * 1000 + g, backend)
              for g, p in enumerate(sim.players)]
    stats: BySide[TeamStats] = {"A": empty_stats(), "B": empty_stats()}
    shouts: dict[int, str] = {}  # made last tick, heard this tick

    async def decide(g: int) -> tuple[Action, ActionRecord]:
        p = sim.players[g]
        st = stats[TEAM_NAMES[p.team]]
        t0 = time.monotonic()
        rec: ActionRecord = {"player": p.name, "agent": agents[g].label}
        obs = sim.observe(g)
        if modes[p.team] != "none" or modes[1 - p.team] == "all":
            obs["shouts"] = heard_by(sim, g, shouts, modes)
        try:
            data, meta = await asyncio.wait_for(agents[g].act(obs), call_timeout)
            # Copy the call's usage onto the log record (bots report none).
            if "model" in meta:
                rec["model"] = meta["model"]
            if "in" in meta:
                rec["in"] = meta["in"]
            if "out" in meta:
                rec["out"] = meta["out"]
            if "cost" in meta:
                rec["cost"] = meta["cost"]
            if "raw" in meta:
                rec["raw"] = meta["raw"]
            st["in"] += meta.get("in", 0)
            st["out"] += meta.get("out", 0)
            st["cost"] += meta.get("cost", 0.0)
        except FatalAPIError:
            st["errors"] += 1
            raise
        except Exception as e:  # a transient failure is a "stay", never a crash
            st["errors"] += 1
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            return Action("stay"), rec
        finally:
            st["calls"] += 1
            st["seconds"] += time.monotonic() - t0
        if modes[p.team] != "none" and isinstance(data, dict):
            say = clean_shout(cfg, data)
            if say:
                rec["say"] = say
                st["shouts"] += 1
        try:
            action = to_action(cfg, p.team, data)
        except (ValueError, KeyError, TypeError) as e:
            st["invalid"] += 1
            rec["error"] = f"invalid action {data!r}: {e}"[:300]
            return Action("stay"), rec
        rec["action"] = action.kind
        rec["target"] = None if action.target is None else [round(c, 2) for c in action.target]
        return action, rec

    # Every player's call for a tick runs at once. Model calls are blocking HTTP in threads,
    # so size the pool to the player count rather than the CPU-based default.
    if own_executor:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=len(sim.players)))

    ticks: list[TickRecord] = [{"tick": -1, "events": [], "frames": [sim.frame()], "score": sim.score[:], "actions": []}]
    while not sim.done:
        t0 = time.monotonic()
        decided = await asyncio.gather(*(decide(g) for g in range(len(sim.players))))
        wall = time.monotonic() - t0
        record = sim.step({g: a for g, (a, _) in enumerate(decided)})
        record["actions"] = [rec for _, rec in decided]
        record["wall_seconds"] = round(wall, 2)
        # Unrounded world-frame actions, so a verifier can replay the match exactly.
        record["exact_actions"] = [(a.kind, None if a.target is None else list(a.target)) for a, _ in decided]
        shouts = {g: rec["say"] for g, (_, rec) in enumerate(decided) if rec.get("say")}
        for e in record["events"]:
            player = e.get("player")
            if e["type"] == "invalid" and player:
                stats["A" if player[0] == "A" else "B"]["invalid"] += 1
        ticks.append(record)
        if on_tick is not None:
            on_tick(record, sim)
        if verbose:
            shown = [e for e in record["events"] if e["type"] in ("goal", "tackle", "interception")]
            if shown or record["tick"] % 10 == 0:
                print(f"  {label}tick {record['tick']:3d}  score {sim.score[0]}-{sim.score[1]}  "
                      + " ".join(f"{e['type']}:{e.get('player')}" for e in shown), flush=True)

    a, b = sim.score
    return {
        "config": cfg.to_dict(),
        "seed": seed,
        "teams": {"A": specs[0], "B": specs[1]},
        "comms": {"A": modes[0], "B": modes[1]},
        "roles": {"A": roles[0], "B": roles[1]},
        "score": {"A": a, "B": b},
        "winner": "A" if a > b else "B" if b > a else None,
        "ticks_played": sim.tick,
        "stats": stats,
        "log": ticks,
    }
