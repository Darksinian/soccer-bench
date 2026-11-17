"""Trusted re-simulation of a logged match: re-scores a game without trusting its log's score.

Every game log records each player's exact action per tick. The verifier rebuilds the match
from the logged config and seed, applies the logged actions, and scores the result. Ticks that
were never logged count as every player standing still. Malformed actions count as "stay".
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, TypedDict

from .sim import Action, Config, Sim


class Verdict(TypedDict):
    score: list[int]      # [team A, team B]
    ticks_played: int
    ticks_logged: int


def _logged_action(entry: object) -> Action:
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        return Action("stay")
    kind, target = entry
    if kind not in ("move", "pass") or not isinstance(target, (list, tuple)) or len(target) < 2:
        return Action("stay")
    try:
        x, y = float(target[0]), float(target[1])
    except (TypeError, ValueError):
        return Action("stay")
    if not (math.isfinite(x) and math.isfinite(y)):
        return Action("stay")
    return Action("move" if kind == "move" else "pass", (x, y))


def replay(config: dict[str, Any], seed: int, logged_ticks: Sequence[Sequence[object] | None]) -> Verdict:
    """config and seed as logged; logged_ticks: per tick, a list over all players of
    [kind, [x, y] | None] in the world frame (a game log's `exact_actions`)."""
    sim = Sim(Config(**config), seed)
    played = 0
    while not sim.done:
        logged = logged_ticks[played] if played < len(logged_ticks) else None
        sim.step({g: _logged_action(logged[g]) if logged and g < len(logged) else Action("stay")
                  for g in range(len(sim.players))})
        played += 1
    return {"score": sim.score[:], "ticks_played": sim.tick, "ticks_logged": min(len(logged_ticks), sim.tick)}
