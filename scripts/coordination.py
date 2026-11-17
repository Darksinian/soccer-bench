"""Coordination patterns per model, from every game on the leaderboard.

  python scripts/coordination.py            # prints the tables and writes results/coordination.json

Movement (measured on the snapshot each decision was made from):
  spacing        mean distance between a team's 4 outfield players
  double_press   share of opponent-possession ticks where 2+ of our outfield players head for
                 the ball (move target within 4 units of it)

What the shouts say is hand-labelled, not keyword-matched: see scripts/talk.py.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccer_eval.games import RESULTS, Game, load_games, name  # noqa: E402
from soccer_eval.schema import MatchResult, Point, Side  # noqa: E402



@dataclass
class Counts:
    ticks: int = 0
    spacing: float = 0.0
    opp_ticks: int = 0
    double_press: int = 0


@dataclass(frozen=True)
class Coordination:
    spacing: float
    double_press: float

    @staticmethod
    def of(c: Counts) -> Coordination:
        def ratio(a: float, b: float) -> float:
            return a / b if b else 0.0
        return Coordination(
            spacing=ratio(c.spacing, c.ticks), double_press=ratio(c.double_press, c.opp_ticks),
        )


def analyse(games: Sequence[Game]) -> dict[str, Coordination]:
    """Metrics per model."""
    return analyse_keyed([(g.match, g.models) for g in games])


def analyse_keyed(matches: Sequence[tuple[MatchResult, dict[Side, str]]]) -> dict[str, Coordination]:
    """Metrics per key, where each match says which key each side counts toward (a model, or
    an experiment condition)."""
    acc: defaultdict[str, Counts] = defaultdict(Counts)
    for m, keys in matches:
        if m.get("stopped_early"):
            continue
        n = int(m["config"]["n_per_team"])
        log = m["log"]
        prev_frame = log[0]["frames"][-1] if log[0]["tick"] == -1 else log[0]["frames"][0]
        for t in log:
            if t["tick"] < 0:
                prev_frame = t["frames"][-1]
                continue
            acts = {a["player"]: a for a in t["actions"]}
            ball = prev_frame["b"]
            holder = prev_frame["h"]
            for team in ("A", "B"):
                c = acc[keys[team]]
                base = 0 if team == "A" else n
                out_idx = list(range(base + 1, base + n))
                pts = [prev_frame["p"][i] for i in out_idx]
                c.ticks += 1
                c.spacing += sum(math.dist(pts[i], pts[j]) for i in range(len(pts))
                                 for j in range(i + 1, len(pts))) / 6
                targets: list[Point] = []
                for i in out_idx:
                    a = acts.get(f"{team}{i - base}")
                    target = a.get("target") if a else None
                    if a and a.get("action") == "move" and target:
                        targets.append(target)
                opp_has = holder >= 0 and ((holder >= n) == (team == "A"))
                if opp_has:
                    c.opp_ticks += 1
                    if sum(1 for tg in targets if math.dist(tg, ball) < 4) >= 2:
                        c.double_press += 1
            prev_frame = t["frames"][-1]
    return {k: Coordination.of(c) for k, c in acc.items()}


def main() -> None:
    games = [g for g in load_games() if g.counts]
    res = analyse(games)
    order = sorted(res, key=lambda k: -res[k].spacing)
    print(f"{len(games)} games\n")
    print(f"{'model':18} {'spacing':>7} {'2-press':>7}")
    for k in order:
        r = res[k]
        print(f"{name(k):18} {r.spacing:7.1f} {r.double_press:7.0%}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "coordination.json").write_text(json.dumps({k: {"name": name(k), **asdict(v)} for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
