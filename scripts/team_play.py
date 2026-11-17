"""Team-play metrics per model, from every game on the leaderboard.

  python scripts/team_play.py        # prints the table and writes results/team_play.json

Each tick's decisions are made from the last frame of the previous tick.

  rest_defence  share of the team's possession ticks with an outfield player 20+ units behind the ball
  counter_goals goals conceded within 5 ticks of losing the ball (interception or tackle) when, just
                before losing it, no outfield player was 15+ units behind the ball

What the shouts say is hand-labelled, not keyword-matched: see scripts/talk.py.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccer_eval.games import RESULTS, Game, load_games  # noqa: E402
from soccer_eval.schema import MatchResult, Side  # noqa: E402


@dataclass
class Counts:
    """Running totals for one team (a model, or an experiment condition)."""
    poss: int = 0
    rest: int = 0
    counter: int = 0


@dataclass(frozen=True)
class TeamPlay:
    rest_defence: float
    counter_goals: int

    @staticmethod
    def of(c: Counts) -> TeamPlay:
        return TeamPlay(rest_defence=c.rest / c.poss if c.poss else 0.0, counter_goals=c.counter)


def analyse(games: Sequence[Game]) -> dict[str, TeamPlay]:
    """Metrics per model."""
    return analyse_keyed([(g.match, g.models) for g in games])


def analyse_keyed(matches: Sequence[tuple[MatchResult, dict[Side, str]]]) -> dict[str, TeamPlay]:
    """Metrics per key, where each match says which key each side counts toward (a model, or
    an experiment condition)."""
    acc: defaultdict[str, Counts] = defaultdict(Counts)
    for m, keys in matches:
        cfg, log = m["config"], m["log"]
        n = int(cfg["n_per_team"])
        for i in range(1, len(log)):
            prev = log[i - 1]["frames"][-1]
            h, bx = prev["h"], prev["b"][0]
            for side in ("A", "B"):
                base, f = (0 if side == "A" else n), (1 if side == "A" else -1)   # A attacks +x
                if h >= 0 and (h < n) == (side == "A"):
                    me = acc[keys[side]]
                    me.poss += 1
                    me.rest += any(bx * f - prev["p"][base + k][0] * f >= 20 for k in range(1, n))
            for e in log[i]["events"]:
                if e["type"] != "goal":
                    continue
                scorer = e["team"]
                conceder: Side = "B" if scorer == "A" else "A"
                for j in range(i, max(1, i - 5) - 1, -1):           # the scorer's last turnover, 5 ticks back at most
                    if any(x["type"] in ("interception", "tackle") and (x.get("player") or "").startswith(scorer) for x in log[j]["events"]):
                        fr = log[j - 1]["frames"][-1]
                        base, f = (0 if conceder == "A" else n), (1 if conceder == "A" else -1)
                        acc[keys[conceder]].counter += not any(fr["b"][0] * f - fr["p"][base + k][0] * f >= 15
                                                               for k in range(1, n))
                        break
    return {k: TeamPlay.of(c) for k, c in acc.items()}


def main() -> None:
    games: list[Game] = [g for g in load_games() if g.counts]
    res = analyse(games)
    order = [r["model"] for r in json.loads((RESULTS / "summary.json").read_text())["standings"]]
    print(f"{'model':28} {'rest':>5} {'counter':>7}")
    for k in order:
        r = res[k]
        print(f"{k:28} {r.rest_defence:5.0%} {r.counter_goals:7}")
    (RESULTS / "team_play.json").write_text(json.dumps({k: asdict(v) for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
