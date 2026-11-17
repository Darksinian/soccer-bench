"""Compare the two conditions of each self-play experiment in runs/experiments/<name>/.

  python scripts/experiments.py        # prints one table per experiment, writes results/experiments.json

Each experiment is one model against itself where only team1 gets the treatment (shouting, or the
formation prompt); teams swap sides every game. Metrics are the ones in coordination.py and
team_play.py, keyed by condition instead of model. Later batches of the same experiment go in
<name>-2/, <name>-3/, ... and are pooled with <name>/.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import coordination  # noqa: E402
import team_play  # noqa: E402
from soccer_eval.games import RESULTS  # noqa: E402
from soccer_eval.schema import MatchResult, SeriesSummary, SeriesTotals, Side  # noqa: E402

EXPERIMENTS = ROOT / "runs" / "experiments"


@dataclass
class Record:
    """One side's results, summed over every batch of an experiment."""
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    cost: float = 0.0

    def add(self, t: SeriesTotals) -> None:
        self.wins += t["wins"]
        self.draws += t["draws"]
        self.losses += t["losses"]
        self.goals_for += t["goals_for"]
        self.goals_against += t["goals_against"]
        self.cost += t["cost"]


@dataclass(frozen=True)
class Condition:
    record: str        # W-D-L
    goals: str         # for:against
    cost: float
    double_press: float
    spacing: float
    rest_defence: float


@dataclass(frozen=True)
class Experiment:
    model: str
    games: int
    conditions: dict[str, Condition]


def condition(summary: SeriesSummary, key: str) -> str:
    """The treatment `key` ("team1" or "team2") played under."""
    if summary.get("roles", {}).get(key):
        return "roles"
    if summary["comms"]["team1"] != summary["comms"]["team2"]:
        return "talk" if summary["comms"][key] != "none" else "silent"
    return "no roles"


def experiment(dirs: list[Path]) -> Experiment:
    games: list[tuple[MatchResult, dict[Side, str]]] = []
    records: defaultdict[str, Record] = defaultdict(Record)
    model = ""
    for d in dirs:
        summary: SeriesSummary = json.loads((d / "summary.json").read_text())
        model = summary["teams"]["team1"]
        for g in summary["games"]:
            m: MatchResult = json.loads((d / f"game_{g['game']:02d}.json").read_text())
            games.append((m, {"A": condition(summary, g["A"]), "B": condition(summary, g["B"])}))
        for key in ("team1", "team2"):
            records[condition(summary, key)].add(summary["totals"][key])
    coord = coordination.analyse_keyed(games)
    play = team_play.analyse_keyed(games)
    out = {}
    for c, r in records.items():
        out[c] = Condition(
            record=f"{r.wins}-{r.draws}-{r.losses}", goals=f"{r.goals_for}:{r.goals_against}",
            cost=round(r.cost, 2), double_press=coord[c].double_press, spacing=coord[c].spacing,
            rest_defence=play[c].rest_defence)
    return Experiment(model, len(games), out)


def main() -> None:
    batches: defaultdict[str, list[Path]] = defaultdict(list)
    for d in sorted(EXPERIMENTS.iterdir()):
        if (d / "summary.json").exists():
            batches[re.sub(r"-\d+$", "", d.name)].append(d)
    res = {}
    for name, dirs in batches.items():
        res[name] = r = experiment(dirs)
        print(f"\n{name}  ({r.model}, {r.games} games)")
        print(f"  {'':9} {'W-D-L':>7} {'goals':>6} {'2+chase':>7} {'spacing':>7} {'rest':>5} {'cost':>6}")
        for c, v in r.conditions.items():
            print(f"  {c:9} {v.record:>7} {v.goals:>6} {v.double_press:7.0%} {v.spacing:7.1f} {v.rest_defence:5.0%} {v.cost:6.2f}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "experiments.json").write_text(json.dumps({k: asdict(v) for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
