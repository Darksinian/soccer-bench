"""What the shouts say, from hand-labelled random samples of the leaderboard games.

  python scripts/talk.py sample     # draws any sample not drawn yet into results/labels/ (unlabelled)
  python scripts/talk.py            # reads the labels, prints the table, writes results/talk.json

No keyword rules: every shout in a sample was read and labelled by hand, with the model's name
hidden and the items shuffled. The label files keep each item's game, tick and player, so any
label can be checked against the replay. Samples are drawn with a fixed seed.

  style.jsonl   SAMPLE_STYLE random outfield shouts per model. Labels (any of):
                  directs     tells a teammate to do something ("B3 hold the middle", "A2, pass to me")
                  announces   says what the speaker itself is doing or will do ("Holding far post")
                  asks        asks for the ball ("open for the pass", "feed me at (70,30)")
  asked.jsonl   SAMPLE_ASKED random passes per model aimed at a teammate (target within 6 units of
                where it stood), with that teammate's shout from the tick before, which is the one
                the passer heard. Labels (one or neither):
                  requests    asks for the pass outright ("A3 pass to me", "lead me toward (70,20)")
                  offers      only says it is open or an outlet ("open at (60,24)", "wide outlet")
                No shout counts as neither.
  home.jsonl    SAMPLE_HOME random attacking ticks per model with an outfield player 20+ units behind
                the ball, with the deepest such player's shout that tick. Label: home (does it say it
                is staying back, covering or guarding against a counter?).
  runs.jsonl    every through ball into a run (a pass aimed 5+ units from the nearest teammate, not
                at the goal mouth, while that teammate moves to within 6 units of the target), with
                the receiver's shout from the tick before. Label: announced (did it say it was
                running to or open at that spot, or ask for the ball there?).
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccer_eval.games import RESULTS, Game, load_games, name  # noqa: E402
from soccer_eval.schema import Side  # noqa: E402

LABELS = RESULTS / "labels"
SAMPLE_STYLE, SAMPLE_ASKED, SAMPLE_HOME, SAMPLE_RUNS = 100, 60, 40, 10_000   # runs: all of them
TASKS = {"style": ("directs", "announces", "asks"), "asked": ("requests", "offers"), "home": ("home",),
         "runs": ("announced",)}


@dataclass
class Item:
    id: str
    model: str
    game: str
    tick: int
    player: str
    shout: str
    labels: list[str] = field(default_factory=list)
    labelled: bool = False


def population(games: list[Game]) -> dict[str, defaultdict[str, list[Item]]]:
    """Every candidate item per task and model, in a fixed order."""
    pop: dict[str, defaultdict[str, list[Item]]] = {t: defaultdict(list) for t in TASKS}
    for g in games:
        m, log = g.match, g.match["log"]
        n = int(m["config"]["n_per_team"])

        def gid(p: str) -> int:
            return (0 if p[0] == "A" else n) + int(p[1:])

        for i, t in enumerate(log):
            if t["tick"] < 0:
                continue
            acts = {a["player"]: a for a in t["actions"]}
            heard = {a["player"]: a.get("say", "").strip() for a in log[i - 1]["actions"]} if i else {}
            prev = log[i - 1]["frames"][-1] if i else t["frames"][0]
            for a in t["actions"]:
                side: Side = "A" if a["player"][0] == "A" else "B"
                model, say = g.model(side), a.get("say", "").strip()
                if say and not a["player"].endswith("0"):
                    pop["style"][model].append(Item("", model, g.id, t["tick"], a["player"], say))
                tgt = a.get("target")
                if a.get("action") == "pass" and tgt and i:
                    base = 0 if side == "A" else n
                    mates = [base + k for k in range(n) if base + k != gid(a["player"])]
                    r = min(mates, key=lambda q: math.dist(prev["p"][q], tgt))
                    recv = f"{side}{r - base}"
                    if math.dist(prev["p"][r], tgt) < 6:
                        pop["asked"][model].append(Item("", model, g.id, t["tick"], recv, heard.get(recv, "")))
                    gx = float(m["config"]["width"]) if side == "A" else 0.0
                    at_goal = abs(tgt[0] - gx) <= 2 and 23.2 <= tgt[1] <= 36.8
                    ra = acts.get(recv)
                    rt = ra.get("target") if ra else None
                    if (not at_goal and math.dist(prev["p"][r], tgt) >= 5 and ra and ra.get("action") == "move"
                            and rt and math.dist(rt, tgt) < 6):
                        pop["runs"][model].append(Item("", model, g.id, t["tick"], recv, heard.get(recv, "")))
            h, bx = prev["h"], prev["b"][0]
            for side in ("A", "B"):
                base, f = (0 if side == "A" else n), (1 if side == "A" else -1)
                if h < 0 or (h < n) != (side == "A"):
                    continue
                behind = {k: bx * f - prev["p"][base + k][0] * f for k in range(1, n)}
                k = max(behind, key=lambda q: behind[q])
                if behind[k] >= 20:
                    p = f"{side}{k}"
                    say = acts[p].get("say", "").strip() if p in acts else ""
                    pop["home"][g.model(side)].append(Item("", g.model(side), g.id, t["tick"], p, say))
    return pop


def sample() -> None:
    LABELS.mkdir(parents=True, exist_ok=True)
    pop = population([g for g in load_games() if g.counts])
    rng = random.Random(0)
    sizes = {"style": SAMPLE_STYLE, "asked": SAMPLE_ASKED, "home": SAMPLE_HOME, "runs": SAMPLE_RUNS}
    for task, by_model in pop.items():
        items = [it for model in sorted(by_model) for it in rng.sample(by_model[model], min(sizes[task], len(by_model[model])))]
        rng.shuffle(items)
        for k, it in enumerate(items):
            it.id = f"{task}-{k:04d}"
            if not it.shout:      # nothing said: nothing to label
                it.labelled = True
        path = LABELS / f"{task}.jsonl"
        if path.exists():       # drawn already (the draw above still runs, so later samples don't change)
            print(f"{task}: {path} exists; delete it to draw a new sample")
            continue
        path.write_text("".join(json.dumps(asdict(it)) + "\n" for it in items))
        print(f"{task}: {len(items)} items -> {path}")


@dataclass(frozen=True)
class Share:
    value: float
    low: float          # 95% Wilson interval
    high: float
    n: int


def wilson(k: int, n: int, z: float = 1.96) -> Share:
    if not n:
        return Share(0.0, 0.0, 0.0, 0)
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return Share(p, max(0.0, mid - half), min(1.0, mid + half), n)


def report() -> None:
    out: dict[str, dict[str, Share]] = defaultdict(dict)
    for task, labels in TASKS.items():
        items = [Item(**json.loads(line)) for line in (LABELS / f"{task}.jsonl").read_text().splitlines()]
        if missing := [it.id for it in items if not it.labelled]:
            sys.exit(f"{task}: {len(missing)} items not labelled yet, e.g. {missing[0]}")
        by_model: defaultdict[str, list[Item]] = defaultdict(list)
        for it in items:
            by_model[it.model].append(it)
        for model, its in by_model.items():
            for lab in labels:
                out[model][f"{task}.{lab}" if len(labels) > 1 else task] = wilson(sum(lab in it.labels for it in its), len(its))
            if task == "asked":
                out[model]["asked.either"] = wilson(sum(bool(it.labels) for it in its), len(its))
    order = [r["model"] for r in json.loads((RESULTS / "summary.json").read_text())["standings"]]
    cols = [f"style.{lab}" for lab in TASKS["style"]] + ["asked.requests", "asked.either", "home", "runs"]

    def cell(s: Share | None) -> str:
        return (f"{s.value:6.0%} ({s.low:3.0%}-{s.high:3.0%})" if s else "-").rjust(15)
    print(f"{'model':18} " + " ".join(f"{c:>15}" for c in cols))
    for m in order:
        print(f"{name(m):18} " + " ".join(cell(out[m].get(c)) for c in cols))
    (RESULTS / "talk.json").write_text(json.dumps({m: {"name": name(m), **{c: asdict(s) for c, s in out[m].items()}} for m in order}, indent=2))


if __name__ == "__main__":
    sample() if sys.argv[1:] == ["sample"] else report()
