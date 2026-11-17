"""Builds the leaderboard from every game in runs/leaderboard/.

  python scripts/leaderboard.py

Each game is re-scored by replaying its logged actions (soccer_eval/verify.py) and flagged if
that disagrees with the live score. A game where more than 5% of one team's calls failed is
left out and listed as excluded.

Writes to results/:
  games.csv        one row per game
  standings.md     ratings (Bradley-Terry, Elo scale, 95% bootstrap CI), record, play stats
  h2h.md           head-to-head score matrix
  ratings.svg      rating chart with confidence intervals
  shouts.csv       every shout: game, tick, team, player, model, action, score at the time
  shouts.jsonl     the same, one JSON object per line
  summary.json     standings and games, for the site
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, astuple, dataclass, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccer_eval.games import LEADERBOARD, MAX_FAILURE_RATE, RESULTS, Game, load_games, name  # noqa: E402
from soccer_eval.schema import Point, Side  # noqa: E402
from soccer_eval.verify import replay  # noqa: E402

# (model a, model b, a's score: 1 win / 0.5 draw / 0 loss)
Result = tuple[str, str, float]


@dataclass
class Totals:
    games: int = 0
    w: int = 0
    d: int = 0
    l: int = 0  # noqa: E741
    gf: int = 0
    ga: int = 0
    calls: int = 0
    errors: int = 0
    invalid: int = 0
    shouts: int = 0
    cost: float = 0.0
    passes: int = 0
    completed: int = 0
    won_ball: int = 0


@dataclass(frozen=True)
class Standing:
    model: str
    name: str
    rating: float
    ci: tuple[float, float]
    w: int
    d: int
    l: int  # noqa: E741
    gf: int
    ga: int
    passes: int
    completed: int
    pass_pct: float
    won_ball: int
    games: int
    shouts_per_call: float
    cost_per_game: float
    errors: int
    calls: int


@dataclass(frozen=True)
class GameRow:
    id: str
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    ticks: int
    ran_on: str
    verified: bool


@dataclass(frozen=True)
class Shout:
    game: str
    tick: int
    team: str
    player: str
    model: str
    action: str | None
    target: Point | None
    shout: str
    score_a: int
    score_b: int


def verified(g: Game) -> bool:
    """Replay both teams' logged actions from the logged config and seed; True if the score matches."""
    m = g.match
    r = replay(m["config"], m["seed"], [t.get("exact_actions") for t in m["log"] if t["tick"] >= 0])
    return r["score"] == [m["score"]["A"], m["score"]["B"]]


# ----- ratings -----

def bradley_terry(results: list[Result], players: list[str], iters: int = 500) -> dict[str, float]:
    """Every pair also gets one virtual draw, which keeps ratings finite for a model that never
    won. Returns Elo-scale ratings centred on 1000."""
    wins: defaultdict[str, float] = defaultdict(float)
    n: defaultdict[tuple[str, str], float] = defaultdict(float)
    for a, b, s in results:
        wins[a] += s
        wins[b] += 1 - s
        n[(a, b)] += 1
        n[(b, a)] += 1
    for i, a in enumerate(players):
        for b in players[i + 1:]:
            wins[a] += 0.5
            wins[b] += 0.5
            n[(a, b)] += 1
            n[(b, a)] += 1
    p = {x: 1.0 for x in players}
    for _ in range(iters):
        new = {}
        for a in players:
            denom = sum(n[(a, b)] / (p[a] + p[b]) for b in players if b != a and n[(a, b)])
            new[a] = wins[a] / denom if denom else p[a]
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        p = {k: v / g for k, v in new.items()}
    return {k: 1000 + 400 * math.log10(v) for k, v in p.items()}


def bootstrap(results: list[Result], players: list[str], reps: int = 1000, seed: int = 0) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    samples: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(reps):
        boot = [results[rng.randrange(len(results))] for _ in results]
        for k, v in bradley_terry(boot, players, iters=200).items():
            samples[k].append(v)
    out = {}
    for k, vals in samples.items():
        vals.sort()
        out[k] = (vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1])
    return out


# ----- charts (static SVG for the README; native tooltips via <title>) -----

SVG_STYLE = """<style>
  .bg { fill: #fcfcfb; } .ink { fill: #0b0b0b; } .ink2 { fill: #52514e; }
  .grid { stroke: #e4e3df; stroke-width: 1; } .axis { stroke: #b5b3ad; stroke-width: 1; }
  .mark { fill: #2a78d6; } .whisker { stroke: #2a78d6; stroke-width: 2; stroke-linecap: round; }
  .ring { stroke: #fcfcfb; stroke-width: 2; }
  text { font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; } .ink { fill: #f0efec; } .ink2 { fill: #c3c2b7; }
    .grid { stroke: #2e2e2c; } .axis { stroke: #555450; }
    .mark { fill: #4c95ec; } .whisker { stroke: #4c95ec; } .ring { stroke: #1a1a19; }
  }
</style>"""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_ticks(lo: float, hi: float, step: float) -> list[float]:
    ticks = []
    v = math.floor(lo / step) * step
    while v <= hi + 1e-9:
        ticks.append(v)
        v += step
    return ticks


def ratings_svg(rows: list[Standing]) -> str:
    W, left, right, top, row_h = 760, 170, 70, 56, 40
    H = top + row_h * len(rows) + 46
    lo = min(r.ci[0] for r in rows) - 20
    hi = max(r.ci[1] for r in rows) + 20
    ticks = nice_ticks(lo, hi, 100 if hi - lo > 300 else 50)
    lo, hi = ticks[0], max(ticks[-1], hi)

    def X(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (W - left - right)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
             f'aria-label="Ratings with 95% confidence intervals">', SVG_STYLE,
             f'<rect class="bg" width="{W}" height="{H}" rx="8"/>',
             '<text class="ink" x="20" y="30" font-size="16" font-weight="600">Head-to-head rating (Elo scale, 95% CI)</text>']
    for t in ticks:
        parts.append(f'<line class="grid" x1="{X(t):.1f}" y1="{top - 8}" x2="{X(t):.1f}" y2="{H - 38}"/>')
        parts.append(f'<text class="ink2" x="{X(t):.1f}" y="{H - 18}" font-size="12" text-anchor="middle">{t:.0f}</text>')
    for i, r in enumerate(rows):
        y = top + i * row_h + row_h / 2
        tip = esc(f"{r.name}: {r.rating:.0f} (95% CI {r.ci[0]:.0f}–{r.ci[1]:.0f}), {r.w}W {r.d}D {r.l}L")
        parts.append(f'<g><title>{tip}</title>'
                     f'<rect x="0" y="{y - row_h / 2:.1f}" width="{W}" height="{row_h}" fill="transparent"/>'
                     f'<text class="ink" x="{left - 14}" y="{y + 4:.1f}" font-size="13" text-anchor="end">{esc(r.name)}</text>'
                     f'<line class="whisker" x1="{X(r.ci[0]):.1f}" y1="{y:.1f}" x2="{X(r.ci[1]):.1f}" y2="{y:.1f}"/>'
                     f'<circle class="mark ring" cx="{X(r.rating):.1f}" cy="{y:.1f}" r="6"/>'
                     f'<text class="ink2" x="{X(r.ci[1]) + 10:.1f}" y="{y + 4:.1f}" font-size="12">{r.rating:.0f}</text></g>')
    parts.append("</svg>")
    return "\n".join(parts)


# ----- main -----

def write_csv(path: Path, rows: list[GameRow] | list[Shout], cls: type[GameRow] | type[Shout]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([fl.name for fl in fields(cls)])
        w.writerows(astuple(r) for r in rows)


def main() -> None:
    every = load_games(LEADERBOARD)
    games = [g for g in every if g.counts]
    excluded = [f"{g.id} (team {', '.join(s for s in ('A', 'B') if g.failure_rate(s) > MAX_FAILURE_RATE)}: "
                f"over {MAX_FAILURE_RATE:.0%} of calls failed)" for g in every if not g.counts]
    if not games:
        sys.exit(f"no games in {LEADERBOARD}")
    RESULTS.mkdir(parents=True, exist_ok=True)

    totals: defaultdict[str, Totals] = defaultdict(Totals)
    results: list[Result] = []
    rows: list[GameRow] = []
    shouts: list[Shout] = []
    h2h: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for g in games:
        m = g.match
        ma, mb = g.model("A"), g.model("B")
        sa, sb = m["score"]["A"], m["score"]["B"]
        results.append((ma, mb, 1.0 if sa > sb else 0.0 if sa < sb else 0.5))
        h2h[(ma, mb)].append(f"{sa}–{sb}")
        h2h[(mb, ma)].append(f"{sb}–{sa}")
        rows.append(GameRow(g.id, ma, mb, sa, sb, m["ticks_played"], m.get("ran_on", ""), verified(g)))
        side: Side
        for side, model, gf, ga in (("A", ma, sa, sb), ("B", mb, sb, sa)):
            t, s = totals[model], m["stats"][side]
            t.games += 1
            t.w += gf > ga
            t.d += gf == ga
            t.l += gf < ga
            t.gf += gf
            t.ga += ga
            t.calls += s["calls"]
            t.errors += s["errors"]
            t.invalid += s["invalid"]
            t.shouts += s.get("shouts", 0)
            t.cost += s["cost"]
        score = [0, 0]
        for tick in m["log"][1:]:
            for e in tick["events"]:
                model = ma if (e.get("player") or "?")[0] == "A" else mb
                if e["type"] == "pass":
                    totals[model].passes += 1
                elif e["type"] == "reception":
                    totals[model].completed += 1
                elif e["type"] in ("interception", "tackle"):
                    totals[model].won_ball += 1
            for a in tick["actions"]:
                if a.get("say"):
                    team = a["player"][0]
                    shouts.append(Shout(g.id, tick["tick"], team, a["player"], ma if team == "A" else mb,
                                        a.get("action"), a.get("target"), a["say"], score[0], score[1]))
            score = tick["score"]

    players = sorted(totals)
    ratings = bradley_terry(results, players)
    ci = bootstrap(results, players)
    table = []
    for p in sorted(players, key=lambda x: -ratings[x]):
        t = totals[p]
        table.append(Standing(
            model=p, name=name(p), rating=ratings[p], ci=ci[p], w=t.w, d=t.d, l=t.l, gf=t.gf, ga=t.ga,
            passes=t.passes, completed=t.completed, pass_pct=100 * t.completed / t.passes if t.passes else 0.0,
            won_ball=t.won_ball, games=t.games, shouts_per_call=t.shouts / t.calls if t.calls else 0.0,
            cost_per_game=t.cost / t.games, errors=t.errors + t.invalid, calls=t.calls))

    write_csv(RESULTS / "games.csv", rows, GameRow)
    write_csv(RESULTS / "shouts.csv", shouts, Shout)
    with open(RESULTS / "shouts.jsonl", "w") as f:
        for sh in shouts:
            f.write(json.dumps(asdict(sh)) + "\n")

    lines = ["| # | Model | Rating (95% CI) | W-D-L | Goals | Passes completed | Ball wins | Shouts per call | Cost per game |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(table, 1):
        flag = f" *(provisional: {r.games} games)*" if r.games < 6 else ""
        lines.append(f"| {i} | {r.name}{flag} | {r.rating:.0f} ({r.ci[0]:.0f}–{r.ci[1]:.0f}) | "
                     f"{r.w}-{r.d}-{r.l} | {r.gf}:{r.ga} | {r.completed}/{r.passes} "
                     f"({r.pass_pct:.0f}%) | {r.won_ball} | {r.shouts_per_call:.2f} | ${r.cost_per_game:.2f} |")
    if any(r.games < 6 for r in table):
        lines += ["", "Provisional: fewer than 6 games, so the interval understates the uncertainty "
                  "(bootstrap resamples often leave out a model's few games)."]
    if excluded:
        lines += ["", "Excluded: " + "; ".join(excluded) + "."]
    unverified = [r.id for r in rows if not r.verified]
    lines += ["", f"{len(rows)} games, {sum(r.calls for r in table)} model calls, "
              f"{sum(r.errors for r in table)} failed or invalid. "
              f"Re-scored by replaying the logged actions: "
              f"{'all agree with the live score' if not unverified else 'not confirmed for ' + ', '.join(unverified)}."]
    (RESULTS / "standings.md").write_text("\n".join(lines) + "\n")

    order = [r.model for r in table]
    hl = ["| row vs column | " + " | ".join(name(m) for m in order) + " |", "|---" * (len(order) + 1) + "|"]
    for row in order:
        hl.append(f"| **{name(row)}** | " + " | ".join("—" if row == col else ", ".join(h2h.get((row, col), [])) or "·"
                                                     for col in order) + " |")
    (RESULTS / "h2h.md").write_text("\n".join(hl) + "\n")

    (RESULTS / "ratings.svg").write_text(ratings_svg(table))
    (RESULTS / "summary.json").write_text(json.dumps(
        {"standings": [asdict(r) for r in table], "games": [asdict(r) for r in rows]}, indent=2))
    print((RESULTS / "standings.md").read_text())
    print((RESULTS / "h2h.md").read_text())
    print(f"{len(shouts)} shouts exported; charts in {RESULTS}/")


if __name__ == "__main__":
    main()
