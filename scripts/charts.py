"""Draws the writeup's charts as SVG files in docs/media/charts/.

  python scripts/leaderboard.py && python scripts/team_play.py && python scripts/coordination.py
  python scripts/charts.py

Reads results/summary.json (model order), results/coordination.json, results/team_play.json,
results/talk.json (hand-labelled shouts) and results/experiments.json. One metric is computed here from the game logs:

  through balls   passes aimed 5+ units from the nearest teammate, not counting passes aimed into
                  the goal mouth (shots); "into a run" when that teammate's own move target on the
                  same tick is within 6 units of the pass target; "arrived" when the next player to
                  take the ball is a teammate
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccer_eval.games import RESULTS, ROOT, Game, family, load_games, name  # noqa: E402
from soccer_eval.schema import Side  # noqa: E402

OUT = ROOT / "docs" / "media" / "charts"

STYLE = """<style>
  .ink { fill: #1b1e22; } .soft { fill: #5f6670; } .track { fill: #eef0ec; }
  .claude { fill: #d9682a; } .gpt { fill: #3b82c4; } .muted { fill: #b9bdb6; }
  text { font-family: 'IBM Plex Sans', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif; }
  @media (prefers-color-scheme: dark) {
    .ink { fill: #eceeea; } .soft { fill: #9aa19b; } .track { fill: #1f2428; }
    .claude { fill: #e8834b; } .gpt { fill: #5d9bd8; } .muted { fill: #4a514c; }
  }
</style>"""

# Every chart is the same width as the writeup's text column, with the same row height and type
# sizes, so they read as one set of figures. Titles live in the figure captions, not the charts.
# Each chart also gets a "-narrow" version for phones: labels above the bars, panels stacked.
WIDTH, LABEL_W, ROW_H, HEAD_H, GAP = 760, 150, 30, 26, 32
NARROW_W, NARROW_ROW_H = 360, 44

Panels = list[tuple[str, list["Bar"]]]


@dataclass(frozen=True)
class Bar:
    label: str
    value: float           # 0..1
    cls: str               # claude, gpt or muted
    note: str = ""         # shown after the percentage
    shown: str = ""        # shown instead of the percentage (e.g. when there is nothing to divide)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def legend_svg(width: int) -> str:
    lx = width - 118
    return (f'<rect class="claude" x="{lx}" y="6" width="10" height="10" rx="2"/><text class="soft" x="{lx + 15}" y="15" font-size="12">Claude</text>'
            f'<rect class="gpt" x="{lx + 66}" y="6" width="10" height="10" rx="2"/><text class="soft" x="{lx + 81}" y="15" font-size="12">GPT</text>')


def value_w(bars: list[Bar], least: float) -> float:
    """Room to leave after the bars for the longest value text (13px figures, 12px note)."""
    return max([least] + [8 * len(b.shown or f"{b.value:.0%}") + (7 * (len(b.note) + 1) if b.note else 0) + 12 for b in bars])


def bar_svg(b: Bar, x0: float, y: float, bar_w: float) -> str:
    """A bar on its track with the value after it; y is the row's top, the bar sits 7 below."""
    w = max(2.0, b.value * bar_w)
    return (f'<rect class="track" x="{x0:.1f}" y="{y + 7}" width="{bar_w:.1f}" height="16" rx="3"/>'
            f'<rect class="{b.cls}" x="{x0:.1f}" y="{y + 7}" width="{w:.1f}" height="16" rx="3"/>'
            f'<text class="ink" x="{x0 + bar_w + 8:.1f}" y="{y + 19}" font-size="13">{b.shown or f"{b.value:.0%}"}'
            f'<tspan class="soft" font-size="12">{(" " + esc(b.note)) if b.note else ""}</tspan></text>')


def bars_svg(label: str, panels: Panels, legend: bool = True) -> str:
    """One or more side-by-side panels of horizontal bars; rows share the first panel's labels."""
    rows = max(len(p[1]) for p in panels)
    top = HEAD_H if legend or any(h for h, _ in panels) else 4
    panel_w = (WIDTH - LABEL_W - GAP * (len(panels) - 1)) / len(panels)
    H = top + rows * ROW_H + 4
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {H}" width="{WIDTH}" height="{H}" role="img" aria-label="{esc(label)}">',
           STYLE]
    if legend:
        out.append(legend_svg(WIDTH))
    for i, (head, bars) in enumerate(panels):
        x0 = LABEL_W + i * (panel_w + GAP)
        bar_w = panel_w - value_w(bars, 90)
        if head:
            out.append(f'<text class="soft" x="{x0:.1f}" y="15" font-size="12" font-weight="600">{esc(head)}</text>')
        for r, b in enumerate(bars):
            y = top + r * ROW_H
            if i == 0:
                out.append(f'<text class="ink" x="{LABEL_W - 12}" y="{y + 19}" font-size="13" text-anchor="end">{esc(b.label)}</text>')
            out.append(bar_svg(b, x0, y, bar_w))
    out.append("</svg>")
    return "\n".join(out)


def bars_svg_narrow(label: str, panels: Panels, legend: bool = True) -> str:
    """The same chart for a phone: each label on its own line above its bar, panels stacked."""
    bar_w = NARROW_W - value_w([b for _, bars in panels for b in bars], 110)
    out: list[str] = []
    y = HEAD_H if legend else 4
    for head, bars in panels:
        if head:
            out.append(f'<text class="soft" x="0" y="{y + 15}" font-size="12" font-weight="600">{esc(head)}</text>')
            y += 24
        for b in bars:
            out.append(f'<text class="ink" x="0" y="{y + 13}" font-size="13">{esc(b.label)}</text>')
            out.append(bar_svg(b, 0, y + 12, bar_w))
            y += NARROW_ROW_H
        y += 12
    H = y - 8
    head = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {NARROW_W} {H}" width="{NARROW_W}" height="{H}" role="img" aria-label="{esc(label)}">'
    return "\n".join([head, STYLE, *([legend_svg(NARROW_W)] if legend else []), *out, "</svg>"])


def chart(label: str, panels: Panels, legend: bool = True) -> tuple[str, str]:
    """The wide and the narrow version of a chart."""
    return bars_svg(label, panels, legend), bars_svg_narrow(label, panels, legend)


Counts = dict[str, tuple[int, int]]


def through_balls(games: list[Game]) -> tuple[Counts, Counts]:
    """Per model (through balls into a run, through balls); into a run or not (through balls
    that arrived, through balls)."""
    tb: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    arrived: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    for g in games:
        log, cfg = g.match["log"], g.match["config"]
        n, W = int(cfg["n_per_team"]), float(cfg["width"])
        for i in range(1, len(log)):
            prev, acts = log[i - 1]["frames"][-1], log[i]["actions"]
            by = {a["player"]: a for a in acts}
            for a in acts:
                tgt = a.get("target")
                if a.get("action") != "pass" or not tgt:
                    continue
                side: Side = "A" if a["player"][0] == "A" else "B"
                base, me = (0 if side == "A" else n), (0 if side == "A" else n) + int(a["player"][1:])
                gx = W if side == "A" else 0.0
                at_goal = abs(tgt[0] - gx) <= 2 and 23.2 <= tgt[1] <= 36.8   # a shot
                mates = [base + k for k in range(n) if base + k != me]
                r = min(mates, key=lambda q: math.dist(prev["p"][q], tgt))
                if at_goal or math.dist(prev["p"][r], tgt) < 5:
                    continue
                ra = by.get(f"{side}{r - base}")
                rt = ra.get("target") if ra else None
                into = bool(ra and ra.get("action") == "move" and rt and math.dist(rt, tgt) < 6)
                t = tb[g.model(side)]
                t[1] += 1
                t[0] += into
                # who takes the ball next, starting after this pass's own event
                evs = [e for j in range(i, len(log)) for e in log[j]["events"]]
                start = next(k for k, e in enumerate(evs) if e["type"] == "pass" and e.get("player") == a["player"])
                nxt = next((e["type"] for e in evs[start + 1:] if e["type"] in ("reception", "interception", "tackle", "goal")), "")
                c = arrived["into a run" if into else "other"]
                c[1] += 1
                c[0] += nxt == "reception"

    def tuples(d: defaultdict[str, list[int]]) -> Counts:
        return {k: (v[0], v[1]) for k, v in d.items()}
    return tuples(tb), tuples(arrived)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    order = [r["model"] for r in json.loads((RESULTS / "summary.json").read_text())["standings"]]
    coord = json.loads((RESULTS / "coordination.json").read_text())
    play = json.loads((RESULTS / "team_play.json").read_text())
    talk = json.loads((RESULTS / "talk.json").read_text())
    exps = json.loads((RESULTS / "experiments.json").read_text())

    def per_model(value: dict[str, float]) -> list[Bar]:
        return [Bar(name(m), value[m], family(m)) for m in order]

    charts = {
        "chase": chart("2+ players head for the ball, by model",
                          [("", per_model({m: coord[m]["double_press"] for m in order}))]),
        "talk-style": chart("How each team talks, by model",
                               [("Tells a teammate what to do", per_model({m: talk[m]["style.directs"]["value"] for m in order})),
                                ("Says what it is doing itself", per_model({m: talk[m]["style.announces"]["value"] for m in order}))]),
        "rest-defence": chart("Someone stays back, by model",
                                 [("", per_model({m: play[m]["rest_defence"] for m in order}))]),
    }
    games = [g for g in load_games() if g.counts]
    tb, arrived = through_balls(games)
    for k, (a, t) in arrived.items():
        print(f"through balls {k}: {a} of {t} arrived ({a / t:.0%})")
    charts["through-balls"] = chart(
        "Through balls into a run, by model",
        [("", [Bar(name(m), tb[m][0] / tb[m][1], family(m), f"of {tb[m][1]}") if m in tb
               else Bar(name(m), 0.0, family(m), shown="none played") for m in order])])

    def exp(key: str, cond: str) -> dict[str, float]:
        return dict(exps[key]["conditions"][cond])

    talk = [("GPT-6.1 Sol", "sol-talk-vs-silent", "gpt"), ("Claude Haiku 5.5", "haiku-talk-vs-silent", "claude"),
            ("GPT-6 Luna", "luna-talk-vs-silent", "gpt")]

    def win_share(rec: str) -> float:
        w, d, lo = (int(x) for x in rec.split("-"))
        return (w + d / 2) / (w + d + lo)

    charts["talk-vs-silent"] = chart(
        "Points won by the talking and the silent side",
        [("Talking side", [Bar(lbl, win_share(str(exp(k, "talk")["record"])), cls, str(exp(k, "talk")["record"])) for lbl, k, cls in talk]),
         ("Silent side", [Bar(lbl, win_share(str(exp(k, "silent")["record"])), "muted", str(exp(k, "silent")["record"])) for lbl, k, cls in talk])])
    form = [("Claude Haiku 5.5", "haiku-roles-vs-none", "claude"), ("GPT-6 Luna", "luna-roles-vs-none", "gpt")]
    charts["formation"] = chart(
        "2+ players head for the ball, with and without a formation",
        [("With a formation", [Bar(lbl, float(exp(k, "roles")["double_press"]), cls) for lbl, k, cls in form]),
         ("Without", [Bar(lbl, float(exp(k, "no roles")["double_press"]), "muted") for lbl, k, cls in form])])
    for k, (wide, narrow) in charts.items():
        (OUT / f"{k}.svg").write_text(wide)
        (OUT / f"{k}-narrow.svg").write_text(narrow)
    print(f"{len(charts)} charts written to {OUT}")


if __name__ == "__main__":
    main()
