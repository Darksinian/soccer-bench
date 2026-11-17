"""Builds the static replay site in site/ from the leaderboard games.

  python scripts/leaderboard.py    # refresh results/summary.json first
  python scripts/build_site.py     # writes site/games.json, site/leaderboard.json, site/replays/*.html
                                   # and site/writeup.html (from docs/writeup.md; needs `markdown`)

site/index.html is hand-written and lists whatever games.json holds. The folder needs no build
step to serve: GitHub Pages (see .github/workflows/pages.yml), Netlify or `python -m http.server`.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccer_eval.games import RESULTS, ROOT, Game, family, load_games, name  # noqa: E402
from soccer_eval.replay import write_replay  # noqa: E402
from soccer_eval.schema import Frame, MatchResult  # noqa: E402

SITE = ROOT / "site"
REPO = "https://github.com/Darksinian/soccer-eval"
LIVE = "https://darksinian.github.io/soccer-eval/"
# Shown in link previews; the card image (site/card.png) is shared with the home page.
FINDINGS_BLURB = "What 58 games of five-a-side soccer between 8 language models show about how they coordinate."
WRITEUP_CSS = """
.writeup { max-width: 760px; display: flex; flex-direction: column; gap: 16px; color: var(--soft); }
.writeup h2 { color: var(--ink); margin-top: 36px; padding-bottom: 12px; border-bottom: 3px solid var(--accent-light); }
.writeup h3 { color: var(--ink); margin-top: 12px; }
.writeup ul, .writeup ol { margin: 0; padding-left: 22px; display: flex; flex-direction: column; gap: 10px; }
.writeup strong { color: var(--ink); }
.writeup code { font-family: var(--mono); font-size: 0.9em; }
.writeup .table-wrap { overflow-x: auto; }
.writeup table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 15px; }
.writeup th, .writeup td { text-align: left; padding: 6px 12px 6px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
.writeup th { font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.writeup img { max-width: 100%; height: auto; display: block; }
.writeup figure { margin: 8px 0 12px; }
.writeup figure img { width: 100%; border-radius: 8px; }
.writeup figcaption { margin-top: 10px; font-size: 14px; line-height: 1.5; color: var(--muted); }
.writeup figcaption b { color: var(--ink); font-weight: 600; }
.writeup blockquote { margin: 0; padding: 12px 16px; border-left: 3px solid var(--accent-light); background: var(--surface);
  color: var(--ink); font-size: 15px; line-height: 1.7; }
.writeup blockquote p { margin: 0; }
.writeup td img { margin-bottom: 4px; }
.writeup th:empty { padding: 0; border: 0; }
.writeup .data th, .writeup .data td { white-space: nowrap; }
.writeup a[href*="?t="] { white-space: nowrap; font-weight: 500; }
"""


@dataclass(frozen=True)
class SiteGame:
    id: str
    a: str
    b: str
    fa: Literal["claude", "gpt"]
    fb: Literal["claude", "gpt"]
    ga: int
    gb: int
    ticks: int
    via: str
    replay: str
    thumb: str


@dataclass(frozen=True)
class SiteModel:
    name: str
    family: Literal["claude", "gpt"]
    rating: int
    ci: tuple[int, int]
    w: int
    d: int
    l: int  # noqa: E741
    games: int
    gf: int
    ga: int
    pass_pct: int
    cost_per_game: float


def key_frame(m: MatchResult) -> Frame:
    """The frame a thumbnail shows: the moment of the first goal, else the middle of the game."""
    for t in m["log"]:
        goal = next((e for e in t["events"] if e["type"] == "goal"), None)
        if goal is not None:
            return t["frames"][min(goal.get("frame", len(t["frames"]) - 1), len(t["frames"]) - 1)]
    return m["log"][len(m["log"]) // 2]["frames"][-1]


def thumb_svg(m: MatchResult) -> str:
    """A small static picture of the pitch at the key frame, in the replay player's colours."""
    cfg, f = m["config"], key_frame(m)
    W, H, n = float(cfg["width"]), float(cfg["height"]), int(cfg["n_per_team"])
    by, bw, bd = (H - float(cfg["box_width"])) / 2, float(cfg["box_width"]), float(cfg["box_depth"])
    gy, gw = (H - float(cfg["goal_width"])) / 2, float(cfg["goal_width"])
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.9" fill="{"#2f6fdb" if g < n else "#e0692d"}"/>'
                   for g, (x, y) in enumerate(f["p"]))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-3 -3 {W + 6:g} {H + 6:g}">'
            f'<rect x="-3" y="-3" width="{W + 6:g}" height="{H + 6:g}" fill="#3f7d4f"/>'
            f'<g fill="none" stroke="#fff" stroke-opacity=".6" stroke-width=".5">'
            f'<rect width="{W:g}" height="{H:g}"/><line x1="{W / 2:g}" x2="{W / 2:g}" y2="{H:g}"/>'
            f'<circle cx="{W / 2:g}" cy="{H / 2:g}" r="9"/><rect y="{by:g}" width="{bd:g}" height="{bw:g}"/>'
            f'<rect x="{W - bd:g}" y="{by:g}" width="{bd:g}" height="{bw:g}"/></g>'
            f'<g fill="#fff" fill-opacity=".35"><rect x="-3" y="{gy:g}" width="3" height="{gw:g}"/>'
            f'<rect x="{W:g}" y="{gy:g}" width="3" height="{gw:g}"/></g>'
            f'{dots}<circle cx="{f["b"][0]:.1f}" cy="{f["b"][1]:.1f}" r="1" fill="#fff" stroke="#000" stroke-width=".3"/></svg>')


def entry(g: Game) -> SiteGame:
    a, b, m = g.model("A"), g.model("B"), g.match
    write_replay(m, SITE / "replays" / f"{g.id}.html", labels={"A": name(a), "B": name(b)})
    (SITE / "thumbs" / f"{g.id}.svg").write_text(thumb_svg(m))
    via = {"valkyrie": "Valkyrie", "openrouter": "OpenRouter"}.get(m.get("ran_on", ""), "")
    return SiteGame(g.id, name(a), name(b), family(a), family(b), m["score"]["A"], m["score"]["B"],
                    m["ticks_played"], via, f"replays/{g.id}.html", f"thumbs/{g.id}.svg")


def build_writeup() -> None:
    """Render docs/writeup.md as site/writeup.html in the site's own style. Repo links go to
    GitHub, links to the live site become relative, and the GIFs are copied to site/media/."""
    import markdown

    text = (ROOT / "docs" / "writeup.md").read_text()
    text = re.sub(r"\A# .*\n+\*[^\n]*\n[^\n]*\*\n+", "", text)  # title and draft note: the page header replaces them
    body = markdown.markdown(text, extensions=["tables", "toc"])
    body = body.replace(f'href="{LIVE}"', 'href="./"').replace(f'href="{LIVE}', 'href="')
    body = re.sub(r'href="\.\./([^"]*)"', lambda m: f'href="{REPO}/blob/main/{m.group(1)}"' if m.group(1) else f'href="{REPO}"', body)
    # An image (optionally linked) with an italic "Figure N." line under it becomes a captioned figure.
    body = re.sub(r'<p>((?:<a [^>]*>)?<img [^>]*/?>(?:</a>)?)\s*(?:<br\s*/?>|</p>\s*<p>)?\s*<em>Figure (\d+)\. (.*?)</em></p>',
                  r'<figure>\1<figcaption><b>Figure \2.</b> \3</figcaption></figure>', body, flags=re.S)
    body = re.sub(r"<table>.*?</table>", lambda m: f'<div class="table-wrap{"" if "<img" in m.group(0) else " data"}">{m.group(0)}</div>',
                  body, flags=re.S)  # data tables keep cells on one line and scroll sideways on phones
    body = re.sub(r'<img ([^>]*)src="media/charts/([\w-]+)\.svg"([^>]*)>',  # phones get the narrow chart
                  r'<picture><source media="(max-width: 600px)" srcset="media/charts/\2-narrow.svg">'
                  r'<img \1src="media/charts/\2.svg"\3></picture>', body)
    (SITE / "media").mkdir(exist_ok=True)
    for gif in re.findall(r'src(?:set)?="media/([^"]+)"', body):
        (SITE / "media" / gif).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "docs" / "media" / gif, SITE / "media" / gif)
    index = (SITE / "index.html").read_text()
    style = re.search(r"<style>(.*?)</style>", index, re.S)
    topbar = re.search(r'<nav class="topbar".*?</nav>', index, re.S)  # same top bar as the home page
    nav = (topbar.group(0).replace('href="#', 'href="./#')
           .replace('<a href="writeup.html">', '<a href="writeup.html" aria-current="page">') if topbar else "")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Soccer-Bench Findings</title>
<meta name="description" content="{FINDINGS_BLURB}">
<meta property="og:type" content="article">
<meta property="og:title" content="Soccer-Bench Findings">
<meta property="og:description" content="{FINDINGS_BLURB}">
<meta property="og:url" content="{LIVE}writeup.html">
<meta property="og:image" content="{LIVE}card.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400..700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{style.group(1) if style else ""}{WRITEUP_CSS}</style>
</head>
<body>
{nav}
<div class="wrap">
  <article class="writeup">
{body}
  </article>
  <footer>Built from <a href="{REPO}/blob/main/docs/writeup.md">docs/writeup.md</a>.</footer>
</div>
</body>
</html>
"""
    (SITE / "writeup.html").write_text(page)


def main() -> None:
    (SITE / "replays").mkdir(parents=True, exist_ok=True)
    (SITE / "thumbs").mkdir(exist_ok=True)
    games = [entry(g) for g in load_games() if g.counts]
    (SITE / "games.json").write_text(json.dumps([asdict(g) for g in games], indent=1))
    standings = json.loads((RESULTS / "summary.json").read_text())["standings"]
    board = [SiteModel(r["name"], family(r["model"]), round(r["rating"]), (round(r["ci"][0]), round(r["ci"][1])),
                       r["w"], r["d"], r["l"], r["games"], r["gf"], r["ga"], round(r["pass_pct"]),
                       round(r["cost_per_game"], 2)) for r in standings]
    (SITE / "leaderboard.json").write_text(json.dumps({
        "models": [asdict(m) for m in board], "games": len(games), "calls": sum(r["calls"] for r in standings),
    }, indent=1))
    (SITE / ".nojekyll").write_text("")
    build_writeup()
    print(f"{len(games)} games written to {SITE}")


if __name__ == "__main__":
    main()
