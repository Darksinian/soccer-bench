"""Play a series between two teams, swapping sides every game.

Examples:
  python run.py --team1 scripted --team2 random --games 4
  python run.py --team1 <provider>/<model> --team2 scripted --games 2 --verbose
  # let teammates shout short messages to each other (heard next tick):
  python run.py --team1 <provider>/<model> --team2 <provider>/<model> --comms team
  # does talking help? same model, only team1 may shout; 4 games at a time:
  python run.py --team1 <provider>/<model> --team2 <provider>/<model> --comms1 team --games 10 --parallel 4
  # one spec per player (index 0 = goalkeeper):
  python run.py --team1 "scripted,<provider>/<model>,<provider>/<model>" --team2 <provider>/<model>@low
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from soccer_eval.agents import FatalAPIError
from soccer_eval.match import play_match
from soccer_eval.replay import write_replay
from soccer_eval.schema import MatchResult, SeriesGame, SeriesSummary, SeriesTotals, Side
from soccer_eval.sim import TEAM_NAMES, Config

COMMS = ["none", "team", "all"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team1", required=True, help="player spec, or comma-separated list of one spec per player")
    ap.add_argument("--team2", required=True)
    ap.add_argument("--games", type=int, default=2, help="teams swap sides every game; use an even number")
    ap.add_argument("--parallel", type=int, default=1, help="games to play at the same time")
    ap.add_argument("--n", type=int, default=Config.n_per_team, help="players per team, including the keeper")
    ap.add_argument("--max-ticks", type=int, default=Config.max_ticks)
    ap.add_argument("--goal-width", type=float, default=Config.goal_width)
    ap.add_argument("--comms", choices=COMMS, default=Config.comms,
                    help="who hears players' shouts: nobody, teammates, or everyone")
    ap.add_argument("--comms1", choices=COMMS, help="override --comms for team1 only")
    ap.add_argument("--comms2", choices=COMMS, help="override --comms for team2 only")
    ap.add_argument("--roles1", action="store_true", help="give team1 a fixed formation in its prompt (needs --n 5)")
    ap.add_argument("--roles2", action="store_true", help="give team2 a fixed formation in its prompt (needs --n 5)")
    ap.add_argument("--backend", choices=["openrouter", "direct"], default="openrouter",
                    help="openrouter: model specs are OpenRouter slugs; direct: anthropic/<id> or openai/<id>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per model call before it counts as 'stay'")
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = Config(n_per_team=args.n, max_ticks=args.max_ticks, comms=args.comms, goal_width=args.goal_width)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    teams = {"team1": args.team1, "team2": args.team2}
    comms = {"team1": args.comms1 or args.comms, "team2": args.comms2 or args.comms}
    roles = {"team1": args.roles1, "team2": args.roles2}
    totals: dict[str, SeriesTotals] = {k: {"wins": 0, "draws": 0, "losses": 0, "goals_for": 0, "goals_against": 0,
                                         "errors": 0, "invalid": 0, "shouts": 0, "cost": 0.0} for k in teams}
    games: list[SeriesGame] = []

    def sides_of(i: int) -> tuple[str, str]:
        return ("team1", "team2") if i % 2 == 0 else ("team2", "team1")

    def label_of(key: str) -> str:
        label = key if comms[key] == "none" else f"{key} · shouts:{comms[key]}"
        return label + " · roles" if roles[key] else label

    async def play(i: int, sem: asyncio.Semaphore) -> MatchResult:
        sides = sides_of(i)
        async with sem:
            print(f"game {i + 1}/{args.games} start: A={label_of(sides[0])} ({teams[sides[0]]})  "
                  f"B={label_of(sides[1])} ({teams[sides[1]]})", flush=True)
            result = await play_match(cfg, (teams[sides[0]], teams[sides[1]]), seed=args.seed + i,
                                      call_timeout=args.timeout, verbose=args.verbose,
                                      comms=(comms[sides[0]], comms[sides[1]]),
                                      roles=(roles[sides[0]], roles[sides[1]]),
                                      label=f"g{i + 1} " if args.parallel > 1 else "", own_executor=False,
                                      backend=args.backend)
        stem = f"game_{i + 1:02d}"
        (out / f"{stem}.json").write_text(json.dumps(result))
        write_replay(result, out / f"{stem}.html", labels={"A": label_of(sides[0]), "B": label_of(sides[1])})
        print(f"game {i + 1} final: A {result['score']['A']} - {result['score']['B']} B  "
              f"({result['ticks_played']} ticks)", flush=True)
        for side in TEAM_NAMES:
            s = result["stats"][side]
            if s["calls"] and (s["errors"] + s["invalid"]) / s["calls"] > 0.05:
                print(f"  WARNING game {i + 1}: {(s['errors'] + s['invalid']) / s['calls']:.0%} of team {side}'s "
                      "calls failed or were invalid; treat this result with caution", flush=True)
        return result

    async def play_all() -> list[MatchResult]:
        # All concurrent games share one thread pool, sized so every player's call can be in flight.
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=max(1, args.parallel) * 2 * cfg.n_per_team))
        sem = asyncio.Semaphore(max(1, args.parallel))
        tasks = [asyncio.create_task(play(i, sem)) for i in range(args.games)]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:
                t.cancel()
            raise

    try:
        results = asyncio.run(play_all())
    except FatalAPIError as e:
        sys.exit(f"\nstopped: {e}\nUnfinished games were discarded; check your OpenRouter key and credit balance.")

    for i, result in enumerate(results):
        sides = sides_of(i)
        for side in TEAM_NAMES:
            key = sides[TEAM_NAMES.index(side)]
            other: Side = "B" if side == "A" else "A"
            t = totals[key]
            gf, ga = result["score"][side], result["score"][other]
            t["goals_for"] += gf
            t["goals_against"] += ga
            if gf > ga:
                t["wins"] += 1
            elif gf < ga:
                t["losses"] += 1
            else:
                t["draws"] += 1
            s = result["stats"][side]
            t["errors"] += s["errors"]
            t["invalid"] += s["invalid"]
            t["shouts"] += s.get("shouts", 0)
            t["cost"] += s["cost"]
        games.append({"game": i + 1, "A": sides[0], "B": sides[1], "score": result["score"],
                      "ticks": result["ticks_played"], "replay": f"game_{i + 1:02d}.html"})

    summary: SeriesSummary = {"teams": teams, "comms": comms, "roles": roles, "config": cfg.to_dict(), "totals": totals, "games": games}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"{'team':6} {'W':>3} {'D':>3} {'L':>3} {'GF':>4} {'GA':>4} {'err':>5} {'inv':>5} {'shouts':>6} {'cost $':>8}  spec")
    for k, t in totals.items():
        print(f"{k:6} {t['wins']:3} {t['draws']:3} {t['losses']:3} {t['goals_for']:4} {t['goals_against']:4} "
              f"{t['errors']:5} {t['invalid']:5} {t['shouts']:6} {t['cost']:8.3f}  {teams[k]} (comms {comms[k]})")
    print(f"\nlogs and replays in {out}/")


if __name__ == "__main__":
    main()
