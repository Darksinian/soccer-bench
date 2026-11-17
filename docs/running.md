# Running games

```bash
pip install -e .        # no dependencies; pip install -e '.[direct]' adds the Anthropic and OpenAI SDKs
export OPENROUTER_API_KEY=...
python run.py --team1 <provider>/<model> --team2 scripted --games 2 --verbose
```

The leaderboard settings are `--n 5 --max-ticks 60 --comms team`.

- A team is one spec for all players, or a comma-separated list with one spec per player
  (index 0 is the goalkeeper): `--team1 "scripted,<provider>/<model>,<other-provider>/<model>"`.
- Specs: `random`, `scripted`, or any OpenRouter model slug. Add `@low|medium|high` to set
  OpenRouter's reasoning effort, e.g. `<provider>/<model>@low`.
- Teams swap sides every game, so use an even `--games`. `--parallel K` plays K games at once (every call in every running game is in flight together).
- `--n` players per team (default 3), `--max-ticks` (default 120), `--timeout` per call.
- `--comms none|team|all` (default `none`): lets each player add `"say": "..."` (up to 80 characters)
  to its reply. Shouts made on one tick are shown to the listeners in the next tick's prompt, since
  everyone acts at once. `team` means only teammates hear them; `all` means opponents hear them too.
  Shouts appear as speech labels in the replay. Comparing `none` against `team` for the same model
  measures how much communication helps. `--comms1` / `--comms2` set it per team, e.g. `--comms1 team` lets only team1
  talk; a team's shouts reach opponents only if that team's mode is `all`.
- `--roles1` / `--roles2` give that team a fixed formation in its prompt (two defenders, two
  forwards, each told its own area); 5 players per team only.

Outputs go to `--out` (default `runs/latest`): `game_XX.json` (full log, including every model's
raw reply), `game_XX.html` (self-contained replay; open it in a browser), and `summary.json`.

Tests: `python -m pytest`. Types: `mypy` (strict; the game logs' shapes are in `soccer_eval/schema.py`).
Rebuild a replay from a log: `python -m soccer_eval.replay runs/x/game_01.json`.

## Cost

A match has up to `2 x n x max_ticks` calls (720 with the defaults), each with about 1k input
tokens plus whatever the model reasons. The summary table and `stats` in each game log report
OpenRouter's cost per team. Smoke-test new models with `--max-ticks 10`. The first 30 leaderboard
games cost about $63 of Anthropic and $64 of OpenAI usage.

## Replay site

[`site/`](../site), live at
[darksinian.github.io/soccer-eval](https://darksinian.github.io/soccer-eval/), is a static site with the leaderboard and every game: a filterable list and a replay viewer (play, scrub, both teams' shouts beside the pitch, a pause on each
goal). It has no build step:

- **GitHub Pages:** in the repository's Settings → Pages, set Source to "GitHub Actions". The
  workflow in [`.github/workflows/pages.yml`](../.github/workflows/pages.yml) then publishes `site/`
  on every push to `main` that changes it. (Pages on a private repository needs a paid GitHub plan;
  on a public one it is free.)
- **Anywhere else:** upload the `site/` folder as is (Netlify Drop, any static host), or run
  `python -m http.server -d site` locally.

To rebuild it after new games: `python scripts/leaderboard.py && python scripts/build_site.py`.
