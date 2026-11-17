#!/bin/bash
# Talk vs silent and roles vs no roles, self-play, leaderboard settings (5v5, 60 ticks, team talk).
cd "$(dirname "$0")/../.."
common="--n 5 --max-ticks 60 --comms team --games 6 --parallel 3"
run() { .venv/bin/python run.py $common "$@" > "runs/experiments/$name.log" 2>&1; }
for m in haiku:anthropic/claude-haiku-5-5 luna:openai/gpt-6-luna; do
  short=${m%%:*}; slug=${m#*:}
  name=$short-talk-vs-silent; run --team1 $slug --team2 $slug --comms2 none --seed 200 --out runs/experiments/$name &
  name=$short-roles-vs-none;  run --team1 $slug --team2 $slug --roles1 --seed 300 --out runs/experiments/$name &
done
wait
