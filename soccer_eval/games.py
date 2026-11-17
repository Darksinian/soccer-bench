"""Finished games on disk, and the models' display names. Shared by the scripts in scripts/.

  runs/leaderboard/<id>.json   every model-vs-model game on the leaderboard (one MatchResult each)
  runs/leaderboard/played.txt  their IDs in the order they were played
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .schema import MatchResult, Side

ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD = ROOT / "runs" / "leaderboard"
RESULTS = ROOT / "results"

NAMES = {
    "openai/gpt-6-astra": "GPT-6 Astra",
    "anthropic/claude-fable-5-2": "Claude Fable 5.2",
    "openai/gpt-6.1-sol": "GPT-6.1 Sol",
    "anthropic/claude-opus-5-5": "Claude Opus 5.5",
    "openai/gpt-5.6-sol": "GPT-5.6 Sol",
    "anthropic/claude-sonnet-5-5": "Claude Sonnet 5.5",
    "openai/gpt-6-luna": "GPT-6 Luna",
    "anthropic/claude-haiku-5-5": "Claude Haiku 5.5",
}

# A game is left off the leaderboard when more than this share of one team's calls failed.
MAX_FAILURE_RATE = 0.05


def canonical(slug: str) -> str:
    """One name per model: drop the effort suffix; OpenRouter writes Claude versions with dots
    (claude-fable-5.2) where Anthropic's own IDs use dashes."""
    base = slug.split("@")[0]
    org, _, model = base.partition("/")
    return f"{org}/{model.replace('.', '-')}" if org == "anthropic" else base


def name(slug: str) -> str:
    return NAMES.get(canonical(slug), slug)


def family(slug: str) -> Literal["claude", "gpt"]:
    return "claude" if "claude" in slug else "gpt"


@dataclass(frozen=True)
class Game:
    id: str
    match: MatchResult

    def model(self, side: Side) -> str:
        """The model playing `side`, under its canonical name."""
        return canonical(self.match["teams"][side][0])

    @property
    def models(self) -> dict[Side, str]:
        return {"A": self.model("A"), "B": self.model("B")}

    def failure_rate(self, side: Side) -> float:
        s = self.match["stats"][side]
        return (s["errors"] + s["invalid"]) / s["calls"] if s["calls"] else 0.0

    @property
    def counts(self) -> bool:
        """Finished, and neither team lost more than MAX_FAILURE_RATE of its calls."""
        return not self.match.get("stopped_early") and all(self.failure_rate(s) <= MAX_FAILURE_RATE for s in ("A", "B"))


def load(path: Path) -> Game:
    m: MatchResult = json.loads(path.read_text())
    return Game(path.stem, m)


def load_games(folder: Path = LEADERBOARD) -> list[Game]:
    """Games in the order they were played, as listed in the folder's played.txt (the rating
    bootstrap resamples in this order); games not listed there come last, by name."""
    listed = (folder / "played.txt").read_text().split() if (folder / "played.txt").exists() else []
    rank = {gid: i for i, gid in enumerate(listed)}
    files = sorted(folder.glob("*.json"), key=lambda f: (rank.get(f.stem, len(rank)), f.stem))
    return [load(f) for f in files]
