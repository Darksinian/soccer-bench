"""Typed shapes of the data the engine passes to players and writes to game logs.

Everything here is plain JSON on disk, so these are TypedDicts: a game log loaded with
`json.loads` has exactly these keys. Points are [x, y] lists, as JSON stores them.
"""
from __future__ import annotations

from typing import Literal, NotRequired, Required, TypedDict

Side = Literal["A", "B"]
Comms = Literal["none", "team", "all"]
ActionKind = Literal["move", "pass", "stay"]
Point = list[float]


class BySide[T](TypedDict):
    A: T
    B: T


# ----- what a player sees each tick (its own team's frame) -----

class PlayerObs(TypedDict):
    id: str
    ours: bool
    keeper: bool
    you: bool
    pos: Point
    vel: Point
    facing: Point
    has_ball: bool


class BallObs(TypedDict):
    pos: Point
    vel: Point
    holder: str | None
    last_touch: str | None


class ShoutObs(TypedDict):
    id: str
    ours: bool
    you: bool
    text: str


class Observation(TypedDict):
    tick: int
    max_ticks: int
    you: str
    team: Side
    is_keeper: bool
    score_us: int
    score_them: int
    players: list[PlayerObs]
    ball: BallObs
    shouts: NotRequired[list[ShoutObs]]


# What a player replies: {"action": ..., "x": ..., "y": ..., "say": ...}. Model replies are
# untrusted JSON, so values are checked when the reply is turned into an Action.
Reply = dict[str, object]

# Usage of one model call ("in" is a keyword, hence the functional form).
CallMeta = TypedDict("CallMeta", {"model": str, "in": int, "out": int, "cost": float, "raw": str}, total=False)


# ----- game log -----

class Frame(TypedDict):
    p: list[Point]  # every player's position, team A first
    b: Point        # ball position
    h: int          # index of the player holding the ball, -1 if loose


Event = TypedDict("Event", {
    "type": Required[str],  # pass, reception, interception, tackle, goal, invalid, ...
    "player": str | None,
    "from": str | None,
    "target": Point,
    "reason": str,
    "team": Side,
    "own_goal": bool,
    "score": list[int],
    "frame": int,
}, total=False)

ActionRecord = TypedDict("ActionRecord", {
    "player": Required[str],
    "agent": Required[str],
    "model": str,
    "in": int,
    "out": int,
    "cost": float,
    "raw": str,
    "say": str,
    "error": str,
    "action": ActionKind,
    "target": Point | None,
}, total=False)

# Unrounded world-frame action per player, so a verifier can replay the match exactly.
ExactAction = tuple[str, Point | None]


class TickRecord(TypedDict):
    tick: int
    events: list[Event]
    frames: list[Frame]
    score: list[int]
    actions: list[ActionRecord]
    wall_seconds: NotRequired[float]
    exact_actions: NotRequired[list[ExactAction]]


TeamStats = TypedDict("TeamStats", {
    "calls": int, "errors": int, "invalid": int, "in": int, "out": int, "cost": float, "seconds": float, "shouts": int,
})


def empty_stats() -> TeamStats:
    return {"calls": 0, "errors": 0, "invalid": 0, "in": 0, "out": 0, "cost": 0.0, "seconds": 0.0, "shouts": 0}


class MatchResult(TypedDict):
    config: dict[str, float | int | str | bool]
    seed: int
    teams: BySide[list[str]]
    comms: BySide[str]
    roles: NotRequired[BySide[bool]]
    score: BySide[int]
    winner: Side | None
    ticks_played: int
    stats: BySide[TeamStats]
    log: list[TickRecord]
    ran_on: NotRequired[str]          # "valkyrie" or "openrouter", for games on the leaderboard
    stopped_early: NotRequired[bool]


# ----- run.py series summary (runs/<name>/summary.json) -----

class SeriesTotals(TypedDict):
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    errors: int
    invalid: int
    shouts: int
    cost: float


class SeriesGame(TypedDict):
    game: int
    A: str  # "team1" or "team2"
    B: str
    score: BySide[int]
    ticks: int
    replay: str


class SeriesSummary(TypedDict):
    teams: dict[str, str]
    comms: dict[str, str]
    roles: dict[str, bool]
    config: dict[str, float | int | str | bool]
    totals: dict[str, SeriesTotals]
    games: list[SeriesGame]
