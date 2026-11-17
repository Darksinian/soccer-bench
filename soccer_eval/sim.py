"""Field physics and match rules. Pure Python; deterministic given the actions and seed.

Coordinates: world frame has team A defending the left goal (x=0) and attacking the
right goal (x=width). Each team observes and acts in its own frame, where it always
attacks toward +x; team B's frame is the world rotated 180 degrees (see `mirror`).
Speeds are in field units per tick.

Possession: a held ball sits just outside the carrier's body, on the side they face, and
swings around them at a limited turn rate when they change direction. Any opponent whose
body touches the ball takes it. Player bodies are solid, so the carrier shields the ball
from defenders behind them.
"""
from __future__ import annotations

import math
import random
from typing import Literal
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from .schema import ActionKind, Event, Frame, Observation, Point, TickRecord

TEAM_NAMES: tuple[Literal["A"], Literal["B"]] = ("A", "B")


@dataclass
class Config:
    n_per_team: int = 3            # includes the goalkeeper, who is always index 0
    width: float = 100.0
    height: float = 60.0
    goal_width: float = 9.6         # was 12.0 before 2026-11-13; each game log records the value used
    box_depth: float = 16.0        # goalkeeper must stay within this distance of its goal line
    box_width: float = 32.0
    player_radius: float = 1.0     # R
    ball_radius: float = 0.4       # b_r
    player_speed: float = 6.0
    carrier_speed: float = 5.0     # speed while holding the ball
    pass_speed: float = 20.0       # constant; the ball never stops on its own
    ball_friction: float = 0.0     # fraction of ball speed lost per tick (0 = constant velocity)
    ball_turn_rate: float = 180.0  # degrees per tick a held ball can swing around its carrier
    max_ticks: int = 120           # 1 tick = 1 simulated second, so 2 minutes
    goals_to_win: int = 3
    steal_immunity_ticks: int = 1  # a player who just won the ball can't lose it straight back
    substeps: int = 20
    comms: str = "none"            # who hears shouts: "none", "team" (teammates) or "all"
    shout_max_chars: int = 80
    roles: bool = False            # give the team a fixed formation in the prompt (5 per team only)

    @property
    def goal_y0(self) -> float:
        return (self.height - self.goal_width) / 2

    @property
    def goal_y1(self) -> float:
        return (self.height + self.goal_width) / 2

    @property
    def box_y0(self) -> float:
        return (self.height - self.box_width) / 2

    @property
    def box_y1(self) -> float:
        return (self.height + self.box_width) / 2

    @property
    def hold_dist(self) -> float:
        """Distance from carrier centre to a held ball's centre (the two just touch)."""
        return self.player_radius + self.ball_radius

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


def mirror(cfg: Config, team: int, xy: Sequence[float]) -> tuple[float, float]:
    """Convert a point between world frame and `team`'s frame (an involution)."""
    x, y = xy
    return (x, y) if team == 0 else (cfg.width - x, cfg.height - y)


def mirror_vec(team: int, v: Sequence[float]) -> tuple[float, float]:
    return (v[0], v[1]) if team == 0 else (-v[0], -v[1])


def angle_diff(a: float, b: float) -> float:
    """Signed smallest rotation from angle b to angle a, in (-pi, pi]."""
    d = (a - b) % (2 * math.pi)
    return d - 2 * math.pi if d > math.pi else d


@dataclass
class Player:
    team: int
    idx: int
    pos: list[float]
    vel: list[float] = field(default_factory=lambda: [0.0, 0.0])     # displacement over the last tick
    facing: list[float] = field(default_factory=lambda: [1.0, 0.0])  # direction of most recent movement
    immune_until: int = -1         # cannot lose the ball to a touch up to and including this tick

    @property
    def is_keeper(self) -> bool:
        return self.idx == 0

    @property
    def name(self) -> str:
        return f"{TEAM_NAMES[self.team]}{self.idx}"


@dataclass
class Ball:
    pos: list[float]
    vel: list[float] = field(default_factory=lambda: [0.0, 0.0])
    holder: int | None = None      # global player index
    angle: float = 0.0             # while held: direction from carrier centre to ball (radians, world)
    ignore: int | None = None      # the passer, until the ball has left their body
    last_touch: int | None = None


@dataclass
class Action:
    kind: ActionKind = "stay"
    target: tuple[float, float] | None = None  # world frame; set for "move" and "pass"


class Sim:
    def __init__(self, cfg: Config, seed: int = 0):
        if cfg.n_per_team < 2:
            raise ValueError("n_per_team must be >= 2 (goalkeeper + at least one outfield player)")
        self.cfg = cfg
        self.rng = random.Random(seed)
        n = cfg.n_per_team
        self.players = [Player(t, i, [0.0, 0.0]) for t in (0, 1) for i in range(n)]
        self.ball = Ball([0.0, 0.0])
        self.tick = 0
        self.score = [0, 0]
        self.kickoff(0)

    # ----- setup -----

    def kickoff(self, team: int) -> None:
        cfg = self.cfg
        mid_y = cfg.height / 2
        for t in (0, 1):
            outfield = [p for p in self.players if p.team == t and not p.is_keeper]
            kicker = outfield[0] if t == team else None
            rest = [p for p in outfield if p is not kicker]
            for p in self.players:
                if p.team != t:
                    continue
                if p.is_keeper:
                    local = (3.0, mid_y)
                elif p is kicker:
                    local = (cfg.width / 2 - cfg.hold_dist, mid_y)  # ball exactly on the centre spot
                else:
                    k = rest.index(p)
                    local = (cfg.width / 2 - 14, cfg.height * (k + 1) / (len(rest) + 1))
                p.pos = list(mirror(cfg, t, local))
                p.vel = [0.0, 0.0]
                p.facing = list(mirror_vec(t, (1.0, 0.0)))
                p.immune_until = -1
            if kicker is not None:
                kicker.immune_until = self.tick
                g = self.players.index(kicker)
                self.ball = Ball([0.0, 0.0], holder=g, last_touch=g,
                                 angle=math.atan2(kicker.facing[1], kicker.facing[0]))
                self._place_held_ball()

    @property
    def done(self) -> bool:
        return max(self.score) >= self.cfg.goals_to_win or self.tick >= self.cfg.max_ticks

    # ----- observation -----

    def observe(self, g: int) -> Observation:
        """Snapshot of the field in player g's team frame."""
        cfg, b, me = self.cfg, self.ball, self.players[g]
        t = me.team

        def pt(xy: Sequence[float]) -> Point:
            return [round(c, 2) for c in mirror(cfg, t, xy)]

        def vec(v: Sequence[float]) -> Point:
            return [round(c, 2) + 0.0 for c in mirror_vec(t, v)]

        return {
            "tick": self.tick,
            "max_ticks": cfg.max_ticks,
            "you": me.name,
            "team": TEAM_NAMES[t],
            "is_keeper": me.is_keeper,
            "score_us": self.score[t],
            "score_them": self.score[1 - t],
            "players": [
                {
                    "id": p.name, "ours": p.team == t, "keeper": p.is_keeper, "you": k == g,
                    "pos": pt(p.pos), "vel": vec(p.vel), "facing": vec(p.facing),
                    "has_ball": b.holder == k,
                }
                for k, p in enumerate(self.players)
            ],
            "ball": {
                "pos": pt(b.pos),
                "vel": vec(b.vel),
                "holder": None if b.holder is None else self.players[b.holder].name,
                "last_touch": None if b.last_touch is None else self.players[b.last_touch].name,
            },
        }

    # ----- simulation -----

    def clamp(self, p: Player, xy: Sequence[float]) -> list[float]:
        cfg, r = self.cfg, self.cfg.player_radius
        x0, x1, y0, y1 = r, cfg.width - r, r, cfg.height - r
        if p.is_keeper:
            y0, y1 = max(y0, cfg.box_y0), min(y1, cfg.box_y1)
            if p.team == 0:
                x1 = cfg.box_depth
            else:
                x0 = cfg.width - cfg.box_depth
        return [min(max(xy[0], x0), x1), min(max(xy[1], y0), y1)]

    def step(self, actions: dict[int, Action]) -> TickRecord:
        cfg, b = self.cfg, self.ball
        events: list[Event] = []
        vels = [[0.0, 0.0] for _ in self.players]

        for g, p in enumerate(self.players):
            a = actions.get(g) or Action()
            if a.kind == "pass" and a.target is not None:
                if b.holder != g:
                    events.append({"type": "invalid", "player": p.name, "reason": "pass without the ball"})
                    continue
                dx, dy = a.target[0] - b.pos[0], a.target[1] - b.pos[1]
                d = math.hypot(dx, dy)
                if d < 1e-6:
                    events.append({"type": "invalid", "player": p.name, "reason": "pass target is the ball's position"})
                    continue
                b.vel = [dx / d * cfg.pass_speed, dy / d * cfg.pass_speed]
                b.holder, b.ignore, b.last_touch = None, g, g
                p.facing = [dx / d, dy / d]
                events.append({"type": "pass", "player": p.name, "target": [round(c, 2) for c in a.target]})
            elif a.kind == "move" and a.target is not None:
                tx, ty = self.clamp(p, a.target)
                dx, dy = tx - p.pos[0], ty - p.pos[1]
                d = math.hypot(dx, dy)
                if d > 1e-9:
                    s = min(cfg.carrier_speed if b.holder == g else cfg.player_speed, d)
                    vels[g] = [dx / d * s, dy / d * s]
                    p.facing = [dx / d, dy / d]

        start = [p.pos[:] for p in self.players]
        frames: list[Frame] = []
        scorer_team: int | None = None
        h = 1.0 / cfg.substeps
        for _ in range(cfg.substeps):
            if b.holder is not None:
                v = vels[b.holder]
                s = math.hypot(*v)
                if s > cfg.carrier_speed:
                    vels[b.holder] = [v[0] / s * cfg.carrier_speed, v[1] / s * cfg.carrier_speed]
            new = [self.clamp(p, (p.pos[0] + vels[g][0] * h, p.pos[1] + vels[g][1] * h))
                   for g, p in enumerate(self.players)]
            self._separate(new)

            if b.holder is None:
                nb = [b.pos[0] + b.vel[0] * h, b.pos[1] + b.vel[1] * h]
                hit = self._first_contact(b.pos, nb, new)
                if hit is not None:
                    g, contact = hit
                    b.pos = contact
                    for k, p in enumerate(self.players):
                        p.pos = new[k]
                    self._take(g, events, "reception" if self._same_team(g, b.last_touch) else "interception")
                else:
                    b.pos = nb
                    scorer_team = self._ball_walls()
            for g, p in enumerate(self.players):
                p.pos = new[g]

            if b.holder is not None:
                self._swing_held_ball(h)
                self._place_held_ball()
                self._touch_steals(events)
                scorer_team = self._held_ball_goal()
            if b.ignore is not None:
                q = self.players[b.ignore].pos
                if math.hypot(b.pos[0] - q[0], b.pos[1] - q[1]) > cfg.hold_dist:
                    b.ignore = None
            frames.append(self.frame())
            if scorer_team is not None:
                break

        if b.holder is None and cfg.ball_friction:
            b.vel = [c * (1 - cfg.ball_friction) for c in b.vel]
        for g, p in enumerate(self.players):
            p.vel = [p.pos[0] - start[g][0], p.pos[1] - start[g][1]]

        tick = self.tick
        self.tick += 1
        if scorer_team is not None:
            self.score[scorer_team] += 1
            lt = self.players[b.last_touch] if b.last_touch is not None else None
            events.append({
                "type": "goal", "team": TEAM_NAMES[scorer_team],
                "player": lt.name if lt else None,
                "own_goal": bool(lt and lt.team != scorer_team),
                "score": self.score[:],
                "frame": len(frames) - 1,  # index within this tick's frames where the ball went in
            })
            if not self.done:
                self.kickoff(1 - scorer_team)
                frames.append(self.frame())
        return {"tick": tick, "events": events, "frames": frames, "score": self.score[:], "actions": []}

    def frame(self) -> Frame:
        b = self.ball
        return {
            "p": [[round(p.pos[0], 2), round(p.pos[1], 2)] for p in self.players],
            "b": [round(b.pos[0], 2), round(b.pos[1], 2)],
            "h": -1 if b.holder is None else b.holder,
        }

    # ----- helpers -----

    def _same_team(self, g: int, other: int | None) -> bool:
        return other is not None and self.players[g].team == self.players[other].team

    def _separate(self, new: list[list[float]], iters: int = 4) -> None:
        """Push overlapping bodies apart so players are solid to each other. Every push in a
        pass is computed from the same positions and applied together, so the result doesn't
        depend on player order (which would favour one team)."""
        r2 = 2 * self.cfg.player_radius
        n = len(new)
        for _ in range(iters):
            shift = [[0.0, 0.0] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    dx, dy = new[j][0] - new[i][0], new[j][1] - new[i][1]
                    d = math.hypot(dx, dy)
                    if d >= r2:
                        continue
                    if d < 1e-9:  # exactly stacked: separate along x, which mirrors cleanly
                        dx, dy, d = 1.0, 0.0, 1.0
                        push = r2 / 2
                    else:
                        push = (r2 - d) / 2
                    ux, uy = dx / d, dy / d
                    shift[i][0] -= ux * push
                    shift[i][1] -= uy * push
                    shift[j][0] += ux * push
                    shift[j][1] += uy * push
            if not any(s[0] or s[1] for s in shift):
                return
            for i in range(n):
                if shift[i][0] or shift[i][1]:
                    new[i] = self.clamp(self.players[i], (new[i][0] + shift[i][0], new[i][1] + shift[i][1]))

    def _first_contact(self, b0: Point, b1: Point, new: list[list[float]]) -> tuple[int, list[float]] | None:
        """Earliest player whose body touches the free ball during this substep (swept test).
        Returns the player and the ball position at the moment of contact."""
        r = self.cfg.hold_dist
        hits: list[tuple[float, int]] = []
        for g, p in enumerate(self.players):
            if g == self.ball.ignore:
                continue
            d0 = (b0[0] - p.pos[0], b0[1] - p.pos[1])
            dd = ((b1[0] - b0[0]) - (new[g][0] - p.pos[0]), (b1[1] - b0[1]) - (new[g][1] - p.pos[1]))
            a = dd[0] ** 2 + dd[1] ** 2
            bq = 2 * (d0[0] * dd[0] + d0[1] * dd[1])
            c = d0[0] ** 2 + d0[1] ** 2 - r * r
            if c <= 0:
                t = 0.0
            elif a < 1e-12:
                continue
            else:
                disc = bq * bq - 4 * a * c
                if disc < 0:
                    continue
                t = (-bq - math.sqrt(disc)) / (2 * a)
                if not 0.0 <= t <= 1.0:
                    continue
            hits.append((t, g))
        if not hits:
            return None
        t_min = min(t for t, _ in hits)
        g = self.rng.choice([g for t, g in hits if t - t_min < 1e-9])
        return g, [b0[0] + (b1[0] - b0[0]) * t_min, b0[1] + (b1[1] - b0[1]) * t_min]

    def _take(self, g: int, events: list[Event], kind: str) -> None:
        """Player g gains the ball where it touched them; it then swings round to their front."""
        b, p = self.ball, self.players[g]
        prev = b.last_touch
        b.holder, b.last_touch, b.ignore, b.vel = g, g, None, [0.0, 0.0]
        b.angle = math.atan2(b.pos[1] - p.pos[1], b.pos[0] - p.pos[0])
        p.facing = [math.cos(b.angle), math.sin(b.angle)]
        self._place_held_ball()
        events.append({"type": kind, "player": p.name,
                       "from": None if prev is None else self.players[prev].name})

    def _carrier(self) -> Player:
        if self.ball.holder is None:
            raise RuntimeError("the ball is loose")
        return self.players[self.ball.holder]

    def _swing_held_ball(self, h: float) -> None:
        b = self.ball
        p = self._carrier()
        diff = angle_diff(math.atan2(p.facing[1], p.facing[0]), b.angle)
        max_turn = math.radians(self.cfg.ball_turn_rate) * h
        b.angle += max(-max_turn, min(max_turn, diff))

    def _place_held_ball(self) -> None:
        """Put the held ball at the carrier's edge. It may cross the opponent's goal line
        (that scores) but never a wall or the carrier's own goal line."""
        cfg, b = self.cfg, self.ball
        p = self._carrier()
        x = p.pos[0] + math.cos(b.angle) * cfg.hold_dist
        y = p.pos[1] + math.sin(b.angle) * cfg.hold_dist
        r = cfg.ball_radius
        y = min(max(y, r), cfg.height - r)
        in_mouth = cfg.goal_y0 <= y <= cfg.goal_y1
        lo = -math.inf if (in_mouth and p.team == 1) else r
        hi = math.inf if (in_mouth and p.team == 0) else cfg.width - r
        b.pos = [min(max(x, lo), hi), y]

    def _touch_steals(self, events: list[Event]) -> None:
        """An opponent whose body touches the held ball takes it."""
        b, cfg = self.ball, self.cfg
        hp = self._carrier()
        if hp.immune_until >= self.tick:
            return
        cands: list[tuple[float, int]] = []
        for g, p in enumerate(self.players):
            if p.team == hp.team:
                continue
            d = math.hypot(p.pos[0] - b.pos[0], p.pos[1] - b.pos[1])
            if d < cfg.hold_dist - 1e-9:
                cands.append((d, g))
        if cands:
            d_min = min(d for d, _ in cands)
            g = self.rng.choice([g for d, g in cands if d - d_min < 1e-9])
            self._take(g, events, "tackle")
            self.players[g].immune_until = self.tick + cfg.steal_immunity_ticks

    def _ball_walls(self) -> int | None:
        """Bounce a free ball off walls; return the scoring team if it entered a goal."""
        cfg, b = self.cfg, self.ball
        r = cfg.ball_radius
        x, y = b.pos
        if x < r or x > cfg.width - r:
            if cfg.goal_y0 <= y <= cfg.goal_y1:
                return 1 if x < r else 0  # left goal is A's, so B scores there
            b.pos[0] = 2 * r - x if x < r else 2 * (cfg.width - r) - x
            b.vel[0] = -b.vel[0]
        if y < r or y > cfg.height - r:
            b.pos[1] = 2 * r - y if y < r else 2 * (cfg.height - r) - y
            b.vel[1] = -b.vel[1]
        return None

    def _held_ball_goal(self) -> int | None:
        """A held ball pushed over the opponent's goal line (between the posts) scores."""
        cfg, b = self.cfg, self.ball
        team = self._carrier().team
        if team == 0 and b.pos[0] > cfg.width - cfg.ball_radius:
            return 0
        if team == 1 and b.pos[0] < cfg.ball_radius:
            return 1
        return None
