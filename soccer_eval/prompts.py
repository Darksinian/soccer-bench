"""Prompt text shown to model-controlled players. Everything is in the player's team frame."""
from __future__ import annotations

from collections.abc import Sequence

from .schema import Observation, PlayerObs, ShoutObs
from .sim import Config


def _comms_text(cfg: Config) -> tuple[str, str, str]:
    """(intro sentence, COMMUNICATION section, output format) for the configured comms mode."""
    if cfg.comms == "none":
        return ("You cannot communicate with your teammates; coordinate by anticipating what they will do.", "",
                '{"action": "move" | "pass" | "stay", "x": <number>, "y": <number>}')
    hearers = "your teammates" if cfg.comms == "team" else "every player on both teams"
    section = f"""
COMMUNICATION
- Each tick you may also shout a short message of up to {cfg.shout_max_chars} characters by adding "say" to your reply. It is heard by {hearers}.
- Everyone acts at the same time, so a shout made this tick is heard at the start of the NEXT tick, together with that tick's snapshot. You will see the shouts made last tick.
- Shouting is optional; leave "say" out or empty to stay quiet. Longer messages are cut off."""
    if cfg.comms == "all":
        section += f"""
- Opponents hear your shouts too. Coordinates in a shout are in the shouter's own frame: an opponent's point (x, y) is ({cfg.width:g} - x, {cfg.height:g} - y) in yours."""
    return (f"You cannot see your teammates' plans, but you can shout short messages to {hearers} (see COMMUNICATION).",
            section,
            '{"action": "move" | "pass" | "stay", "x": <number>, "y": <number>, "say": "<optional shout>"}')


# Formation roles by player index, used when cfg.roles is on; the keeper (0) keeps its usual role.
ROLES = {1: "a defender (y < 30 side)", 2: "a defender (y > 30 side)",
         3: "a forward (y < 30 side)", 4: "a forward (y > 30 side)"}


def _roles_text(cfg: Config) -> str:
    if not cfg.roles:
        return ""
    if cfg.n_per_team != 5:
        raise ValueError("roles needs 5 players per team")
    mx, my = cfg.width / 2, cfg.height / 2
    return f"""
YOUR TEAM'S FORMATION
- Player 0 is the goalkeeper.
- Players 1 and 2 are defenders and stay in your own half (x below {mx:g}): player 1 on the y < {my:g} side, player 2 on the y > {my:g} side.
- Players 3 and 4 are forwards and play in the opponent's half (x above {mx:g}): player 3 on the y < {my:g} side, player 4 on the y > {my:g} side.
- Hold your area. Leave it to chase the ball only if you are the closest teammate to it.
"""


def system_prompt(cfg: Config) -> str:
    W, H = cfg.width, cfg.height
    g0, g1 = cfg.goal_y0, cfg.goal_y1
    intro, comms_section, output_format = _comms_text(cfg)
    return f"""You are ONE player in a simulated 2D soccer match. Each team has {cfg.n_per_team} players (player 0 is the goalkeeper). Every player on both teams is controlled by a separate AI that sees the same snapshot and chooses an action at the same time as everyone else. {intro}

FIELD (all coordinates are from your team's point of view)
- The field spans x from 0 to {W:g} and y from 0 to {H:g}, with walls on all four sides.
- You ATTACK the goal on the right: the line x = {W:g} between y = {g0:g} and y = {g1:g}.
- You DEFEND the goal on the left: the line x = 0 between y = {g0:g} and y = {g1:g}.
- Players are circles of radius {cfg.player_radius:g}. The ball is a circle of radius {cfg.ball_radius:g}.
- Each goalkeeper must stay in its own box. Your keeper: x from 0 to {cfg.box_depth:g}, y from {cfg.box_y0:g} to {cfg.box_y1:g}. Opponent keeper: x from {W - cfg.box_depth:g} to {W:g}, same y range. Outfield players can go anywhere.

TIME
- One tick is one second. The match ends after {cfg.max_ticks} ticks or when a team reaches {cfg.goals_to_win} goals.

ACTIONS (choose exactly one per tick)
- "move" toward target (x, y): you walk in a straight line toward the target at up to {cfg.player_speed:g} units per tick ({cfg.carrier_speed:g} if you hold the ball) and stop if you reach it. If you hold the ball it moves with you.
- "pass" toward target (x, y): only allowed while you hold the ball. The ball leaves from its current position in a straight line toward the target at a constant {cfg.pass_speed:g} units per tick. It does NOT stop at the target: it keeps going and bounces off walls until it touches any player (teammate, opponent or goalkeeper), who then holds it. Shooting is a pass toward the opponent goal. You stay still on the tick you pass.
- "stay": do nothing.

BODIES AND POSSESSION
- Player bodies are solid: players push each other apart and cannot pass through one another.
- A held ball sits just outside its carrier's body ({cfg.hold_dist:g} units from the carrier's centre) on the side the carrier is FACING, which is the direction of their most recent movement. When the carrier changes direction, the ball swings around them at up to {cfg.ball_turn_rate:g} degrees per tick, so a sharp turn takes time and the ball passes through the space beside the carrier.
- A free ball attaches to the first player whose body touches it, at the point where it touched, then swings to their front.
- Tackling: any opponent whose body touches the ball takes it, including a held ball. Because the ball is in front of its carrier, defenders win it by getting in the carrier's path; the carrier's body shields it from behind. A player who has just won the ball by tackling cannot lose it for the rest of that tick and the next {cfg.steal_immunity_ticks} tick(s).

SCORING
- A goal is scored when the ball crosses a goal line between the posts: by a pass or shot, or by a carrier pushing the held ball over the opponent goal line. A passed or deflected ball that enters your own goal scores for the opponents; a held ball cannot be carried into its carrier's own goal.
- After a goal, everyone resets and the team that conceded kicks off.
{comms_section}{_roles_text(cfg)}
OUTPUT
Reply with a single JSON object and nothing else:
{output_format}
x and y are ignored for "stay"."""


def _xy(v: Sequence[float]) -> str:
    return f"({v[0]:g}, {v[1]:g})"


def user_prompt(obs: Observation, roles: bool = False) -> str:
    me = next(p for p in obs["players"] if p["you"])
    ours = [p for p in obs["players"] if p["ours"]]
    theirs = [p for p in obs["players"] if not p["ours"]]
    other = "B" if obs["team"] == "A" else "A"
    ball = obs["ball"]

    if ball["holder"]:
        holder = next(p for p in obs["players"] if p["id"] == ball["holder"])
        side = "your teammate" if holder["ours"] else "opponent"
        if holder["you"]:
            side = "YOU"
        ball_line = (f"held by {side} {holder['id']}; the ball is at {_xy(ball['pos'])}, "
                     f"carrier at {_xy(holder['pos'])} facing {_xy(holder['facing'])}")
    elif ball["vel"] == [0.0, 0.0]:
        ball_line = f"loose and still at {_xy(ball['pos'])}"
    else:
        ball_line = f"moving freely: at {_xy(ball['pos'])} with velocity {_xy(ball['vel'])} per tick"
    if ball["last_touch"]:
        ball_line += f"; last touched by {ball['last_touch']}"

    def row(p: PlayerObs) -> str:
        tags = []
        if p["keeper"]:
            tags.append("goalkeeper")
        if p["you"]:
            tags.append("YOU")
        if p["has_ball"]:
            tags.append(f"HAS THE BALL, facing {_xy(p['facing'])}")
        tag = f" [{', '.join(tags)}]" if tags else ""
        return f"- {p['id']}{tag}: position {_xy(p['pos'])}, moved {_xy(p['vel'])} last tick"

    shouts = []
    if "shouts" in obs:
        def who(s: ShoutObs) -> str:
            return "you" if s["you"] else "teammate" if s["ours"] else "opponent"
        heard = [f"- {s['id']} ({who(s)}): \"{s['text']}\"" for s in obs["shouts"]]
        shouts = ["Shouts heard from last tick:", *(heard or ["- (none)"]), ""]

    left = obs["max_ticks"] - obs["tick"]
    role = "the goalkeeper" if me["keeper"] else "an outfield player"
    if roles and not me["keeper"]:
        role = ROLES[int(me["id"][1:])]
    return "\n".join([
        f"Tick {obs['tick']} of {obs['max_ticks']} ({left} ticks left).",
        f"Score: your team ({obs['team']}) {obs['score_us']} - {obs['score_them']} opponents ({other}).",
        f"You are {me['id']}, {role}, at {_xy(me['pos'])}.",
        "",
        f"Ball: {ball_line}.",
        "",
        f"Your team {obs['team']} (attacking the goal at x = right edge):",
        *map(row, ours),
        "",
        f"Opponent team {other} (attacking your goal at x = 0):",
        *map(row, theirs),
        "",
        *shouts,
        "Choose your action for this tick.",
    ])
