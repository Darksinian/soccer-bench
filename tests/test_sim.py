"""Physics and harness checks. Run with: python -m pytest tests  (or python tests/test_sim.py)"""
import asyncio
import json
import math
import sys
from pathlib import Path
from collections.abc import Sequence
from email.message import Message
from typing import Any, NoReturn
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccer_eval.agents import OpenRouterAgent, parse_action
from soccer_eval.match import expand_team, play_match
from soccer_eval.schema import Observation
from soccer_eval.sim import Action, Config, Sim


def empty_field(cfg: Config | None = None) -> Sim:
    """A sim with every player parked in a corner so tests can place them freely."""
    sim = Sim(cfg or Config(n_per_team=2))
    for p in sim.players:
        p.pos = [50.0, 59.0] if not p.is_keeper else ([5.0, 22.0] if p.team == 0 else [95.0, 22.0])
        p.immune_until = -1
    return sim


def free_ball(sim: Sim, pos: Sequence[float], vel: Sequence[float]) -> None:
    b = sim.ball
    b.pos, b.vel, b.holder, b.ignore, b.last_touch = list(pos), list(vel), None, None, None


def test_constant_speed_and_wall_bounce() -> None:
    sim = empty_field()
    free_ball(sim, (50, 10), (0, -20))
    sim.step({})
    # 10 units to y=0.4 then reflects: 20 - 9.6 = 10.4 back up -> y = 0.4 + 10.4
    assert math.isclose(sim.ball.pos[1], 10.8, abs_tol=1e-6), sim.ball.pos
    assert sim.ball.vel == [0, 20]


def test_no_tunnelling() -> None:
    sim = empty_field()
    a1 = sim.players[1]
    a1.pos = [60.0, 30.0]
    free_ball(sim, (40, 30.5), (100, 0))  # 5 units per substep, much more than R + b_r
    sim.step({})
    assert sim.ball.holder == 1


def hold(sim: Sim, g: int, pos: Sequence[float], angle: float = 0.0) -> None:
    """Give player g the ball at `pos`, with the ball on the side given by `angle`."""
    p = sim.players[g]
    p.pos, p.facing = list(pos), [math.cos(angle), math.sin(angle)]
    b = sim.ball
    b.holder, b.angle, b.vel, b.ignore, b.last_touch = g, angle, [0.0, 0.0], None, g
    sim._place_held_ball()


def test_held_ball_sits_in_front_of_carrier() -> None:
    sim = empty_field()
    hold(sim, 1, (40, 30))
    assert sim.ball.pos == [40 + sim.cfg.hold_dist, 30]


def test_passer_does_not_catch_own_pass_but_opponent_does() -> None:
    sim = empty_field()
    hold(sim, 1, (40, 30))
    sim.players[3].pos = [55.0, 30.0]
    events = sim.step({1: Action("pass", (90, 30))})["events"]
    assert sim.ball.holder == 3
    assert [e["type"] for e in events] == ["pass", "interception"]


def test_backward_pass_goes_through_passer() -> None:
    sim = empty_field()
    hold(sim, 1, (40, 30))
    sim.step({1: Action("pass", (10, 30))})
    assert sim.ball.holder is None and sim.ball.pos[0] < 40


def test_tackle_from_front_wins_ball_body_shields_from_behind() -> None:
    for defender_x, expect_steal in [(44.0, True), (36.0, False)]:
        sim = empty_field()
        hold(sim, 1, (40, 30))
        sim.players[3].pos = [defender_x, 30.0]
        sim.step({3: Action("move", (40, 30))})
        assert (sim.ball.holder == 3) == expect_steal, (defender_x, sim.ball.holder)


def test_turning_swings_ball_through_the_side() -> None:
    # Defender stands beside the carrier. Reversing direction swings the ball past it.
    sim = empty_field()
    hold(sim, 1, (40, 30))
    sim.players[3].pos = [40.0, 32.6]
    sim.step({1: Action("move", (39.5, 30.0001))})
    assert sim.ball.holder == 3

    # Without the defender, the ball ends up on the new front after a 180 degree turn.
    sim = empty_field()
    hold(sim, 1, (40, 30))
    sim.step({1: Action("move", (39.5, 30.0001))})
    assert sim.ball.holder == 1 and sim.ball.pos[0] < sim.players[1].pos[0]


def test_bodies_do_not_overlap() -> None:
    sim = empty_field()
    sim.players[1].pos, sim.players[3].pos = [40.0, 30.0], [50.0, 30.0]
    free_ball(sim, (50, 5), (0, 0))
    sim.step({1: Action("move", (50, 30)), 3: Action("move", (40, 30))})
    a, b = sim.players[1].pos, sim.players[3].pos
    assert math.dist(a, b) >= 2 * sim.cfg.player_radius - 1e-6


def test_shot_scores_and_own_goal_counts_for_opponent() -> None:
    sim = empty_field()
    free_ball(sim, (90, 30), (20, 0))
    sim.ball.last_touch = 1
    ev = sim.step({})["events"]
    assert sim.score == [1, 0] and ev[-1]["type"] == "goal" and not ev[-1]["own_goal"]

    sim = empty_field()
    free_ball(sim, (10, 30), (-20, 0))
    sim.ball.last_touch = 1
    sim.players[0].pos = [5.0, 15.0]  # A's keeper out of the way
    ev = sim.step({})["events"]
    assert sim.score == [0, 1] and ev[-1]["own_goal"]


def test_carrying_ball_over_line_scores_but_not_into_own_goal() -> None:
    sim = empty_field()
    hold(sim, 1, (97, 30))
    sim.step({1: Action("move", (100, 30))})
    assert sim.score == [1, 0]

    sim = empty_field()
    hold(sim, 0, (2, 30), angle=math.pi)  # A's keeper, ball on the goal side
    sim.step({0: Action("move", (0, 30))})
    assert sim.score == [0, 0] and sim.ball.pos[0] >= sim.cfg.ball_radius


def test_keeper_stays_in_box() -> None:
    sim = empty_field()
    for _ in range(20):
        sim.step({0: Action("move", (80, 59))})
    x, y = sim.players[0].pos
    assert x <= sim.cfg.box_depth and y <= sim.cfg.box_y1


def test_observations_are_mirror_symmetric() -> None:
    sim = Sim(Config())
    sim.kickoff(0)
    obs_a = sim.observe(1)
    sim.kickoff(1)
    obs_b = sim.observe(sim.cfg.n_per_team + 1)
    def strip(o: Observation) -> list[tuple[bool, bool, list[float], bool]]:
        return [(p["ours"], p["keeper"], p["pos"], p["has_ball"]) for p in o["players"]]

    a, b = strip(obs_a), strip(obs_b)
    n = sim.cfg.n_per_team
    assert a == b[n:] + b[:n]  # same picture, with the team order swapped


def test_parse_action_variants() -> None:
    assert parse_action('{"action":"stay","x":0,"y":0}')["action"] == "stay"
    assert parse_action('Sure!\n```json\n{"action": "move", "x": 1, "y": 2}\n```')["x"] == 1
    assert parse_action('I will pass. {"action": "pass", "x": 100, "y": 30} done')["action"] == "pass"


def test_team_specs() -> None:
    assert expand_team("scripted", 3) == ["scripted"] * 3
    assert expand_team("scripted, x/y , x/y@low", 3) == ["scripted", "x/y", "x/y@low"]


def test_openrouter_agent_with_stubbed_http() -> None:
    calls = []

    def fake_post(self: OpenRouterAgent, body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        return {"model": body["model"], "choices": [{"message": {"content": '{"action":"move","x":70,"y":20}'}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 30, "cost": 0.001}}

    with patch.object(OpenRouterAgent, "_post", fake_post):
        cfg = Config(max_ticks=3)
        result = asyncio.run(play_match(cfg, ("scripted,fake/model@low,fake/model@low", "random")))
    assert len(calls) == 2 * 3
    assert calls[0]["reasoning"] == {"effort": "low"} and calls[0]["model"] == "fake/model"
    assert "max_tokens" not in calls[0]
    assert "You are A1" in calls[0]["messages"][1]["content"] or "You are A2" in calls[0]["messages"][1]["content"]
    assert result["stats"]["A"]["in"] == 900 * 6 and result["stats"]["A"]["errors"] == 0


def test_physics_is_side_symmetric() -> None:
    """A game kicked off by B must be the exact mirror of one kicked off by A."""
    from soccer_eval.agents import ScriptedAgent
    from soccer_eval.match import to_action
    from soccer_eval.sim import mirror

    cfg = Config(max_ticks=60)
    n = cfg.n_per_team

    def run(first: int) -> list[tuple[list[list[float]], list[float]]]:
        sim, agent = Sim(cfg), ScriptedAgent(cfg)
        states: list[tuple[list[list[float]], list[float]]] = []
        sim.kickoff(first)
        while not sim.done:
            sim.step({g: to_action(cfg, sim.players[g].team, agent.decide(sim.observe(g))) for g in range(2 * n)})
            states.append(([p.pos[:] for p in sim.players], sim.ball.pos[:]))
        return states

    for (pa, ba), (pb, bb) in zip(run(0), run(1)):
        for g in range(2 * n):
            assert all(abs(x - y) < 1e-6 for x, y in zip(mirror(cfg, 1, pa[g]), pb[(g + n) % (2 * n)]))
        assert all(abs(x - y) < 1e-6 for x, y in zip(mirror(cfg, 1, ba), bb))


def test_billing_error_stops_the_match() -> None:
    import io
    import urllib.error
    import urllib.request

    from soccer_eval.agents import FatalAPIError

    def refuse(req: urllib.request.Request, timeout: float | None = None) -> NoReturn:
        raise urllib.error.HTTPError(req.full_url, 402, "Payment Required", Message(), io.BytesIO(b'{"error":"credits"}'))

    agent = OpenRouterAgent(Config(), "fake/model", None)
    with patch.object(urllib.request, "urlopen", refuse):
        import os
        os.environ.setdefault("OPENROUTER_API_KEY", "test")
        try:
            agent._post({"model": "fake/model"})
            raise AssertionError("expected FatalAPIError")
        except FatalAPIError:
            pass


def test_shouts_are_heard_next_tick_by_teammates_only() -> None:
    from soccer_eval.prompts import system_prompt

    seen = []

    def fake_post(self: OpenRouterAgent, body: dict[str, Any]) -> dict[str, Any]:
        user = body["messages"][1]["content"]
        me = user.split("You are ")[1].split(",")[0]
        seen.append((me, user))
        say = {"action": "stay", "x": 0, "y": 0, "say": f"  {me} here\nover   " + "x" * 200}
        return {"choices": [{"message": {"content": json.dumps(say)}}], "usage": {}}

    with patch.object(OpenRouterAgent, "_post", fake_post):
        cfg = Config(max_ticks=2, comms="team")
        result = asyncio.run(play_match(cfg, ("fake/m", "fake/m")))

    first_tick = [a for a in result["log"][1]["actions"]]
    assert all(len(a["say"]) == cfg.shout_max_chars and a["say"].startswith(a["player"] + " here over x") for a in first_tick)
    tick1_prompts = {me: user for me, user in seen[6:]}
    assert 'A2 (teammate): "A2 here over' in tick1_prompts["A1"]
    assert 'A1 (you): "A1 here over' in tick1_prompts["A1"]
    assert "B1" not in tick1_prompts["A1"].split("Shouts heard")[1]
    assert "Shouts heard from last tick:\n- (none)" in seen[0][1]
    assert "COMMUNICATION" in system_prompt(cfg) and "COMMUNICATION" not in system_prompt(Config())
    assert result["stats"]["A"]["shouts"] == 6


def test_no_shouts_without_comms() -> None:
    obs = Sim(Config()).observe(1)
    from soccer_eval.prompts import user_prompt
    assert "Shouts" not in user_prompt(obs)


def test_per_team_comms() -> None:
    seen = []

    def fake_post(self: OpenRouterAgent, body: dict[str, Any]) -> dict[str, Any]:
        user = body["messages"][1]["content"]
        me = user.split("You are ")[1].split(",")[0]
        seen.append((me, body["messages"][0]["content"], user))
        reply = {"action": "stay", "x": 0, "y": 0, "say": f"{me} talking"}
        return {"choices": [{"message": {"content": json.dumps(reply)}}], "usage": {}}

    with patch.object(OpenRouterAgent, "_post", fake_post):
        result = asyncio.run(play_match(Config(max_ticks=2), ("fake/m", "fake/m"), comms=("team", "none")))

    assert result["comms"] == {"A": "team", "B": "none"}
    assert result["stats"]["A"]["shouts"] == 6 and result["stats"]["B"]["shouts"] == 0
    for me, system, user in seen:
        if me.startswith("A"):
            assert "COMMUNICATION" in system and "Shouts heard" in user
        else:
            assert "COMMUNICATION" not in system and "Shouts heard" not in user and "talking" not in user


def test_verifier_replay_matches_live_match_and_counts_missing_ticks_as_stay() -> None:
    from soccer_eval.verify import replay

    for teams, n in ((("random", "scripted"), 3), (("scripted", "scripted"), 5)):
        cfg = Config(n_per_team=n, max_ticks=40)
        live = asyncio.run(play_match(cfg, teams, seed=3, comms=("team", "team")))
        r = replay(cfg.to_dict(), 3, [t.get("exact_actions") for t in live["log"][1:]])
        assert r["score"] == [live["score"]["A"], live["score"]["B"]]
        assert r["ticks_logged"] == live["ticks_played"]

    empty = replay(Config(n_per_team=3, max_ticks=40).to_dict(), 0, [])
    assert empty["ticks_logged"] == 0 and empty["score"] == [0, 0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def test_roles_add_formation_only_for_that_side() -> None:
    from soccer_eval.prompts import system_prompt

    seen = {}

    def fake_post(self: OpenRouterAgent, body: dict[str, Any]) -> dict[str, Any]:
        user = body["messages"][1]["content"]
        seen[user.split("You are ")[1].split(",")[0]] = (body["messages"][0]["content"], user)
        return {"choices": [{"message": {"content": '{"action": "stay"}'}}], "usage": {}}

    with patch.object(OpenRouterAgent, "_post", fake_post):
        result = asyncio.run(play_match(Config(n_per_team=5, max_ticks=1), ("fake/m", "fake/m"), roles=(True, False)))

    assert result["roles"] == {"A": True, "B": False}
    assert "FORMATION" in seen["A3"][0] and "FORMATION" not in seen["B3"][0]
    assert "You are A3, a forward (y < 30 side)" in seen["A3"][1]
    assert "You are A0, the goalkeeper" in seen["A0"][1]
    assert "You are B3, an outfield player" in seen["B3"][1]
    assert "FORMATION" not in system_prompt(Config(n_per_team=5))
