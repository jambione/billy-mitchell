"""Unit tests for SMB 1-3 lift-band detection and frame search helpers."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.abstractions import Observation, Step
from billy.games.smb.lift_search import (LIFT_GOAL_X, is_lift_death, lift_approach_zone,
                                         lift_band, lift_cacheable, lift_stall_visit_cap,
                                         lift_zone, lift_zone_at, moves_to_plan, Move)


def _obs(level: str, x: int, on_ground: bool = True) -> Observation:
    og = on_ground

    class _Raw:
        on_ground = og
    return Observation(
        frame=1, progress=x, score=0, level_label=level,
        level_key=(0, 2, 3), dead=False, summary="", ascii_map="", raw=_Raw(),
    )


def test_lift_band_detects_1_3_on_ground_range():
    assert lift_band(_obs("1-3", 636))
    assert lift_band(_obs("1-3", 590))
    assert lift_band(_obs("1-3", 600))
    assert lift_band(_obs("1-3", 760))
    assert lift_band(_obs("1-3", 790))
    assert not lift_band(_obs("1-3", 550))
    assert lift_approach_zone(_obs("1-3", 500))
    assert not lift_approach_zone(_obs("1-3", 600))
    assert lift_band(_obs("1-3", 800))
    assert not lift_band(_obs("1-3", 810))
    assert not lift_band(_obs("1-2", 636))
    assert lift_zone(_obs("1-3", 636, on_ground=False))
    assert not lift_band(_obs("1-3", 636, on_ground=False))


def test_lift_cacheable_requires_full_crossing():
    assert not lift_cacheable(735)
    assert not lift_cacheable(LIFT_GOAL_X - 1)
    assert lift_cacheable(LIFT_GOAL_X)
    assert lift_cacheable(735, crossed=True)


def test_lift_stall_visit_cap_default_is_high():
    assert lift_stall_visit_cap() >= 32


def test_is_lift_death_and_zone_at():
    assert is_lift_death("1-3", 740)
    assert is_lift_death("1-3", LIFT_GOAL_X + 20)
    assert not is_lift_death("1-2", 740)
    assert not is_lift_death("1-3", 400)
    assert lift_zone_at("1-3", 590)
    assert not lift_zone_at("1-3", 500)


def test_moves_to_plan_roundtrip():
    plan = moves_to_plan([Move("idle", 0, 44), Move("jump", 131, 16)])
    assert plan == [Step(44, 0), Step(16, 131)]


@pytest.mark.skipif(not os.path.isfile("data/rl/states/smb_1_3_lift.state"),
                    reason="lift savestate not present")
def test_frame_lift_search_runs_on_savestate():
    from billy.games.smb import SmbGame
    from billy.games.smb.lift_search import frame_lift_search

    game = SmbGame()
    session = game.system.connect()
    with open("data/rl/states/smb_1_3_lift.state", "rb") as f:
        session.reset()
        session.env.em.set_state(f.read())
        session._refresh_ram()

    def observe():
        st = session.read_state()
        return game.observe(st.frame, st.ram)

    start = observe()
    plan, reach, _crossed = frame_lift_search(session, observe, min_gain=8, depth=2, beam=8)
    session.close()
    assert start.level_label == "1-3"
    assert reach >= start.progress or plan is None