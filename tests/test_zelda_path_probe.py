"""Unit tests for Zelda path probe helpers (no ROM)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.games.zelda.path_probe import (  # noqa: E402
    keyframe_reason,
    load_plan,
    parse_script,
    plan_from_jsonable,
    plan_to_jsonable,
    scene_record,
)
from billy.games.zelda.perception import Scene  # noqa: E402
from billy.games.zelda.start_cave import FULL_FROM_APPROACH  # noqa: E402
from billy.systems.nes import controller as c  # noqa: E402


def _scene(**kw) -> Scene:
    defaults = dict(
        frame=1, link_x=120, link_y=141, direction=0, game_mode=5,
        current_level=0, map_location=119, next_location=119,
        health=3, max_hearts=3, partial_heart=0, triforce_pieces=0,
        sword_level=0, rupees=0, keys=0, bombs=0, scrolling=False,
        visited_screens=(119,),
    )
    defaults.update(kw)
    return Scene(**defaults)


def test_parse_script():
    plan = parse_script("left:35,up:25")
    assert len(plan) == 2
    assert plan[0].frames == 35 and plan[0].buttons == c.LEFT
    assert plan[1].buttons == c.UP


def test_load_plan_module():
    plan = load_plan("billy.games.zelda.start_cave:FULL_FROM_APPROACH")
    assert list(plan) == list(FULL_FROM_APPROACH)


def test_plan_json_roundtrip():
    data = plan_to_jsonable(FULL_FROM_APPROACH[:3])
    back = plan_from_jsonable(data)
    assert len(back) == 3


def test_scene_record_fields():
    row = scene_record(_scene(game_mode=11, link_y=212))
    assert row["screen"] == 119
    assert row["link"] == [120, 212]
    assert row["interior_phase"] == "text"
    assert row["faq_phase"] == "wooden_sword"


def test_keyframe_on_screen_change():
    a = scene_record(_scene(map_location=119), t=0)
    b = scene_record(_scene(map_location=120), t=1)
    assert keyframe_reason(a, b) == "screen_120"