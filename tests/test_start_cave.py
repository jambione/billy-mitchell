"""ROM-verified start-cave macro — plan structure and phase detection."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.games.zelda.perception import Scene  # noqa: E402
from billy.games.zelda.start_cave import (  # noqa: E402
    APPROACH_PLAN,
    CAVE_ATTEMPT_TIMEOUT_FRAMES,
    CLIMB_PLAN,
    ENTER_PLAN,
    EXIT_PLAN,
    FULL_FROM_APPROACH,
    INTERIOR_PLAN,
    SWORD_PICKUP_PLAN,
    TEXT_PLAN,
    cave_attempt_exhausted,
    cave_quest_active,
    has_wooden_sword,
    interior_phase,
    macro_candidates,
    phase_plan,
)
from billy.systems.nes import controller as c  # noqa: E402


def _scene(*, link_y: int, sword_level: int = 0, in_cave: bool = True) -> Scene:
    return Scene(
        frame=1,
        link_x=112,
        link_y=link_y,
        direction=0,
        game_mode=11 if in_cave else 5,
        current_level=0,
        map_location=119,
        next_location=119,
        health=3,
        max_hearts=3,
        partial_heart=0,
        triforce_pieces=0,
        sword_level=sword_level,
        rupees=0,
        keys=0,
        bombs=0,
        scrolling=False,
        visited_screens=(119,),
    )


def test_plan_structure():
    assert len(TEXT_PLAN) == 35
    assert all(step.buttons == c.A for step in TEXT_PLAN)
    assert len(CLIMB_PLAN) == 18
    assert all(step.buttons == c.UP for step in CLIMB_PLAN)
    assert len(SWORD_PICKUP_PLAN) == 6
    assert SWORD_PICKUP_PLAN[0].buttons == c.RIGHT
    assert len(EXIT_PLAN) == 1 and EXIT_PLAN[0].buttons == c.DOWN
    assert INTERIOR_PLAN == TEXT_PLAN + CLIMB_PLAN + SWORD_PICKUP_PLAN + EXIT_PLAN
    assert FULL_FROM_APPROACH == APPROACH_PLAN + ENTER_PLAN + INTERIOR_PLAN


def test_interior_phase_detection():
    assert interior_phase(_scene(link_y=210, in_cave=True)) == "text"
    assert interior_phase(_scene(link_y=150, in_cave=True)) == "climb"
    assert interior_phase(_scene(link_y=141, sword_level=0, in_cave=True)) == "pickup"
    assert interior_phase(_scene(link_y=141, sword_level=1, in_cave=True)) == "exit"
    assert interior_phase(_scene(link_y=205, sword_level=1, in_cave=True)) == "exit"
    assert interior_phase(_scene(link_y=141, in_cave=False)) == "overworld"


def test_phase_plan_keys():
    assert phase_plan("text") == TEXT_PLAN
    assert phase_plan("pickup") == SWORD_PICKUP_PLAN
    assert phase_plan("unknown") == []


def test_cave_quest_and_timeout():
    assert has_wooden_sword(1)
    assert not has_wooden_sword(0)
    assert cave_quest_active(119, 0, in_cave=False)
    assert not cave_quest_active(119, 1, in_cave=False)
    assert cave_quest_active(119, 0, in_cave=True)
    assert not cave_quest_active(119, 0, in_cave=True, cave_gave_up=True)
    assert not cave_attempt_exhausted(100, 0)
    assert cave_attempt_exhausted(CAVE_ATTEMPT_TIMEOUT_FRAMES, 0)
    assert not cave_attempt_exhausted(CAVE_ATTEMPT_TIMEOUT_FRAMES, 1)


def test_macro_candidates_nonempty():
    cands = macro_candidates()
    assert cands
    assert INTERIOR_PLAN in cands
    assert SWORD_PICKUP_PLAN in cands


def test_walkthrough_skips_sword_after_timeout():
    from billy.games.zelda.walkthrough import current_phase, route_step

    assert current_phase(
        map_location=119, sword_level=0, max_hearts=3,
        visited={119}, in_cave=False, cave_gave_up=True) == "east_to_sea"
    step = route_step(
        119, sword_level=0, max_hearts=3, visited={119},
        in_cave=False, link_x=120, link_y=141, cave_gave_up=True)
    assert step is not None
    assert step.button == c.RIGHT