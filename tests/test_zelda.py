"""Zelda adapter — boots and is controllable via the top-down reflex.

Requires the Zelda ROM to be present in stable-retro's experimental integration
(user-supplied ROM, copyright). Skips cleanly when the integration isn't available.
"""
import os
import sys
from collections import deque

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.games.zelda import ZeldaGame  # noqa: E402
from billy.games.zelda.curiosity import (
    cave_approach_button,
    curious_exit,
    needs_cave_approach,
    requires_start_cave_inspection,
)  # noqa: E402
from billy.games.zelda.vision import detect_cave_mouths  # noqa: E402
from billy.games.zelda.explore import pick_explore_direction  # noqa: E402
from billy.games.zelda.hazard_hooks import ZeldaHazardHooks  # noqa: E402
from billy.games.zelda.perception import build_scene  # noqa: E402
from billy.games.zelda.walkthrough import (
    LEVEL_1_SCREEN,
    SEA_EAST_SCREEN,
    grid_to_screen,
)  # noqa: E402
from billy.systems.nes import controller as c  # noqa: E402


def _has_rom() -> bool:
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import stable_retro as retro
        from stable_retro.data import Integrations
        env = retro.make("LegendOfZeldaPRG0-Nes", render_mode="rgb_array",
                         inttype=Integrations.EXPERIMENTAL)
        env.close()
        return True
    except Exception:
        return False


def test_zelda_level_semantics():
    game = ZeldaGame()
    ow = ("overworld", 119)
    ow2 = ("overworld", 120)
    dg = ("dungeon-1", 0)
    assert not game.level_cleared(ow, ow2)
    assert game.screen_changed(ow, ow2)
    assert game.level_cleared(ow, dg)
    assert not game.search_area_advance(ow, ow2)


def test_walkthrough_grid_mapping():
    assert grid_to_screen(8, 8) == 119
    assert grid_to_screen(16, 8) == SEA_EAST_SCREEN
    assert grid_to_screen(8, 4) == LEVEL_1_SCREEN


def test_explore_anti_oscillation():
    """Prefer unvisited east over backtracking west."""
    visited = {119, 120}
    btn, label, dest = pick_explore_direction(
        120, visited, recent=deque([119, 120]), sword_level=1)
    assert dest != 119


def test_explore_start_cave_before_sword():
    btn, label, dest = pick_explore_direction(
        119, {119}, sword_level=0, link_x=120, link_y=141)
    assert dest == 119
    assert btn in (c.LEFT, c.UP)
    assert "inspect" in label or "walkthrough" in label


def test_explore_east_after_sword():
    btn, label, dest = pick_explore_direction(
        119, {119}, sword_level=1, link_x=120, link_y=141)
    assert btn == c.RIGHT and dest == 120
    assert "east" in label


def test_curiosity_start_screen_targets_nw_cave():
    curious = curious_exit(119, {119}, sword_level=0, link_x=120, link_y=141)
    assert curious is not None
    btn, label, dest = curious
    assert dest == 119
    assert btn in (c.LEFT, c.UP)


def test_start_screen_wont_wander_before_sword():
    assert requires_start_cave_inspection(119, {119}, sword_level=0)
    assert not requires_start_cave_inspection(119, {119}, sword_level=1)
    btn, label, dest = pick_explore_direction(
        119, {119}, sword_level=0, link_x=120, link_y=141)
    assert dest == 119
    btn2, _, dest2 = pick_explore_direction(
        119, {119}, sword_level=1, link_x=120, link_y=141)
    assert dest2 == 120


def test_curiosity_continues_north_through_cave_chain():
    btn, label, dest = pick_explore_direction(
        103, {119, 103}, sword_level=1, max_hearts=5)
    assert btn == c.UP and dest == 87
    assert "curious" in label or "walkthrough" in label


def test_cave_approach_when_far_from_mouth():
    assert needs_cave_approach(119, 120, 141)
    approach = cave_approach_button(119, 120, 141)
    assert approach is not None
    btn, note = approach
    assert btn in (c.LEFT, c.UP)
    assert "inspect-cave-approach" in note
    assert not needs_cave_approach(119, 65, 85)


def test_vision_detects_nw_black_square_cave():
    import numpy as np
    frame = np.full((224, 240, 3), 180, dtype=np.uint8)
    # NW cave mouth (16×16) — tile column 3, row 4
    frame[64:80, 48:64] = 0
    mouths = detect_cave_mouths(frame)
    assert mouths
    x, y = mouths[0]
    assert x < 100
    assert 60 <= y <= 90


def test_curious_exit_prefers_visible_black_square_cave():
    curious = curious_exit(
        119, {119}, cave_mouths=((60, 76),), sword_level=1, link_x=120, link_y=141)
    assert curious is not None
    btn, label, dest = curious
    assert btn == c.RIGHT and dest == 120


def test_objective_score_weights_exploration():
    ram = bytearray(0x800)
    ram[112] = 100
    ram[132] = 141
    ram[18] = 5
    ram[16] = 0
    ram[235] = 119
    ram[1647] = 0x22
    ram[1645] = 5
    ram[1649] = 1
    ram[1623] = 1
    ram[1569] = 119
    ram[1570] = 120
    s = build_scene(bytes(ram))
    base = s.objective_score()
    ram[1571] = 135
    s2 = build_scene(bytes(ram))
    assert s2.objective_score() > base


def test_zelda_hazard_hooks_combat_zone():
    hooks = ZeldaHazardHooks()
    ram = bytearray(0x800)
    ram[18] = 5
    ram[16] = 0
    ram[235] = 119
    ram[1647] = 0x22
    ram[113] = 140
    ram[133] = 140
    ram[848] = 1
    scene = build_scene(bytes(ram))
    from billy.abstractions import Observation
    obs = Observation(
        frame=1, progress=500, score=0, level_label=scene.room_label,
        level_key=("overworld", 119), dead=False, summary="", ascii_map="",
        raw=scene, elevation=scene.link_y)
    assert hooks.in_special_zone(obs)
    assert hooks.stall_break_exempt(obs)


def test_zelda_perception_ignores_phantom_enemy_slots():
    """Slots with stale x/y but enemy_type=0 must not count as on-screen enemies."""
    ram = bytearray(0x800)
    ram[18] = 5
    ram[16] = 0
    ram[235] = 103
    ram[1647] = 0x22
    ram[113] = 112
    ram[133] = 125
    ram[848] = 0
    ram[114] = 140
    ram[134] = 140
    ram[849] = 7
    scene = build_scene(bytes(ram))
    assert scene.enemy_count() == 1
    assert scene.enemies[0].enemy_type == 7


def test_zelda_perception_reads_ground_items():
    ram = bytearray(0x800)
    ram[18] = 11
    ram[16] = 0
    ram[235] = 119
    ram[1647] = 0x22
    ram[113] = 120
    ram[133] = 128
    ram[173] = 2
    ram[112] = 112
    ram[132] = 140
    scene = build_scene(bytes(ram))
    assert scene.item_count() == 1
    assert scene.items[0].x == 120
    assert scene.enemy_count() == 0
    near = scene.nearest_ground_item()
    assert near is not None
    dx, dy, item = near
    assert item.item_type == 2


def test_zelda_perception_decodes_start_state():
    ram = bytearray(0x800)
    ram[112] = 120
    ram[132] = 141
    ram[18] = 5
    ram[16] = 0
    ram[235] = 119
    ram[1647] = 0x22
    ram[1645] = 12
    ram[1569] = 119
    scene = build_scene(bytes(ram), frame=1)
    assert scene.in_play
    assert scene.link_x == 120
    assert scene.health == 2
    assert scene.max_hearts == 3
    assert scene.rupees == 12
    assert scene.room_label == "overworld #119"


@pytest.mark.skipif(not _has_rom(), reason="Zelda ROM not in stable-retro experimental integration")
def test_zelda_boots_and_advances():
    game = ZeldaGame()
    assert game.RETRO_GAME == "LegendOfZeldaPRG0-Nes"
    session = game.system.connect()
    session.wait_until_live()
    obs = game.boot(session)
    assert "overworld" in obs.level_label
    assert obs.progress > 0 and not obs.dead
    from billy.games.zelda.reflex import ZeldaReflex
    assert isinstance(game.make_reflex(), ZeldaReflex)
    session.close()