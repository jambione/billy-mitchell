"""Unit tests for east march, dungeon nav, and expanded play modes (no ROM)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.abstractions import Observation, Step  # noqa: E402
from billy.games.zelda.dungeon import read_dungeon_state  # noqa: E402
from billy.games.zelda.dungeon_nav import (  # noqa: E402
    dungeon_combat_decision,
    dungeon_explore_decision,
    dungeon_key_decision,
    pick_dungeon_direction,
)
from billy.games.zelda.east_march import (  # noqa: E402
    EAST_EDGE_FRAMES,
    EAST_EDGE_PLAN,
    EAST_HOP_FRAMES,
    EAST_HOP_PLAN,
    _fight_button,
    east_march_active,
    east_march_approach_decision,
    east_march_at_lip,
    east_march_combat_decision,
    east_march_cross_decision,
    east_march_decision,
    east_march_lane_decision,
    east_march_screen_clear,
    east_march_cross_reps,
    east_march_needs_cross,
    east_march_screen_cross_plan,
    east_march_threatened,
    east_march_lip_walk_plan,
    east_march_walk_cross_plan,
    is_east_march_walk_cross_plan,
    east_march_transition_plan,
    is_east_march_cross_plan,
    is_east_march_plan,
)
from billy.games.zelda.hazard_hooks import ZeldaHazardHooks  # noqa: E402
from billy.games.zelda.perception import DUNGEON_MODES, IN_PLAY_MODES, build_scene  # noqa: E402
from billy.knowledge.cache import CacheEntry  # noqa: E402
from billy.systems.nes import controller as c  # noqa: E402


def _scene(**overrides):
    ram = bytearray(0x800)
    ram[18] = overrides.get("game_mode", 5)
    ram[16] = overrides.get("current_level", 0)
    ram[235] = overrides.get("map_location", 119)
    ram[112] = overrides.get("link_x", 200)
    ram[132] = overrides.get("link_y", 141)
    ram[232] = 255 if not overrides.get("scrolling", False) else 0
    ram[1647] = overrides.get("hearts_byte", 0x22)
    ram[1623] = overrides.get("sword_level", 1)
    ram[1646] = overrides.get("keys", 0)
    ram[1649] = overrides.get("triforce", 0)
    ram[1569] = overrides.get("visited0", 119)
    if "enemy_x" in overrides:
        ram[114] = overrides["enemy_x"]
        ram[134] = overrides["enemy_y"]
        ram[849] = overrides.get("enemy_type", 7)
    if overrides.get("dungeon"):
        ram[16] = 1
        ram[249] = 0x03
    return build_scene(bytes(ram))


def test_in_play_modes_include_scroll_and_dungeon():
    assert 4 in IN_PLAY_MODES
    assert 10 in IN_PLAY_MODES
    assert 16 in IN_PLAY_MODES
    for mode in DUNGEON_MODES:
        assert mode in IN_PLAY_MODES


def test_dungeon_mode_sets_in_dungeon():
    scene = _scene(game_mode=9, current_level=0)
    assert scene.in_dungeon
    scene2 = _scene(game_mode=5, current_level=1)
    assert scene2.in_dungeon


def test_east_march_active_post_sword_row8():
    scene = _scene(map_location=120, sword_level=1, link_x=200, link_y=141)
    assert east_march_active(scene, visited={119, 120})
    scene_pre = _scene(map_location=119, sword_level=0)
    assert not east_march_active(scene_pre, visited={119})


def test_east_march_decision_commits_hop():
    scene = _scene(map_location=119, link_x=196, link_y=141)
    decision = east_march_decision(scene, c.RIGHT, "east→#120")
    assert decision is not None
    assert decision.plan == EAST_HOP_PLAN
    assert decision.plan[0].frames == EAST_HOP_FRAMES


def test_east_march_decision_blocked_by_enemy():
    scene = _scene(
        map_location=120, link_x=200, link_y=141,
        enemy_x=220, enemy_y=140, enemy_type=7,
    )
    assert not east_march_screen_clear(scene)
    assert east_march_threatened(scene)
    assert east_march_decision(scene, c.RIGHT, "east→#121") is None


def test_east_march_combat_decision_fights_ahead():
    scene = _scene(
        map_location=120, link_x=200, link_y=141,
        enemy_x=220, enemy_y=140, enemy_type=7,
    )
    fight = east_march_combat_decision(scene)
    assert fight is not None
    assert len(fight.plan) >= 2
    assert "east-march-fight" in fight.note


def test_east_march_combat_approaches_distant_enemy():
    scene = _scene(
        map_location=120, link_x=80, link_y=133,
        enemy_x=200, enemy_y=140, enemy_type=7,
    )
    fight = east_march_combat_decision(scene)
    assert fight is not None
    assert "east-march-approach" in fight.note


def test_east_march_decision_ignores_far_behind_enemy():
    scene = _scene(
        map_location=120, link_x=200, link_y=141,
        enemy_x=40, enemy_y=140, enemy_type=7,
    )
    assert not east_march_screen_clear(scene)
    assert not east_march_threatened(scene)
    assert east_march_decision(scene, c.RIGHT, "east→#121") is None


def test_east_march_lane_recenters_row():
    scene = _scene(map_location=120, link_x=160, link_y=120)
    lane = east_march_lane_decision(scene, c.RIGHT)
    assert lane is not None
    assert "lane-down" in lane.note


def test_is_east_march_plan():
    assert is_east_march_plan(EAST_HOP_PLAN)
    assert is_east_march_plan(EAST_EDGE_PLAN)
    assert not is_east_march_plan([Step(8, c.RIGHT)])


def test_east_march_transition_plan_edge_vs_hop():
    edge_scene = _scene(map_location=120, link_x=224, link_y=141)
    hop_scene = _scene(map_location=119, link_x=196, link_y=141)
    mid_lip = _scene(map_location=120, link_x=210, link_y=141)
    low_scene = _scene(map_location=120, link_x=120, link_y=141)
    assert east_march_transition_plan(edge_scene) == EAST_EDGE_PLAN
    assert east_march_transition_plan(mid_lip) is None
    assert east_march_transition_plan(hop_scene) == EAST_HOP_PLAN
    assert east_march_transition_plan(low_scene) is None


def test_east_march_edge_with_enemies_at_lip():
    scene = _scene(
        map_location=120, link_x=224, link_y=141,
        enemy_x=100, enemy_y=128, enemy_type=7,
    )
    assert east_march_transition_plan(scene) == EAST_EDGE_PLAN


def test_east_march_at_lip_requires_true_edge():
    assert east_march_at_lip(_scene(map_location=120, link_x=224, link_y=141))
    assert not east_march_at_lip(_scene(map_location=120, link_x=210, link_y=141))


def test_row_lock_never_chases_south_or_west():
    assert _fight_button(8, 24, row_lock=True) == c.RIGHT
    assert _fight_button(0, 40, row_lock=True) == c.RIGHT
    assert _fight_button(-32, -20, row_lock=True) == c.RIGHT


def test_east_march_cross_reps_scale_with_screen():
    assert east_march_cross_reps(119) == 35
    assert east_march_cross_reps(122) == 50
    assert east_march_cross_reps(123) == 60
    assert east_march_cross_reps(124) == 66


def test_east_march_needs_cross_deep_screens():
    west = _scene(map_location=123, link_x=80, link_y=141)
    mid = _scene(map_location=123, link_x=210, link_y=141)
    s120 = _scene(map_location=120, link_x=80, link_y=141)
    assert east_march_needs_cross(west)
    assert not east_march_needs_cross(mid)
    assert east_march_needs_cross(s120)


def test_is_east_march_cross_plan():
    assert is_east_march_cross_plan(east_march_screen_cross_plan(map_location=120))
    assert not is_east_march_cross_plan(EAST_EDGE_PLAN)
    assert not is_east_march_cross_plan([Step(12, c.UP)])


def test_east_march_cross_decision_alternates():
    scene = _scene(map_location=120, link_x=40, link_y=141, enemy_x=80, enemy_y=128, enemy_type=7)
    d0 = east_march_cross_decision(scene, tick=0)
    d1 = east_march_cross_decision(scene, tick=1)
    assert d0 is not None and d1 is not None
    assert d0.plan[0].buttons != d1.plan[0].buttons


def test_east_march_approach_walks_lip():
    scene = _scene(map_location=120, link_x=120, link_y=141)
    step = east_march_approach_decision(scene, c.RIGHT)
    assert step is not None
    assert "approach-lip" in step.note


def test_stale_cache_stales_north_replay_on_row8_east_march():
    hooks = ZeldaHazardHooks()
    scene = _scene(map_location=120, sword_level=1, link_x=120, link_y=141)
    from billy.abstractions import Observation
    obs = Observation(
        frame=1, progress=scene.objective_score(), score=0,
        level_label=scene.room_label, level_key=("overworld", 120),
        dead=False, summary="", ascii_map="", raw=scene, elevation=scene.link_y)
    north = CacheEntry(plan=[Step(48, c.UP)], reach_after=2000)
    assert hooks.stale_cache(obs, north)


def test_stale_cache_allows_east_hop_post_sword():
    hooks = ZeldaHazardHooks()
    scene = _scene(map_location=120, sword_level=1, link_x=200, link_y=141)
    obs = Observation(
        frame=1, progress=scene.objective_score(), score=0,
        level_label=scene.room_label, level_key=("overworld", 120),
        dead=False, summary="", ascii_map="", raw=scene, elevation=scene.link_y)
    cached = CacheEntry(plan=EAST_HOP_PLAN, reach_after=2000)
    assert not hooks.stale_cache(obs, cached)


def test_stale_cache_allows_short_combat_replay_on_screen_121():
    """Learn-from-death survivors on #121 are short — must not be staled like lip hops."""
    hooks = ZeldaHazardHooks()
    scene = _scene(
        map_location=121, sword_level=1, link_x=120, link_y=141,
        enemy_x=160, enemy_y=141, visited0=121,
    )
    ram = bytearray(0x800)
    ram[18] = 5
    ram[235] = 121
    ram[112] = 120
    ram[132] = 141
    ram[1623] = 1
    ram[1647] = 0x22
    ram[1569] = 121
    ram[1570] = 120
    ram[1571] = 121
    ram[114] = 160
    ram[134] = 141
    ram[849] = 7
    scene = build_scene(bytes(ram))
    obs = Observation(
        frame=1, progress=scene.objective_score(), score=0,
        level_label=scene.room_label, level_key=("overworld", 121),
        dead=False, summary="", ascii_map="", raw=scene, elevation=scene.link_y)
    learned = CacheEntry(
        plan=[Step(12, c.LEFT), Step(14, c.mask(c.LEFT, c.B))],
        reach_after=obs.progress + 40,
    )
    assert not hooks.stale_cache(obs, learned)


def test_stale_cache_stales_short_edge_hop_on_screen_120():
    hooks = ZeldaHazardHooks()
    scene = _scene(map_location=120, sword_level=1, link_x=120, link_y=141)
    obs = Observation(
        frame=1, progress=scene.objective_score(), score=0,
        level_label=scene.room_label, level_key=("overworld", 120),
        dead=False, summary="", ascii_map="", raw=scene, elevation=scene.link_y)
    cached = CacheEntry(plan=EAST_EDGE_PLAN, reach_after=2000)
    assert hooks.stale_cache(obs, cached)


def test_pick_dungeon_direction_prefers_unvisited():
    btn, label = pick_dungeon_direction(12, {12})
    assert btn in (c.DOWN, c.RIGHT, c.LEFT)
    assert "dungeon-" in label
    assert "#12" not in label


def test_dungeon_combat_decision_swords_near_enemy():
    scene = _scene(
        dungeon=True, map_location=12, link_x=120, link_y=120,
        enemy_x=140, enemy_y=120, enemy_type=7,
    )
    decision = dungeon_combat_decision(scene)
    assert decision is not None
    assert "dungeon-fight" in decision.note


def test_dungeon_key_decision_at_right_edge():
    scene = _scene(
        dungeon=True, map_location=12, keys=1, link_x=224, link_y=120,
    )
    decision = dungeon_key_decision(scene)
    assert decision is not None
    assert "dungeon-key-right" in decision.note


def test_walk_cross_plan_is_right_only():
    plan = east_march_walk_cross_plan(map_location=123)
    assert is_east_march_walk_cross_plan(plan)
    assert not is_east_march_walk_cross_plan(east_march_screen_cross_plan(map_location=123))


def test_stale_cache_stales_sword_cross_on_screen_123():
    hooks = ZeldaHazardHooks()
    scene = _scene(map_location=123, sword_level=1, link_x=80, link_y=141)
    obs = Observation(
        frame=1, progress=scene.objective_score(), score=0,
        level_label=scene.room_label, level_key=("overworld", 123),
        dead=False, summary="", ascii_map="", raw=scene, elevation=scene.link_y)
    sword = CacheEntry(
        plan=east_march_screen_cross_plan(map_location=123), reach_after=4000)
    assert hooks.stale_cache(obs, sword)
    lip = CacheEntry(plan=east_march_lip_walk_plan(map_location=123), reach_after=4000)
    assert not hooks.stale_cache(obs, lip)
    long_walk = CacheEntry(plan=east_march_walk_cross_plan(map_location=123), reach_after=4000)
    assert hooks.stale_cache(obs, long_walk)


def test_dungeon_explore_decision_emits_step():
    scene = _scene(dungeon=True, map_location=12, link_x=120, link_y=120)
    decision = dungeon_explore_decision(scene, {12}, [])
    assert decision is not None
    assert decision.plan[0].frames > 0