"""Zelda combat: the sword is the A button (not the B item slot), and engagement range is
health-gated by the sword beam (full health reaches across the screen; below full is melee only).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.abstractions import Observation  # noqa: E402
from billy.games.zelda.dungeon_nav import dungeon_combat_decision  # noqa: E402
from billy.games.zelda.perception import build_scene  # noqa: E402
from billy.games.zelda.reflex import ZeldaReflex, _sword, combat_candidates  # noqa: E402
from billy.systems.nes import controller as c  # noqa: E402

# Heart byte 0x066F is 0-indexed: 3/3 hearts -> health 2, max_hearts 3 (byte 0x22); 0x0670 is the
# current heart's partial fill (0xFF = full). Beam fires only at completely full health.
FULL = 0x22   # 3/3 hearts; pair with partial=0xFF -> full (sword beam ready)
HURT = 0x21   # 2/3 hearts -> below full (melee only)


def _obs(**over) -> Observation:
    ram = bytearray(0x800)
    ram[18] = over.get("game_mode", 5)          # in-play, not a cave
    ram[16] = over.get("current_level", 0)      # overworld
    ram[235] = over.get("map_location", 72)     # not START(119); grid row != 8 -> not east-march
    ram[112] = over.get("link_x", 120)
    ram[132] = over.get("link_y", 120)
    ram[232] = 255                              # not scrolling
    ram[1647] = over.get("hearts_byte", HURT)
    ram[1648] = over.get("partial", 0xFF)       # current heart fill (0xFF = full)
    ram[1623] = over.get("sword_level", 1)
    ram[1569] = over.get("visited0", 72)
    if "enemy_x" in over:
        ram[114] = over["enemy_x"]
        ram[134] = over["enemy_y"]
        ram[849] = over.get("enemy_type", 7)
    if over.get("dungeon"):
        ram[16] = 1
        ram[249] = 0x03
    scene = build_scene(bytes(ram))
    return Observation(
        frame=1, progress=scene.objective_score(), score=0,
        level_label=scene.room_label, level_key=(scene.realm, ram[235]),
        dead=False, summary="", ascii_map="", raw=scene, elevation=scene.link_y)


def _stepped(obs: Observation):
    reflex = ZeldaReflex()
    reflex.reset(obs)
    return reflex.step(obs)


# --- the button itself --------------------------------------------------------------------------

def test_sword_helper_presses_A_not_B():
    plan = _sword(c.RIGHT)
    assert plan[0].buttons & c.A, "sword must press A"
    assert not (plan[0].buttons & c.B), "sword must NOT press the B item slot"


def test_combat_candidates_swing_with_A_never_B():
    obs = _obs(enemy_x=140, enemy_y=120)
    plans = combat_candidates(obs)
    assert any(any(s.buttons & c.A for s in p) for p in plans), "some candidate must swing (A)"
    for p in plans:
        for s in p:
            assert not (s.buttons & c.B), f"no candidate may press B: {p}"


def test_dungeon_combat_swings_with_A():
    scene = _obs(dungeon=True, map_location=12, link_x=120, link_y=120,
                 enemy_x=140, enemy_y=120).raw
    decision = dungeon_combat_decision(scene)
    assert decision is not None
    assert any(s.buttons & c.A for s in decision.plan), "dungeon swing must press A"
    assert all(not (s.buttons & c.B) for s in decision.plan), "dungeon swing must not press B"


# --- health-gated engagement (the sword beam) ---------------------------------------------------

def test_below_full_stabs_adjacent_enemy_with_A():
    # health 2 but a depleted partial heart -> below full (no beam) yet above the retreat floor.
    # dx=20 (<= ATTACK_RANGE) -> melee; below full still fights what's in reach.
    decision = _stepped(_obs(hearts_byte=0x22, partial=100, link_x=120, enemy_x=140, enemy_y=120))
    assert "sword enemy" in decision.note
    assert "[beam]" not in decision.note
    assert any(s.buttons & c.A for s in decision.plan)


def test_full_health_beam_engages_distant_enemy():
    # dx=100 (> ATTACK_RANGE) -> only reachable via the full-health beam.
    decision = _stepped(_obs(hearts_byte=FULL, link_x=120, enemy_x=220, enemy_y=120))
    assert "sword enemy" in decision.note
    assert "[beam]" in decision.note
    assert any(s.buttons & c.A for s in decision.plan)


def test_below_full_ignores_distant_enemy():
    # Same distant enemy, but below full health -> out of melee, not engaged (left to the march).
    decision = _stepped(_obs(hearts_byte=0x22, partial=100, link_x=120, enemy_x=220, enemy_y=120))
    assert "sword enemy" not in decision.note
