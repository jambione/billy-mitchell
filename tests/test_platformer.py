"""Tests for the shared platformer reflex — parity with SMB's behaviour + the candidate builders.

The real regression guard is the 1-1 clear benchmark; these lock the structural contract so a
future game reusing PlatformerReflex (e.g. SMB2-Japan) can't silently drift.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.games.common import platformer  # noqa: E402
from billy.games.common.platformer import PhysicsProfile, PlatformerReflex, PlatformerView  # noqa: E402
from billy.games.smb import tuning  # noqa: E402
from billy.games.smb.perception import build_scene  # noqa: E402
from billy.systems.nes import controller  # noqa: E402


def _obs(scene, progress=40):
    return type("O", (), {"raw": scene, "progress": progress})()


def test_smb_profile_matches_legacy_constants():
    p = tuning.PROFILE
    assert p.reflex_step_frames == tuning.REFLEX_STEP_FRAMES
    assert p.jump_max_frames == tuning.JUMP_MAX_FRAMES
    assert p.stomp_range == tuning.STOMP_RANGE
    assert p.enemy_react_px == 72  # SMB's early-reaction range


def test_scene_satisfies_platformer_view():
    sc = build_scene(bytes(0x800), 0)
    assert isinstance(sc, PlatformerView)


def test_danger_candidates_shape_is_stable():
    r = PlatformerReflex(tuning.PROFILE)
    sc = build_scene(bytes(0x800), 0)
    cands = r.danger_candidates(_obs(sc))
    # 2 big jumps + 2 patience stomps + 3 wall-jumps = 7 (the verified Phase-0 spread)
    assert len(cands) == 7
    assert all(isinstance(c, list) and c for c in cands)


def _floor_ram():
    ram = bytearray(0x800)
    ram[0x03B8] = 100  # mario_y -> 116, floor row below
    for sx in range(16):  # solid floor across page 0 at the row beneath Mario
        ram[0x500 + 7 * 16 + sx] = 1
    return bytes(ram)


def test_cruise_on_flat_ground():
    # Solid floor, no enemies/pits => first exchange just cruises right (no escalation).
    r = PlatformerReflex(tuning.PROFILE)
    r.reset(_obs(build_scene(_floor_ram(), 0)))
    d = r.step(_obs(build_scene(_floor_ram(), 1)))
    assert not d.needs_billy
    assert d.plan and controller.names_from_mask(d.plan[0].buttons)


def test_gap_jumper_respects_profile_bounds():
    cands = platformer.gap_jumper(width=3, profile=PhysicsProfile())
    assert cands and all(isinstance(c, list) for c in cands)
