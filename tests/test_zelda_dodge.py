"""Zelda projectile perception + dodge: Billy reads monster shots from the high object slots and
sidesteps an incoming rock's lane (identified by velocity across two frames), then resumes combat.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.abstractions import Observation  # noqa: E402
from billy.games.zelda.perception import build_scene  # noqa: E402
from billy.games.zelda.reflex import ZeldaReflex  # noqa: E402
from billy.systems.nes import controller as c  # noqa: E402

SLOT = 14                       # a high object slot; X at 0x70+14=0x7E, Y at 0x84+14=0x92
SLOT_X, SLOT_Y = 0x70 + SLOT, 0x84 + SLOT


def _obs(link=(120, 120), shot=None, hearts_byte=0x21, partial=0xFF, map_location=72) -> Observation:
    ram = bytearray(0x800)
    ram[18] = 5                 # in-play, not cave
    ram[16] = 0                 # overworld
    ram[235] = map_location     # not START(119), grid row != 8 -> not east-march
    ram[112], ram[132] = link   # Link X/Y (object slot 0)
    ram[232] = 255
    ram[1647] = hearts_byte
    ram[1648] = partial
    ram[1623] = 1               # has wooden sword
    ram[1569] = map_location
    if shot is not None:
        ram[SLOT_X], ram[SLOT_Y] = shot
    scene = build_scene(bytes(ram))
    return Observation(frame=1, progress=scene.objective_score(), score=0,
                       level_label=scene.room_label, level_key=(scene.realm, map_location),
                       dead=False, summary="", ascii_map="", raw=scene, elevation=scene.link_y)


def _dodge_after(frame1_shot, frame2_shot, link=(120, 120)):
    """Two consecutive frames so the reflex can measure the shot's velocity; return frame-2 note+plan."""
    reflex = ZeldaReflex()
    o1 = _obs(link=link, shot=frame1_shot)
    reflex.reset(o1)
    reflex.step(o1)                                   # frame 1 records the object
    dec = reflex.step(_obs(link=link, shot=frame2_shot))
    return dec


# --- perception ---------------------------------------------------------------------------------

def test_scene_reads_high_slot_objects():
    s = _obs(shot=(160, 120)).raw
    assert (SLOT, 160, 120) in s.objects
    assert s.object_positions()[SLOT] == (160, 120)


def test_no_objects_when_slots_empty():
    assert _obs(shot=None).raw.objects == ()


# --- dodge ---------------------------------------------------------------------------------------

def test_dodges_incoming_horizontal_shot():
    # Shot on Link's row (y=120), moving left toward Link (160 -> 156). Sidestep vertically.
    dec = _dodge_after((160, 120), (156, 120))
    assert "dodge shot" in dec.note
    assert dec.plan[0].buttons in (c.UP, c.DOWN)


def test_dodges_incoming_vertical_shot():
    # Shot on Link's column (x=120), moving down toward Link (60 -> 64). Sidestep horizontally.
    dec = _dodge_after((120, 60), (120, 64))
    assert "dodge shot" in dec.note
    assert dec.plan[0].buttons in (c.LEFT, c.RIGHT)


def test_no_dodge_for_slow_walking_object():
    # 1px/frame is an enemy walk, not a shot -> no dodge (falls through to normal play).
    dec = _dodge_after((140, 120), (139, 120))
    assert "dodge shot" not in dec.note


def test_no_dodge_when_shot_off_lane():
    # Fast, but 40px off Link's row -> it won't connect, so don't flinch.
    dec = _dodge_after((160, 80), (156, 80))
    assert "dodge shot" not in dec.note


def test_no_dodge_for_receding_shot():
    # Fast + on the row, but moving AWAY from Link (120 -> 130) -> no need to dodge.
    dec = _dodge_after((124, 120), (134, 120))
    assert "dodge shot" not in dec.note
