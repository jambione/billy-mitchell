"""Unit tests for the RAM->Scene decoding and plan encoding (no FCEUX needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.systems.nes import controller  # noqa: E402
from billy.games.smb.perception import build_scene  # noqa: E402


def _blank_ram() -> bytearray:
    return bytearray(0x800)


def test_mario_position_and_speed():
    ram = _blank_ram()
    ram[0x6D] = 2      # x page
    ram[0x86] = 0x10   # x within page  => world x = 2*256 + 16 = 528
    ram[0x03B8] = 100  # screen y => mario_y = 116
    ram[0x57] = 0xFE   # x-speed -2 (signed)
    scene = build_scene(bytes(ram), frame=42)
    assert scene.mario_x == 528
    assert scene.mario_y == 116
    assert scene.x_speed == -2
    assert scene.frame == 42


def test_world_stage_lives_time_coins():
    ram = _blank_ram()
    ram[0x075F] = 0    # world 0 -> "1"
    ram[0x075C] = 0    # stage 0 -> "1"
    ram[0x075A] = 2    # lives byte
    ram[0x075E] = 7    # coins
    ram[0x07F8], ram[0x07F9], ram[0x07FA] = 3, 5, 0  # time 350
    scene = build_scene(bytes(ram), frame=0)
    assert scene.world_stage == "1-1"
    assert scene.lives == 2
    assert scene.coins == 7
    assert scene.time == 350


def test_enemy_detection_and_ahead():
    ram = _blank_ram()
    ram[0x6D], ram[0x86] = 0, 50          # mario x = 50
    ram[0x03B8] = 100                      # mario y = 116
    ram[0x0F] = 1                          # enemy slot 0 alive
    ram[0x6E], ram[0x87] = 0, 80           # enemy x = 80 (30px ahead)
    ram[0xCF] = 100                        # enemy y = 124
    scene = build_scene(bytes(ram), frame=0)
    assert len(scene.enemies) == 1
    assert scene.enemies[0].x == 80
    assert scene.enemy_ahead(within=48) is True
    assert scene.enemy_ahead(within=16) is False


def test_dying_state_flags():
    ram = _blank_ram()
    ram[0x000E] = 0x0B                     # dying
    assert build_scene(bytes(ram), 0).is_dying is True
    ram2 = _blank_ram()
    ram2[0x00B5] = 2                        # fell below the floor
    assert build_scene(bytes(ram2), 0).is_dying is True
    assert build_scene(bytes(_blank_ram()), 0).is_dying is False


def test_gap_info_distance_and_width():
    ram = _blank_ram()
    ram[0x6D], ram[0x86] = 0, 64       # mario x = 64 (tile col 4, aligned)
    ram[0x03B8] = 100                   # mario_y = 116 -> floor_y = 132 -> sub_y 6
    # Solid floor on page 0 row 6, but punch a 2-tile pit at tile cols 6,7.
    for sx in range(16):
        ram[0x500 + 6 * 16 + sx] = 1
    ram[0x500 + 6 * 16 + 6] = 0
    ram[0x500 + 6 * 16 + 7] = 0
    scene = build_scene(bytes(ram), 0)
    info = scene.gap_info()
    assert info is not None
    dist_px, width = info
    assert width == 2
    assert dist_px == 32                # 2 tiles ahead, aligned: 2*16 - 0
    assert scene.gap_ahead(lookahead_tiles=3) is True


def test_gap_info_none_on_solid_floor():
    ram = _blank_ram()
    ram[0x03B8] = 100
    for sx in range(16):
        ram[0x500 + 6 * 16 + sx] = 1
    assert build_scene(bytes(ram), 0).gap_info() is None


def test_in_play_guard_rejects_overflow_frames():
    # Death / transition frames: world-x and world/stage read 0xFF -> must be flagged not-in-play.
    ram = _blank_ram()
    ram[0x6D], ram[0x86] = 0xFF, 0xFF      # world-x would be 65535
    ram[0x075F], ram[0x075C] = 0xFF, 0xFF  # world/stage would be "256-256"
    scene = build_scene(bytes(ram), 0)
    assert scene.in_play is False
    # A normal in-level frame is in_play.
    good = _blank_ram()
    good[0x6D], good[0x86] = 1, 0x20       # x = 288
    assert build_scene(bytes(good), 0).in_play is True


def test_observe_reuses_last_good_on_death_frame():
    from billy.games.smb import SmbGame  # imports retro via NesSystem; run under the venv

    game = SmbGame()
    good = _blank_ram()
    good[0x6D], good[0x86] = 0, 200        # x = 200, world 1-1
    o1 = game.observe(0, bytes(good))
    assert o1.progress == 200 and o1.level_label == "1-1" and o1.dead is False

    death = _blank_ram()
    death[0x6D], death[0x86] = 0xFF, 0xFF  # overflow x
    death[0x075F], death[0x075C] = 0xFF, 0xFF
    death[0x000E] = 0x0B                    # dying
    o2 = game.observe(1, bytes(death))
    assert o2.progress == 200              # reused last good, NOT 65535
    assert o2.level_label == "1-1"         # NOT "256-256"
    assert o2.level_key == (0, 0)
    assert o2.dead is True                  # death is still reported


def test_plan_encoding_roundtrip():
    plan = [controller.Step(12, controller.mask(controller.RIGHT, controller.B)),
            controller.Step(20, controller.mask(controller.RIGHT, controller.B, controller.A))]
    blob = controller.encode_plan(plan)
    assert blob[0] == 2                                  # nsteps
    assert int.from_bytes(blob[1:3], "little") == 12     # first duration
    assert blob[3] == controller.RIGHT | controller.B    # first mask
    assert controller.plan_frames(plan) == 32


def test_button_name_mask_roundtrip():
    m = controller.mask_from_names(["right", "A"])
    assert m == controller.RIGHT | controller.A
    assert set(controller.names_from_mask(m)) == {"right", "A"}


def test_mask_from_names_tolerates_model_formats():
    R, B, A = controller.RIGHT, controller.B, controller.A
    assert controller.mask_from_names("rightA") == R | A      # concatenated
    assert controller.mask_from_names("right,A") == R | A     # comma
    assert controller.mask_from_names("right+B+A") == R | B | A
    assert controller.mask_from_names("right A") == R | A     # space
    assert controller.mask_from_names(["right", "B"]) == R | B
    # garbage / missing must never raise
    for junk in (None, 5, {"x": 1}, ["right", None], "xyz"):
        controller.mask_from_names(junk)
    assert controller.mask_from_names("xyz") == 0   # no button letters => no buttons
