"""SMW scaffold: SNES controller vocabulary, WRAM perception fixtures, level identity.

No ROM needed — perception runs on synthetic 128KB WRAM images (see games/smw/STATUS.md for
the live bring-up checklist)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.games.smw import ram_map as R
from billy.games.smw.game import SmwGame
from billy.games.smw.perception import build_scene
from billy.systems.snes import controller as sc


# --- SNES controller: logical vocabulary + physical translation ---------------------------
def test_logical_bits_match_nes_layout():
    from billy.systems.nes import controller as nc
    for name in ("A", "B", "UP", "DOWN", "LEFT", "RIGHT", "SELECT", "START"):
        assert getattr(sc, name) == getattr(nc, name), f"{name} bit diverged from NES layout"


def test_retro_names_translate_roles_to_snes_pad():
    assert sc.RETRO_NAMES["A"] == "B"      # logical jump -> SNES B
    assert sc.RETRO_NAMES["B"] == "Y"      # logical run  -> SNES Y
    assert sc.RETRO_NAMES["SPIN"] == "A"   # spin jump    -> SNES A


def test_mask_roundtrip_including_snes_buttons():
    m = sc.mask(sc.RIGHT, sc.A, sc.SPIN)
    names = sc.names_from_mask(m)
    assert set(names) == {"right", "A", "spin"}
    assert sc.mask_from_names(names) == m


def test_spin_jump_plan_shape():
    plan = sc.spin_jump_right(jump_frames=20)
    assert len(plan) == 1 and plan[0].frames == 20
    assert plan[0].buttons & sc.SPIN and plan[0].buttons & sc.RIGHT


# --- perception on synthetic WRAM ----------------------------------------------------------
def _wram(**kw) -> bytes:
    ram = bytearray(0x20000)
    ram[R.GAME_MODE] = kw.get("mode", R.GAME_MODE_LEVEL)
    x = kw.get("x", 300)
    y = kw.get("y", 384)
    ram[R.PLAYER_X], ram[R.PLAYER_X + 1] = x & 0xFF, x >> 8
    ram[R.PLAYER_Y], ram[R.PLAYER_Y + 1] = y & 0xFF, y >> 8
    ram[R.PLAYER_STATE] = kw.get("state", 0)
    ram[R.PLAYER_IN_AIR] = kw.get("in_air", 0)
    ram[R.ON_GROUND] = kw.get("on_ground", 1)
    ram[R.POWERUP] = kw.get("powerup", 0)
    ram[R.LIVES] = kw.get("lives", 5)
    ram[R.COINS] = kw.get("coins", 12)
    ram[R.TRANSLEVEL] = kw.get("translevel", 0x29)
    ram[R.EVENTS_TRIGGERED] = kw.get("events", 0)
    score = kw.get("score", 4210) // 10
    ram[R.SCORE] = score & 0xFF
    ram[R.SCORE + 1] = (score >> 8) & 0xFF
    for i, (sx, sy, status) in enumerate(kw.get("sprites", [])):
        ram[R.SPRITE_STATUS + i] = status
        ram[R.SPRITE_X_LO + i], ram[R.SPRITE_X_HI + i] = sx & 0xFF, sx >> 8
        ram[R.SPRITE_Y_LO + i], ram[R.SPRITE_Y_HI + i] = sy & 0xFF, sy >> 8
    return bytes(ram)


def test_scene_reads_player_and_state():
    s = build_scene(_wram(x=513, y=400, powerup=3, coins=7, score=12340), frame=1)
    assert s.mario_x == 513 and s.mario_y == 400
    assert s.in_play and s.on_ground and not s.is_dying
    assert s.powerup == 3 and s.size == 1
    assert s.coins == 7 and s.score == 12340


def test_scene_dying_and_airborne():
    assert build_scene(_wram(state=9), 0).is_dying
    s = build_scene(_wram(in_air=1, on_ground=0), 0)
    assert not s.on_ground


def test_sprites_become_relative_enemies():
    s = build_scene(_wram(x=300, y=384, sprites=[(340, 384, 8), (200, 384, 0)]), 0)
    assert s.enemies == [(40, 0)]           # dead slot (status 0) skipped
    assert s.nearest_enemy() == (40, 0)
    assert s.enemy_ahead(within=48)


def test_game_level_identity_and_clear_semantics():
    g = SmwGame()
    o1 = g.observe(1, _wram(translevel=0x29, events=0))
    o2 = g.observe(2, _wram(translevel=0x29, events=1))   # beat a level -> events bumped
    assert g.level_cleared(o1.level_key, o2.level_key)
    o3 = g.observe(3, _wram(translevel=0x2A, events=1))   # walked the map -> screen change
    assert not g.level_cleared(o2.level_key, o3.level_key)
    assert g.screen_changed(o2.level_key, o3.level_key)


def test_not_in_play_reuses_last_good():
    g = SmwGame()
    good = g.observe(1, _wram(x=500))
    faded = g.observe(2, _wram(x=9999, mode=0x00))   # fade/menu: garbage x must be ignored
    assert faded.progress == good.progress == 500
