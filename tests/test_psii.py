"""Phantasy Star II bring-up: Genesis controller, RAM Scene decode, exploration progress."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.games.psii import ram_map as M
from billy.games.psii.game import PsiiGame
from billy.games.psii.perception import build_scene
from billy.systems.genesis import controller as c


# --- controller ------------------------------------------------------------------------------

def test_genesis_shared_bits_match_nes_layout():
    from billy.systems.nes import controller as nes
    for name in ("A", "B", "START", "UP", "DOWN", "LEFT", "RIGHT"):
        assert getattr(c, name) == getattr(nes, name), f"logical {name} bit must be shared"


def test_genesis_retro_names_translate_primary_to_physical_c():
    # PSII confirm/talk is the physical C button; the logical vocabulary calls it A.
    assert c.RETRO_NAMES["A"] == "C"
    assert c.RETRO_NAMES["SELECT"] == "MODE"
    assert c.RETRO_NAMES["GEN_A"] == "A"
    assert "B" not in c.RETRO_NAMES          # identity fallback


def test_genesis_mask_roundtrip():
    m = c.mask(c.LEFT, c.A, c.GEN_A)
    names = set(c.names_from_mask(m))
    assert names == {"left", "A", "gen_a"}
    assert c.mask_from_names(["left", "A", "gen_a"]) == m


def test_genesis_no_bit_collisions():
    bits = list(c.BUTTON_BITS.values())
    assert len(bits) == len(set(bits))
    combined = 0
    for b in bits:
        assert combined & b == 0
        combined |= b


# --- perception ------------------------------------------------------------------------------

def _ram(x=984, y=40, place_a=9, place_b=9, outdoor=1, menu=0):
    r = bytearray(0x10000)
    r[M.PLAYER_X_HI], r[M.PLAYER_X_HI + 1] = x >> 8, x & 0xFF
    r[M.PLAYER_Y_HI], r[M.PLAYER_Y_HI + 1] = y >> 8, y & 0xFF
    r[M.PLACE_A], r[M.PLACE_B] = place_a, place_b
    r[M.OUTDOOR] = outdoor
    r[M.MENU_OPEN] = menu
    return bytes(r)


def test_scene_decodes_position_and_place():
    s = build_scene(_ram(x=984, y=104, place_a=9, place_b=9, outdoor=1), frame=1)
    assert (s.x, s.y) == (984, 104)
    assert s.place == (9, 9, 1)
    assert s.outdoor and s.in_play and not s.menu_open
    assert s.tile == (984 // 16, 104 // 16)


def test_scene_transition_fade_is_not_in_play():
    # Between-maps fades zero the place bytes; that must never become a level_key.
    s = build_scene(_ram(place_a=0, place_b=0), frame=1)
    assert not s.in_play


def test_scene_menu_flag():
    assert build_scene(_ram(menu=1), frame=1).menu_open


# --- game progress ---------------------------------------------------------------------------

def test_exploration_progress_is_monotone_and_resumes_per_place():
    g = PsiiGame()
    p1 = g.observe(1, _ram(x=100, y=100)).progress
    p2 = g.observe(2, _ram(x=132, y=100)).progress          # new tile
    p3 = g.observe(3, _ram(x=100, y=100)).progress          # re-tread: no drop, no gain
    assert p2 > p1 and p3 == p2
    # hop to the house...
    g.observe(4, _ram(x=64, y=64, place_a=6, place_b=6, outdoor=0))
    # ...and back: Paseo's count RESUMES (new ground still beats re-treading)
    p4 = g.observe(5, _ram(x=100, y=100)).progress
    assert p4 == p2
    p5 = g.observe(6, _ram(x=164, y=100)).progress
    assert p5 > p4


def test_place_hop_changes_level_key_but_never_clears():
    g = PsiiGame()
    town = g.observe(1, _ram()).level_key
    house = g.observe(2, _ram(place_a=6, place_b=6, outdoor=0)).level_key
    assert town != house
    assert g.screen_changed(town, house)
    assert not g.level_cleared(town, house)
    assert not g.search_area_advance(town, house)


def test_transition_frames_keep_last_good_key():
    g = PsiiGame()
    town = g.observe(1, _ram())
    fade = g.observe(2, _ram(place_a=0, place_b=0))
    assert fade.level_key == town.level_key          # fade must not flap identity
    assert fade.progress == town.progress


# --- guide loader suffix support --------------------------------------------------------------

def test_guide_loader_finds_md_suffix(tmp_path, monkeypatch):
    from billy.knowledge import guide as guide_mod

    class FakeSystem:
        name = "genesis"

    class FakeGame:
        system = FakeSystem()

    (tmp_path / "walkthrough" / "genesis").mkdir(parents=True)
    (tmp_path / "walkthrough" / "genesis" / "psii.md").write_text(
        "Paseo:\n\nWalk outside to Mota and fight monsters to build levels.\n")
    monkeypatch.setattr(guide_mod.config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(guide_mod, "GUIDES_DIR", tmp_path / "guides")
    monkeypatch.setattr(guide_mod, "llm", __import__("types").SimpleNamespace(health=lambda: False),
                        raising=False)
    lib = guide_mod.load_guide_for(FakeGame(), "psii")
    assert lib is not None and len(lib) > 0
