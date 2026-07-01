"""Gamepad mapping persistence: defaults → saved calibration → env overrides."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.systems.nes.pad_map import DEFAULTS, describe, load_pad_map, save_pad_map


def test_defaults_when_no_file(tmp_path):
    m = load_pad_map(tmp_path / "missing.json")
    assert m["A"] == DEFAULTS["A"] and m["use_hat"] == DEFAULTS["use_hat"]


def test_saved_calibration_overrides_defaults(tmp_path):
    p = tmp_path / "pad_map.json"
    save_pad_map({"A": 5, "B": 3, "use_hat": False, "deadzone": 0.5, "invert_y": True}, p)
    m = load_pad_map(p)
    assert m["A"] == 5 and m["B"] == 3
    assert m["use_hat"] is False and m["deadzone"] == 0.5 and m["invert_y"] is True
    assert m["START"] == DEFAULTS["START"]   # unassigned roles keep defaults


def test_env_overrides_saved_map(tmp_path, monkeypatch):
    p = tmp_path / "pad_map.json"
    save_pad_map({"A": 5}, p)
    monkeypatch.setenv("BILLY_PAD_A", "9")
    monkeypatch.setenv("BILLY_PAD_USE_HAT", "0")
    monkeypatch.setenv("BILLY_PAD_DEADZONE", "0.55")
    m = load_pad_map(p)
    assert m["A"] == 9
    assert m["use_hat"] is False
    assert m["deadzone"] == 0.55


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    p = tmp_path / "pad_map.json"
    p.write_text("{not json")
    m = load_pad_map(p)
    assert m["A"] == DEFAULTS["A"]


def test_save_strips_unknown_keys(tmp_path):
    p = save_pad_map({"A": 4, "name": "SN30", "junk": 1}, tmp_path / "m.json")
    saved = json.loads(p.read_text())
    assert saved == {"A": 4}


def test_snes_extra_roles_roundtrip(tmp_path):
    p = save_pad_map({"A": 1, "SPIN": 0, "L": 6, "R": 7}, tmp_path / "m.json")
    m = load_pad_map(p)
    assert m["SPIN"] == 0 and m["L"] == 6 and m["R"] == 7


def test_describe_omits_unassigned():
    text = describe({"A": 2, "B": 1, "FINISH": -1, "use_hat": True,
                     "deadzone": 0.4, "invert_y": False})
    assert "A=2" in text and "FINISH" not in text
