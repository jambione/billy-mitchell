"""Gamepad mapping persistence: defaults → saved calibration → env overrides."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.systems.nes.pad_map import DEFAULTS, describe, load_pad_map, save_pad_map


def test_defaults_when_no_file(tmp_path):
    m = load_pad_map(tmp_path / "missing.json")
    assert m["A"] == DEFAULTS["A"] and m["use_hat"] == DEFAULTS["use_hat"]
    assert m["invert_x"] is False and m["invert_y"] is False


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
    monkeypatch.setenv("BILLY_PAD_INVERT_X", "1")
    m = load_pad_map(p)
    assert m["A"] == 9
    assert m["use_hat"] is False
    assert m["deadzone"] == 0.55
    assert m["invert_x"] is True


def test_invert_x_and_invert_y_are_independent(tmp_path):
    p = tmp_path / "m.json"
    save_pad_map({"invert_x": True, "invert_y": False}, p)
    m = load_pad_map(p)
    assert m["invert_x"] is True and m["invert_y"] is False


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


def test_dirs_roundtrip(tmp_path):
    dirs = {"LEFT": {"src": "button", "idx": 13},
            "RIGHT": {"src": "hat_x", "sign": 1, "rest": 0.0},
            "UP": {"src": "stick_y", "sign": -1, "rest": 0.02}}
    p = save_pad_map({"A": 2, "dirs": dirs}, tmp_path / "m.json")
    m = load_pad_map(p)
    assert m["dirs"] == dirs


# --- calibrated per-direction movement decoding (fake joystick, no window) ----------------
def _viewer_with(joy, joy_map):
    from billy.systems.nes.retro_session import _Viewer
    v = _Viewer.__new__(_Viewer)      # skip __init__: no pyglet/window needed for _joy_mask
    v._c = __import__("billy.systems.nes.controller", fromlist=["c"])
    v.joystick = joy
    v._joy_map = joy_map
    v._finish = False
    return v


class _FakeJoy:
    def __init__(self, buttons=(), hat=(0, 0), stick=(0.0, 0.0)):
        self.buttons = [i in buttons for i in range(16)]
        self.hat_x, self.hat_y = hat
        self.x, self.y = stick


def test_dirs_button_dpad(tmp_path):
    from billy.systems.nes import controller as c
    dirs = {"LEFT": {"src": "button", "idx": 13}, "RIGHT": {"src": "button", "idx": 14}}
    v = _viewer_with(_FakeJoy(buttons=(13,)), {"dirs": dirs, "deadzone": 0.4})
    assert v._joy_mask() & c.LEFT
    assert not v._joy_mask() & c.RIGHT


def test_dirs_hat_with_offcenter_rest():
    from billy.systems.nes import controller as c
    # A hat that RESTS at y=1 (the SN30 float): DOWN spec is rest-relative, so resting
    # produces no input, and deviation past the rest fires it.
    dirs = {"DOWN": {"src": "hat_y", "sign": -1, "rest": 1.0}}
    at_rest = _viewer_with(_FakeJoy(hat=(0, 1)), {"dirs": dirs, "deadzone": 0.4})
    assert not at_rest._joy_mask() & c.DOWN
    held = _viewer_with(_FakeJoy(hat=(0, -1)), {"dirs": dirs, "deadzone": 0.4})
    assert held._joy_mask() & c.DOWN


def test_dirs_hat_exact_tuple_for_scrambled_hats():
    from billy.systems.nes import controller as c
    # The SN30's hat arrives rotated: physically holding DOWN emits (-1,-1). The tuple spec
    # matches EXACTLY that — and must not fire LEFT/RIGHT the way per-axis signs would.
    dirs = {"DOWN": {"src": "hat", "value": [-1, -1]},
            "LEFT": {"src": "hat", "value": [-1, 1]}}
    held_down = _viewer_with(_FakeJoy(hat=(-1, -1)), {"dirs": dirs, "deadzone": 0.4})
    m = held_down._joy_mask()
    assert m & c.DOWN and not (m & c.LEFT) and not (m & c.RIGHT)
    at_rest = _viewer_with(_FakeJoy(hat=(0, 1)), {"dirs": dirs, "deadzone": 0.4})
    assert at_rest._joy_mask() == 0


def test_dirs_generic_axis_source():
    from billy.systems.nes import controller as c
    # A pad that puts the stick on rz: generic axis specs read any HID axis by name.
    dirs = {"RIGHT": {"src": "axis_rz", "sign": 1, "rest": 0.0}}
    j = _FakeJoy()
    j.rz = 0.9
    v = _viewer_with(j, {"dirs": dirs, "deadzone": 0.4})
    assert v._joy_mask() & c.RIGHT
    j.rz = 0.0
    assert not v._joy_mask() & c.RIGHT


def test_dirs_stick_sign_and_deadzone():
    from billy.systems.nes import controller as c
    dirs = {"LEFT": {"src": "stick_x", "sign": -1, "rest": 0.0},
            "UP": {"src": "stick_y", "sign": -1, "rest": 0.0}}
    v = _viewer_with(_FakeJoy(stick=(-0.9, -0.1)), {"dirs": dirs, "deadzone": 0.4})
    m = v._joy_mask()
    assert m & c.LEFT                  # -0.9 past the deadzone
    assert not m & c.UP                # -0.1 inside the deadzone


def test_dirs_present_disables_legacy_stick():
    from billy.systems.nes import controller as c
    # With dirs defined, raw stick motion that has no spec must NOT leak through legacy logic.
    dirs = {"LEFT": {"src": "button", "idx": 13}}
    v = _viewer_with(_FakeJoy(stick=(0.9, 0.0)), {"dirs": dirs, "deadzone": 0.4})
    assert not v._joy_mask() & c.RIGHT
