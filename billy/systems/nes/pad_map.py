"""Persistent gamepad mapping — calibrate once, teleop forever.

Precedence (lowest to highest): built-in defaults (8Bitdo SN30 Pro on macOS, the pad this was
first tuned on) → `data/pad_map.json` (written by `teleop.py calibrate`) → `BILLY_PAD_*` env
vars (quick one-off overrides, e.g. borrowing a friend's pad without touching your saved map).

Mapping keys:
  Role names ("A", "B", "START", "SELECT", "FINISH", and console extras like "SPIN") map to a
  pad button INDEX (-1 = unassigned). Roles are LOGICAL — "A" is jump on every console; the
  viewer resolves them against the active controller module, so one saved map drives NES and
  SNES titles alike.
  Movement options: "use_hat" (d-pad hat for left/right; some pads' hats never center —
  calibration detects this), "deadzone" (analog stick), "invert_x"/"invert_y" (stick sign
  differs across pads/OS backends — either axis can come in flipped independently).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ... import config

PAD_MAP_FILE = config.DATA_DIR / "pad_map.json"

# Button-index roles a wizard can assign (console extras included; harmless if absent).
ROLE_KEYS = ("A", "B", "START", "SELECT", "FINISH", "SPIN", "X", "L", "R")
OPTION_KEYS = ("use_hat", "deadzone", "invert_x", "invert_y", "dirs")

# Per-direction movement specs (the robust path, written by `teleop.py calibrate`):
#   "dirs": {"LEFT": {"src": "button", "idx": 13},
#            "DOWN": {"src": "hat", "value": [-1, -1]},
#            "UP": {"src": "axis_y", "sign": -1, "rest": 0.02}, ...}
# Each direction is driven by whatever the calibration saw change when the user held it —
# a plain button, the hat as an EXACT TUPLE (some pads' hats arrive rotated/scrambled from
# the OS HID layer, so per-axis signs lie — the tuple emitted during the hold IS the
# direction), or any HID axis (x/y/z/rx/ry/rz, with resting offset and sign).
# When "dirs" is present it fully owns movement; the legacy use_hat/invert_x/invert_y stick
# heuristics apply only to maps without it (pre-directional calibrations and the defaults).

DEFAULTS = {
    # 8Bitdo SN30 Pro (Bluetooth, macOS), verified via pad-debug: JUMP=2, RUN=1.
    "A": 2, "B": 1, "SELECT": 8, "START": 7, "FINISH": -1,
    # This pad's hat_y floats (never centers) so vertical comes from the stick; hat_x is fine.
    "use_hat": True, "deadzone": 0.4, "invert_x": False, "invert_y": False,
}


def load_pad_map(path: str | Path | None = None) -> dict:
    m = dict(DEFAULTS)
    p = Path(path) if path is not None else PAD_MAP_FILE
    if p.is_file():
        try:
            saved = json.loads(p.read_text())
            m.update({k: v for k, v in saved.items()
                      if k in ROLE_KEYS or k in OPTION_KEYS})
        except (json.JSONDecodeError, OSError):
            pass   # a corrupt map must never break teleop; recalibrate to fix

    for role in ROLE_KEYS:
        v = os.environ.get(f"BILLY_PAD_{role}")
        if v is not None:
            try:
                m[role] = int(v)
            except ValueError:
                pass
    if "BILLY_PAD_USE_HAT" in os.environ:
        m["use_hat"] = os.environ["BILLY_PAD_USE_HAT"] == "1"
    if "BILLY_PAD_DEADZONE" in os.environ:
        try:
            m["deadzone"] = float(os.environ["BILLY_PAD_DEADZONE"])
        except ValueError:
            pass
    if "BILLY_PAD_INVERT_X" in os.environ:
        m["invert_x"] = os.environ["BILLY_PAD_INVERT_X"] == "1"
    if "BILLY_PAD_INVERT_Y" in os.environ:
        m["invert_y"] = os.environ["BILLY_PAD_INVERT_Y"] == "1"
    return m


def save_pad_map(mapping: dict, path: str | Path | None = None) -> Path:
    p = Path(path) if path is not None else PAD_MAP_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in mapping.items() if k in ROLE_KEYS or k in OPTION_KEYS}
    p.write_text(json.dumps(keep, indent=2) + "\n")
    return p


def describe(mapping: dict) -> str:
    roles = "  ".join(f"{k}={mapping[k]}" for k in ROLE_KEYS
                      if k in mapping and mapping[k] is not None and mapping[k] >= 0)
    opts = (f"use_hat={mapping.get('use_hat')}  deadzone={mapping.get('deadzone')}  "
            f"invert_x={mapping.get('invert_x')}  invert_y={mapping.get('invert_y')}")
    return f"{roles}\n  {opts}"
