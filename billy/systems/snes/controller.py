"""The SNES controller — LOGICAL button vocabulary, physically translated per console.

The engine, the shared platformer reflex, and every banked plan speak the *logical* names the
NES established: A = jump, B = run/attack, plus the d-pad. On SNES those roles live on
different physical buttons (SMW jumps with SNES "B", runs with SNES "Y"), so this module keeps
the SAME bit layout for the shared roles and adds the SNES-only buttons at higher bits. The
retro session translates logical names to the console's physical buttons via RETRO_NAMES.

That one design choice is what lets `games/common/platformer.py` (which builds plans with
logical A/B) drive a SNES title with zero reflex changes — the cross-console carry-forward.
"""
from __future__ import annotations

from ...abstractions import Controller, Step, encode_plan, plan_frames  # noqa: F401 (re-export)

# Logical bits 0-7 match the NES layout exactly (shared roles, shared plans).
A = 1 << 0        # jump        -> SNES physical "B"
B = 1 << 1        # run/attack  -> SNES physical "Y"
SELECT = 1 << 2
START = 1 << 3
UP = 1 << 4
DOWN = 1 << 5
LEFT = 1 << 6
RIGHT = 1 << 7
# SNES-only buttons at higher bits (new capabilities, e.g. SMW's spin jump).
SPIN = 1 << 8     # spin jump   -> SNES physical "A"
X = 1 << 9        # alt action  -> SNES physical "X"
L = 1 << 10
R = 1 << 11
NEUTRAL = 0

BUTTON_BITS = {"A": A, "B": B, "select": SELECT, "start": START,
               "up": UP, "down": DOWN, "left": LEFT, "right": RIGHT,
               "spin": SPIN, "x": X, "l": L, "r": R}

# Logical name (upper) -> stable-retro env.buttons physical name for the SNES core.
RETRO_NAMES = {"A": "B", "B": "Y", "SPIN": "A", "X": "X", "L": "L", "R": "R"}

# Extra teleop keyboard keys the viewer binds beyond the NES set (Z=jump, X=run inherited).
VIEWER_KEYS = {"C": SPIN, "S": X, "Q": L, "W": R}

_BTN_TOKENS = ["select", "start", "right", "spin", "down", "left", "up", "x", "l", "r", "a", "b"]


def _extract_buttons(text: str) -> list[str]:
    import re
    found: list[str] = []
    for chunk in re.split(r"[^a-zA-Z]+", text.lower()):
        i = 0
        while i < len(chunk):
            for tok in _BTN_TOKENS:
                if chunk.startswith(tok, i):
                    found.append(tok.upper() if tok in ("a", "b", "x", "l", "r") else tok)
                    i += len(tok)
                    break
            else:
                i += 1
    return found


def mask(*buttons: int) -> int:
    m = 0
    for b in buttons:
        m |= b
    return m


def mask_from_names(names: object) -> int:
    if names is None:
        return 0
    text = " ".join(str(x) for x in names) if isinstance(names, (list, tuple)) else str(names)
    m = 0
    for tok in _extract_buttons(text):
        m |= BUTTON_BITS.get(tok, BUTTON_BITS.get(tok.lower(), 0))
    return m


def names_from_mask(m: int) -> list[str]:
    return [name for name, bit in BUTTON_BITS.items() if m & bit]


class SnesController(Controller):
    name = "snes"
    neutral = NEUTRAL
    buttons = BUTTON_BITS

    def mask_from_names(self, names: object) -> int:
        return mask_from_names(names)

    def names_from_mask(self, mask: int) -> list[str]:
        return names_from_mask(mask)


# --- movement helpers (logical-role identical to the NES module) --------------------------
def hold(buttons: int, frames: int) -> Step:
    return Step(frames, buttons)


def idle(frames: int) -> list[Step]:
    return [Step(frames, NEUTRAL)]


def press_start(frames: int = 4, gap: int = 6) -> list[Step]:
    return [Step(frames, START), Step(gap, NEUTRAL)]


def run_right(frames: int, sprint: bool = True) -> list[Step]:
    return [Step(frames, mask(RIGHT, B) if sprint else RIGHT)]


def run_left(frames: int, sprint: bool = True) -> list[Step]:
    return [Step(frames, mask(LEFT, B) if sprint else LEFT)]


def jump_right(run_frames: int = 0, jump_frames: int = 16, sprint: bool = True) -> list[Step]:
    steps: list[Step] = []
    move = mask(RIGHT, B) if sprint else RIGHT
    if run_frames:
        steps.append(Step(run_frames, move))
    steps.append(Step(jump_frames, mask(move, A)))
    return steps


def spin_jump_right(jump_frames: int = 20, sprint: bool = True) -> list[Step]:
    """SMW's spin jump: bounces off hazards a normal jump can't touch (a NEW capability the
    logical vocabulary exposes to search as extra candidates, not a reflex rewrite)."""
    move = mask(RIGHT, B) if sprint else RIGHT
    return [Step(jump_frames, mask(move, SPIN))]
