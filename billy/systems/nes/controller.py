"""The NES controller (input device for the NES system).

Owns the NES button bit-layout and name<->bitmask encoding. The bit order here MUST match
BUTTON_ORDER in systems/nes/bridge.lua. Plan/Step encoding lives in the engine
(abstractions); this module is just the device + a few NES movement helpers.
"""
from __future__ import annotations

import re

from ...abstractions import Controller, Step, encode_plan, plan_frames  # noqa: F401 (re-export)

# NES button bits (bit0..bit7) — must match bridge.lua's BUTTON_ORDER.
A = 1 << 0
B = 1 << 1
SELECT = 1 << 2
START = 1 << 3
UP = 1 << 4
DOWN = 1 << 5
LEFT = 1 << 6
RIGHT = 1 << 7
NEUTRAL = 0
BUTTON_BITS = {"A": A, "B": B, "select": SELECT, "start": START,
               "up": UP, "down": DOWN, "left": LEFT, "right": RIGHT}

# Tokens longest-first so a greedy scan splits concatenations like "rightA".
_BTN_TOKENS = ["select", "start", "right", "down", "left", "up", "a", "b"]


def _extract_buttons(text: str) -> list[str]:
    """Pull canonical button names out of whatever a model emitted: a list, "right,A",
    "right+A", "right A", or even a concatenation like "rightA"."""
    found: list[str] = []
    for chunk in re.split(r"[^a-zA-Z]+", text.lower()):
        i = 0
        while i < len(chunk):
            for tok in _BTN_TOKENS:
                if chunk.startswith(tok, i):
                    found.append("A" if tok == "a" else "B" if tok == "b" else tok)
                    i += len(tok)
                    break
            else:
                i += 1
    return found


def mask(*buttons: int) -> int:
    """OR together button bits, e.g. mask(RIGHT, B, A) -> run + jump."""
    m = 0
    for b in buttons:
        m |= b
    return m


def mask_from_names(names: object) -> int:
    """Bitmask from any shape a model emits (['right','A'], 'right,A', 'rightA', None, …)."""
    if names is None:
        return 0
    text = " ".join(str(x) for x in names) if isinstance(names, (list, tuple)) else str(names)
    m = 0
    for tok in _extract_buttons(text):
        m |= BUTTON_BITS[tok]
    return m


def names_from_mask(m: int) -> list[str]:
    return [name for name, bit in BUTTON_BITS.items() if m & bit]


class NesController(Controller):
    name = "nes"
    neutral = NEUTRAL
    buttons = BUTTON_BITS

    def mask_from_names(self, names: object) -> int:
        return mask_from_names(names)

    def names_from_mask(self, mask: int) -> list[str]:
        return names_from_mask(mask)


# --- NES movement helpers (used by NES games' reflex/boot) -------------------------------
def hold(buttons: int, frames: int) -> Step:
    return Step(frames, buttons)


def idle(frames: int) -> list[Step]:
    return [Step(frames, NEUTRAL)]


def press_start(frames: int = 4, gap: int = 6) -> list[Step]:
    return [Step(frames, START), Step(gap, NEUTRAL)]


def run_right(frames: int, sprint: bool = True) -> list[Step]:
    return [Step(frames, mask(RIGHT, B) if sprint else RIGHT)]


def jump_right(run_frames: int = 0, jump_frames: int = 16, sprint: bool = True) -> list[Step]:
    steps: list[Step] = []
    move = mask(RIGHT, B) if sprint else RIGHT
    if run_frames:
        steps.append(Step(run_frames, move))
    steps.append(Step(jump_frames, mask(move, A)))
    return steps
