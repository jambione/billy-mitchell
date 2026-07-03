"""The Genesis controller — LOGICAL button vocabulary, physically translated per console.

Same design as the SNES module: the engine and every banked plan speak the logical names the
NES established — A = primary action (confirm/interact), B = secondary (cancel/run) — and the
retro session translates to the console's physical buttons via RETRO_NAMES. On the Genesis
pad the primary action lives on physical "C" (Phantasy Star II: C confirms/talks/advances
text, B cancels — verified empirically during the PSII boot-dance bring-up), so logical A
maps to "C". The physical "A" button is exposed as a new capability bit (GEN_A), and SELECT
maps to the pad's MODE.

Genesis gotcha (found the hard way): stable-retro's FILTERED action space silently strips
START on this console, so the system must create sessions with Actions.ALL — see
systems/genesis/system.py.
"""
from __future__ import annotations

from ...abstractions import Controller, Step, encode_plan, plan_frames  # noqa: F401 (re-export)

# Logical bits 0-7 match the NES layout exactly (shared roles, shared plans).
A = 1 << 0        # primary: confirm / talk / advance text  -> Genesis physical "C"
B = 1 << 1        # secondary: cancel                       -> Genesis physical "B"
SELECT = 1 << 2   #                                          -> Genesis physical "MODE"
START = 1 << 3
UP = 1 << 4
DOWN = 1 << 5
LEFT = 1 << 6
RIGHT = 1 << 7
# Genesis-only button at a higher bit (a new capability, seldom used by PSII).
GEN_A = 1 << 8    #                                          -> Genesis physical "A"
NEUTRAL = 0

BUTTON_BITS = {"A": A, "B": B, "select": SELECT, "start": START,
               "up": UP, "down": DOWN, "left": LEFT, "right": RIGHT,
               "gen_a": GEN_A}

# Logical name (upper) -> stable-retro env.buttons physical name for the Genesis core.
RETRO_NAMES = {"A": "C", "SELECT": "MODE", "GEN_A": "A"}

# Extra teleop keyboard keys beyond the NES set (Z=A/confirm, X=B/cancel inherited).
VIEWER_KEYS = {"S": GEN_A}

_BTN_TOKENS = ["select", "start", "right", "gen_a", "down", "left", "up", "a", "b"]


def _extract_buttons(text: str) -> list[str]:
    import re
    found: list[str] = []
    for chunk in re.split(r"[^a-zA-Z_]+", text.lower()):
        i = 0
        while i < len(chunk):
            for tok in _BTN_TOKENS:
                if chunk.startswith(tok, i):
                    found.append(tok.upper() if tok in ("a", "b") else tok)
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


class GenesisController(Controller):
    name = "genesis"
    neutral = NEUTRAL
    buttons = BUTTON_BITS

    def mask_from_names(self, names: object) -> int:
        return mask_from_names(names)

    def names_from_mask(self, mask: int) -> list[str]:
        return names_from_mask(mask)


# --- movement helpers ----------------------------------------------------------------------
def hold(buttons: int, frames: int) -> Step:
    return Step(frames, buttons)


def idle(frames: int) -> list[Step]:
    return [Step(frames, NEUTRAL)]


def press_start(frames: int = 8, gap: int = 30) -> list[Step]:
    return [Step(frames, START), Step(gap, NEUTRAL)]


def confirm(frames: int = 8, gap: int = 30) -> list[Step]:
    """Tap the primary action (logical A -> physical C): talk / confirm / advance text."""
    return [Step(frames, A), Step(gap, NEUTRAL)]


def cancel(frames: int = 8, gap: int = 30) -> list[Step]:
    """Tap the secondary action (logical B): back out of menus."""
    return [Step(frames, B), Step(gap, NEUTRAL)]
