"""ROM-verified macro for the start-screen NW cave (wooden sword).

Discovered via stable-retro PRG0 probing (June 2026): after text dismiss and
climbing to link_y≈141, a short RIGHT → DOWN → UP+LEFT → LEFT → A sequence
sets current_sword @ 1623 to 1.  Long DOWN exits back to overworld mode 5.

Bank this plan via the Director cache once verified; replay skips search on
later attempts.
"""
from __future__ import annotations

from ...abstractions import Plan, Step
from ...systems.nes import controller as c
from .tuning import START_SCREEN

# Frames spent in cave interior without sword before FAQ east-march fallback.
CAVE_ATTEMPT_TIMEOUT_FRAMES = 900

# --- overworld: walk from boot position to NW mouth ---------------------------------

APPROACH_PLAN: Plan = [
    Step(35, c.LEFT),
    Step(25, c.UP),
    Step(15, c.LEFT),
    Step(15, c.UP),
]

ENTER_PLAN: Plan = [
    Step(50, c.mask(c.UP, c.LEFT)),
    Step(40, c.LEFT),
    Step(40, c.mask(c.UP, c.LEFT)),
]

# --- cave interior (mode 11) --------------------------------------------------------

TEXT_PLAN: Plan = [Step(12, c.A)] * 35

CLIMB_PLAN: Plan = [Step(4, c.UP)] * 18

# At link≈(112,141): duck under lip, then UP+LEFT into old man — sword RAM → 1.
SWORD_PICKUP_PLAN: Plan = [
    Step(4, c.RIGHT),
    Step(8, c.DOWN),
    Step(16, c.DOWN),
    Step(12, c.mask(c.UP, c.LEFT)),
    Step(12, c.LEFT),
    Step(8, c.A),
]

EXIT_PLAN: Plan = [
    Step(120, c.DOWN),
]

INTERIOR_PLAN: Plan = TEXT_PLAN + CLIMB_PLAN + SWORD_PICKUP_PLAN + EXIT_PLAN

FULL_FROM_APPROACH: Plan = APPROACH_PLAN + ENTER_PLAN + INTERIOR_PLAN


def has_wooden_sword(sword_level: int) -> bool:
    return sword_level >= 1


def cave_quest_active(
    map_location: int,
    sword_level: int,
    *,
    in_cave: bool,
    cave_frames: int = 0,
    cave_gave_up: bool = False,
) -> bool:
    """True while Billy must finish (or time out) the start-cave wooden sword."""
    if has_wooden_sword(sword_level) or cave_gave_up:
        return False
    if in_cave:
        return True
    return map_location == START_SCREEN


def cave_attempt_exhausted(cave_frames: int, sword_level: int) -> bool:
    return cave_frames >= CAVE_ATTEMPT_TIMEOUT_FRAMES and not has_wooden_sword(sword_level)


def interior_phase(scene) -> str:
    """Where we are inside the start-cave macro."""
    if not scene.in_cave:
        return "overworld"
    if has_wooden_sword(scene.sword_level):
        return "exit"
    if scene.link_y >= 200:
        return "text"
    if scene.link_y > 145:
        return "climb"
    return "pickup"


def phase_plan(phase: str) -> Plan:
    return {
        "text": TEXT_PLAN,
        "climb": CLIMB_PLAN,
        "pickup": SWORD_PICKUP_PLAN,
        "exit": EXIT_PLAN,
    }.get(phase, [])


def macro_candidates() -> list[Plan]:
    """Micro-search seeds for the Director when the macro drifts."""
    return [
        INTERIOR_PLAN,
        TEXT_PLAN + CLIMB_PLAN + SWORD_PICKUP_PLAN,
        SWORD_PICKUP_PLAN,
        CLIMB_PLAN + SWORD_PICKUP_PLAN,
        [Step(60, c.A)],
        [Step(40, c.UP)] + SWORD_PICKUP_PLAN,
    ]