"""Billy Curiosity — intrinsic pull toward unexplored interesting features.

Route priorities come from the FAQ walkthrough (walkthrough.py).  Cave mouths are
black squares detected by vision; static map knowledge is the fallback in unit tests.
"""
from __future__ import annotations

from ...systems.nes import controller as c
from .tuning import (
    BLOCKED_NEIGHBORS,
    DUNGEON_ENTRANCE_SCREENS,
    SCREEN_EAST,
    SCREEN_NORTH,
    SCREEN_SOUTH,
    SCREEN_WEST,
    START_SCREEN,
)
from .walkthrough import (
    LEVEL_1_STAIRS,
    NORTH_CAVE_CHAIN,
    START_CAVE_MOUTH,
    START_CAVE_STAND,
    route_step,
)

# Overworld screens with a visible cave mouth Link must walk into (not edge-hop).
CAVE_MOUTH_TILES: dict[int, tuple[int, int]] = {
    START_SCREEN: START_CAVE_MOUTH,
    103: (112, 56),
    87: (112, 56),
    71: (112, 56),
    55: (112, 56),
    39: (112, 56),
    23: (112, 56),
    7: (112, 56),
    LEVEL_1_STAIRS: (120, 56),
}

START_CAVE_ROUTE: tuple[int, ...] = NORTH_CAVE_CHAIN

_DIR_BY_DELTA: dict[int, tuple[int, str]] = {
    SCREEN_NORTH: (c.UP, "north"),
    SCREEN_SOUTH: (c.DOWN, "south"),
    SCREEN_EAST: (c.RIGHT, "east"),
    SCREEN_WEST: (c.LEFT, "west"),
}

APPROACH_RANGE = 20


def requires_start_cave_inspection(
    here: int,
    visited: set[int],
    recent: tuple[int, ...] | list[int] = (),
    *,
    sword_level: int = 0,
    in_cave: bool = False,
) -> bool:
    """On the start screen, enter the NW cave for the wooden sword (FAQ step 1)."""
    if here != START_SCREEN:
        return False
    if sword_level >= 1:
        return False
    if in_cave:
        return True
    return True


def start_cave_inspection_action(
    link_x: int,
    link_y: int,
    *,
    cave_mouths: tuple[tuple[int, int], ...] = (),
) -> tuple[int, str, int]:
    """Approach or enter the NW cave — stay on screen 119 (not the north edge)."""
    mouth = _mouth_target(START_SCREEN, link_x, link_y, cave_mouths) or START_CAVE_STAND
    tx, ty = mouth
    dist = abs(link_x - tx) + abs(link_y - ty)
    if dist <= APPROACH_RANGE:
        return c.mask(c.UP, c.LEFT), "walkthrough-enter-nw-cave", START_SCREEN
    return c.LEFT, "inspect-start-cave-approach", START_SCREEN


def start_cave_inspection_exit() -> tuple[int, str, int]:
    """Legacy alias — callers should prefer start_cave_inspection_action."""
    return c.LEFT, "inspect-start-cave", START_SCREEN


def _button_to(dest: int, here: int) -> tuple[int, str] | None:
    delta = dest - here
    return _DIR_BY_DELTA.get(delta)


def next_curious_dest(here: int, visited: set[int]) -> int | None:
    """Next unexplored screen along a curiosity route from `here`."""
    if here in START_CAVE_ROUTE:
        idx = START_CAVE_ROUTE.index(here)
        for dest in START_CAVE_ROUTE[idx + 1:]:
            if dest not in visited:
                return dest
    if here in DUNGEON_ENTRANCE_SCREENS and here not in visited:
        return here
    return None


def _mouth_target(
    here: int,
    link_x: int,
    link_y: int,
    cave_mouths: tuple[tuple[int, int], ...],
) -> tuple[int, int] | None:
    static = CAVE_MOUTH_TILES.get(here)
    if cave_mouths:
        mouth = min(cave_mouths, key=lambda m: abs(m[0] - link_x) + abs(m[1] - link_y))
        if here == START_SCREEN and mouth[0] > 90 and static is not None:
            return static
        return mouth
    return static


def curious_exit(
    here: int,
    visited: set[int],
    *,
    cave_mouths: tuple[tuple[int, int], ...] = (),
    recent: tuple[int, ...] | list[int] = (),
    sword_level: int = 0,
    max_hearts: int = 3,
    in_cave: bool = False,
    link_x: int = 0,
    link_y: int = 0,
) -> tuple[int, str, int] | None:
    """Return (button, label, dest) when curiosity or the FAQ has a strong target."""
    faq = route_step(
        here,
        sword_level=sword_level,
        max_hearts=max_hearts,
        visited=visited,
        in_cave=in_cave,
        link_x=link_x,
        link_y=link_y,
    )
    if faq is not None:
        return faq.button, faq.label, faq.dest

    if requires_start_cave_inspection(
            here, visited, recent, sword_level=sword_level, in_cave=in_cave):
        return start_cave_inspection_action(link_x, link_y, cave_mouths=cave_mouths)

    blocked = BLOCKED_NEIGHBORS.get(here, frozenset())
    if cave_mouths:
        dest = here + SCREEN_NORTH
        if dest not in blocked:
            return c.UP, "curious-black-square-cave", dest

    dest = next_curious_dest(here, visited)
    if dest is None:
        return None
    if dest == here:
        pair = _DIR_BY_DELTA.get(SCREEN_NORTH)
        if pair is None:
            return None
        btn, label = pair
        return btn, f"curious-dungeon-{label}", dest
    pair = _button_to(dest, here)
    if pair is None:
        return None
    btn, label = pair
    kind = "cave" if here in START_CAVE_ROUTE or dest in START_CAVE_ROUTE else "dungeon"
    return btn, f"curious-{kind}-{label}", dest


def curiosity_bonus(here: int, dest: int, visited: set[int]) -> int:
    """Exploration score boost: higher = more curious about this exit."""
    target = next_curious_dest(here, visited)
    if target is None:
        return 0
    if dest == target:
        return 4
    if dest in START_CAVE_ROUTE and dest not in visited:
        return 2
    if dest in DUNGEON_ENTRANCE_SCREENS and dest not in visited:
        return 2
    return 0


def needs_cave_approach(
    here: int,
    link_x: int,
    link_y: int,
    *,
    cave_mouths: tuple[tuple[int, int], ...] = (),
) -> bool:
    """True when Link should walk to the black-square cave mouth first."""
    mouth = _mouth_target(here, link_x, link_y, cave_mouths)
    if mouth is None:
        return False
    tx, ty = mouth
    return abs(link_x - tx) + abs(link_y - ty) > APPROACH_RANGE


def cave_approach_button(
    here: int,
    link_x: int,
    link_y: int,
    *,
    cave_mouths: tuple[tuple[int, int], ...] = (),
) -> tuple[int, str] | None:
    """Walk toward a visible black-square cave mouth."""
    mouth = _mouth_target(here, link_x, link_y, cave_mouths)
    if mouth is None:
        return None
    tx, ty = mouth
    dx, dy = tx - link_x, ty - link_y
    if abs(dx) + abs(dy) <= APPROACH_RANGE:
        return None
    if abs(dy) >= abs(dx):
        btn = c.UP if dy < 0 else c.DOWN
        label = "up" if dy < 0 else "down"
    else:
        btn = c.LEFT if dx < 0 else c.RIGHT
        label = "left" if dx < 0 else "right"
    note = f"inspect-cave-approach-{label}" if here == START_SCREEN else f"curious-approach-{label}"
    return btn, note