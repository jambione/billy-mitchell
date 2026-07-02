"""First-quest route knowledge from Dan Simpson's FAQ (walkthrough/NES/zelda).

Screen ids match the stable-retro PRG0 map: origin (grid 8,8) = screen 119,
+1 east, -1 west, +16 south, -16 north.

See STATUS.md in this package for accomplishments, ROM blockers, and path forward.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...systems.nes import controller as c
from .start_cave import cave_quest_active
from .tuning import (
    SCREEN_EAST,
    SCREEN_NORTH,
    SCREEN_SOUTH,
    SCREEN_WEST,
    START_SCREEN,
)

# FAQ overworld grid origin (8, 8) — "Where you start"
ORIGIN_GRID = (8, 8)

# NW black-square cave mouth on the start screen (ROM + FAQ §1 HEART 1 / wooden sword).
START_CAVE_MOUTH = (60, 76)
START_CAVE_STAND = (72, 88)

# Key screens derived from the FAQ Hyrule map (FIRST QUEST, row Y / column X).
LEVEL_1_SCREEN = 55          # grid (8, 4) — "The Eagle" entrance marker
SEA_EAST_SCREEN = 127        # grid (16, 8) — east coast after "right 8 screens"
BOMB_SHOP_SCREEN = 111         # grid (16, 7) — north of the sea for bomb shop
WHITE_SWORD_SCREEN = 151     # grid (11, 1) — waterfall cave (needs 5 hearts)

# North forest cave chain toward the Level-1 stairs (screen 6) — alternate FAQ path.
NORTH_CAVE_CHAIN: tuple[int, ...] = (119, 103, 87, 71, 55, 39, 23, 7, 6)
LEVEL_1_STAIRS = 6


def grid_to_screen(gx: int, gy: int) -> int:
    """FAQ map coordinate → overworld screen id (Y grows downward on the map)."""
    ox, oy = ORIGIN_GRID
    return START_SCREEN + (gx - ox) + (gy - oy) * 16


def screen_to_grid(screen: int) -> tuple[int, int]:
    ox, oy = ORIGIN_GRID
    delta = screen - START_SCREEN
    gy = oy + delta // 16
    gx = ox + delta % 16
    return gx, gy


@dataclass(frozen=True)
class RouteStep:
    """One committed walkthrough direction from the current screen."""
    phase: str
    button: int
    label: str
    dest: int


def _adjacent(here: int, delta: int) -> int:
    return here + delta


def current_phase(
    *,
    map_location: int,
    sword_level: int,
    max_hearts: int,
    visited: set[int],
    in_cave: bool,
    cave_gave_up: bool = False,
    cave_frames: int = 0,
) -> str:
    """Highest-priority FAQ milestone Billy is working on."""
    if cave_quest_active(
            map_location, sword_level, in_cave=in_cave,
            cave_frames=cave_frames, cave_gave_up=cave_gave_up):
        return "wooden_sword"
    gx, gy = screen_to_grid(map_location)
    if (SEA_EAST_SCREEN not in visited and gy == 8
            and START_SCREEN <= map_location < SEA_EAST_SCREEN):
        return "east_to_sea"
    if sword_level == 1 and max_hearts < 5 and LEVEL_1_SCREEN not in visited:
        return "pre_level1_hearts"
    if LEVEL_1_SCREEN not in visited and map_location != LEVEL_1_STAIRS:
        return "level_1_approach"
    return "explore"


def route_step(
    here: int,
    *,
    sword_level: int,
    max_hearts: int,
    visited: set[int],
    in_cave: bool,
    link_x: int,
    link_y: int,
    cave_gave_up: bool = False,
    cave_frames: int = 0,
) -> RouteStep | None:
    """Return the FAQ route direction when we have a clear next step."""
    phase = current_phase(
        map_location=here,
        sword_level=sword_level,
        max_hearts=max_hearts,
        visited=visited,
        in_cave=in_cave,
        cave_gave_up=cave_gave_up,
        cave_frames=cave_frames,
    )

    if phase == "wooden_sword":
        if here != START_SCREEN:
            delta = START_SCREEN - here
            if delta in (SCREEN_NORTH, SCREEN_SOUTH, SCREEN_EAST, SCREEN_WEST):
                btn = {
                    SCREEN_NORTH: c.UP,
                    SCREEN_SOUTH: c.DOWN,
                    SCREEN_EAST: c.RIGHT,
                    SCREEN_WEST: c.LEFT,
                }[delta]
                label = {c.UP: "north", c.DOWN: "south", c.RIGHT: "east", c.LEFT: "west"}[btn]
                return RouteStep(phase, btn, f"walkthrough-return-{label}", START_SCREEN)
            return None
        if in_cave:
            return None   # reflex owns cave interior
        dist = abs(link_x - START_CAVE_STAND[0]) + abs(link_y - START_CAVE_STAND[1])
        if dist <= 24:
            return RouteStep(phase, c.LEFT, "walkthrough-enter-nw-cave", here)
        return None   # approach handled by cave_approach_button

    if phase == "east_to_sea":
        if here < SEA_EAST_SCREEN:
            dest = _adjacent(here, SCREEN_EAST)
            return RouteStep(phase, c.RIGHT, "walkthrough-east-to-sea", dest)
        return None

    if phase == "level_1_approach":
        if here == START_SCREEN:
            # FAQ: from origin go right, up 4, then left → (8, 4) = screen 55
            dest = _adjacent(here, SCREEN_EAST)
            return RouteStep(phase, c.RIGHT, "walkthrough-level1-right", dest)
        if here in NORTH_CAVE_CHAIN:
            idx = NORTH_CAVE_CHAIN.index(here)
            for dest in NORTH_CAVE_CHAIN[idx + 1:]:
                if dest not in visited:
                    delta = dest - here
                    btn = {
                        SCREEN_NORTH: c.UP,
                        SCREEN_SOUTH: c.DOWN,
                        SCREEN_EAST: c.RIGHT,
                        SCREEN_WEST: c.LEFT,
                    }.get(delta)
                    if btn is None:
                        continue
                    label = {c.UP: "north", c.DOWN: "south", c.RIGHT: "east", c.LEFT: "west"}[btn]
                    return RouteStep(phase, btn, f"walkthrough-cave-{label}", dest)
        gx, gy = screen_to_grid(here)
        if gx < 8:
            return RouteStep(phase, c.RIGHT, "walkthrough-level1-east", _adjacent(here, SCREEN_EAST))
        if gy > 4:
            return RouteStep(phase, c.UP, "walkthrough-level1-north", _adjacent(here, SCREEN_NORTH))
        if gx > 8:
            return RouteStep(phase, c.LEFT, "walkthrough-level1-west", _adjacent(here, SCREEN_WEST))
        if here != LEVEL_1_SCREEN:
            return RouteStep(phase, c.UP, "walkthrough-level1-north", _adjacent(here, SCREEN_NORTH))

    return None


def phase_summary(
    *,
    map_location: int,
    sword_level: int,
    max_hearts: int,
    visited: set[int],
    in_cave: bool,
    cave_gave_up: bool = False,
    cave_frames: int = 0,
) -> str:
    phase = current_phase(
        map_location=map_location,
        sword_level=sword_level,
        max_hearts=max_hearts,
        visited=visited,
        in_cave=in_cave,
        cave_gave_up=cave_gave_up,
        cave_frames=cave_frames,
    )
    labels = {
        "wooden_sword": "FAQ: wooden sword (NW cave)",
        "east_to_sea": f"FAQ: east to sea (#{SEA_EAST_SCREEN})",
        "pre_level1_hearts": "FAQ: hearts before Level 1",
        "level_1_approach": f"FAQ: Level 1 (#{LEVEL_1_SCREEN})",
        "explore": "explore",
    }
    return labels.get(phase, phase)