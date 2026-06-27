"""Exploration direction scoring — testable without the emulator."""
from __future__ import annotations

from collections import deque

from ...systems.nes import controller as c
from .curiosity import (
    curiosity_bonus,
    curious_exit,
    requires_start_cave_inspection,
)
from .tuning import (
    BLOCKED_NEIGHBORS,
    DUNGEON_ENTRANCE_SCREENS,
    SCREEN_EAST,
    SCREEN_NORTH,
    SCREEN_SOUTH,
    SCREEN_WEST,
    START_SCREEN,
)

_DIR_CANDIDATES = (
    (c.RIGHT, SCREEN_EAST, "east"),
    (c.DOWN, SCREEN_SOUTH, "south"),
    (c.UP, SCREEN_NORTH, "north"),
    (c.LEFT, SCREEN_WEST, "west"),
)


def adjacent_screens(here: int) -> list[tuple[int, int, str]]:
    """(button, dest_screen_id, label) for each cardinal neighbor."""
    return [(btn, here + delta, label) for btn, delta, label in _DIR_CANDIDATES]


def pick_explore_direction(
    here: int,
    visited: set[int],
    *,
    recent: deque[int] | list[int] = (),
    prefer_dungeons: bool = True,
    use_curiosity: bool = True,
    cave_mouths: tuple[tuple[int, int], ...] = (),
    sword_level: int = 0,
    max_hearts: int = 3,
    in_cave: bool = False,
    link_x: int = 0,
    link_y: int = 0,
) -> tuple[int, str, int]:
    """Return (button, label, dest_screen) with anti-oscillation and dungeon bias.

    Priority:
      0. Billy Curiosity — unexplored cave / dungeon hook from this screen
      1. Unvisited screen, not a recent backtrack
      2. Unvisited screen (even if recent — better than ping-pong)
      3. Adjacent dungeon-entrance screen we haven't committed to recently
      4. Any direction not leading to the screen we just left
      5. Default east
    """
    if use_curiosity and requires_start_cave_inspection(
            here, visited, recent, sword_level=sword_level, in_cave=in_cave):
        from .curiosity import start_cave_inspection_action
        return start_cave_inspection_action(link_x, link_y, cave_mouths=cave_mouths)

    if use_curiosity:
        curious = curious_exit(
            here, visited, cave_mouths=cave_mouths, recent=recent,
            sword_level=sword_level, max_hearts=max_hearts, in_cave=in_cave,
            link_x=link_x, link_y=link_y)
        if curious is not None:
            return curious

    recent_set = set(recent)
    last = recent[-1] if recent else None
    blocked = BLOCKED_NEIGHBORS.get(here, frozenset())
    options = [(b, d, l) for b, d, l in adjacent_screens(here) if d not in blocked]
    if not options:
        options = list(adjacent_screens(here))

    def score(btn: int, dest: int, label: str) -> tuple[int, int, int]:
        unvisited = dest not in visited
        backtrack = dest in recent_set or dest == last
        dungeon = prefer_dungeons and dest in DUNGEON_ENTRANCE_SCREENS
        curious = curiosity_bonus(here, dest, visited) if use_curiosity else 0
        # Higher is better: tuple comparison
        rank = (
            curious,
            1 if unvisited and not backtrack else 0,
            1 if unvisited else 0,
            1 if dungeon and not backtrack else 0,
            0 if backtrack else 1,
        )
        return (sum(rank), -dest if unvisited else 0, dest)

    ranked = sorted(
        ((score(btn, dest, label), btn, label, dest) for btn, dest, label in options),
        reverse=True,
    )
    _, btn, label, dest = ranked[0]
    return btn, label, dest