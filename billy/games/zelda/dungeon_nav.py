"""Level 1+ dungeon navigation — keys, doors, room exploration."""
from __future__ import annotations

from ...abstractions import Decision, Step
from ...systems.nes import controller as c
from .tuning import REFLEX_FRAMES

_KEY_HOLD = 24
_CARDINALS = (
    (c.UP, "up"),
    (c.RIGHT, "right"),
    (c.DOWN, "down"),
    (c.LEFT, "left"),
)


def pick_dungeon_direction(
    here: int,
    visited: set[int],
    *,
    recent: tuple[int, ...] | list[int] = (),
) -> tuple[int, str]:
    """Greedy dungeon explore: prefer unvisited room guesses (cache learns true graph)."""
    last = recent[-1] if recent else None
    options = [
        (c.UP, here - 16, "north"),
        (c.DOWN, here + 16, "south"),
        (c.RIGHT, here + 1, "east"),
        (c.LEFT, here - 1, "west"),
    ]
    unvisited = [(b, d, n) for b, d, n in options if 0 <= d <= 255 and d not in visited]
    if unvisited:
        btn, dest, name = unvisited[0]
        return btn, f"dungeon-{name}→#{dest}"
    non_back = [(b, d, n) for b, d, n in options if 0 <= d <= 255 and d != last]
    if non_back:
        btn, dest, name = non_back[0]
        return btn, f"dungeon-{name}→#{dest}"
    return c.UP, "dungeon-idle"


def dungeon_key_decision(scene) -> Decision | None:
    """Use a key on a blocking door at a screen edge."""
    if not scene.in_dungeon or scene.keys <= 0:
        return None
    dung = scene.dungeon
    if dung is None or not dung.locked_doors:
        return None
    if scene.at_right_edge:
        return Decision([Step(_KEY_HOLD, c.mask(c.A, c.RIGHT))], note="dungeon-key-right")
    if scene.at_left_edge:
        return Decision([Step(_KEY_HOLD, c.mask(c.A, c.LEFT))], note="dungeon-key-left")
    if scene.at_top_edge:
        return Decision([Step(_KEY_HOLD, c.mask(c.A, c.UP))], note="dungeon-key-up")
    if scene.at_bottom_edge:
        return Decision([Step(_KEY_HOLD, c.mask(c.A, c.DOWN))], note="dungeon-key-down")
    return None


def dungeon_combat_decision(scene) -> Decision | None:
    """Basic sword swing when a dungeon enemy is adjacent."""
    if not scene.in_dungeon or scene.enemy_count() == 0:
        return None
    near = scene.nearest_enemy(within=56)
    if near is None:
        return None
    dx, dy = near
    if abs(dx) >= abs(dy):
        btn = c.RIGHT if dx >= 0 else c.LEFT
    else:
        btn = c.DOWN if dy >= 0 else c.UP
    return Decision(
        [Step(REFLEX_FRAMES, btn), Step(REFLEX_FRAMES * 2, c.mask(btn, c.B))],
        note=f"dungeon-fight ({dx},{dy})",
    )


def dungeon_explore_decision(scene, visited: set[int], recent) -> Decision | None:
    """One step of dungeon exploration when no combat override."""
    btn, label = pick_dungeon_direction(
        scene.map_location, visited, recent=tuple(recent))
    return Decision([Step(REFLEX_FRAMES, btn)], note=f"dungeon {label}")