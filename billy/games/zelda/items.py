"""Ground-item perception — drops are NOT enemies."""
from __future__ import annotations

from dataclasses import dataclass

N_SLOTS = 6
DROP_TYPE_BASE = 173
ENEMY_X_BASE = 113
ENEMY_Y_BASE = 133


@dataclass(frozen=True)
class GroundItem:
    slot: int
    x: int
    y: int
    item_type: int


def read_ground_items(ram: bytes) -> list[GroundItem]:
    out: list[GroundItem] = []
    for i in range(N_SLOTS):
        item_type = ram[DROP_TYPE_BASE + i]
        if item_type == 0:
            continue
        x = ram[ENEMY_X_BASE + i]
        y = ram[ENEMY_Y_BASE + i]
        if x == 0 and y == 0:
            continue
        out.append(GroundItem(slot=i + 1, x=x, y=y, item_type=item_type))
    return out


def nearest_item(
    items: list[GroundItem],
    link_x: int,
    link_y: int,
    within: int = 96,
) -> tuple[int, int, GroundItem] | None:
    """Return (dx, dy, item) to the closest ground item."""
    best: tuple[int, int, GroundItem] | None = None
    best_dist = within + 1
    for item in items:
        dx = item.x - link_x
        dy = item.y - link_y
        dist = abs(dx) + abs(dy)
        if dist < best_dist:
            best_dist = dist
            best = (dx, dy, item)
    return best


def walk_toward(dx: int, dy: int) -> int:
    """Single-step walk button toward (dx, dy)."""
    from ...systems.nes import controller as c

    if abs(dy) >= abs(dx):
        return c.UP if dy < 0 else c.DOWN
    return c.LEFT if dx < 0 else c.RIGHT