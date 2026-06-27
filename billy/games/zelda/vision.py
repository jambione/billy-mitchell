"""Screen vision for Zelda — detect black-square cave mouths from RGB frames."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# NES Zelda viewport (stable-retro rgb_array).
FRAME_W = 240
FRAME_H = 224
TILE = 16

# Playfield band — skip HUD (tile row 0) and bottom status.
SCAN_TILE_Y0 = 1
SCAN_TILE_Y1 = 6

# A cave mouth is a dark 16×16 (or 16×32) square tile cluster.
DARK_MAX = 48
DARK_MEAN = 30
MIN_SOLID_RATIO = 0.55


def _tile_dark(frame: "np.ndarray", x: int, y: int) -> bool:
    patch = frame[y:y + TILE, x:x + TILE]
    if patch.shape[0] < TILE or patch.shape[1] < TILE:
        return False
    solid = (patch.max(axis=2) <= DARK_MAX).mean()
    return solid >= MIN_SOLID_RATIO and patch.mean() <= DARK_MEAN


def _merge_tiles(tiles: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Group 16px tiles that touch orthogonally into blobs."""
    remaining = set(tiles)
    blobs: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        blob = {seed}
        stack = [seed]
        while stack:
            tx, ty = stack.pop()
            for nx, ny in ((tx - 1, ty), (tx + 1, ty), (tx, ty - 1), (tx, ty + 1)):
                if (nx, ny) in remaining:
                    remaining.remove((nx, ny))
                    blob.add((nx, ny))
                    stack.append((nx, ny))
        blobs.append(blob)
    return blobs


def _blob_is_cave_square(blob: set[tuple[int, int]]) -> bool:
    xs = [t[0] for t in blob]
    ys = [t[1] for t in blob]
    w = (max(xs) - min(xs) + 1) * TILE
    h = (max(ys) - min(ys) + 1) * TILE
    area = len(blob) * TILE * TILE
    if w > 80 or h > 48 or area < TILE * TILE:
        return False
    aspect = max(w, h) / max(min(w, h), 1)
    return aspect <= 3.0 and h <= 48


def detect_cave_mouths(frame: "np.ndarray", *, prefer_nw: bool = True) -> list[tuple[int, int]]:
    """Return link-space (x, y) targets for visible black-square cave mouths."""
    import numpy as np

    img = np.asarray(frame)
    if img.ndim != 3 or img.shape[0] < SCAN_TILE_Y1 * TILE:
        return []

    dark_tiles: set[tuple[int, int]] = set()
    scan_x1 = 6 if prefer_nw else 12
    for ty in range(SCAN_TILE_Y0, SCAN_TILE_Y1):
        for tx in range(0, scan_x1):
            x, y = tx * TILE, ty * TILE
            if _tile_dark(img, x, y):
                dark_tiles.add((tx, ty))

    candidates: list[tuple[tuple[int, int], float]] = []
    for blob in _merge_tiles(dark_tiles):
        if not _blob_is_cave_square(blob):
            continue
        tx = sum(t[0] for t in blob) / len(blob)
        ty = sum(t[1] for t in blob) / len(blob)
        px = int(tx * TILE + TILE // 2)
        py = int(ty * TILE + TILE // 2)
        link_x = max(16, min(220, px))
        link_y = max(48, min(200, py))
        # NW caves: leftmost wins, then highest (smallest ty).
        rank = tx * 20 + ty
        candidates.append(((link_x, link_y), rank))

    if not candidates:
        return []
    candidates.sort(key=lambda item: item[1])
    return [candidates[0][0]]