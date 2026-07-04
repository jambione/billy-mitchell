"""Pure frame primitives for pixel perception (numpy only — fast enough for the hot loop).

Everything here is stateless and unit-testable on synthetic frames. The design constraint:
`observe()` runs after every committed chunk and search rollout step, so the whole pipeline
must cost ~a millisecond, which rules out heavyweight vision. NES-era games make this
tractable: flat background colors, sprite-scale moving objects, integer-pixel scrolling.
"""
from __future__ import annotations

import numpy as np

# Default playfield band for NES frames (224x240): skip the HUD rows on top.
HUD_ROWS = 32


def to_gray(frame: np.ndarray) -> np.ndarray:
    """uint8 luminance. NOT channel-max: distinct NES colors often share a max channel
    (SMB's red player and blue sky both peak at 252), which would make sprites invisible."""
    if frame.ndim == 2:
        return frame
    f = frame.astype(np.uint16)
    return ((f[..., 0] * 77 + f[..., 1] * 151 + f[..., 2] * 28) >> 8).astype(np.uint8)


def estimate_scroll_cost(prev: np.ndarray, cur: np.ndarray, *, max_dx: int = 56,
                         rows: int = 8, hud_rows: int = HUD_ROWS,
                         expect: int = 0) -> tuple[int, float]:
    """(scroll_dx, match_cost). A high match_cost even at the best shift means the frames
    do NOT depict a scrolled version of the same scene — a cut (respawn, warp, fade)."""
    g1, g2 = to_gray(prev).astype(np.int16), to_gray(cur).astype(np.int16)
    h, w = g1.shape
    ys = np.linspace(hud_rows + 4, h - 8, rows).astype(int)
    strip1, strip2 = g1[ys], g2[ys]

    lo, hi = -max_dx, max_dx
    costs = np.full(hi - lo + 1, np.inf)
    for i, dx in enumerate(range(lo, hi + 1)):
        if dx >= 0:
            a, b = strip1[:, dx:], strip2[:, :w - dx]
        else:
            a, b = strip1[:, :w + dx], strip2[:, -dx:]
        if a.shape[1] < 32:
            continue
        costs[i] = np.abs(a - b).mean()
    best = costs.min()
    # The tie-break exists ONLY for periodic ambiguity (tiled backgrounds give near-EQUAL
    # minima a tile apart). Keep the window tight — a shallow cost slope around the true
    # minimum is not a tie, and a generous window lets `expect` drag the answer off it.
    tol = max(0.05, best * 0.02)
    # No-evidence case: a (near-)uniform strip costs ~the same at every shift. If standing
    # still (dx=0) is within the tie window, there is no scroll signal — report 0 rather than
    # letting `expect` hallucinate motion on a blank scene.
    if costs[-lo] <= best + tol:
        return 0, float(best)
    tied = np.flatnonzero(costs <= best + tol)
    dxs = tied + lo
    return int(dxs[np.argmin(np.abs(dxs - expect))]), float(best)


def estimate_scroll(prev: np.ndarray, cur: np.ndarray, *, max_dx: int = 56,
                    rows: int = 8, hud_rows: int = HUD_ROWS, expect: int = 0) -> int:
    """Horizontal camera scroll between two frames, in pixels (positive = world moved left,
    i.e. the camera advanced right).

    Correlates a handful of playfield rows across candidate shifts. Tiled backgrounds repeat
    every 16px, so among near-tied candidates we prefer the one closest to `expect` (the
    caller's velocity guess) — continuity beats a marginally better match a tile away."""
    dx, _ = estimate_scroll_cost(prev, cur, max_dx=max_dx, rows=rows,
                                 hud_rows=hud_rows, expect=expect)
    return dx


def background_color(frame: np.ndarray, hud_rows: int = HUD_ROWS) -> np.ndarray:
    """Dominant playfield color (the sky). Mode over a coarse sample of the upper playfield."""
    band = frame[hud_rows:hud_rows + 64:4, ::4]
    flat = band.reshape(-1, band.shape[-1]) if band.ndim == 3 else band.reshape(-1, 1)
    # Pack RGB to a single int for a cheap mode.
    packed = (flat[:, 0].astype(np.int32) << 16)
    if flat.shape[1] > 1:
        packed |= (flat[:, 1].astype(np.int32) << 8) | flat[:, 2].astype(np.int32)
    vals, counts = np.unique(packed, return_counts=True)
    top = int(vals[np.argmax(counts)])
    return np.array([(top >> 16) & 0xFF, (top >> 8) & 0xFF, top & 0xFF], dtype=np.uint8)


def ground_distance(frame: np.ndarray, bg: np.ndarray, x0: int, x1: int, y0: int, *,
                    max_scan: int = 28, min_frac: float = 0.4) -> int | None:
    """Rows from y0 down to the first SUPPORTED row across columns [x0, x1) — the local
    ground line under a sprite's feet. A row is supported when at least min_frac of its
    pixels differ clearly from the background color (ground top, pipe lip, brick).
    Returns None when nothing solid appears within max_scan rows (a gap/pit below).

    Static, current-frame-only evidence: unlike motion, it works on a sprite that is
    standing still, and it never needs history — which is what makes the on_ground verdict
    reproducible across passes."""
    h = frame.shape[0]
    y0 = max(0, min(h - 1, y0))
    band = frame[y0:min(h, y0 + max_scan), max(0, x0):max(0, x1)]
    if band.size == 0:
        return None
    diff = np.abs(band.astype(np.int16) - bg.astype(np.int16))
    solid = (diff.max(axis=-1) if band.ndim == 3 else diff) > 40
    rows = np.flatnonzero(solid.mean(axis=1) >= min_frac)
    return int(rows[0]) if rows.size else None


def motion_mask(prev: np.ndarray, cur: np.ndarray, scroll_dx: int, *,
                hud_rows: int = HUD_ROWS, thresh: int = 24) -> np.ndarray:
    """Boolean mask of pixels that moved DIFFERENTLY from the global scroll — sprites.
    The scrolled background cancels out; anything left is an object with its own motion.
    Diffed per-channel (max over channels) so a color change with equal luminance still
    registers — NES sprites are palette-distinct more reliably than brightness-distinct."""
    a3 = prev if prev.ndim == 3 else prev[..., None]
    b3 = cur if cur.ndim == 3 else cur[..., None]
    h, w = a3.shape[:2]
    dx = scroll_dx
    if dx >= 0:
        a, b = a3[:, dx:], b3[:, :w - dx]
        off = 0
    else:
        a, b = a3[:, :w + dx], b3[:, -dx:]
        off = -dx
    diff = (np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)) > thresh
    mask = np.zeros((h, w), dtype=bool)
    mask[:, off:off + diff.shape[1]] = diff
    mask[:hud_rows] = False
    return mask


def find_blobs(mask: np.ndarray, *, cell: int = 8, min_cells: int = 2,
               max_blobs: int = 12) -> list[tuple[int, int, int, int]]:
    """Connected clusters in a boolean mask, as (x, y, w, h) pixel boxes.

    Works on a coarse cell grid (cell x cell downsample + flood fill over a few dozen cells)
    — sprite-scale precision without scipy. Returns largest-first."""
    h, w = mask.shape
    gh, gw = h // cell, w // cell
    grid = mask[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).any(axis=(1, 3))
    seen = np.zeros_like(grid, dtype=bool)
    blobs: list[tuple[int, int, int, int, int]] = []
    for gy in range(gh):
        for gx in range(gw):
            if not grid[gy, gx] or seen[gy, gx]:
                continue
            stack = [(gy, gx)]
            seen[gy, gx] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1),
                               (cy - 1, cx - 1), (cy - 1, cx + 1),
                               (cy + 1, cx - 1), (cy + 1, cx + 1)):
                    if 0 <= ny < gh and 0 <= nx < gw and grid[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) < min_cells:
                continue
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            blobs.append((min(xs) * cell, min(ys) * cell,
                          (max(xs) - min(xs) + 1) * cell, (max(ys) - min(ys) + 1) * cell,
                          len(cells)))
    blobs.sort(key=lambda b: -b[4])
    return [(x, y, bw, bh) for x, y, bw, bh, _ in blobs[:max_blobs]]


def frame_fingerprint(frame: np.ndarray, hud_rows: int = HUD_ROWS) -> int:
    """Coarse scene identity: quantized color histogram of the playfield, hashed.

    Stable across scrolling within a level (palette + tile mix barely change), different
    across levels/areas (new palette). Used with hysteresis by the tracker."""
    band = frame[hud_rows::8, ::8]
    if band.ndim == 3:
        q = (band >> 6).reshape(-1, 3)          # 2 bits/channel -> 64 colors
        codes = (q[:, 0] << 4) | (q[:, 1] << 2) | q[:, 2]
    else:
        codes = (band >> 2).reshape(-1)
    hist = np.bincount(codes, minlength=64)
    # Keep the 8 dominant color codes as the signature (order-insensitive, scroll-tolerant).
    top = tuple(sorted(np.argsort(hist)[-8:].tolist()))
    return hash(top) & 0x7FFFFFFF


def fingerprint_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Normalized L1 distance between two color histograms (0 identical, 2 disjoint)."""
    a = hist_a / max(1, hist_a.sum())
    b = hist_b / max(1, hist_b.sum())
    return float(np.abs(a - b).sum())


def color_histogram(frame: np.ndarray, hud_rows: int = HUD_ROWS) -> np.ndarray:
    """The quantized playfield histogram behind frame_fingerprint (for soft comparisons)."""
    band = frame[hud_rows::8, ::8]
    if band.ndim == 3:
        q = (band >> 6).reshape(-1, 3)
        codes = (q[:, 0] << 4) | (q[:, 1] << 2) | q[:, 2]
    else:
        codes = (band >> 2).reshape(-1)
    return np.bincount(codes, minlength=64)
