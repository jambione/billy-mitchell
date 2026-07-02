"""Scene -> fixed-length observation vector, and the discrete action set.

Reuses the existing RAM perception (`Scene`) rather than pixels: it's far more sample-efficient and
keeps RL on the same footing as the rest of Billy (RAM-derived state). The feature vector is the
local tile window + Mario's dynamic state + the nearest enemies + the derived gap/obstacle probes —
exactly the signals the hand-crafted reflex already reasons over.

The action set is a focused SMB controller vocabulary (NES button combos), mapped to the engine's
controller bitmasks so a learned action becomes a normal `Step` the Session can execute.
"""
from __future__ import annotations

import numpy as np

from ..games.smb.perception import COL_AHEAD, COL_BEHIND, GRID_ROWS, Scene
from ..systems.nes import controller as C

# --- discrete action vocabulary (name tuples -> controller masks) -----------------------
ACTION_NAMES: list[tuple[str, ...]] = [
    (),                       # NOOP
    ("right",),
    ("right", "B"),           # run right
    ("right", "A"),           # jump right
    ("right", "A", "B"),      # run-jump right
    ("A",),                   # jump in place
    ("left",),
    ("left", "B"),
    ("left", "A"),
    ("down",),                # duck / enter pipe
]
ACTION_MASKS: list[int] = [C.mask_from_names(list(n)) for n in ACTION_NAMES]
N_ACTIONS = len(ACTION_MASKS)

_TILE_COLS = COL_BEHIND + COL_AHEAD + 1          # 13
_N_TILES = GRID_ROWS * _TILE_COLS               # 13 * 13 = 169
_MAX_ENEMIES = 4                                 # nearest N enemies as (dx, dy)
# tiles + [x_speed, on_ground, size, mario_y] + enemies*2 + [gap_dist,gap_w,obs_dist,obs_h,block]
OBS_DIM = _N_TILES + 4 + _MAX_ENEMIES * 2 + 5


def featurize(scene: Scene) -> np.ndarray:
    """Encode a Scene into a normalized float32 vector of length OBS_DIM (stable across frames)."""
    out = np.zeros(OBS_DIM, dtype=np.float32)
    i = 0

    # local solid-tile window (binary), padded/truncated to the expected shape
    flat = [t for row in scene.tiles for t in row]
    for k in range(_N_TILES):
        out[i + k] = 1.0 if (k < len(flat) and flat[k]) else 0.0
    i += _N_TILES

    # Mario dynamic state
    out[i + 0] = np.clip(scene.x_speed / 40.0, -1.0, 1.0)
    out[i + 1] = 1.0 if scene.on_ground else 0.0
    out[i + 2] = scene.size / 2.0
    out[i + 3] = np.clip(scene.mario_y / 240.0, 0.0, 1.0)
    i += 4

    # nearest enemies ahead/around, as relative (dx, dy) normalized
    rel = sorted(((e.x - scene.mario_x, e.y - scene.mario_y) for e in scene.enemies),
                 key=lambda d: abs(d[0]))
    for k in range(_MAX_ENEMIES):
        if k < len(rel):
            out[i + k * 2] = np.clip(rel[k][0] / 128.0, -1.0, 1.0)
            out[i + k * 2 + 1] = np.clip(rel[k][1] / 128.0, -1.0, 1.0)
    i += _MAX_ENEMIES * 2

    # derived hazard probes (reuse the perception helpers)
    gap = scene.gap_info()
    obst = scene.obstacle_ahead()
    block = scene.block_above_ahead()
    out[i + 0] = np.clip((gap[0] if gap else 128) / 128.0, 0.0, 1.0)
    out[i + 1] = (gap[1] if gap else 0) / 8.0
    out[i + 2] = np.clip((obst[0] if obst else 64) / 64.0, 0.0, 1.0)
    out[i + 3] = (obst[1] if obst else 0) / 4.0
    out[i + 4] = np.clip((block if block is not None else 64) / 64.0, 0.0, 1.0)
    return out
