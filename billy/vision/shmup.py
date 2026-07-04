"""ShmupTracker — pixel perception for a FIXED-CAMERA shooter (no RAM map, no scroll).

The platformer PixelTracker bakes in horizontal-scroll progress and a ground line; a vertical
shoot-'em-up has neither. This tracker keeps only what the engine's learning loop actually
needs, derived from pixels alone:

  progress   — SURVIVAL frames. A shmup has no spatial position, but the emulator is
               deterministic from boot, so "how long this input trajectory keeps the ship
               alive" is a monotonic, reproducible score — and a tape that survives longer is
               exactly what search should prefer. (Score OCR is a later refinement.)
  player     — the controlled ship: the persistent sprite low on the screen. Confirmable by
               input-response (it moves horizontally with LEFT/RIGHT) — the boot does that
               probe; in play we track it by nearest-neighbor continuation.
  enemies    — the other moving sprites (for the reflex's dodge/aim candidates).
  dead       — a LIFE LOST. The ship explodes and vanishes for a beat before respawning; a
               sustained loss of the tracked ship (longer than any occlusion) is a death. This
               is the frequent, learnable wall — "survive longer before the first hit" — not
               the all-lives terminal. Validated in the probe against the integration's lives
               counter (ground truth only, never used here).

Honest v1 limits: no appearance model (nearest-neighbor track can jump to an enemy that
overlaps the ship); survival-only progress ignores score; death is a lost-ship heuristic, so
its timing lags the explosion by the loss window.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import background_color, find_blobs

_HUD_ROWS = 24              # Genesis Airstriker HUD band (Lives/Score) along the top
_REACQUIRE = 44            # px the ship can plausibly move between observes
_BRIGHT = 40               # per-channel distance from the space background = a sprite pixel
_DEATH_DROP = 0.6          # bright area falling below this fraction of baseline = a death
_MIN_FIELD = 200           # baseline bright area must exceed this for a drop to be meaningful


def bright_mask(frame: np.ndarray, hud_rows: int) -> np.ndarray:
    """Sprite pixels on a flat (black space) background: everything far from the dominant
    background color. Appearance, not motion — so an idle ship is still found."""
    bg = background_color(frame, hud_rows)
    diff = np.abs(frame.astype(np.int16) - bg.astype(np.int16))
    mask = (diff.max(axis=-1) if frame.ndim == 3 else diff) > _BRIGHT
    mask[:hud_rows] = False
    return mask


def bright_blobs(frame: np.ndarray, hud_rows: int) -> list[tuple[int, int, int, int]]:
    """Appearance-based sprite blobs (see bright_mask). Robust for fixed-camera shooters."""
    return find_blobs(bright_mask(frame, hud_rows))


@dataclass
class ShmupView:
    frame: int
    progress: int               # survival frames
    player: tuple | None        # (x, y, w, h)
    enemies: list = field(default_factory=list)
    dead: bool = False
    in_play: bool = True

    def summary(self) -> str:
        p = f"({self.player[0]},{self.player[1]})" if self.player else "lost"
        return (f"shmup survived={self.progress} ship={p} "
                f"enemies={len(self.enemies)} {'DEAD' if self.dead else ''}").strip()


class ShmupTracker:
    def __init__(self, hud_rows: int = _HUD_ROWS):
        self.hud_rows = hud_rows
        self._prev: np.ndarray | None = None
        self.player: tuple | None = None
        self._player_v = (0.0, 0.0)
        self._lost = 0
        self._alive = 0            # survival counter (progress) — observes before first death
        self._dead = False
        self._area_ema: float | None = None   # slow baseline of bright-sprite area

    # --- player pick ----------------------------------------------------------------------
    def _pick_player(self, blobs: list[tuple[int, int, int, int]], h: int) -> tuple | None:
        """The ship is the persistent sprite low on the screen. Track by nearest-neighbor to
        the last position; bootstrap to the bottom-most reasonable blob."""
        if not blobs:
            return None
        if self.player is not None:
            px = self.player[0] + self.player[2] / 2 + self._player_v[0]
            py = self.player[1] + self.player[3] / 2 + self._player_v[1]
            best, bd = None, _REACQUIRE + self._lost * 12
            for b in blobs:
                cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                d = abs(cx - px) + abs(cy - py)
                if d < bd:
                    best, bd = b, d
            return best
        # Bootstrap: prefer sprite-sized blobs in the lower half, bottom-most first.
        cands = [b for b in blobs
                 if b[1] + b[3] > h // 2 and 6 <= b[2] <= 48 and 6 <= b[3] <= 48]
        cands.sort(key=lambda b: -(b[1] + b[3]))     # lowest on screen wins
        return cands[0] if cands else None

    # --- main entry -----------------------------------------------------------------------
    def update(self, frame: np.ndarray, frame_no: int) -> ShmupView:
        h, w = frame.shape[:2]
        # Appearance-based: every sprite stands out from the flat background, moving or not.
        mask = bright_mask(frame, self.hud_rows)
        blobs = find_blobs(mask)
        field_area = int(mask.sum())

        picked = self._pick_player(blobs, h)
        if picked is not None:
            if self.player is not None:
                self._player_v = (0.6 * self._player_v[0] + 0.4 * (picked[0] - self.player[0]),
                                  0.6 * self._player_v[1] + 0.4 * (picked[1] - self.player[1]))
            self.player, self._lost = picked, 0
        else:
            self._lost += 1

        enemies = [b for b in blobs if b is not picked]

        # SURVIVAL always advances — the progress signal a longer trajectory earns. It must
        # NOT be gated on the (best-effort, fragile) pixel death below, or a spurious death
        # latch would freeze progress and starve the learning loop of a reach signal.
        self._alive += 1
        # DEATH-from-pixels (view.dead) is a DIAGNOSTIC only: the sprite area COLLAPSES when a
        # life is lost (the ship + surrounding wave clear, roughly halving field area). It's
        # clean under a stable-firing policy but confounded by the player's own bullet area,
        # so the adapter uses the integration's lives for the real terminal, not this. Kept so
        # the probe can grade it against ground truth.
        if (not self._dead and self._area_ema is not None
                and self._area_ema > _MIN_FIELD
                and field_area < _DEATH_DROP * self._area_ema):
            self._dead = True
        self._area_ema = (float(field_area) if self._area_ema is None
                          else 0.85 * self._area_ema + 0.15 * field_area)

        self._prev = frame
        view = ShmupView(frame=frame_no, progress=self._alive, player=self.player,
                         enemies=enemies, dead=self._dead, in_play=not self._dead)
        return view

    def rebase(self) -> None:
        """New attempt from the boot state — clear all per-run state."""
        self._prev = None
        self.player = None
        self._player_v = (0.0, 0.0)
        self._lost = 0
        self._alive = 0
        self._dead = False
        self._area_ema = None

    # time-travel support (the tracked-session wrapper snapshots __dict__ wholesale)
    respawned = rebase
