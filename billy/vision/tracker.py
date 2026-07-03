"""PixelTracker — stateful pixel perception: a frame stream in, engine signals out.

Produces the minimum the Director's contracts need, from pixels alone:
  progress   — cumulative camera scroll + player screen x (the generic `Observation.progress`)
  player     — screen box of the controlled sprite, tracked across frames
  on_ground  — feet supported by non-background pixels and vertical velocity ~0
  dead       — player fell below the viewport, or was lost into a scene blackout
  area_seq   — level identity: bumps when the scene's color signature changes for good

Honest design notes (v1):
- The player is found MOTION-first (sprites move differently from the scrolled background)
  and tracked by nearest-neighbor continuation; the boot wiggle in the adapter guarantees the
  player is moving when tracking starts. There is no appearance model yet.
- Death detection is the weakest signal (pit falls are solid; on-the-spot enemy deaths rely
  on the post-death blackout). The SMB parity probe measures exactly how weak.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import (HUD_ROWS, background_color, color_histogram, estimate_scroll_cost,
                   find_blobs, fingerprint_distance, motion_mask, to_gray)

_REACQUIRE_RADIUS = 48      # px: how far the player can plausibly move between observes
_LOST_LIMIT = 6             # updates without the player before we consider them gone
_AREA_DIST = 0.75           # histogram distance that means "different scene"
_AREA_SUSTAIN = 3           # consecutive different-scene updates before switching areas
_MAX_SPRITE = 64            # px: a "player" bigger than this is a merge with scenery
_CUT_COST = 28              # scroll match cost above which the scene CUT (respawn/warp)


@dataclass
class PixelView:
    frame: int
    progress: int
    camera_x: int
    player: tuple | None        # (x, y, w, h) on screen
    on_ground: bool
    dead: bool
    in_play: bool
    area_seq: int               # monotonic scene counter (level identity)
    scroll_dx: int
    blobs: list = field(default_factory=list)

    def summary(self) -> str:
        p = f"({self.player[0]},{self.player[1]})" if self.player else "lost"
        return (f"pixel area#{self.area_seq} progress={self.progress} player={p} "
                f"on_ground={self.on_ground} sprites={len(self.blobs)}")


class PixelTracker:
    def __init__(self, hud_rows: int = HUD_ROWS):
        self.hud_rows = hud_rows
        self._prev: np.ndarray | None = None
        self._prev_frame_no = 0
        self.camera_x = 0
        self.player: tuple | None = None
        self._player_v = (0.0, 0.0)
        self._px_ema: float | None = None    # smoothed player-center x (progress component)
        self._bottoms: list[int] = []        # recent player bottom-y (ground stability)
        self._lost = 0
        self._fell = False
        self._on_ground = False       # last verdict — reused when there's no fresh evidence
        self._vx_per_frame = 0.0
        self.area_seq = 0
        self._area_hist: np.ndarray | None = None
        self._area_diff_run = 0
        self._blackout_run = 0
        self._last_view: PixelView | None = None

    # --- helpers -------------------------------------------------------------------------
    @staticmethod
    def _is_blackout(gray: np.ndarray) -> bool:
        """Loading/death/transition screens: nearly uniform, nearly dark."""
        return bool(gray.std() < 8 or (gray < 24).mean() > 0.92)

    def _pick_player(self, blobs: list[tuple[int, int, int, int]],
                     h: int) -> tuple | None:
        if not blobs:
            return None
        if self.player is not None:
            px = self.player[0] + self.player[2] / 2 + self._player_v[0]
            py = self.player[1] + self.player[3] / 2 + self._player_v[1]
            best, bd = None, _REACQUIRE_RADIUS + self._lost * 16
            for b in blobs:
                cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                d = abs(cx - px) + abs(cy - py)
                if d < bd:
                    best, bd = b, d
            if best is not None:
                return best
            return None
        # Bootstrap: prefer a sprite-sized blob in the lower 2/3 of the screen (platformer
        # players live near the ground; HUD/score popups live on top).
        cands = [b for b in blobs if b[1] > h // 3 and 6 <= b[2] <= 64 and 6 <= b[3] <= 64]
        return cands[0] if cands else blobs[0]

    # --- main entry ------------------------------------------------------------------------
    def update(self, frame: np.ndarray, frame_no: int) -> PixelView:
        gray = to_gray(frame)
        h, w = gray.shape

        if self._is_blackout(gray):
            # Not in play. A blackout right after losing the player corroborates a death.
            dead = self._fell or (self.player is None and self._lost >= _LOST_LIMIT
                                  and self._last_view is not None)
            view = PixelView(frame=frame_no, progress=self._progress(), camera_x=self.camera_x,
                             player=None, on_ground=False, dead=dead, in_play=False,
                             area_seq=self.area_seq, scroll_dx=0)
            self._prev = None          # scroll estimation restarts after the transition
            self._blackout_run += 1
            self._last_view = view
            return view
        if self._blackout_run >= 2:
            # A SUSTAINED blackout was a transition (death screen, level load) — whatever
            # comes after it is a fresh scene, so progress re-bases. (A single dark flash
            # doesn't count; some games strobe on hits.)
            self._rebase()
        self._blackout_run = 0

        # --- scene identity (level/area change) ------------------------------------------
        hist = color_histogram(frame, self.hud_rows)
        if self._area_hist is None:
            self._area_hist = hist
        elif fingerprint_distance(self._area_hist, hist) > _AREA_DIST:
            self._area_diff_run += 1
            if self._area_diff_run >= _AREA_SUSTAIN:
                self.area_seq += 1
                self._area_hist = hist
                self._area_diff_run = 0
                self._rebase()             # progress re-bases per area, like RAM perception
                self._prev = None
        else:
            self._area_diff_run = 0
            # Slow-adapt the reference so gradual palette shifts don't false-trigger.
            self._area_hist = (self._area_hist * 15 + hist) // 16

        # --- scroll + sprites --------------------------------------------------------------
        scroll = 0
        blobs: list[tuple[int, int, int, int]] = []
        if self._prev is not None:
            dframes = max(1, frame_no - self._prev_frame_no)
            expect = int(self._vx_per_frame * dframes)
            scroll, match_cost = estimate_scroll_cost(self._prev, frame, expect=expect,
                                                      hud_rows=self.hud_rows)
            if match_cost > _CUT_COST:
                # Even the best shift is a terrible match: the scene CUT (respawn after a
                # death, pipe warp, screen load) rather than scrolled. Progress must re-base —
                # otherwise every respawn inherits the previous life's camera and the same
                # world spot never maps to the same progress twice (cache poison).
                self._rebase()
                self._prev = frame
                self._prev_frame_no = frame_no
                view = PixelView(frame=frame_no, progress=self._progress(),
                                 camera_x=self.camera_x, player=None, on_ground=False,
                                 dead=False, in_play=True, area_seq=self.area_seq,
                                 scroll_dx=0)
                self._last_view = view
                return view
            self._vx_per_frame = 0.7 * self._vx_per_frame + 0.3 * (scroll / dframes)
            self.camera_x += scroll
            blobs = find_blobs(motion_mask(self._prev, frame, scroll,
                                           hud_rows=self.hud_rows))

        # --- player tracking ----------------------------------------------------------------
        picked = self._pick_player(blobs, h)
        reliable = picked is not None and picked[2] <= _MAX_SPRITE and picked[3] <= _MAX_SPRITE
        on_ground = False
        if picked is not None:
            if self.player is not None:
                self._player_v = (0.6 * self._player_v[0]
                                  + 0.4 * (picked[0] - self.player[0]),
                                  0.6 * self._player_v[1]
                                  + 0.4 * (picked[1] - self.player[1]))
            self.player, self._lost = picked, 0
            cx = picked[0] + picked[2] / 2
            self._px_ema = cx if self._px_ema is None else 0.7 * self._px_ema + 0.3 * cx
            if reliable:
                # An oversized match is the player merged with scenery/enemies — track the
                # position but don't trust its BOX for ground/fall judgments.
                x, y, bw, bh = picked
                bottom = y + bh
                self._bottoms = (self._bottoms + [bottom])[-3:]
                feet = frame[min(h - 1, bottom + 1):min(h, bottom + 5), x:x + bw]
                bg = background_color(frame, self.hud_rows)
                solid = bool(feet.size) and (
                    np.abs(feet.astype(np.int16) - bg.astype(np.int16)).max(axis=-1)
                    > 40).mean() > 0.4
                # Grounded = supported AND the bottom edge has been STABLE (box jitter from
                # 8px cell quantization makes instantaneous vy useless).
                stable = (len(self._bottoms) >= 2
                          and max(self._bottoms) - min(self._bottoms) <= 8)
                on_ground = solid and stable
                if on_ground:
                    self._fell = False
                if bottom >= h - 8 and self._player_v[1] > 2:
                    self._fell = True        # small sprite, at the bottom edge, moving down
        else:
            self._lost += 1
            # No motion = no new evidence. A player who stopped MOVING did not stop STANDING:
            # things in the air fall (and would produce motion), so keep the last verdict.
            on_ground = self._on_ground if self.player is not None else False
            if self.player is not None and self.player[1] + self.player[3] >= h - 24 \
                    and self._player_v[1] > 2:
                self._fell = True
            if self._lost >= _LOST_LIMIT:
                self.player = None
                on_ground = False

        dead = self._fell and (self._lost >= 2
                               or (reliable and picked[1] + picked[3] >= h - 2))

        self._on_ground = on_ground
        self._prev = frame
        self._prev_frame_no = frame_no
        view = PixelView(frame=frame_no, progress=self._progress(), camera_x=self.camera_x,
                         player=self.player, on_ground=on_ground, dead=dead, in_play=True,
                         area_seq=self.area_seq, scroll_dx=scroll, blobs=blobs)
        self._last_view = view
        return view

    def _rebase(self) -> None:
        """Progress restarts from zero at a scene discontinuity — a level entry always reads
        the same progress this way, which is what makes pixel buckets reproducible."""
        self.camera_x = 0
        self._fell = False
        self._lost = 0
        self.player = None
        self._px_ema = None
        self._bottoms = []
        self._player_v = (0.0, 0.0)
        self._vx_per_frame = 0.0
        self._on_ground = False

    def respawned(self) -> None:
        """The engine restored a checkpoint — clear death/motion state and re-base progress
        (the cut detector also catches this, but an explicit signal is cheaper and exact)."""
        self._rebase()
        self._prev = None

    def _progress(self) -> int:
        # Smoothed player x: blob boxes are cell-quantized (±8px jitter) and the occasional
        # wrong pick (an enemy) would spike raw progress; the EMA keeps deltas honest.
        px = self._px_ema if self._px_ema is not None else 0.0
        return max(0, int(self.camera_x + px))
