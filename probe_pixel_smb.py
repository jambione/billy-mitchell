#!/usr/bin/env python3
"""Parity probe: pixel perception vs RAM ground truth on real SMB frames.

Replays the banked 1-1 tape (deterministic, realistic play) in commit-sized chunks, feeding
each frame's rgb to PixelTracker while reading the RAM scene as truth. Reports:
  - progress tracking:   correlation + error of per-chunk deltas (pixel vs mario_x)
  - player acquisition:  % of observes with the player tracked
  - on_ground agreement: % vs RAM (when tracked)
  - death detection:     lag (frames) on a scripted pit death, false-positive count on the
                         clean run
  - cost:                ms per tracker update (hot-loop budget)

    BILLY_HEADLESS=1 .venv/bin/python probe_pixel_smb.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("BILLY_HEADLESS", "1")
os.environ.setdefault("BILLY_TURBO", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from billy.abstractions import Step
from billy.games.smb.game import SmbGame
from billy.games.smb.perception import build_scene
from billy.systems.nes import controller as C
from billy.vision import PixelTracker

CHUNK = 6   # frames per observe — mirrors the Director's live commit chunking


def load_tape(level_key):
    for line in open("data/tapes.jsonl"):
        e = json.loads(line)
        if e.get("level_key") == list(level_key):
            return [Step(f, b) for f, b in e["plan"]]
    return None


def flat_steps(plan):
    out = []
    for s in plan:
        out.extend([s.buttons] * s.frames)
    return out


def main() -> int:
    game = SmbGame()
    session = game.system.connect()
    session.reset()
    session.save_state(9)   # boot state — respawn point for death cycles

    # Sprint-hop play: run+periodic jumps genuinely traverses 1-1 (bare reflex without the
    # search stack stalls at the first hazard; a tape replayed here desyncs mid-level).
    # Deaths are fine — reload and continue; the death chunks are excluded from progress
    # scoring the same way the engine excludes not-in-play frames.
    tracker = PixelTracker()
    truth_x, pixel_p, og_truth, og_pixel, tracked = [], [], [], [], []
    false_deaths = 0
    times = []
    frame_no = 0
    hop = 0
    life = 0
    while frame_no < 4200:
        # Vary the hop cadence per life so lives cross the SAME ground via DIFFERENT
        # trajectories — otherwise identical inputs from an identical boot state make the
        # cross-life reproducibility metric trivially zero.
        period = 4 + (life % 3)
        mask = C.mask(C.RIGHT, C.B, C.A) if (hop // period) % 2 else C.mask(C.RIGHT, C.B)
        hop += 1
        session.send_plan([Step(CHUNK, mask)])
        frame_no += CHUNK
        st = session.read_state()
        scene = build_scene(st.ram, st.frame)
        if scene.is_dying or not scene.in_play:
            session.load_state(9)
            tracker.respawned()
            life += 1
            truth_x.append(None)      # segment boundary — progress totals sum per life
            pixel_p.append(None)
            continue
        t0 = time.perf_counter()
        view = tracker.update(st.rgb, frame_no)
        times.append(time.perf_counter() - t0)
        if not view.in_play:
            continue
        truth_x.append(scene.mario_x)
        pixel_p.append(view.progress)
        tracked.append(view.player is not None)
        if view.player is not None:
            og_truth.append(bool(scene.on_ground))
            og_pixel.append(view.on_ground)
        if view.dead and not scene.is_dying:
            false_deaths += 1

    # Per-life segments (None = respawn boundary): deltas and totals within lives only.
    segs, cur = [], ([], [])
    for t_, p_ in zip(truth_x, pixel_p):
        if t_ is None:
            if len(cur[0]) > 3:
                segs.append(cur)
            cur = ([], [])
        else:
            cur[0].append(t_)
            cur[1].append(p_)
    if len(cur[0]) > 3:
        segs.append(cur)
    dt = np.concatenate([np.diff(s[0]) for s in segs])
    dp = np.concatenate([np.diff(s[1]) for s in segs])
    keep = np.abs(dt) < 60                     # skip RAM x wraps (in-level pipes)
    corr = float(np.corrcoef(dt[keep], dp[keep])[0, 1]) if keep.sum() > 4 else 0.0
    mae = float(np.abs(dt[keep] - dp[keep]).mean())
    truth_total = sum(s[0][-1] - s[0][0] for s in segs)
    pixel_total = sum(s[1][-1] - s[1][0] for s in segs)
    total_err = abs(truth_total - pixel_total)
    # THE gate that matters to the engine: REPRODUCIBILITY. Pixel progress is its own
    # coordinate system (a constant offset vs RAM is harmless), but the cache keys 16px
    # buckets on it — so the same world spot must read the same pixel progress every life.
    # Interpolate each life's pixel progress onto a common truth-x grid and measure the
    # spread across lives.
    grid_lo = max(min(s[0]) for s in segs) + 8
    grid_hi = min(max(s[0]) for s in segs) - 8
    spreads = []
    if grid_hi - grid_lo > 64 and len(segs) >= 3:
        grid = np.arange(grid_lo, grid_hi, 16)
        per_life = []
        for s in segs:
            t_arr, p_arr = np.array(s[0], dtype=float), np.array(s[1], dtype=float)
            order = np.argsort(t_arr)
            t_s, p_s = t_arr[order], p_arr[order]
            t_u, idx = np.unique(t_s, return_index=True)
            per_life.append(np.interp(grid, t_u, p_s[idx]))
        stack = np.vstack(per_life)
        spreads = stack.std(axis=0)
    repro_mean = float(np.mean(spreads)) if len(spreads) else float("inf")
    repro_p95 = float(np.percentile(spreads, 95)) if len(spreads) else float("inf")
    # Secondary: drift vs truth within a life (should be ~constant, i.e. flat).
    drift = np.concatenate([
        np.abs((np.array(s[1]) - s[1][0]) - (np.array(s[0]) - s[0][0])) for s in segs])
    drift_mae, drift_p95 = float(drift.mean()), float(np.percentile(drift, 95))
    og_t, og_p = np.array(og_truth), np.array(og_pixel)
    og_agree = float((og_t == og_p).mean()) if len(og_t) else 0.0
    # Error DIRECTION matters more than the rate: a false negative only misses a cache
    # lookup; a false positive lets the engine snapshot an AIRBORNE state (non-reproducible
    # replay — invariant violation).
    og_fp = float((og_p & ~og_t).sum() / max(1, (~og_t).sum()))   # airborne read as grounded
    og_fn = float((~og_p & og_t).sum() / max(1, og_t.sum()))      # grounded read as airborne

    n_obs = sum(len(s[0]) for s in segs)
    print("== sprint-hop 1-1 run ==")
    print(f"observes={n_obs} across {len(segs)} live(s)  "
          f"player tracked={100 * np.mean(tracked):.0f}%")
    print(f"progress: delta-corr={corr:.3f}  delta-MAE={mae:.2f}px  "
          f"total-travel err={total_err:.0f}px of {truth_total}px "
          f"({100 * total_err / max(1, truth_total):.1f}%)")
    print(f"progress drift vs truth within a life: mean={drift_mae:.1f}px  "
          f"p95={drift_p95:.1f}px")
    print(f"cross-life reproducibility (same spot, {len(segs)} lives): "
          f"spread mean={repro_mean:.1f}px  p95={repro_p95:.1f}px  (cache bucket = 16px)")
    print(f"on_ground agreement={100 * og_agree:.0f}% (n={len(og_t)})  "
          f"false-grounded={100 * og_fp:.0f}% of airborne  "
          f"missed-grounded={100 * og_fn:.0f}% of grounded")
    print(f"false deaths={false_deaths}")
    print(f"tracker cost: mean={1000 * np.mean(times):.2f}ms  "
          f"p95={1000 * np.percentile(times, 95):.2f}ms per observe")

    # --- scripted pit death: walk straight into 1-1's first pit -------------------------
    session.reset()
    tracker2 = PixelTracker()
    frame_no = 0
    # run right toward the first pit (~x=1550 area contains pits; simplest: sprint and jump
    # never — Mario walks into the first pit around x~1547)
    ram_dead_at = pixel_dead_at = None
    for i in range(0, 3000, CHUNK):
        session.send_plan([Step(CHUNK, C.mask(C.RIGHT, C.B))])
        frame_no += CHUNK
        st = session.read_state()
        scene = build_scene(st.ram, st.frame)
        view = tracker2.update(st.rgb, frame_no)
        if ram_dead_at is None and scene.is_dying:
            ram_dead_at = frame_no
        if pixel_dead_at is None and view.dead:
            pixel_dead_at = frame_no
        if ram_dead_at is not None and (pixel_dead_at is not None
                                        or frame_no - ram_dead_at > 240):
            break
    print("\n== scripted pit death ==")
    print(f"RAM dead at frame {ram_dead_at}; pixel dead at {pixel_dead_at} "
          f"(lag {None if pixel_dead_at is None or ram_dead_at is None else pixel_dead_at - ram_dead_at} frames)")
    ok = (repro_mean <= 8 and repro_p95 <= 16 and og_agree > 0.75 and false_deaths == 0
          and pixel_dead_at is not None and ram_dead_at is not None
          and pixel_dead_at - ram_dead_at <= 120)
    print(f"\nPARITY {'✅ PASS' if ok else '❌ NOT YET'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
