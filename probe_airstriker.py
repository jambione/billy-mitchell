#!/usr/bin/env python3
"""Parity probe: ShmupTracker (pixels only) vs the integration's info on real Airstriker frames.

Airstriker-Genesis ships FREE with stable-retro (no ROM to import) — the honest "play a game
Billy has no RAM map for" test. This drives the ship with a few varied strafe-and-fire
policies and checks what the learning loop needs from pixels:
  - player tracking:  % of in-play observes with the ship tracked
  - survival progress: monotonic frames-alive (the tape-search reward)
  - game over:        pixel-detected terminal vs info['gameover'] (lag, false positives)

info (gameover/lives/score) is GROUND TRUTH for grading only — the tracker never sees it.

    BILLY_HEADLESS=1 .venv/bin/python probe_airstriker.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("BILLY_HEADLESS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import stable_retro as retro

from billy.vision.shmup import ShmupTracker

CHUNK = 4


def _act(env, n, **kw):
    a = np.zeros(n, dtype=np.int8)
    for k in kw:
        a[env.buttons.index(k)] = 1
    return a


def run_life(env, n, tracker, policy: int, max_frames: int = 4000):
    """Play one attempt (boot -> game over) under a strafe/fire policy; return per-frame
    (tracked, survived, pixel_dead, truth_gameover) plus timing."""
    env.reset()
    z = np.zeros(n, dtype=np.int8)
    # Boot: a fixed START-then-settle dance drops us into clean gameplay. (The integration's
    # `gameover` field is a counter, not a flag, and `terminated` fires on a fixed timer — so
    # the boot must NOT gate on them; the adapter gates on pixels going live.)
    info = {"lives": 3}
    for i in range(120):
        _, _, _, _, info = env.step(_act(env, n, START=1) if i < 8 else z)
    tracker.rebase()
    start_lives = int(info.get("lives", 3))
    rows = []
    times = []
    frame_no = 0
    swing = 0
    pixel_dead_at = truth_dead_at = None
    while frame_no < max_frames:
        swing += 1
        # Realistic shmup policies — the agent ALWAYS fires (fire is free and always-on, and
        # it keeps the field populated so the area-collapse death signal has a stable
        # baseline); movement varies so tracking/death aren't validated on one trajectory.
        if policy == 0:
            a = _act(env, n, B=1)                                   # hold-fire
        elif policy == 1:
            a = _act(env, n, B=1, LEFT=1)                           # strafe to a corner
        elif policy == 2:
            d = "LEFT" if (swing // 8) % 2 else "RIGHT"
            a = _act(env, n, B=1, **{d: 1})                         # slow sweep
        else:
            d = "LEFT" if (swing // 4) % 2 else "RIGHT"
            a = _act(env, n, B=1, UP=1, **{d: 1})                   # fast dodge + advance
        info = {}
        for _ in range(CHUNK):
            _, _, term, trunc, info = env.step(a)
            frame_no += 1
        rgb = np.asarray(env.render())
        t0 = time.perf_counter()
        v = tracker.update(rgb, frame_no)
        times.append(time.perf_counter() - t0)
        # Death ground truth = the FIRST life lost (lives dropped below the start count).
        # (Not `terminated` — for this ROM it fires on a fixed timer, not on death.)
        truth_dead = int(info.get("lives", start_lives)) < start_lives
        rows.append((v.player is not None, v.progress, v.dead, truth_dead))
        if v.dead and pixel_dead_at is None:
            pixel_dead_at = frame_no
        if truth_dead and truth_dead_at is None:
            truth_dead_at = frame_no
        # stop once both agree it's over, or well past the truth death
        if truth_dead_at is not None and (pixel_dead_at is not None
                                          or frame_no - truth_dead_at > 120):
            break
    return rows, times, pixel_dead_at, truth_dead_at


def main() -> int:
    env = retro.make("Airstriker-Genesis-v0", render_mode="rgb_array",
                     inttype=retro.data.Integrations.STABLE)
    n = len(env.buttons)
    tracker = ShmupTracker()

    tracked_all = []
    times_all = []
    lags = []
    false_pos = 0
    mono_ok = True
    print("== Airstriker: ShmupTracker vs integration ground truth ==")
    for policy in range(4):
        rows, times, pdead, tdead = run_life(env, n, tracker, policy)
        inplay = [r for r in rows if not r[3]]
        tracked = np.mean([r[0] for r in inplay]) if inplay else 0.0
        tracked_all.append(tracked)
        times_all += times
        # survival must be monotonic non-decreasing while in play
        surv = [r[1] for r in rows]
        mono_ok = mono_ok and all(b >= a for a, b in zip(surv, surv[1:]))
        # false positive: pixel death fired while truth still in play
        fp = sum(1 for r in rows if r[2] and not r[3])
        false_pos += fp
        lag = (pdead - tdead) if (pdead is not None and tdead is not None) else None
        lags.append(lag)
        print(f"  policy {policy}: tracked={100*tracked:.0f}%  survived_max={max(surv)}  "
              f"truth_over@{tdead}  pixel_dead@{pdead}  lag={lag}  false_pos={fp}")

    env.close()
    tracked_mean = float(np.mean(tracked_all))
    good_lags = [l for l in lags if l is not None]
    clean = [l for l in good_lags if -30 <= l <= 120]
    print(f"\nplayer tracked mean={100*tracked_mean:.0f}%   survival monotonic={mono_ok}")
    print(f"tracker cost: mean={1000*np.mean(times_all):.2f}ms/observe")
    # THE Phase-1 gate: the reusable perception. Ship tracking is what a per-game adapter
    # would otherwise have to hand-engineer; getting it from pixels on a new console/genre is
    # the generality result. (Survival progress is RAM-free and monotonic.)
    perception_ok = tracked_mean > 0.95 and mono_ok and np.mean(times_all) < 2e-3
    print(f"\nPERCEPTION {'PASS' if perception_ok else 'NOT YET'} "
          f"(ship-tracking + survival — the reusable, generalizable layer)")
    # Diagnostic (not a gate): pixel death-from-collapse is clean under a stable-firing
    # policy but confounded by the player's own bullets when they camp/idle, so it's
    # best-effort. A robust terminal likely wants the integration's lives/score.
    print(f"death-from-pixels: clean on {len(clean)}/4 policies  lags={lags}  "
          f"false_pos={false_pos}  (best-effort; fragile under bullet-area swings)")
    return 0 if perception_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
