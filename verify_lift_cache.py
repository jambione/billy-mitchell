#!/usr/bin/env python3
"""Verify a cached 1-3 lift crossing and capture a settled pit-edge savestate."""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("BILLY_HEADLESS", "1")

from billy.abstractions import Step
from billy.games.smb import SmbGame
from billy.games.smb.capture_util import settle_mario
from billy.knowledge.cache import SolutionCache, bucket_of
from billy.systems.nes import controller as C

LEVEL_1_3 = (0, 2, 3)


def _load_plan(cache: SolutionCache, x: int, y: int = 0):
    entry = cache.get(LEVEL_1_3, x, y)
    if entry is None:
        return None, None
    return entry.plan, entry.reach_after


def main() -> int:
    p = argparse.ArgumentParser(description="Verify cached lift plan on a clone.")
    p.add_argument("--x", type=int, default=496, help="progress bucket anchor to look up")
    p.add_argument("--y", type=int, default=0)
    p.add_argument("--savestate-out", default="data/rl/states/smb_1_3_lift.state")
    p.add_argument("--from-state", default="", help="optional: restore this state before replay")
    args = p.parse_args()

    cache = SolutionCache()
    plan, reach = _load_plan(cache, args.x, args.y)
    if not plan:
        # list best 1-3 entries
        rows = [(e.reach_after, k, e.plan) for k, e in cache.entries.items() if k[0] == LEVEL_1_3]
        rows.sort(reverse=True)
        print("[verify] no entry at x=%s; top 1-3 cache entries:" % args.x)
        for r, k, pl in rows[:8]:
            print(f"  bucket={k[1]} yband={k[2]} reach={r} plan={pl}")
        if not rows:
            return 1
        plan, reach = rows[0][2], rows[0][0]
        print(f"[verify] using best: reach={reach}")

    game = SmbGame()
    session = game.system.connect()
    if args.from_state:
        with open(args.from_state, "rb") as f:
            session.reset()
            session.env.em.set_state(f.read())
            session._refresh_ram()
    else:
        session.reset()

    def observe():
        st = session.read_state()
        return game.observe(st.frame, st.ram)

    start = observe()
    print(f"[verify] start {start.level_label} x={start.progress} ground={start.raw.on_ground}")

    session.send_plan(plan)
    after = observe()
    for _ in range(60):
        if after.dead or after.progress >= 900:
            break
        session.send_plan([Step(4, C.mask(C.RIGHT, C.B))])
        after = observe()

    print(f"[verify] after plan x={after.progress} dead={after.dead} "
          f"(cached reach={reach})")
    if after.dead or after.progress < 850:
        session.close()
        return 1

    ok, _ = settle_mario(session, observe)
    settled = observe()
    print(f"[verify] settled x={settled.progress} vx={settled.raw.x_speed} "
          f"ground={settled.raw.on_ground}")
    if ok and settled.raw.on_ground:
        os.makedirs(os.path.dirname(args.savestate_out) or ".", exist_ok=True)
        with open(args.savestate_out, "wb") as f:
            f.write(session.clone_state())
        print(f"[verify] saved settled savestate -> {args.savestate_out}")

    # export composed plan for section bootstrap
    out_plan = "data/rl/lift_cached.plan.json"
    os.makedirs(os.path.dirname(out_plan) or ".", exist_ok=True)
    with open(out_plan, "w") as f:
        json.dump({"plan": [[s.frames, s.buttons] for s in plan],
                   "reach_after": after.progress, "start_x": start.progress}, f)
    print(f"[verify] wrote {out_plan}")
    session.close()
    return 0 if after.progress >= 850 and not after.dead else 1


if __name__ == "__main__":
    raise SystemExit(main())