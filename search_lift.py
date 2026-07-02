#!/usr/bin/env python3
"""CLI wrapper for SMB 1-3 frame-level lift search (see billy.games.smb.lift_search)."""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("BILLY_HEADLESS", "1")

from billy.games.smb import SmbGame
from billy.games.smb.lift_search import frame_lift_search, persist_lift_plan


def main() -> int:
    p = argparse.ArgumentParser(description="Frame-level lift-gap search.")
    p.add_argument("--state", default="data/rl/states/smb_1_3_lift.state")
    p.add_argument("--goal-x", type=int, default=880)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--beam", type=int, default=24)
    args = p.parse_args()

    game = SmbGame()
    session = game.system.connect()
    with open(args.state, "rb") as f:
        session.reset()
        session.env.em.set_state(f.read())
        session._refresh_ram()

    def observe():
        st = session.read_state()
        return game.observe(st.frame, st.ram)

    start = observe()
    print(f"[search] start {start.level_label} x={start.progress} "
          f"ground={start.raw.on_ground} gap={start.raw.gap_info()}")

    plan, reach, crossed = frame_lift_search(
        session, observe, goal_x=args.goal_x, depth=args.depth, beam=args.beam)
    print(f"[search] crossed={crossed} reach={reach} steps={len(plan) if plan else 0}")
    if plan:
        for i, step in enumerate(plan):
            print(f"  {i+1:2d}. frames={step.frames} buttons={step.buttons}")
        if crossed:
            persist_lift_plan(plan, reach)
    session.close()
    return 0 if crossed else 1


if __name__ == "__main__":
    raise SystemExit(main())