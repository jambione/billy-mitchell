#!/usr/bin/env python3
"""Probe SMB 1-3 lift geometry from a savestate: log Mario + lift object positions over time."""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("BILLY_HEADLESS", "1")

LIFT_ID = 0x25


def _objects(ram: bytes) -> list[dict]:
    from billy.games.smb.perception import _u8
    out = []
    for slot in range(6):
        if _u8(ram, 0x0F + slot) == 0:
            continue
        eid = _u8(ram, 0x16 + slot)
        ex = _u8(ram, 0x6E + slot) * 256 + _u8(ram, 0x87 + slot)
        ey = _u8(ram, 0xCF + slot) + 24
        out.append({"slot": slot, "id": eid, "x": ex, "y": ey})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Probe lift trajectory in SMB 1-3.")
    p.add_argument("--state", default="data/rl/states/smb_1_3_lift.state")
    p.add_argument("--frames", type=int, default=360, help="idle frames to sample")
    p.add_argument("--action", choices=["idle", "right", "right_b"], default="idle")
    args = p.parse_args()

    from billy.abstractions import Step
    from billy.games.smb import SmbGame
    from billy.systems.nes import controller as C

    game = SmbGame()
    session = game.system.connect()
    with open(args.state, "rb") as f:
        session.reset()
        session.env.em.set_state(f.read())
        session._refresh_ram()

    def observe():
        st = session.read_state()
        return game.observe(st.frame, st.ram)

    masks = {"idle": 0, "right": C.RIGHT, "right_b": C.mask(C.RIGHT, C.B)}
    step = Step(4, masks[args.action])

    obs = observe()
    print(f"[probe] start {obs.level_label} x={obs.progress} y={obs.raw.mario_y} "
          f"ground={obs.raw.on_ground}")
    lifts = []
    for i in range(args.frames // step.frames):
        session.send_plan([step])
        obs = observe()
        objs = _objects(obs.raw.ram)
        lift = next((o for o in objs if o["id"] == LIFT_ID), None)
        if lift:
            lifts.append((obs.progress, lift["x"], lift["y"]))
        if i % 15 == 0 or lift:
            gap = scene_gap(obs)
            print(f"  f={i*step.frames:3d} mario=({obs.progress},{obs.raw.mario_y}) "
                  f"ground={obs.raw.on_ground} lift={lift} gap={gap} dead={obs.dead}")
        if obs.dead:
            print("[probe] Mario died")
            break

    if lifts:
        xs = [l[1] for l in lifts]
        ys = [l[2] for l in lifts]
        print(f"[probe] lift x range {min(xs)}-{max(xs)}, y range {min(ys)}-{max(ys)}, "
              f"samples={len(lifts)}")
        static = max(xs) - min(xs) < 8 and max(ys) - min(ys) < 8
        print(f"[probe] lift appears {'STATIC' if static else 'MOVING'} over sample window")
    else:
        print("[probe] no lift object (0x25) seen — check savestate position")
    session.close()
    return 0


def scene_gap(obs) -> str:
    g = obs.raw.gap_info()
    return f"{g[0]}px,w{g[1]}" if g else "none"


if __name__ == "__main__":
    raise SystemExit(main())