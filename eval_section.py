#!/usr/bin/env python3
"""Evaluate a trained section sub-policy: run it deterministically from the savestate N times and
report cross-rate + reach distribution. This is the standalone gate before wiring into Billy.

    .venv/bin/python eval_section.py --model data/rl/section_1_3 --episodes 30
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("BILLY_HEADLESS", "1")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/rl/section_1_3")
    p.add_argument("--state", default="data/rl/states/smb_1_3_section.state")
    p.add_argument("--goal-x", type=int, default=700)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--stochastic", action="store_true", help="sample actions (default: deterministic)")
    args = p.parse_args()

    from stable_baselines3 import PPO

    from billy.rl.section_env import SECTION_ACTIONS, SectionEnv

    env = SectionEnv(args.state, goal_x=args.goal_x)
    model = PPO.load(args.model, device="cpu")
    crosses, reaches = 0, []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, r, term, trunc, info = env.step(int(action))
            done = term or trunc
        crosses += int(info["crossed"])
        reaches.append(info["best_x"])
        tag = "CROSS" if info["crossed"] else ("dead" if info["dead"] else "stall")
        print(f"  ep {ep + 1:2d}: best_x={info['best_x']:4d} {tag}")
    env.close()
    reaches.sort()
    print(f"\ncross-rate {crosses}/{args.episodes} = {crosses / args.episodes:.0%} | "
          f"median reach {reaches[len(reaches) // 2]} | min {reaches[0]} max {reaches[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
