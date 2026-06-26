#!/usr/bin/env python3
"""Train a HAZARD-SCOPED PPO sub-policy to cross ONE hard section (default: 1-3's tree-top
platform/lift chain), resetting from a savestate at the section entrance.

    .venv/bin/python train_section.py --timesteps 400000 --n-envs 8 \
        --state data/rl/states/smb_1_3_section.state --out data/rl/section_1_3

Why scoped (vs the whole-level PPO that failed twice): every episode is on-task at the hazard, so
it's sample-efficient; the env's death-dominant reward + sustained-jump actions fix the v1 (no
sustained jumps) and v2 (rush-and-die) failure modes. The saved .zip is loaded at runtime by the
section controller, invoked only when Billy is at this hazard.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("BILLY_HEADLESS", "1")

STATE = "data/rl/states/smb_1_3_section.state"
GOAL_X = 700


def make_env(state: str, goal_x: int, seed: int = 0):
    from billy.rl.section_env import SectionEnv

    def _init():
        env = SectionEnv(state, goal_x=goal_x)
        env.reset(seed=seed)
        return env
    return _init


def build_vec_env(n_envs: int, state: str, goal_x: int):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    fns = [make_env(state, goal_x, seed=i) for i in range(n_envs)]
    vec = DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns, start_method="spawn")
    return VecMonitor(vec)


class _Report:
    """Tiny callback: print rolling cross-rate so we can SEE if the policy is learning to cross."""

    def __init__(self):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _CB(BaseCallback):
            def _on_step(self) -> bool:
                for info in self.locals.get("infos", []):
                    if info.get("crossed") or info.get("dead"):
                        outer.results.append(1 if info.get("crossed") else 0)
                        outer.results[:] = outer.results[-200:]
                        outer.reach.append(info.get("best_x", 0))
                        outer.reach[:] = outer.reach[-200:]
                if self.num_timesteps - outer.last >= 20000 and outer.results:
                    rate = sum(outer.results) / len(outer.results)
                    avg_reach = sum(outer.reach) / max(1, len(outer.reach))
                    mx = max(outer.reach) if outer.reach else 0
                    print(f"[section] t={self.num_timesteps} cross_rate≈{rate:.0%} "
                          f"avg_reach≈{avg_reach:.0f} max_reach={mx} "
                          f"(last {len(outer.results)} terms)", flush=True)
                    outer.last = self.num_timesteps
                return True

        self.results: list[int] = []
        self.reach: list[int] = []
        self.last = 0
        self.cb = _CB()


def main() -> int:
    p = argparse.ArgumentParser(description="Train a hazard-scoped section-crossing sub-policy.")
    p.add_argument("--timesteps", type=int, default=400_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--state", default=STATE)
    p.add_argument("--goal-x", type=int, default=GOAL_X)
    p.add_argument("--out", default="data/rl/section_1_3")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    p.add_argument("--resume", default="")
    args = p.parse_args()

    from stable_baselines3 import PPO

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vec = build_vec_env(args.n_envs, args.state, args.goal_x)
    report = _Report()

    if args.resume:
        model = PPO.load(args.resume, env=vec, device=args.device)
        print(f"[section] resumed from {args.resume}")
    else:
        # Higher entropy than the whole-level run: the crossing bonus is sparse, so the policy
        # must EXPLORE the jump-timing before it ever sees +cross.
        model = PPO("MlpPolicy", vec, device=args.device, verbose=0, n_steps=512,
                    batch_size=256, gae_lambda=0.95, gamma=0.99, ent_coef=0.03)
    model.learn(total_timesteps=args.timesteps, callback=report.cb, progress_bar=False)
    model.save(args.out)
    vec.close()
    print(f"[section] saved -> {args.out}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
