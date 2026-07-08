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


def make_env(state: str, goal_x: int, seed: int = 0, *, landing_waits: int = 0,
             randomize_frames: int = 36, max_steps: int = 220, start_x: int = 126,
             back_x: int = 80, level_label: str = "1-3",
             milestones: tuple[tuple[int, float], ...] = ((300, 20.0), (500, 50.0)),
             game=None):
    from billy.rl.section_env import SectionEnv

    def _init():
        env = SectionEnv(state, level_label=level_label, goal_x=goal_x,
                         landing_waits=landing_waits,
                         randomize_frames=randomize_frames, max_steps=max_steps,
                         start_x=start_x, back_x=back_x, milestones=milestones,
                         game=game)
        env.reset(seed=seed)
        return env
    return _init


def build_vec_env(n_envs: int, state: str, goal_x: int, **env_kw):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    fns = [make_env(state, goal_x, seed=i, **env_kw) for i in range(n_envs)]
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
    p.add_argument("--state", default=STATE,
                   help="savestate path, or comma-separated list for multi-phase training")
    p.add_argument("--goal-x", type=int, default=GOAL_X)
    p.add_argument("--out", default="data/rl/section_1_3")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    p.add_argument("--resume", default="")
    p.add_argument("--landing-waits", type=int, default=0,
                   help="wait actions on reset so airborne savestates land before training")
    p.add_argument("--randomize-frames", type=int, default=36,
                   help="random cruise on reset (0 = fixed entry, good for lift timing)")
    p.add_argument("--max-steps", type=int, default=220)
    p.add_argument("--start-x", type=int, default=126)
    p.add_argument("--back-x", type=int, default=80)
    p.add_argument("--demo", action="append", default=[],
                   help="teleop .demo.json to behavior-clone as a warm start (repeatable). "
                        "One good human crossing turns PPO-from-scratch into fine-tuning.")
    p.add_argument("--bc-epochs", type=int, default=200)
    p.add_argument("--game", default="smb", choices=["smb", "smb_lost"],
                   help="which ROM integration the savestate belongs to")
    p.add_argument("--level-label", default="",
                   help="level label for this section (default: 1-3, or 1-1 for smb_lost)")
    args = p.parse_args()

    from stable_baselines3 import PPO
    from run import GAMES

    game_cls = GAMES[args.game]
    level_label = args.level_label or ("1-1" if args.game == "smb_lost" else "1-3")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    milestones = ((300, 20.0), (500, 50.0))
    if "lift" in args.out or args.landing_waits:
        milestones = ((760, 25.0), (800, 45.0), (850, 70.0), (900, 100.0))
    elif args.game == "smb_lost":
        milestones = ((1050, 25.0), (1065, 50.0), (1080, 100.0))
    states = [s.strip() for s in args.state.split(",") if s.strip()]

    # BC warm-start pass 1: map demos to the action vocabulary and collect on-trajectory
    # (obs, action) pairs by replaying them in a DETERMINISTIC env (no start randomization).
    # Done before build_vec_env — stable-retro allows one emulator per process, so the
    # collection env must be closed before an in-process (n_envs=1) training env exists.
    bc_pairs = []
    if args.demo:
        from billy.rl.bc import collect_bc_pairs_from_plan, load_demo
        from billy.rl.section_env import SectionEnv

        env = SectionEnv(states[0], level_label=level_label, goal_x=args.goal_x,
                         landing_waits=args.landing_waits, randomize_frames=0,
                         max_steps=args.max_steps, start_x=args.start_x, back_x=args.back_x,
                         milestones=milestones, game=game_cls)
        for demo_path in args.demo:
            plan = load_demo(demo_path)
            pairs = collect_bc_pairs_from_plan(env, plan)
            bc_pairs.extend(pairs)
            print(f"[section] demo {demo_path}: {len(plan)} steps -> "
                  f"{len(pairs)} BC pairs")
        env.close()

    vec = build_vec_env(args.n_envs, states if len(states) > 1 else states[0], args.goal_x,
                        landing_waits=args.landing_waits,
                        randomize_frames=args.randomize_frames,
                        max_steps=args.max_steps, start_x=args.start_x, back_x=args.back_x,
                        level_label=level_label, milestones=milestones, game=game_cls)
    report = _Report()

    if args.resume:
        model = PPO.load(args.resume, env=vec, device=args.device)
        print(f"[section] resumed from {args.resume}")
    else:
        # Higher entropy than the whole-level run: the crossing bonus is sparse, so the policy
        # must EXPLORE the jump-timing before it ever sees +cross.
        model = PPO("MlpPolicy", vec, device=args.device, verbose=0, n_steps=512,
                    batch_size=256, gae_lambda=0.95, gamma=0.99, ent_coef=0.03)
    if bc_pairs:
        from billy.rl.bc import bc_pretrain
        loss = bc_pretrain(model, bc_pairs, epochs=args.bc_epochs)
        print(f"[section] BC warm-start on {len(bc_pairs)} pairs (final loss {loss:.3f}) — "
              f"PPO now fine-tunes a policy that already knows the crossing")
    model.learn(total_timesteps=args.timesteps, callback=report.cb, progress_bar=False)
    model.save(args.out)
    vec.close()
    print(f"[section] saved -> {args.out}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
