#!/usr/bin/env python3
"""Train a PPO controller for Billy against the in-process SMB env.

    .venv/bin/python train_rl.py --timesteps 200000 --n-envs 4 --out data/rl/ppo_smb
    .venv/bin/python train_rl.py --timesteps 50000  --imitate 4000   # BC warm-start first

Vectorized training uses SubprocVecEnv (one emulator per process — a stable-retro constraint). The
optional `--imitate` step warm-starts the policy by behavior-cloning the hand-crafted reflex, so PPO
doesn't start from random flailing. The saved `.zip` is what `--rl` / LearnedReflex loads at runtime.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("BILLY_HEADLESS", "1")   # never open a window during training


def make_env(seed: int = 0):
    """Factory for one env (must be importable for SubprocVecEnv's spawn)."""
    from billy.rl.env import BillyMarioEnv

    def _init():
        env = BillyMarioEnv()
        env.reset(seed=seed)
        return env
    return _init


def _has_tensorboard() -> bool:
    import importlib.util
    return importlib.util.find_spec("tensorboard") is not None


def build_vec_env(n_envs: int):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    fns = [make_env(seed=i) for i in range(n_envs)]
    vec = DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns, start_method="spawn")
    return VecMonitor(vec)   # episode reward/length stats for logging


def behavior_clone(model, steps: int, device: str) -> None:
    """Warm-start the policy by imitating the hand-crafted reflex (cross-entropy on its actions)."""
    import numpy as np
    import torch

    from billy.games.smb import SmbGame
    from billy.rl import features
    from billy.rl.env import BillyMarioEnv

    # nearest-action lookup: map a reflex button-mask to the closest discrete action index
    def mask_to_action(mask: int) -> int:
        best, best_overlap = 0, -1
        for idx, m in enumerate(features.ACTION_MASKS):
            overlap = bin(m & mask).count("1") - bin(m & ~mask).count("1")
            if overlap > best_overlap:
                best, best_overlap = idx, overlap
        return best

    env = BillyMarioEnv()
    reflex = SmbGame().make_reflex()
    obs, _ = env.reset()
    reflex.reset(type("O", (), {"progress": 40})())
    X, Y = [], []
    for _ in range(steps):
        st = env.session.read_state()
        scene = env.game.observe(st.frame, st.ram).raw
        decision = reflex.step(type("O", (), {"raw": scene, "progress": scene.mario_x})())
        plan = decision.plan or [type("S", (), {"buttons": 0})()]
        action = mask_to_action(plan[0].buttons)
        X.append(features.featurize(scene)); Y.append(action)
        obs, r, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
    env.close()

    Xt = torch.as_tensor(np.array(X), dtype=torch.float32, device=device)
    Yt = torch.as_tensor(np.array(Y), dtype=torch.long, device=device)
    opt = torch.optim.Adam(model.policy.parameters(), lr=3e-4)
    print(f"[bc] behavior-cloning on {len(X)} demos…")
    for epoch in range(8):
        perm = torch.randperm(len(Xt), device=device)
        total = 0.0
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            dist = model.policy.get_distribution(Xt[b])
            loss = -dist.log_prob(Yt[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss)
        print(f"[bc] epoch {epoch + 1}/8 loss {total:.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description="Train Billy's PPO controller.")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--out", default="data/rl/ppo_smb")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    p.add_argument("--imitate", type=int, default=0, help="BC warm-start steps from the reflex (0=off)")
    p.add_argument("--resume", default="", help="continue training from this .zip")
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vec = build_vec_env(args.n_envs)

    tb_log = "data/rl/tb" if _has_tensorboard() else None   # tensorboard is optional
    if args.resume:
        model = PPO.load(args.resume, env=vec, device=args.device)
        print(f"[rl] resumed from {args.resume}")
    else:
        model = PPO("MlpPolicy", vec, device=args.device, verbose=1, n_steps=512,
                    batch_size=256, gae_lambda=0.95, gamma=0.99, ent_coef=0.01,
                    tensorboard_log=tb_log)
        if args.imitate:
            behavior_clone(model, args.imitate, args.device)

    ckpt = CheckpointCallback(save_freq=max(1, 50_000 // args.n_envs),
                              save_path="data/rl/ckpt", name_prefix="ppo_smb")
    model.learn(total_timesteps=args.timesteps, callback=ckpt, progress_bar=False)
    model.save(args.out)
    vec.close()
    print(f"[rl] saved policy -> {args.out}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
