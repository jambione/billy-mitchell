"""Behavior-cloning warm-start: turn ONE human demo into a section sub-policy's prior.

A teleop demo is an exact per-frame button stream — the richest possible supervision for a
timing hazard (the moving lift, knockback fights). PPO from scratch needs ~10^5-10^6 steps to
stumble onto a crossing; cloning the demo first initializes the policy ON the crossing, so PPO
only has to make it robust (~10^4 steps). This is the lever that turns "a little demo help"
into a trained, verified, bankable sub-policy.

Pipeline: demo Plan -> map to the section action vocabulary (`demo_to_actions`, pure/testable)
-> replay mapped actions in the SectionEnv collecting (features, action) pairs on-trajectory
(`collect_bc_pairs`) -> supervised pretrain of the PPO policy head (`bc_pretrain`). The result
still flows through the same safety machinery: the sub-policy only SEEDS micro-search on a
clone; nothing replays blind.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..abstractions import Plan, Step
from ..systems.nes import controller as C
from .section_env import SECTION_ACTIONS

# Buttons that carry gameplay meaning for similarity scoring (START/SELECT excluded).
_RELEVANT = C.mask_from_names(["right", "left", "up", "down", "A", "B"])


def load_demo(path: str | Path) -> Plan:
    """Read a `.demo.json` written by teleop.py (`{"steps": [[frames, buttons], ...]}`)."""
    raw = json.loads(Path(path).read_text())
    return [Step(f, b) for f, b in raw["steps"]]


def _similarity(demo_mask: int, action_mask: int) -> float:
    """Per-frame agreement on the relevant button bits (1.0 = identical input)."""
    diff = (demo_mask ^ action_mask) & _RELEVANT
    return 1.0 - bin(diff).count("1") / 6.0


def demo_to_actions(plan: Plan, actions=SECTION_ACTIONS) -> list[int]:
    """Greedily map an exact demo input stream onto the discrete section-action vocabulary.

    At each point, every candidate action is scored by average per-frame button agreement over
    its hold window; the best-scoring action is emitted and its window consumed. Longer holds
    win ties (fewer, more decisive actions — matching how the policy acts at inference). The
    mapping is lossy by design: BC only needs the *gist* of the demo; PPO fine-tuning restores
    the precision against the actual environment.
    """
    frames: list[int] = []
    for s in plan:
        frames.extend([s.buttons] * s.frames)

    vocab = [(C.mask_from_names(list(names)), hold, k)
             for k, (names, hold) in enumerate(actions)]
    idxs: list[int] = []
    i = 0
    while i < len(frames):
        best_k, best_hold, best_score = 0, vocab[0][1], -1.0
        for mask, hold, k in vocab:
            window = frames[i:i + hold]
            score = sum(_similarity(m, mask) for m in window) / hold
            score += hold * 1e-4          # tiebreak: prefer the longer, more decisive action
            if score > best_score:
                best_score, best_k, best_hold = score, k, hold
        idxs.append(best_k)
        i += best_hold
    return idxs


def collect_bc_pairs(env, action_idxs: list[int]) -> list[tuple]:
    """Replay the mapped actions once in a (deterministic) SectionEnv, recording the
    (feature_vector, action) pair at each decision point — on-trajectory supervision.
    Stops early on death/cross so BC never learns from a post-mortem tail."""
    pairs: list[tuple] = []
    obs, _info = env.reset()
    for a in action_idxs:
        pairs.append((obs, int(a)))
        obs, _r, terminated, truncated, info = env.step(int(a))
        if terminated or truncated:
            if info.get("dead"):
                pairs = pairs[:-4] if len(pairs) > 4 else []   # drop the doomed final inputs
            break
    return pairs


def bc_pretrain(model, pairs: list[tuple], *, epochs: int = 200, lr: float = 3e-4) -> float:
    """Supervised pretrain of an SB3 PPO policy on (obs, action) pairs. Returns final loss."""
    import numpy as np
    import torch

    if not pairs:
        return 0.0
    policy = model.policy
    obs = torch.as_tensor(np.stack([o for o, _ in pairs]),
                          dtype=torch.float32, device=policy.device)
    acts = torch.as_tensor([a for _, a in pairs], device=policy.device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss = torch.tensor(0.0)
    policy.set_training_mode(True)
    for _ in range(epochs):
        _values, log_prob, _entropy = policy.evaluate_actions(obs, acts)
        loss = -log_prob.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    policy.set_training_mode(False)
    return float(loss)
