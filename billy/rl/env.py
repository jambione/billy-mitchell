"""Gymnasium environment over the existing in-process emulator + RAM perception.

This reuses the *same* Game/Session/perception the rest of Billy uses (no separate sim): the env
boots the game through `Game.boot`, steps it with the engine's `Step`/Session contract, and turns
each `Observation`'s `Scene` into the feature vector from `features.py`. So a policy trained here
sees exactly what the deployed `LearnedReflex` will see at inference time.

One emulator per process (a stable-retro constraint), so vectorized training must use SubprocVecEnv
(separate processes) — see train_rl.py.
"""
from __future__ import annotations

import os
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..abstractions import Step
from ..games.smb import SmbGame
from . import features


class BillyMarioEnv(gym.Env):
    """A single-level SMB controller environment. Episode ends on death, level clear, or step cap."""

    metadata = {"render_modes": []}

    def __init__(self, game_factory: Callable[[], object] = SmbGame, frame_skip: int = 4,
                 max_steps: int = 2000, progress_w: float = 0.1, score_w: float = 0.01,
                 death_penalty: float = 15.0, clear_bonus: float = 50.0, time_penalty: float = 0.01):
        super().__init__()
        os.environ.setdefault("BILLY_HEADLESS", "1")   # training is always headless
        self.game = game_factory()
        self.session = self.game.system.connect()
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.w = dict(progress=progress_w, score=score_w, death=death_penalty,
                      clear=clear_bonus, time=time_penalty)

        self.action_space = spaces.Discrete(features.N_ACTIONS)
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(features.OBS_DIM,), dtype=np.float32)
        self._steps = 0
        self._prev_x = 0
        self._prev_score = 0
        self._start_level: tuple = ()

    # --- gym API ------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.game.boot(self.session)   # boots to the level start (1-1), checkpoints slot 0
        self._steps = 0
        self._prev_x = obs.progress
        self._prev_score = obs.score
        self._start_level = obs.level_key
        return features.featurize(obs.raw), {"x": obs.progress, "level": obs.level_label}

    def step(self, action: int):
        mask = features.ACTION_MASKS[int(action)]
        self.session.send_plan([Step(self.frame_skip, mask)])
        st = self.session.read_state()
        obs = self.game.observe(st.frame, st.ram)
        self._steps += 1

        cleared = obs.level_key > self._start_level
        terminated = bool(obs.dead or cleared)
        truncated = self._steps >= self.max_steps

        dx = obs.progress - self._prev_x
        reward = (dx * self.w["progress"]
                  + (obs.score - self._prev_score) * self.w["score"]
                  - self.w["time"])
        if obs.dead:
            reward -= self.w["death"]
        if cleared:
            reward += self.w["clear"]
        self._prev_x, self._prev_score = obs.progress, obs.score

        info = {"x": obs.progress, "level": obs.level_label, "cleared": cleared, "dead": obs.dead}
        return features.featurize(obs.raw), float(reward), terminated, truncated, info

    def close(self):
        close = getattr(self.session, "close", None)
        if close:
            close()
