"""A HAZARD-SCOPED Gymnasium env: learn to cross ONE hard section, not a whole level.

Motivation (see the RL notes): training a whole-level policy failed twice — v1 (4-frame skip)
never learned sustained jumps and was slow; v2 (macro-jumps) learned a profitable RUSH-AND-DIE
because the death penalty was tiny next to the progress reward banked first. The hand-crafted
reflex already clears 1-1/1-2; RL is only worth it at the spots the geometry reflex can't chain.

So this env is deliberately narrow:
  • it RESETS from a savestate captured at the section entrance (no replaying earlier levels —
    every episode is on-task, so training is sample-efficient);
  • the action set uses SUSTAINED jumps (A held for a full arc) so the policy can actually land on
    the small tree-top platforms 1-3 demands;
  • the reward is DEATH-DOMINANT — death (-DEATH) dwarfs the most progress an episode can bank, so
    surviving-and-crossing strictly beats rushing into the pit;
  • the episode ENDS the moment the section is crossed (mario_x >= goal_x) with a large bonus.

It reuses the same RAM perception + feature vector as the rest of Billy (features.featurize), so a
policy trained here drops into the deployed controller seeing exactly what it trained on. Parameterized
by (state, level_label, start_x, goal_x) so the same machinery scopes any future hazard section.
"""
from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..abstractions import Step
from ..games.smb import SmbGame
from ..systems.nes import controller as C
from . import features

# Focused platforming vocabulary. Each entry is (button-names, hold_frames): jump actions HOLD A
# for a full arc so the policy can clear wide pits and land on small platforms; ground moves use a
# short skip so steering stays responsive.
SECTION_ACTIONS: list[tuple[tuple[str, ...], int]] = [
    (("right", "B"), 4),            # run right (cruise)
    (("right", "A", "B"), 26),      # run-jump right, full arc (the section-crossing move)
    (("right", "A", "B"), 14),      # shorter run-jump (short platform gaps)
    (("A",), 22),                   # jump straight up (vertical adjust)
    (("right",), 4),                # walk right (precise approach)
    (("left", "B"), 4),             # back up to re-line a jump
    ((), 4),                        # wait a beat (let a moving platform come)
]
N_SECTION_ACTIONS = len(SECTION_ACTIONS)


class SectionEnv(gym.Env):
    """Cross a single hard section, resetting from a savestate at its entrance."""

    metadata = {"render_modes": []}

    def __init__(self, state_path: str, level_label: str = "1-3", start_x: int = 126,
                 goal_x: int = 700, max_steps: int = 220,
                 progress_w: float = 0.1, death_penalty: float = 40.0,
                 cross_bonus: float = 150.0, time_penalty: float = 0.01,
                 back_x: int = 80, randomize_frames: int = 36,
                 milestones: tuple[tuple[int, float], ...] = ((300, 20.0), (500, 50.0))) -> None:
        super().__init__()
        os.environ.setdefault("BILLY_HEADLESS", "1")
        self.state_path = state_path
        self.level_label = level_label
        self.start_x, self.goal_x, self.back_x = start_x, goal_x, back_x
        self.max_steps = max_steps
        # On reset, advance a random few frames of cruise so the policy sees a SPREAD of entry
        # positions/velocities/enemy-phases (not one fixed savestate) — that's what makes it robust
        # to wherever Billy actually arrives at the hazard live, instead of overfitting one trajectory.
        self.randomize_frames = randomize_frames
        # Latched checkpoint bonuses (x_threshold -> one-time reward) densify the path to the cross
        # so the policy isn't crushed into the "never jump -> never die" idle optimum.
        self.milestones = tuple(sorted(milestones))
        self.w = dict(progress=progress_w, death=death_penalty,
                      cross=cross_bonus, time=time_penalty)

        self.game = SmbGame()
        self.session = self.game.system.connect()
        with open(state_path, "rb") as f:
            self._snapshot = f.read()

        self.action_space = spaces.Discrete(N_SECTION_ACTIONS)
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(features.OBS_DIM,), dtype=np.float32)
        self._steps = 0
        self._best_x = start_x

    # --- helpers ------------------------------------------------------------------------
    def _observe(self):
        st = self.session.read_state()
        return self.game.observe(st.frame, st.ram)

    def _restore(self):
        self.session.reset()
        self.session.env.em.set_state(self._snapshot)
        self.session._refresh_ram()

    # --- gym API ------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._restore()
        if self.randomize_frames:
            # Cruise a random spell to diversify the entry state (stay on the start ground).
            n = int(self.np_random.integers(0, self.randomize_frames + 1))
            if n:
                self.session.send_plan([Step(n, C.mask_from_names(["right", "B"]))])
        obs = self._observe()
        self._steps = 0
        self._best_x = obs.progress
        self._hit = set()                       # milestone thresholds already paid this episode
        return features.featurize(obs.raw), {"x": obs.progress, "level": obs.level_label}

    def step(self, action: int):
        names, hold = SECTION_ACTIONS[int(action)]
        mask = C.mask_from_names(list(names))
        self.session.send_plan([Step(hold, mask)])
        obs = self._observe()
        self._steps += 1

        x = obs.progress
        dead = obs.dead or x < self.back_x          # fell behind the section entrance = failed run
        crossed = x >= self.goal_x

        # Progress only on NEW ground (no reward for inching back and forth at the pit edge).
        gained = max(0, x - self._best_x)
        self._best_x = max(self._best_x, x)
        reward = gained * self.w["progress"] - self.w["time"]
        # Latched checkpoint bonuses: reward reaching each milestone once, so partial crossings
        # pull the policy forward instead of it freezing to avoid the death penalty entirely.
        for thr, bonus in self.milestones:
            if x >= thr and thr not in self._hit:
                self._hit.add(thr)
                reward += bonus
        if dead:
            reward -= self.w["death"]
        if crossed:
            reward += self.w["cross"]

        terminated = bool(dead or crossed)
        truncated = self._steps >= self.max_steps
        info = {"x": x, "crossed": crossed, "dead": dead, "best_x": self._best_x}
        return features.featurize(obs.raw), float(reward), terminated, truncated, info

    def close(self):
        close = getattr(self.session, "close", None)
        if close:
            close()
