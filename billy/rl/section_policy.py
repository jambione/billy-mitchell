"""Hazard-scoped RL sub-policies, wired in the architecture-faithful way: they SEED micro-search.

A whole-level RL policy lost to the hand-crafted reflex twice (slow / rush-and-die). But a NARROW
sub-policy trained only to cross one section the geometry reflex can't chain (1-3's tree-top
platform/lift hops) is a different bet — and the safe way to use it is NOT to let it drive live play
(that would forgo Billy's verified-exact-replay guarantees). Instead, at a registered hazard the
controller rolls the policy out CLOSED-LOOP on a CLONE of the live state (invisible, like search),
records the exact button sequence it produced, and offers THAT as one more micro-search candidate.

So the RL contributes the crossing SKILL; the existing machinery still owns correctness:
  • _micro_search verifies the recorded sequence survives AND advances before committing it;
  • the cache banks the (deterministic) button steps, so the crossing COMPOUNDS like any solution —
    next pass replays it verbatim, no RL inference on the hot loop.
Registered per (level_label, x-range), so it can only ever fire at its hazard — 1-1/1-2 are untouched.
Degrades to a no-op if torch/SB3 or the model file is missing (reflex-only build keeps working).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_DEBUG = os.environ.get("BILLY_DEBUG_SECTION", "0") == "1"

from ..abstractions import Observation, Plan, Step
from ..systems.nes import controller as C
from . import features
from .section_env import SECTION_ACTIONS


@dataclass
class Section:
    label: str           # level_label this section belongs to, e.g. "1-3"
    x_lo: int            # trigger only while Billy is on-ground within [x_lo, x_hi]
    x_hi: int
    goal_x: int          # stop the rollout once across (matches the training goal)
    model_path: str
    max_steps: int = 64  # cap the closed-loop rollout (each step = one held action)


class SectionController:
    """Registry of section sub-policies that seed micro-search with a learned crossing candidate."""

    def __init__(self, sections: list[Section]) -> None:
        self.sections: list[tuple[Section, object]] = []
        for sec in sections:
            model = self._load(sec.model_path)
            if model is not None:
                self.sections.append((sec, model))

    @staticmethod
    def _load(path: str):
        try:
            from stable_baselines3 import PPO
            model = PPO.load(path, device="cpu")
            print(f"[section] loaded hazard sub-policy: {path}")
            return model
        except Exception as e:   # missing deps / file / version skew -> no-op
            print(f"[section] no sub-policy ({type(e).__name__}: {e})")
            return None

    def __len__(self) -> int:
        return len(self.sections)

    def _match(self, obs: Observation):
        on_ground = getattr(obs.raw, "on_ground", True)
        if not on_ground:
            return None
        for sec, model in self.sections:
            if obs.level_label == sec.label and sec.x_lo <= obs.progress <= sec.x_hi:
                return sec, model
        return None

    _MIN_GAIN = 40   # only commit a crossing that makes real forward progress

    def cross(self, obs: Observation, session, observe):
        """If Billy is at a registered hazard, roll the sub-policy out CLOSED-LOOP on a clone and,
        if it survives and makes real forward progress, return (plan, reach). This rollout IS the
        verification — the recorded button sequence is deterministic, so committing+banking it is a
        proven crossing. Returns None if not at a hazard, or the policy didn't get through cleanly.

        Crucially we stop at the section goal (no post-coast): the caller banks THIS crossing and
        treats whatever lies past the section as a fresh hazard — so a death further down the level
        never taints a candidate that genuinely cleared this section."""
        match = self._match(obs)
        if match is None:
            return None
        sec, model = match
        snap = session.clone_state()
        plan: list[Step] = []
        reach = obs.progress
        dying = False
        with session.search_mode():            # rollout frames stay invisible
            cur = obs
            for _ in range(sec.max_steps):
                dying = getattr(cur.raw, "is_dying", False)
                if dying or cur.progress >= sec.goal_x:
                    break
                action, _ = model.predict(features.featurize(cur.raw), deterministic=True)
                names, hold = SECTION_ACTIONS[int(action)]
                step = Step(hold, C.mask_from_names(list(names)))
                session.send_plan([step])
                plan.append(step)
                cur = observe()
                reach = max(reach, cur.progress)
        session.restore(snap)                  # leave the live state exactly as we found it
        observe()
        ok = (not dying) and plan and reach > obs.progress + self._MIN_GAIN
        if _DEBUG:
            print(f"[section] fired @{obs.level_label} x={obs.progress} -> {len(plan)} steps, "
                  f"reach={reach}, dying={dying}, commit={bool(ok)}", flush=True)
        return (plan, reach) if ok else None


def default_smb_sections() -> list[Section]:
    """The hazards we've trained sub-policies for. 1-3's tree-top platform/lift chain (x~120-700)."""
    return [
        Section(label="1-3", x_lo=100, x_hi=560, goal_x=700,
                model_path="data/rl/section_1_3"),
    ]
