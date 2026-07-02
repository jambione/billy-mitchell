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

import json
import os
from dataclasses import dataclass
from pathlib import Path

_DEBUG = os.environ.get("BILLY_DEBUG_SECTION", "0") == "1"
_LIFT_CACHE_PLAN = Path("data/rl/lift_cached.plan.json")

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
    landing_waits: int = 0   # post-goal noop waits so airborne crossings land before commit


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

    def _try_cached_plan(self, sec: Section, obs: Observation, session, observe):
        """Replay a banked frame-level plan (solution cache / lift bootstrap) on a clone."""
        from ..knowledge.cache import SolutionCache

        plans: list[Plan] = []
        cache = SolutionCache()
        entry = cache.get(obs.level_key, obs.progress, obs.elevation)
        if entry and entry.reach_after > obs.progress + self._MIN_GAIN:
            plans.append(entry.plan)
        if _LIFT_CACHE_PLAN.is_file() and "lift" in sec.model_path:
            try:
                raw = json.loads(_LIFT_CACHE_PLAN.read_text())
                plans.append([Step(f, b) for f, b in raw["plan"]])
            except Exception:
                pass
        if not plans:
            return None
        snap = session.clone_state()
        best: tuple[Plan, int] | None = None
        with session.search_mode():
            for plan in plans:
                session.restore(snap)
                cur = obs
                for step in plan:
                    if cur.dead or cur.progress >= sec.goal_x:
                        break
                    session.send_plan([step])
                    cur = observe()
                for _ in range(24):
                    if cur.dead or cur.progress >= sec.goal_x:
                        break
                    session.send_plan([Step(4, C.mask(C.RIGHT, C.B))])
                    cur = observe()
                if not cur.dead and cur.progress > obs.progress + self._MIN_GAIN:
                    ok = True
                    if "lift" in sec.model_path:
                        try:
                            from ..games.smb.lift_search import lift_cacheable
                            ok = lift_cacheable(cur.progress)
                        except ImportError:
                            pass
                    if ok and (best is None or cur.progress > best[1]):
                        best = (plan, cur.progress)
        session.restore(snap)
        observe()
        return best

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
        cached = self._try_cached_plan(sec, obs, session, observe)
        if cached is not None:
            return cached
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
            # Tree-top crossings often end mid-arc; coast with waits so Mario lands on the
            # long platform before we bank the sequence (else the live replay falls into the pit).
            if not dying and sec.landing_waits and cur.progress >= sec.goal_x:
                for _ in range(sec.landing_waits):
                    step = Step(4, C.mask_from_names([]))
                    session.send_plan([step])
                    plan.append(step)
                    cur = observe()
                    reach = max(reach, cur.progress)
                    if cur.dead:
                        dying = True
                        break
                    if getattr(cur.raw, "on_ground", False):
                        break
        # Lift section: coast after rollout so a mid-arc crossing can land alive past the pit.
        if not dying and "lift" in sec.model_path:
            for _ in range(24):
                step = Step(4, C.mask_from_names([]))
                session.send_plan([step])
                plan.append(step)
                cur = observe()
                reach = max(reach, cur.progress)
                if cur.dead:
                    dying = True
                    break
            if not dying:
                for _ in range(24):
                    step = Step(4, C.mask_from_names(["right", "B"]))
                    session.send_plan([step])
                    plan.append(step)
                    cur = observe()
                    reach = max(reach, cur.progress)
                    if cur.dead:
                        dying = True
                        break
        session.restore(snap)                  # leave the live state exactly as we found it
        observe()
        landed = not sec.landing_waits or getattr(cur.raw, "on_ground", True)
        ok = (not dying) and landed and plan and reach > obs.progress + self._MIN_GAIN
        if ok and "lift" in sec.model_path:
            try:
                from ..games.smb.lift_search import lift_cacheable, verified_alive
                ok = lift_cacheable(reach)
                if ok and plan:
                    vok, vreach = verified_alive(session, observe, plan, goal_x=sec.goal_x)
                    ok = vok
                    reach = max(reach, vreach)
            except ImportError:
                pass
        if _DEBUG:
            print(f"[section] fired @{obs.level_label} x={obs.progress} -> {len(plan)} steps, "
                  f"reach={reach}, dying={dying}, commit={bool(ok)}", flush=True)
        return (plan, reach) if ok else None


def default_smb_sections() -> list[Section]:
    """Hazard-scoped sub-policies wired into Billy's section controller."""
    sections = [
        # Tree-top platform hops (x~120-700). landing_waits: land on the long platform past it.
        Section(label="1-3", x_lo=100, x_hi=560, goal_x=700,
                model_path="data/rl/section_1_3", landing_waits=8),
    ]
    # Moving-lift gap (x~700-860 → goal past the lift). Model is optional — if missing or
    # unloadable the controller degrades to search-only at this band (see SectionController._load).
    lift = Section(label="1-3", x_lo=560, x_hi=860, goal_x=900,
                   model_path="data/rl/section_1_3_lift", max_steps=96, landing_waits=2)
    if os.path.isfile(f"{lift.model_path}.zip") or os.path.isfile(lift.model_path):
        sections.append(lift)
    return sections
