"""LearnedReflex — a PPO policy that plugs in as the Tier-1 ReflexPolicy.

Design (how RL coexists with cache / search / LLM):
  The Director's order is cache -> reflex -> micro-search -> LLM. LearnedReflex slots in at the
  reflex tier, wrapping the hand-crafted `PlatformerReflex` as a FALLBACK. Per step:
    • if the fallback flags a hazard/stuck (`needs_billy`) -> we return THAT, so the proven
      cache / micro-search / learn-from-death / LLM machinery handles deadly spots unchanged
      (this preserves Billy's 1-1 clear and the compounding memory);
    • otherwise (routine play) -> we return the RL policy's action.
  So RL improves moment-to-moment movement/positioning while the verified search still owns the
  lethal hazards. The fallback also supplies `advance_plan` / `danger_candidates` /
  `expanded_candidates`, so micro-search and learn-from-death keep working untouched.

  Set `rl_handles_hazards=True` to let the policy act even at hazards (more RL-forward, but it
  forgoes the verified search at those spots — opt in once the policy is strong).

If the model can't be loaded (no torch/SB3, or missing file), it degrades to the pure fallback —
the reflex-only build keeps working.
"""
from __future__ import annotations

from ..abstractions import Decision, Observation, Plan, ReflexPolicy, Step
from . import features


class LearnedReflex(ReflexPolicy):
    def __init__(self, model_path: str, fallback: ReflexPolicy, frame_skip: int = 4,
                 deterministic: bool = True, rl_handles_hazards: bool = False) -> None:
        self.fallback = fallback
        self.frame_skip = frame_skip
        self.deterministic = deterministic
        self.rl_handles_hazards = rl_handles_hazards
        self.model = self._load(model_path)

    @staticmethod
    def _load(model_path: str):
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path, device="cpu")  # inference on CPU is plenty for one step
            print(f"[rl] loaded PPO policy from {model_path}")
            return model
        except Exception as e:  # missing deps / file / version skew -> fall back gracefully
            print(f"[rl] no policy ({type(e).__name__}: {e}); using hand-crafted reflex")
            return None

    # --- ReflexPolicy: lifecycle + search hooks delegate to the fallback ----------------
    def reset(self, obs: Observation) -> None:
        self.fallback.reset(obs)

    def note_level_advance(self, obs: Observation) -> None:
        self.fallback.note_level_advance(obs)

    def advance_plan(self, obs: Observation) -> Plan:
        return self.fallback.advance_plan(obs)

    def danger_candidates(self, obs: Observation) -> list[Plan]:
        return self.fallback.danger_candidates(obs)

    def expanded_candidates(self, obs: Observation) -> list[Plan]:
        fn = getattr(self.fallback, "expanded_candidates", None)
        return fn(obs) if fn else self.fallback.danger_candidates(obs)

    def add_reflex_rule(self, rule) -> None:
        add = getattr(self.fallback, "add_reflex_rule", None)
        if add:
            add(rule)

    # --- the per-exchange decision ------------------------------------------------------
    def step(self, obs: Observation) -> Decision:
        decision = self.fallback.step(obs)
        # No model, or a hazard the verified machinery should own -> defer to the fallback path.
        if self.model is None or (decision.needs_billy and not self.rl_handles_hazards):
            return decision
        action, _ = self.model.predict(features.featurize(obs.raw),
                                       deterministic=self.deterministic)
        mask = features.ACTION_MASKS[int(action)]
        return Decision([Step(self.frame_skip, mask)], note="rl")
