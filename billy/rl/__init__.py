"""Optional reinforcement-learning tier.

A learned PPO policy that plugs in as a `ReflexPolicy` (the fast Tier-1 controller), trained against
a Gymnasium wrapper over the SAME in-process emulator + RAM perception the rest of Billy uses. RL is
strictly optional: import this package only when training or running with `--rl`, so the reflex-only
build keeps working without torch/stable-baselines3 installed.

Design: RL does NOT replace Billy's compounding memory — it replaces the *hand-crafted reflex*. The
Director's policy order is unchanged: SolutionCache (exact replay) -> reflex tier (now optionally a
LearnedReflex) -> micro-search / learn-from-death -> LLM. So the cache still owns deterministic
hazard replay, search still discovers/verifies, the LLM still handles novelty — RL just makes the
moment-to-moment routine play smarter and is swappable.
"""
