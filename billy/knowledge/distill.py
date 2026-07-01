"""Skill distillation — the cross-game exponential: banked wins become transferable tactics.

The SolutionCache replays a win only at its EXACT spot. Distillation additionally turns each
*significant* banked maneuver (a real crossing, not a two-step shuffle) into a `sequence`
Skill: the proven plan plus an embedding of the situation it solved. On any later hazard —
same level, another level, another game on the same console — whose situation reads similar,
the plan is offered as one more micro-search candidate. Search verifies it on a clone before
anything commits, so transfer can only ever *win* rollouts, never cause a death (the
exact-replay invariant is untouched).

Heuristic labeling keeps this on the hot loop (no LLM call at bank time): the skill's
retrieval text IS the observation summary where it worked, which is exactly what retrieval
matches against later. An optional offline LLM pass could enrich names/descriptions, but the
embedding of the raw situation already carries the transfer signal.
"""
from __future__ import annotations

import hashlib

from ..abstractions import Plan, plan_frames
from .skills import SkillLibrary

# A maneuver must gain at least this much progress to be worth generalizing. Below it, the
# reflex/search re-derives the move trivially and the library would silt up with noise.
MIN_GAIN_PX = 60
# Cap the library's distilled share: retrieval is top-k, but an unbounded library slows the
# cosine scan and dilutes retrieval quality. Oldest distilled skills are dropped first.
MAX_SEQUENCE_SKILLS = 64


def plan_signature(plan: Plan) -> str:
    """Stable content hash of an exact plan (dedupe key)."""
    raw = ";".join(f"{s.frames},{s.buttons}" for s in plan)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def distill_solution(skills: SkillLibrary, *, summary: str, level_label: str,
                     plan: Plan, start_x: int, reach: int, source: str,
                     console: str = "nes", min_gain: int = MIN_GAIN_PX) -> bool:
    """Distill one banked solution into a transferable sequence skill. Returns True if added.

    Gates: real progress (reach-start ≥ min_gain), a non-trivial plan, no duplicate plan
    already distilled. `summary` should be the Observation.summary at the plan's START — the
    situation the plan solves, which is what future retrieval must match."""
    gained = reach - start_x
    if gained < min_gain or not plan or plan_frames(plan) < 8 or not summary:
        return False
    sig = plan_signature(plan)
    if skills.has_signature(sig):
        return False

    _prune_sequences(skills)
    name = f"maneuver {level_label}@{start_x} (+{gained}px)"
    skills.add(
        name=name,
        description=summary,
        kind="sequence",
        payload={"plan": [[s.frames, s.buttons] for s in plan], "console": console,
                 "sig": sig, "gained": gained, "source": source},
    )
    return True


def _prune_sequences(skills: SkillLibrary) -> None:
    seq = [s for s in skills.skills if s.kind == "sequence"]
    while len(seq) >= MAX_SEQUENCE_SKILLS:
        oldest = seq.pop(0)
        skills.skills.remove(oldest)
