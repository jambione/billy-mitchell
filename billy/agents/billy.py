"""Billy — the Tier-2 strategist (cocky LLM player), game-agnostic.

Consulted only at decision points. Given an Observation (the game's text summary + map),
retrieved lessons, and recent events, he returns a short controller plan plus trash talk.
Plan parsing uses the active system's Controller, so this works for any game/console. Every
failure path degrades to a safe idle so the control loop never stalls on a flaky local model.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import config, llm
from ..abstractions import Controller, Observation, Plan, Step
from ..knowledge import Lesson
from ..persona import BILLY_SYSTEM


@dataclass
class BillyDecision:
    plan: Plan
    trash_talk: str
    reasoning: str


def decide(obs: Observation, lessons: list[Lesson], recent_events: list[str],
           controller: Controller, *, memory: str = "") -> BillyDecision:
    user = _build_prompt(obs, lessons, recent_events, memory=memory)
    try:
        data = llm.chat_json(
            [{"role": "system", "content": BILLY_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.7, max_tokens=400,
        )
    except llm.LLMError:
        return _fallback(controller, "The cartridge hiccuped — watch a legend improvise.")

    plan = _parse_plan(data.get("plan", []), controller)
    if not plan:
        return _fallback(controller, "Trivial. I'll make this look easy.")
    return BillyDecision(
        plan=plan,
        trash_talk=str(data.get("trash_talk", "")).strip() or "Watch and learn.",
        reasoning=str(data.get("reasoning", "")).strip(),
    )


def _build_prompt(obs: Observation, lessons: list[Lesson], recent_events: list[str],
                  *, memory: str = "") -> str:
    # Rank lessons by quality (impact per use) and show effectiveness
    if lessons:
        sorted_lessons = sorted(lessons, key=lambda l: l.quality(), reverse=True)
        lesson_lines = []
        for l in sorted_lessons:
            quality = f"[quality: {l.quality():.0f}]" if l.uses > 0 else "[new]"
            lesson_lines.append(f"- {quality} At {l.situation}: {l.tactic}")
        lesson_block = "\n".join(lesson_lines)
    else:
        lesson_block = "- (none yet; start building your knowledge base)"

    events = ", ".join(recent_events[-6:]) or "just started"
    memory_block = f"{memory}\n\n" if memory else ""
    return (
        f"GAME STATE:\n{obs.summary}\n\n"
        f"MAP (M=you, E=enemy, #=solid, space=air):\n{obs.ascii_map}\n\n"
        f"{memory_block}"
        f"RECENT EVENTS: {events}\n\n"
        f"YOUR BEST LESSONS (ranked by effectiveness):\n{lesson_block}\n\n"
        f"Your next move: apply a high-quality lesson if it fits, or adapt as needed. JSON object only."
    )


def _parse_plan(raw_plan: object, controller: Controller) -> list[Step]:
    """Convert Billy's JSON plan into validated, frame-budgeted Steps via the controller."""
    steps: list[Step] = []
    total = 0
    if not isinstance(raw_plan, list):
        return steps
    for item in raw_plan:
        if not isinstance(item, dict):
            continue
        names = item.get("buttons") or []   # tolerate missing / null buttons
        try:
            frames = int(item.get("frames", 8))
        except (TypeError, ValueError):
            frames = 8
        frames = max(1, min(frames, config.BILLY_PLAN_MAX_FRAMES - total))
        if frames <= 0:
            break
        steps.append(Step(frames, controller.mask_from_names(names)))
        total += frames
        if total >= config.BILLY_PLAN_MAX_FRAMES:
            break
    return steps


def _fallback(controller: Controller, line: str) -> BillyDecision:
    return BillyDecision(plan=[Step(8, controller.neutral)], trash_talk=line,
                         reasoning="fallback: hold position")
