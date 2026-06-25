"""Coach — the Tier-3 analyst.

After an attempt ends (death / clear / safety cap), the Coach reviews a compact replay of the
trajectory and distills ONE reusable lesson, which the Director writes to the knowledge base.
Keeping reflection out of Billy's own context keeps his prompts lean and his ego intact.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import llm
from ..persona import COACH_SYSTEM


@dataclass
class TrajectoryStep:
    """One sampled moment of an attempt, for film study."""
    x: int
    summary: str
    action: str
    event: str


@dataclass
class CoachLesson:
    situation: str
    tactic: str
    outcome: str


def reflect(trajectory: list[TrajectoryStep], outcome: str, world_stage: str) -> CoachLesson | None:
    """Return a lesson distilled from one attempt, or None if the model can't be reached."""
    if not trajectory:
        return None
    replay = "\n".join(
        f"x={s.x:>4} | {s.event:<11} | did: {s.action:<22} | {s.summary}"
        for s in _thin(trajectory, keep=24)
    )
    user = (
        f"LEVEL: {world_stage}\nHOW IT ENDED: {outcome}\n"
        f"FARTHEST X REACHED: {max(s.x for s in trajectory)}\n\n"
        f"REPLAY (one line per sampled moment):\n{replay}\n\n"
        f"Extract ONE concrete lesson for next time. JSON object only."
    )
    try:
        data = llm.chat_json(
            [{"role": "system", "content": COACH_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.3, max_tokens=300,
        )
    except llm.LLMError:
        return None
    situation = str(data.get("situation", "")).strip()
    tactic = str(data.get("tactic", "")).strip()
    if not situation or not tactic:
        return None
    return CoachLesson(situation=situation, tactic=tactic,
                       outcome=str(data.get("outcome", outcome)).strip())


def _thin(traj: list[TrajectoryStep], keep: int) -> list[TrajectoryStep]:
    """Downsample a long trajectory, always keeping the final (decisive) moments."""
    if len(traj) <= keep:
        return traj
    head_n = keep - 6
    step = max(1, len(traj) // head_n)
    return traj[::step][:head_n] + traj[-6:]
