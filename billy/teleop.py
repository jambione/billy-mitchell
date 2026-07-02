"""Human-in-the-loop demonstration capture — a learning accelerator.

A human demo is just an exact button sequence, the same object the micro-search produces.
So it flows through Billy's existing pipeline without violating the "exact-replay only"
invariant: capture the player's input as a `Plan`, verify on a cloned state that it survives
AND advances `progress`, then bank it to the SolutionCache keyed to the start position. Every
later attempt replays it for free — the wall becomes a one-time teaching moment.

This module is game-agnostic: it speaks only to the `Session`/`Game`/cache contracts. The live
keyboard capture lives in the NES viewer; the standalone CLI is `teleop_zelda.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .abstractions import Plan, Step
from .knowledge.cache import SolutionCache, bucket_of


class TeleopRecorder:
    """Accumulates per-frame button masks, run-length-encoded into a compact Plan.

    Consecutive frames with the same mask collapse into one Step, so a 90-frame RIGHT hold is
    one `Step(90, RIGHT)` rather than 90 single-frame steps — identical to how reflex plans look.
    """

    def __init__(self) -> None:
        self._steps: list[Step] = []
        self._cur_mask: int | None = None
        self._cur_frames: int = 0

    def record(self, mask: int, frames: int = 1) -> None:
        if frames <= 0:
            return
        if mask == self._cur_mask:
            self._cur_frames += frames
        else:
            self._flush()
            self._cur_mask = mask
            self._cur_frames = frames

    def _flush(self) -> None:
        if self._cur_mask is not None and self._cur_frames > 0:
            self._steps.append(Step(self._cur_frames, self._cur_mask))
        self._cur_mask = None
        self._cur_frames = 0

    def plan(self) -> Plan:
        steps = list(self._steps)
        if self._cur_mask is not None and self._cur_frames > 0:
            steps.append(Step(self._cur_frames, self._cur_mask))
        return steps

    def frame_count(self) -> int:
        return sum(s.frames for s in self.plan())


@dataclass
class DemoResult:
    """Outcome of replaying a captured demo from its start state (verification gate)."""
    survived: bool
    start_progress: int
    end_progress: int
    end_level_key: tuple
    end_label: str
    frames: int
    min_progress: int = 8

    @property
    def advanced(self) -> bool:
        return self.end_progress > self.start_progress + self.min_progress

    @property
    def bankable(self) -> bool:
        """The same gate every banked solution passes: survive AND make real progress."""
        return self.survived and self.advanced

    def summary(self) -> str:
        verdict = "BANKABLE" if self.bankable else (
            "died" if not self.survived else "no-progress")
        return (f"{verdict}: {self.start_progress}->{self.end_progress} "
                f"(+{self.end_progress - self.start_progress}) over {self.frames}f "
                f"end={self.end_label}")


def verify_demo(session, game, start_state: bytes, plan: Plan, *,
                min_progress: int = 8) -> DemoResult:
    """Restore the start state on the (cloned) env, replay the captured plan, report the result.

    Confirms the run-length-encoded plan deterministically reproduces a surviving, progressing
    run — exactly the gate a micro-search survivor passes before it is banked. Runs invisibly
    under `search_mode()` so verification never flashes on screen.
    """
    def observe():
        st = session.read_state()
        return game.observe(st.frame, st.ram, getattr(st, "rgb", None))

    with session.search_mode():
        session.restore(start_state)
        start = observe()
        session.send_plan(plan)
        end = observe()

    return DemoResult(
        survived=not end.dead,
        start_progress=start.progress,
        end_progress=end.progress,
        end_level_key=end.level_key,
        end_label=end.level_label,
        frames=sum(s.frames for s in plan),
        min_progress=min_progress,
    )


def bank_demo(cache: SolutionCache, start_obs, plan: Plan, reach: int) -> tuple:
    """Bank a verified demo to the SolutionCache, keyed exactly as the Director looks it up:
    `cache.get(level_key, progress, elevation)`. force=True so it replaces any stale entry at the
    node (a human survivor should win over whatever was thrashing there)."""
    cache.put(start_obs.level_key, start_obs.progress, plan, reach,
              y=start_obs.elevation, force=True)
    return bucket_of(start_obs.level_key, start_obs.progress, start_obs.elevation)
