"""SMB hazard hooks — lift gap, pit approaches, and 2-2 pit compounding."""
from __future__ import annotations

import os

from ...abstractions import Observation, Plan, Step
from ...hazard_hooks import HazardHooks
from ...stuck_trainer import StuckRemedy
from ...knowledge.cache import CacheEntry
from ...systems.nes import controller as C
from ..common.platformer import gap_jumper
from .lift_search import (
    LIFT_APPROACH_LO,
    LIFT_GOAL_X,
    LIFT_X_HI,
    LIFT_X_LO,
    frame_lift_search,
    is_lift_death,
    lift_approach_zone,
    lift_cacheable,
    lift_crossing_search,
    lift_search_zone,
    lift_zone,
    lift_zone_at,
    persist_lift_plan,
)
from .tuning import PROFILE

# 2-2 area-2 pit — deaths observed at x≈1124–1155 in test runs.
PIT_22_LO = 980
PIT_22_HI = 1160          # approach / search band (through pit lip)
PIT_22_DEATH_LO = 1080
PIT_22_DEATH_HI = 1180
PIT_22_GOAL_X = 1170      # verified landing must reach past the lip
PIT_COAST_FRAMES = 150
PIT_CLEAR_MARGIN = 16


def pit_approach_zone(obs: Observation) -> bool:
    return obs.level_label == "2-2" and PIT_22_LO <= obs.progress <= PIT_22_HI


def is_pit_death(level_label: str, death_x: int) -> bool:
    return (level_label == "2-2"
            and PIT_22_DEATH_LO <= death_x <= PIT_22_DEATH_HI)


def pit_goal_x(death_x: int = 0) -> int:
    """Minimum x a verified pit crossing must reach before banking."""
    if death_x > 0:
        return max(PIT_22_GOAL_X, death_x + PIT_CLEAR_MARGIN)
    return PIT_22_GOAL_X


def pit_cacheable(reach: int, death_x: int = 0) -> bool:
    """Bank only if the survivor clears the lip with margin (matches verified_pit_crossing)."""
    return reach >= pit_goal_x(death_x)


def verified_pit_crossing(session, observe, plan: Plan, *, goal_x: int,
                          start_x: int) -> tuple[bool, int]:
    """Replay `plan` on a clone; True only if Mario is alive on solid ground past `goal_x`."""
    snap = session.clone_state()
    with session.search_mode():
        session.send_plan(plan)
        obs = observe()
        best = obs.progress
        if obs.dead or getattr(obs.raw, "is_dying", False):
            session.restore(snap)
            observe()
            return False, best
        coasted = 0
        while coasted < PIT_COAST_FRAMES and not obs.dead:
            if obs.progress >= goal_x and getattr(obs.raw, "on_ground", True):
                break
            session.send_plan([Step(4, C.mask(C.RIGHT, C.B))])
            obs = observe()
            best = max(best, obs.progress)
            coasted += 4
        ok = (not obs.dead
              and not getattr(obs.raw, "is_dying", False)
              and obs.progress >= goal_x
              and obs.progress > start_x + 8)
    session.restore(snap)
    observe()
    return ok, best


class SmbHazardHooks:
    """SMB-specific hazard behaviour consumed by the game-agnostic Director."""

    def commit_chunk_size(self, obs: Observation, default: int) -> int:
        if obs.level_label == "1-3" and (obs.progress >= LIFT_X_LO - 64 or lift_zone(obs)):
            return 1
        return default

    def in_special_zone(self, obs: Observation) -> bool:
        return lift_zone(obs) or pit_approach_zone(obs)

    def stall_break_exempt(self, obs: Observation) -> bool:
        if lift_zone(obs) or pit_approach_zone(obs):
            return True
        return getattr(obs.raw, "pipe_entry_spot", lambda: None)() is not None

    def stale_cache(self, obs: Observation, cached: CacheEntry | None) -> bool:
        if cached is None:
            return False
        if lift_zone(obs) and not lift_cacheable(cached.reach_after):
            return True
        if pit_approach_zone(obs) and not pit_cacheable(cached.reach_after):
            return True
        return False

    def pit_death(self, level_label: str, death_x: int) -> bool:
        return is_pit_death(level_label, death_x)

    def is_special_death(self, level_label: str, death_x: int) -> bool:
        return is_lift_death(level_label, death_x) or is_pit_death(level_label, death_x)

    def learn_horizon_frames(self, level_label: str, death_x: int) -> int | None:
        if is_pit_death(level_label, death_x):
            return max(PIT_COAST_FRAMES, 180)
        if is_lift_death(level_label, death_x):
            return 180
        return None

    def learn_runway_action(self, level_label: str, death_x: int, runway: int,
                            default_horizon: int) -> str | None:
        if is_lift_death(level_label, death_x):
            if runway > default_horizon:
                return "continue"
        return None

    def learn_cacheable(self, level_label: str, death_x: int, reach: int) -> bool:
        if is_lift_death(level_label, death_x):
            return lift_cacheable(reach)
        if is_pit_death(level_label, death_x):
            return pit_cacheable(reach, death_x)
        return True

    def try_frame_search(self, session, observe, obs: Observation, *, deep: bool,
                         min_gain: int) -> tuple[Plan | None, int, bool]:
        if not lift_search_zone(obs):
            return None, obs.progress, False
        finder = lift_crossing_search if deep else frame_lift_search
        plan, reach, crossed = finder(session, observe, min_gain=min_gain)
        if plan and lift_cacheable(reach, crossed=crossed):
            if crossed:
                persist_lift_plan(plan, reach)
            return plan, reach, crossed
        return None, reach, False

    def try_pit_search(self, session, observe, obs: Observation, *,
                       death_x: int, min_gain: int) -> tuple[Plan | None, int]:
        if not pit_approach_zone(obs):
            return None, obs.progress
        scene = obs.raw
        gap = scene.gap_info() if hasattr(scene, "gap_info") else None
        width = gap[1] if gap else 3
        goal = pit_goal_x(death_x)
        start_x = obs.progress
        # Wider hold sweep + short run-ups — partial jumps were banking at x≈1146 then dying at 1155.
        candidates = gap_jumper(width, PROFILE)
        candidates += gap_jumper(max(width + 1, 4), PROFILE)
        runups = (0, 6, 10, 14)
        expanded: list[Plan] = []
        for plan in candidates:
            expanded.append(plan)
            for r in runups:
                if r:
                    expanded.append([Step(r, C.mask(C.RIGHT, C.B))] + list(plan))
        snap = session.clone_state()
        best_plan, best_reach = None, start_x
        for plan in expanded:
            ok, reach = verified_pit_crossing(
                session, observe, plan, goal_x=goal, start_x=start_x)
            if ok and reach > best_reach and reach > start_x + min_gain:
                best_plan, best_reach = plan, reach
        session.restore(snap)
        observe()
        return best_plan, best_reach

    def cacheable_reach(self, obs: Observation, reach: int, *, crossed: bool = False) -> bool:
        if lift_zone(obs):
            return lift_cacheable(reach, crossed=crossed)
        if pit_approach_zone(obs):
            return pit_cacheable(reach)
        return True

    def section_bankable(self, obs: Observation, reach: int) -> bool:
        if lift_zone(obs):
            return lift_cacheable(reach)
        return True

    def replay_death_drop_reason(self, level_label: str, replay_x: int,
                                 death_x: int) -> str | None:
        if lift_zone_at(level_label, replay_x) or is_lift_death(level_label, death_x):
            return "lift_commit_fail"
        if is_pit_death(level_label, death_x) or (
                level_label == "2-2" and PIT_22_LO <= replay_x <= PIT_22_HI):
            return "pit_commit_fail"
        return None

    def extra_candidates(self, obs: Observation) -> list[Plan]:
        if not pit_approach_zone(obs):
            return []
        scene = obs.raw
        gap = scene.gap_info() if hasattr(scene, "gap_info") else None
        if gap is None:
            return []
        return gap_jumper(gap[1], PROFILE)

    def approach_capture_band(self, level_label: str, death_x: int) -> tuple[int, int] | None:
        if is_lift_death(level_label, death_x):
            return LIFT_APPROACH_LO, LIFT_X_HI
        if is_pit_death(level_label, death_x):
            return PIT_22_LO, PIT_22_HI - 32
        return None

    def approach_snapshot_band(self, obs: Observation) -> tuple[int, int] | None:
        if lift_search_zone(obs):
            return LIFT_APPROACH_LO, LIFT_X_HI
        if pit_approach_zone(obs):
            return PIT_22_LO, PIT_22_HI
        return None

    def stuck_remedy(self, level_label: str, death_x: int) -> StuckRemedy | None:
        if is_lift_death(level_label, death_x):
            states = (
                "data/rl/states/smb_1_3_lift.state",
                "data/rl/states/smb_1_3_lift_751.state",
                "data/rl/states/smb_1_3_lift_636.state",
                "data/rl/states/smb_1_3_section.state",
            )
            return StuckRemedy(
                kind="frame_search",
                level_label=level_label,
                death_x=death_x,
                goal_x=LIFT_GOAL_X,
                savestate_paths=states,
                bank_x_lo=LIFT_X_LO - 64,
                bank_x_hi=LIFT_X_HI,
                section_out="data/rl/section_1_3_lift",
                section_timesteps=int(os.environ.get("BILLY_STUCK_TRAIN_STEPS", "100000")),
            )
        if is_pit_death(level_label, death_x):
            return StuckRemedy(
                kind="pit_search",
                level_label=level_label,
                death_x=death_x,
                goal_x=pit_goal_x(death_x),
                savestate_paths=("data/rl/states/smb_2_2_pit.state",),
                bank_x_lo=PIT_22_LO,
                bank_x_hi=PIT_22_HI,
            )
        return None