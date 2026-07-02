"""Frame-granular search for SMB 1-3's moving-lift gap (x≈600–760).

Macro-action reflex/RL search caps around x≈841; the lift needs precise wait/ride timing.
Used by the Director when Billy is on-ground in the lift band.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ...abstractions import Observation, Plan, Step
from ...systems.nes import controller as C

LIFT_LEVEL = "1-3"
LIFT_X_LO = 560
LIFT_X_HI = 800   # include pit-lip deaths (x≈770) for search/cache/replay
LIFT_APPROACH_LO = 450
LIFT_GOAL_X = 880
LIFT_COAST_FRAMES = 90
_LIFT_PLAN_FILE = Path("data/rl/lift_cached.plan.json")

# Wait-heavy vocabulary — the lift is timing, not jump-spam.
_BUTTONS: list[tuple[str, int]] = [
    ("idle", C.NEUTRAL),
    ("right", C.RIGHT),
    ("run", C.mask(C.RIGHT, C.B)),
    ("jump", C.mask(C.RIGHT, C.B, C.A)),
    ("hop", C.mask(C.RIGHT, C.A)),
    ("up", C.A),
]
_DURATIONS = (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 60, 80)


@dataclass(frozen=True)
class Move:
    label: str
    mask: int
    frames: int

    def step(self) -> Step:
        return Step(self.frames, self.mask)


def lift_zone(obs: Observation) -> bool:
    """1-3 x-range for the lift gap (airborne or on-ground)."""
    return obs.level_label == LIFT_LEVEL and LIFT_X_LO <= obs.progress <= LIFT_X_HI


def lift_band(obs: Observation) -> bool:
    on_ground = getattr(obs.raw, "on_ground", True)
    return lift_zone(obs) and on_ground


def lift_approach_zone(obs: Observation) -> bool:
    """Tree-top handoff → lift lip (learn-from-death searches here too)."""
    return (obs.level_label == LIFT_LEVEL
            and LIFT_APPROACH_LO <= obs.progress < LIFT_X_LO)


def lift_search_zone(obs: Observation) -> bool:
    """Any 1-3 band where frame/bootstrap lift search applies."""
    return lift_zone(obs) or lift_approach_zone(obs)


def lift_zone_at(level_label: str, x: int) -> bool:
    """Lift x-band without a full Observation (cache invalidation, etc.)."""
    return level_label == LIFT_LEVEL and LIFT_X_LO <= x <= LIFT_X_HI


def lift_cacheable(reach: int, *, crossed: bool = False) -> bool:
    """Only bank lift solutions that clear the pit lip — partial reach is lethal."""
    return crossed or reach >= LIFT_GOAL_X


def is_lift_death(level_label: str, death_x: int) -> bool:
    """Deaths in the lift approach trigger frame-granular learn-from-death."""
    return (level_label == LIFT_LEVEL
            and LIFT_X_LO <= death_x <= LIFT_GOAL_X + 48)


def lift_stall_visit_cap() -> int | None:
    """Lift timing needs many search tries — don't stall-break at the default cap.

    Returns None to disable stall-breaking entirely in the lift zone."""
    if os.environ.get("BILLY_LIFT_STALL_BREAK", "0") == "1":
        return int(os.environ.get("BILLY_LIFT_MAX_BUCKET_VISITS",
                                 os.environ.get("BILLY_MAX_BUCKET_VISITS", "8")))
    return int(os.environ.get("BILLY_LIFT_MAX_BUCKET_VISITS", "32"))


def _enabled() -> bool:
    return os.environ.get("BILLY_LIFT_FRAME_SEARCH", "1") == "1"


def search_params() -> tuple[int, int, int]:
    """(depth, beam, idle_max) — tunable for learn-from-death / bootstrap."""
    depth = int(os.environ.get("BILLY_LIFT_SEARCH_DEPTH", "6"))
    beam = int(os.environ.get("BILLY_LIFT_SEARCH_BEAM", "32"))
    idle_max = int(os.environ.get("BILLY_LIFT_IDLE_MAX", "180"))
    return depth, beam, idle_max


def verified_alive(session, observe, plan: Plan, *, goal_x: int = LIFT_GOAL_X) -> tuple[bool, int]:
    """Replay `plan` on a clone; True only if Mario finishes alive past goal_x."""
    snap = session.clone_state()
    with session.search_mode():
        session.send_plan(plan)
        obs = observe()
        best = obs.progress
        coasted = 0
        while coasted < LIFT_COAST_FRAMES and not obs.dead and obs.progress < goal_x:
            session.send_plan([Step(4, C.mask(C.RIGHT, C.B))])
            obs = observe()
            best = max(best, obs.progress)
            coasted += 4
        ok = (not obs.dead and not getattr(obs.raw, "is_dying", False)
              and obs.progress >= goal_x)
    session.restore(snap)
    observe()
    return ok, best


def moves_to_plan(moves: list[Move]) -> Plan:
    return [m.step() for m in moves]


def _rollout(session, observe, snap: bytes, moves: list[Move],
             goal_x: int, min_gain: int) -> tuple[bool, bool, int]:
    session.restore(snap)
    start = observe()
    best = start.progress
    for mv in moves:
        session.send_plan([mv.step()])
        obs = observe()
        best = max(best, obs.progress)
        if obs.dead:
            return False, False, best
    obs = observe()
    coasted = 0
    while coasted < LIFT_COAST_FRAMES and not obs.dead and obs.progress < goal_x:
        session.send_plan([Step(4, C.mask(C.RIGHT, C.B))])
        obs = observe()
        best = max(best, obs.progress)
        coasted += 4
    crossed = (not obs.dead and obs.progress >= goal_x
               and obs.progress > start.progress + min_gain)
    progressed = (not obs.dead and obs.progress > start.progress + min_gain)
    return progressed, crossed, best


def _idle_sweep(session, observe, snap: bytes, goal_x: int,
                min_gain: int, *, idle_max: int = 120) -> tuple[Plan | None, int, bool]:
    """Fast path: single hold-neutral durations (the dominant lift move)."""
    start = observe()
    best_plan: Plan | None = None
    best_reach = start.progress
    crossed = False
    for wait in range(1, idle_max + 1):
        session.restore(snap)
        session.send_plan([Step(wait, C.NEUTRAL)])
        obs = observe()
        reach = obs.progress
        if not obs.dead:
            coasted = 0
            while coasted < LIFT_COAST_FRAMES and not obs.dead and obs.progress < goal_x:
                session.send_plan([Step(4, C.mask(C.RIGHT, C.B))])
                obs = observe()
                reach = max(reach, obs.progress)
                coasted += 4
            if not obs.dead and reach > best_reach:
                best_reach = reach
                best_plan = [Step(wait, C.NEUTRAL)]
            if not obs.dead and obs.progress >= goal_x and reach > start.progress + min_gain:
                return [Step(wait, C.NEUTRAL)], reach, True
    session.restore(snap)
    observe()
    if best_plan and best_reach > start.progress + min_gain:
        return best_plan, best_reach, crossed
    return None, best_reach, False


def _beam_search(session, observe, snap: bytes, goal_x: int, min_gain: int,
                 depth: int, beam: int) -> tuple[Plan | None, int, bool]:
    start_x = observe().progress
    frontier: list[list[Move]] = [[]]
    best_moves: list[Move] = []
    best_x = start_x
    for _ in range(depth):
        candidates: list[tuple[list[Move], int, bool]] = []
        for moves in frontier:
            for label, mask in _BUTTONS:
                for dur in _DURATIONS:
                    trial = moves + [Move(label, mask, dur)]
                    progressed, crossed, reach = _rollout(
                        session, observe, snap, trial, goal_x, min_gain)
                    if crossed:
                        session.restore(snap)
                        observe()
                        return moves_to_plan(trial), reach, True
                    if progressed and reach > best_x:
                        best_x, best_moves = reach, trial
                    if progressed:
                        candidates.append((trial, reach, crossed))
        if not candidates:
            break
        candidates.sort(key=lambda c: c[1], reverse=True)
        frontier = [c[0] for c in candidates[:beam]]
    session.restore(snap)
    observe()
    if best_moves and best_x > start_x + min_gain:
        return moves_to_plan(best_moves), best_x, False
    return None, best_x, False


def _search_at_snap(session, observe, snap: bytes, *, goal_x: int, min_gain: int,
                    depth: int, beam: int, idle_max: int) -> tuple[Plan | None, int, bool]:
    with session.search_mode():
        plan, reach, crossed = _idle_sweep(
            session, observe, snap, goal_x, min_gain, idle_max=idle_max)
        if crossed:
            return plan, reach, True
        if plan is None or reach < goal_x - 80:
            bplan, reach, crossed = _beam_search(
                session, observe, snap, goal_x, min_gain, depth=depth, beam=beam)
            if crossed and bplan is not None:
                return bplan, reach, True
            if bplan is not None:
                plan = bplan
    return plan, reach, crossed


def _section_lift_prefixes(session, observe, snap: bytes, *, max_steps: int = 6) -> list[Plan]:
    """Collect growing prefixes of the lift section RL rollout (for bootstrap combos)."""
    try:
        from stable_baselines3 import PPO
        from ...rl import features
        from ...rl.section_env import SECTION_ACTIONS
    except ImportError:
        return [[]]
    model_path = "data/rl/section_1_3_lift"
    if not (os.path.isfile(f"{model_path}.zip") or os.path.isfile(model_path)):
        return [[]]
    ppo = PPO.load(model_path, device="cpu")
    prefixes: list[Plan] = [[]]
    session.restore(snap)
    obs = observe()
    prefix: Plan = []
    for _ in range(max_steps):
        if obs.dead:
            break
        action, _ = ppo.predict(features.featurize(obs.raw), deterministic=True)
        names, hold = SECTION_ACTIONS[int(action)]
        step = Step(hold, C.mask_from_names(list(names)))
        prefix = prefix + [step]
        prefixes.append(list(prefix))
        session.send_plan([step])
        obs = observe()
    session.restore(snap)
    observe()
    return prefixes


def _chain_frame_search(session, observe, snap: bytes, *, goal_x: int, min_gain: int,
                        depth: int, beam: int, idle_max: int,
                        max_hops: int = 3) -> tuple[Plan | None, int, bool]:
    """Resume frame search from partial survivors (approach → lip → crossing)."""
    base = snap
    session.restore(base)
    best_reach = observe().progress
    full: Plan = []
    cur = base
    for _ in range(max_hops):
        plan, _reach, crossed = _search_at_snap(
            session, observe, cur, goal_x=goal_x, min_gain=min_gain,
            depth=depth, beam=beam, idle_max=idle_max)
        if crossed and plan is not None:
            full = full + plan
            session.restore(base)
            observe()
            ok, verified = verified_alive(session, observe, full, goal_x=goal_x)
            if ok:
                return full, verified, True
            best_reach = max(best_reach, verified)
            break
        if plan is None:
            break
        full = full + plan
        session.restore(base)
        observe()
        ok, verified = verified_alive(session, observe, full, goal_x=goal_x)
        if ok:
            return full, verified, True
        best_reach = max(best_reach, verified)
        session.restore(cur)
        observe()
        for step in plan:
            session.send_plan([step])
            if observe().dead:
                break
        if observe().dead:
            break
        cur = session.clone_state()
        if observe().progress >= goal_x - 16:
            break
    session.restore(base)
    observe()
    return (full if full else None), best_reach, False


def lift_bootstrap_search(session, observe, *, goal_x: int = LIFT_GOAL_X,
                          min_gain: int = 8) -> tuple[Plan | None, int, bool]:
    """Lift-phase search: random entry offsets + section prefixes + frame idle/beam."""
    if not _enabled():
        return None, 0, False
    depth, beam, idle_max = search_params()
    base = session.clone_state()
    trials: list[tuple[Plan, bytes]] = [( [], base)]

    # Random cruise offsets diversify lift vertical phase (like SectionEnv randomize_frames).
    stride = int(os.environ.get("BILLY_LIFT_BOOTSTRAP_STRIDE", "8"))
    max_offset = int(os.environ.get("BILLY_LIFT_BOOTSTRAP_OFFSETS", "40"))
    for offset in range(0, max_offset + 1, max(4, stride)):
        session.restore(base)
        if offset:
            session.send_plan([Step(offset, C.mask(C.RIGHT, C.B))])
        obs = observe()
        if obs.dead:
            continue
        trials.append(([], session.clone_state()))

    for pre_plan, snap in trials:
        for prefix in _section_lift_prefixes(session, observe, snap):
            combo_pre = pre_plan + prefix
            session.restore(snap)
            if combo_pre:
                session.send_plan(combo_pre)
            if observe().dead:
                continue
            mid = session.clone_state()
            plan, reach, crossed = _chain_frame_search(
                session, observe, mid, goal_x=goal_x, min_gain=min_gain,
                depth=depth, beam=beam, idle_max=idle_max)
            if plan is None:
                continue
            full = combo_pre + plan
            ok, best = verified_alive(session, observe, full, goal_x=goal_x)
            if ok:
                session.restore(base)
                observe()
                return full, best, True
    session.restore(base)
    observe()
    chained, reach, crossed = _chain_frame_search(
        session, observe, base, goal_x=goal_x, min_gain=min_gain,
        depth=depth, beam=beam, idle_max=idle_max)
    if chained:
        ok, best = verified_alive(session, observe, chained, goal_x=goal_x)
        if ok:
            return chained, best, True
    return None, observe().progress, False


def frame_lift_search(session, observe, *, goal_x: int = LIFT_GOAL_X,
                      min_gain: int = 40, depth: int = -1, beam: int = -1
                      ) -> tuple[Plan | None, int, bool]:
    """Search on a clone; leave live state untouched. Returns (plan, reach, crossed_goal)."""
    if not _enabled():
        return None, 0, False
    d, b, idle_max = search_params()
    depth = d if depth < 0 else depth
    beam = b if beam < 0 else beam
    snap = session.clone_state()
    plan, reach, crossed = _search_at_snap(
        session, observe, snap, goal_x=goal_x, min_gain=min_gain,
        depth=depth, beam=beam, idle_max=idle_max)
    session.restore(snap)
    observe()
    return plan, reach, crossed


def lift_crossing_search(session, observe, *, goal_x: int = LIFT_GOAL_X,
                         min_gain: int = 8) -> tuple[Plan | None, int, bool]:
    """Full lift crossing search: bootstrap (phase-aware) then plain frame search."""
    start_x = observe().progress
    for finder in (lift_bootstrap_search, frame_lift_search):
        plan, reach, crossed = finder(session, observe, goal_x=goal_x, min_gain=min_gain)
        if not plan:
            continue
        ok, best = verified_alive(session, observe, plan, goal_x=goal_x)
        if ok:
            return plan, best, True
    return None, start_x, False


def persist_lift_plan(plan: Plan, reach_after: int) -> None:
    """Write a verified crossing for section_policy bootstrap."""
    _LIFT_PLAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LIFT_PLAN_FILE.write_text(json.dumps({
        "plan": [[s.frames, s.buttons] for s in plan],
        "reach_after": reach_after,
        "kind": "frame_lift_search",
    }))