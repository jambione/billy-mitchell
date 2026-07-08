"""The Director — the game-agnostic engine loop.

Drives any Game through the abstract contracts: observe/act, boot, then a continuous
playthrough — reflex for routine play, and at each hazard a **cache-first policy**: replay the
exact verified solution if we've solved this spot before, else **search on a cloned state**
(invisible to the live run) for a surviving sequence and **remember it**. The LLM is consulted
only when search finds nothing. This is what makes the learning compound across attempts.
It never references a specific game or system.
"""
from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path

from . import config, metrics
from .abstractions import Game, Observation, Plan, Step, plan_frames
from .agents import billy, coach
from .agents.rolling_memory import RollingGameMemory
from .commentary import Commentator
from .hazard_hooks import HazardHooks
from .knowledge import KnowledgeBase, RouteGraph, SolutionCache, SkillLibrary, TapeLibrary
from .knowledge.cache import bucket_of
from .knowledge.tape import append_plan
from .learning import LearningLedger, format_learning_line
from .stuck_trainer import (
    StuckTracker, auto_state_path, remediate, write_trail_snapshot,
)

_IDLE: Plan = [Step(2, 0)]   # neutral input — to consume a pending frame or coast a cutscene

# a level/area advance (e.g. entering a pipe) dominates any reach
_TRANSITION_BONUS = 100_000
_MIN_PROGRESS_PX = 8   # a "solution" must actually advance, else it's a stall, not an escape


def rollout_candidate(session, observe, reflex, game, plan: Plan,
                      settle: int = config.SEARCH_HORIZON_FRAMES,
                      *, min_progress: int = _MIN_PROGRESS_PX) -> tuple:
    """Run one candidate from the CURRENT (already-restored) state, THEN coast forward `settle`
    frames watching for a delayed death. Module-level so the parallel search workers evaluate
    candidates with EXACTLY the code the serial path uses (see Director._rollout for the full
    rationale: post-candidate settle budget, transition bonus, detour horizon)."""
    start = observe()
    start_level, base_x = start.level_key, start.progress
    # Execute the candidate in short chunks so a MID-PLAN death or area transition is seen when
    # it happens. A blind full-plan send scores the observation at plan END — by then a death's
    # dying flags can have decayed through the level-reload frames, letting a plan that crossed
    # a transition and died in the new area report "survived + advanced" (which is exactly how a
    # poisoned exit-pipe plan got banked and replayed Billy into 1-3's first pit forever).
    obs = start
    advanced = False
    for step in plan:
        remaining = step.frames
        while remaining > 0 and not advanced:
            n = min(16, remaining)
            session.send_plan([Step(n, step.buttons)])
            remaining -= n
            obs = observe()
            if obs.dead:
                return False, obs.progress, obs.progress, obs.elevation
            advanced = game.search_area_advance(start_level, obs.level_key)
        if advanced:
            break   # verified up to the crossing; the tail is the NEXT area's problem
    reached = obs.progress
    detour = obs.progress < base_x
    coasted = 0
    while coasted < settle and not obs.dead and not advanced:
        coast = reflex.advance_plan(obs)
        session.send_plan(coast)
        obs = observe()
        advanced = game.search_area_advance(start_level, obs.level_key)
        reached = max(reached, obs.progress)
        coasted += max(1, plan_frames(coast))
        if coasted >= config.SEARCH_HORIZON_FRAMES and (
                not detour or reached > base_x + min_progress):
            break
    if advanced:
        reached = base_x + _TRANSITION_BONUS
    return (not obs.dead), reached, obs.progress, obs.elevation


class Director:
    def __init__(self, game: Game, kb: KnowledgeBase, use_llm: bool = True,
                 cache: SolutionCache | None = None, skills: SkillLibrary | None = None,
                 tapes: TapeLibrary | None = None, sections=None, guide=None) -> None:
        self.game = game
        self.hooks: HazardHooks = game.hazard_hooks()
        self.session = game.system.connect()
        self.controller = game.system.controller
        self.reflex = game.make_reflex()
        self.kb = kb
        gid = str(getattr(game, "cli_name", "") or "smb")
        self.cache = cache if cache is not None else SolutionCache(game_id=gid)
        self.tapes = tapes if tapes is not None else TapeLibrary()
        self.routes = RouteGraph()   # discovered level topology (clears/warps/screens)
        # Reads the route graph to plan toward game completion (prefers discovered warps). The
        # game may supply a progress rank for its own topology (SMB world/stage is the default).
        from .strategist import RouteStrategist
        rank = getattr(game, "route_rank", None)
        self.strategist = RouteStrategist(self.routes, rank=rank) if rank else RouteStrategist(self.routes)
        self.skills = skills if skills is not None else SkillLibrary()  # cross-game transferable tactics
        # Optional ingested walkthrough (knowledge/guide.py): seeds search candidates and
        # informs the LLM prompt. Advice, not authority — everything it suggests is verified.
        self.guide = guide
        self._tape_record: list = []
        self._tape_key: tuple = ()
        self._tape_replay: list | None = None
        self._tape_mode = False
        self._evolver = None                # lazy TapeEvolver for reactive (tape-evolving) games
        self._learned_buckets: set = set()
        self._reachback_miss: set = set()   # per-attempt: buckets where reachback verify failed
        self._reachback_reported: set = set()   # per-session: spots already reported (quiet logs)
        self._taught_demo_used: set = set()  # per-attempt: BC demo start-x already warped to
        self._last_pit_death_x: int = 0
        # Optional hazard-scoped RL sub-policies: at a registered section they SEED micro-search with
        # a learned crossing candidate (verified+banked like any solution). None = pure reflex/search.
        self.sections = sections
        # Optional parallel micro-search: N emulator workers evaluate candidates concurrently
        # (BILLY_PARALLEL_SEARCH=<n>). Serial path is the default and the regression baseline.
        self.pool = None
        if config.PARALLEL_SEARCH > 0:
            from .search_pool import SearchPool
            self.pool = SearchPool(game, config.PARALLEL_SEARCH)
        self.use_llm = use_llm
        # Eval mode: end each attempt at the FIRST level clear so every attempt is a fresh run of
        # the same starting level. This exposes the compounding curve (search-per-clear falls and
        # clear-time drops as the cache fills) instead of the checkpoint marching forward.
        self.repeat_level = os.environ.get("BILLY_REPEAT_LEVEL", "0") == "1"
        self.recent: deque[str] = deque(maxlen=12)
        self.memory = RollingGameMemory() if use_llm else None
        self.commentator = Commentator()
        self.best_score = 0
        self.fastest_clear_frames: int | None = None
        self.cur_level: tuple = ()
        self._prev_best_x = 0   # furthest x reached so far (for frames-to-frontier metric)
        self.ledger = LearningLedger()
        self.stuck = StuckTracker()

    def _bank_solution(self, lk, x: int, plan: Plan, reach: int, *, y: int = 0,
                       force: bool = False, source: str = "search",
                       summary: str = "", level_label: str = "") -> None:
        prev = self.cache.get(lk, x, y)
        prev_reach = prev.reach_after if prev else -1
        self.cache.put(lk, x, plan, reach, y=y, force=force)
        entry = self.cache.get(lk, x, y)
        if entry and (prev is None or entry.reach_after > prev_reach):
            self.ledger.bank(lk, x, reach, source)
            # Cross-game exponential: a SIGNIFICANT banked maneuver also becomes a transferable
            # sequence Skill (seeds search at similar situations; never blind-replayed). Callers
            # that can't provide the plan-start summary simply skip distillation. Transition-bonus
            # reaches are excluded (the +100k px "gain" is an area-advance sentinel, not distance).
            if (config.DISTILL and summary and reach - x < self._TRANSITION_BONUS):
                from .knowledge.distill import distill_solution
                if distill_solution(self.skills, summary=summary,
                                    level_label=level_label or str(lk), plan=plan,
                                    start_x=x, reach=reach, source=source,
                                    console=getattr(self.game.system, "name", "nes")):
                    print(f"  [skill] 🧬 distilled {level_label}@{x} (+{reach - x}px) "
                          f"→ transferable tactic")

    def _drop_solution(self, lk, x: int, *, y: int = 0, reason: str = "fail") -> None:
        self.cache.record_fail(lk, x, y)
        self.ledger.drop(lk, x, reason)

    # --- lock-step helper ---------------------------------------------------------------
    def _observe(self) -> Observation:
        st = self.session.read_state()
        return self.game.observe(st.frame, st.ram, getattr(st, "rgb", None))

    def _game_id(self) -> str:
        """Stable per-game id used to scope lessons/checkpoints (cli_name, else game name)."""
        return str(getattr(self.game, "cli_name", "") or self.game.name)

    # --- micro-search: evaluate candidates on a CLONE so the live run never visibly rewinds ----
    _TRANSITION_BONUS = _TRANSITION_BONUS   # class alias (module constant is the source of truth)

    def _rollout(self, plan: Plan, settle: int = config.SEARCH_HORIZON_FRAMES) -> tuple[bool, int]:
        """Run the candidate, THEN coast forward `settle` frames and watch for a delayed death.

        `settle` is a budget for the coast AFTER the candidate (not a total that the candidate's own
        frames consume) — critical, because the killer case is a hazard just past where the candidate
        lands (e.g. a Goomba 15px beyond a jump's landing). Coasting forward with reflex.advance_plan
        after every candidate guarantees that imminent death is simulated instead of being declared a
        false 'survivor'. Returns (survived, farthest).

        A candidate that ADVANCES the level_key — most importantly entering a pipe, which warps Mario
        to a new area where x resets to a low value — would otherwise look like NO progress (x didn't
        grow). We detect the advance and report a dominating `reached` so the search prefers and banks
        it; this is what lets Billy take 1-2's mandatory exit pipe instead of running at it forever.

        A candidate that ended BEHIND its start is a detour (retreat/drop) that needs the long
        horizon to recover; forward candidates cap at SEARCH_HORIZON (a forward stall stays a
        stall). Body lives in module-level `rollout_candidate` so the parallel search workers run
        exactly this code."""
        return rollout_candidate(self.session, self._observe, self.reflex, self.game,
                                 plan, settle, min_progress=self._MIN_PROGRESS_PX)

    _MIN_PROGRESS_PX = _MIN_PROGRESS_PX   # class alias (module constant is the source of truth)

    def _micro_search(self, candidates: list[Plan], start_x: int, early_exit: bool = False,
                      settle: int = config.SEARCH_HORIZON_FRAMES,
                      level_key: tuple = ()) -> tuple[Plan, bool, int]:
        """Try each candidate on a cloned state; return (best_plan, made_progress, reach_after).

        Runs inside the session's search_mode + a state clone, so none of the candidate frames
        are displayed and the live game is left exactly where it started (invisible search). A
        candidate only counts as a real escape if it SURVIVES *and* moves forward — a plan that
        merely avoids death without advancing is a stall and must not be cached/replayed.

        `settle` is the post-candidate coast budget. The expanded fallback passes a LONGER one so a
        detour candidate (retreat left, drop off a dead-end ledge) can recover and show NET forward
        progress on the path it reaches — a short horizon would only see the leftward dip and reject
        it, which is why Billy couldn't escape 1-2's dead-end upper ledge.

        With `early_exit`, return the FIRST surviving+advancing candidate instead of scanning all
        of them. Used for the dense expanded grid (~40 candidates): once one works we don't need the
        best, just a survivor to bank — this keeps the brute-force fallback's cost bounded."""
        snap = self.session.clone_state()
        best_plan, best_score, best_reach = candidates[0], -10 ** 9, start_x
        # Parallel path: ship the clone + candidates to the worker pool (same rollout code);
        # scoring below is identical either way. None => serial (pool off/too few/failed).
        pooled = (self.pool.evaluate(snap, candidates, settle, self._MIN_PROGRESS_PX)
                  if self.pool is not None else None)
        with self.session.search_mode():
            for i, plan in enumerate(candidates):
                if pooled is not None:
                    survived, reached, end_x, end_y = pooled[i]
                else:
                    survived, reached, end_x, end_y = self._rollout(plan, settle)
                score = reached if survived else reached - 100_000  # death ≫ worse than short
                dead = bool(survived and level_key and self.cache.is_dead(level_key, end_x, end_y))
                if survived:
                    # ROUTE-AWARENESS: prefer the lower/grounded road on comparable x (a gentle
                    # gravity bias against gratuitous climbs into dead-ends / over low exits), and
                    # heavily avoid a candidate that lands on a node already proven to dead-end.
                    score += end_y * config.ELEVATION_TIEBREAK
                    if dead:
                        score -= config.ROUTE_DEAD_PENALTY
                if score > best_score:
                    best_score, best_plan, best_reach = score, plan, reached
                if pooled is None:
                    self.session.restore(snap)
                if early_exit and survived and not dead and reached > start_x + self._MIN_PROGRESS_PX:
                    break   # good-enough survivor found — stop the expensive grid here
        self.session.restore(snap)   # back to the live pre-search position, nothing shown
        self._observe()
        made_progress = best_score >= 0 and best_reach > start_x + self._MIN_PROGRESS_PX
        return best_plan, made_progress, best_reach

    _SNAP_CHUNK = 6   # commit live plans in chunks this many frames long (dense snapshot trail)

    def _commit(self, plan: Plan, safe_history, last_snap_x: int,
                *, record_tape: bool = True, interruptible: bool = False) -> tuple[Observation, int]:
        """Execute a committed (live) plan in small same-button chunks, appending a snapshot to the
        trail every ~tile of new progress. Splitting a Step into equal sub-steps of the same button
        mask is identical input frame-for-frame, so behaviour is unchanged — but the dense trail
        guarantees learn-from-death has a runway snapshot before the next death, even mid-jump.

        `interruptible` (routine + search; never a verified REPLAY whose banked trajectory must
        run to its end): stop early when an on-ground chunk crosses a bucket holding a cached
        solution that reaches further than we already are, so the decision loop can pick up a
        human Remix demo mid-approach. Without this cut, a weak search plan flies through the
        demo's key airborne and the taught line never fires."""
        obs = self._observe()
        start_key = obs.level_key
        start_bucket = bucket_of(obs.level_key, obs.progress, obs.elevation)
        start_progress = obs.progress
        executed: list[Step] = []
        for step in plan:
            remaining = step.frames
            while remaining > 0:
                chunk = min(self.hooks.commit_chunk_size(obs, self._SNAP_CHUNK), remaining)
                self.session.send_plan([Step(chunk, step.buttons)])
                remaining -= chunk
                executed.append(Step(chunk, step.buttons))
                obs = self._observe()
                if obs.dead:
                    return obs, last_snap_x
                if obs.level_key != start_key:
                    # Level/area boundary crossed mid-plan. The tail past this point was never
                    # verified (rollouts stop scoring at a transition) — do NOT replay it blind
                    # into the new area. Return so the main loop handles the clear/screen change
                    # (checkpoint, tape hand-off, fresh decisions). The executed prefix IS the
                    # committed input, so it still extends the old level's tape.
                    if record_tape and executed:
                        append_plan(self._tape_record, executed)
                    return obs, last_snap_x
                if (interruptible and getattr(obs.raw, "on_ground", True)
                        and bucket_of(obs.level_key, obs.progress, obs.elevation) != start_bucket):
                    hit = (self.cache.get(obs.level_key, obs.progress, obs.elevation)
                           or self.cache.best_in_bucket(obs.level_key, obs.progress))
                    if hit is not None and hit.reach_after > max(start_progress, obs.progress) + self._MIN_PROGRESS_PX:
                        if record_tape and executed:
                            append_plan(self._tape_record, executed)
                        return obs, last_snap_x
                # Only snapshot ON-GROUND states: a cached solution keyed here is then replayed
                # from the SAME reproducible state next pass (an airborne snapshot wouldn't match
                # Mario's exact mid-jump state on the next approach, so its replay would drift).
                if (obs.progress >= last_snap_x + config.CACHE_BUCKET_PX
                        and getattr(obs.raw, "on_ground", True)):
                    snap = self.session.clone_state()
                    safe_history.append((obs.progress, obs.elevation, obs.level_key, snap))
                    band = self.hooks.approach_snapshot_band(obs)
                    if band and band[0] <= obs.progress <= band[1]:
                        self._approach_trail.append(
                            (obs.progress, obs.elevation, obs.level_key, snap))
                    last_snap_x = obs.progress
        if record_tape and executed:
            append_plan(self._tape_record, executed)
        return obs, last_snap_x

    def _verify(self, plan: Plan, horizon: int = config.LEARN_HORIZON_FRAMES) -> bool:
        """Does this plan still survive from the CURRENT live state? Checked on a clone (invisible).
        Because the clone is the exact live state, a plan that survives here will reproduce
        identically when committed — so a verified replay is deterministic. A cached solution that
        no longer survives (e.g. a moving enemy has shifted phase since it was learned) fails here,
        and the caller live-searches fresh with the enemy where it ACTUALLY is now."""
        snap = self.session.clone_state()
        with self.session.search_mode():
            lived, *_ = self._rollout(plan, horizon)
        self.session.restore(snap)
        self._observe()
        return lived

    def _evaluate(self, plan: Plan, settle: int = config.LEARN_HORIZON_FRAMES) -> tuple[bool, int]:
        """Run a plan on a clone of the CURRENT live state and report (survived, reach) — for
        scoring an LLM-improvised plan before trusting/caching it. Invisible; leaves live untouched."""
        snap = self.session.clone_state()
        with self.session.search_mode():
            survived, reach, *_ = self._rollout(plan, settle)
        self.session.restore(snap)
        self._observe()
        return survived, reach

    def _cached_at(self, obs: Observation, *, on_ground: bool, in_special: bool):
        """Cache lookup; stale entries count as a miss (reflex owns the spot)."""
        if not (on_ground or in_special):
            return None
        cached = self.cache.get(obs.level_key, obs.progress, obs.elevation)
        # Exact y-band miss: a Remix demo banked at yband 6 is invisible when Mario lands on
        # yband 5 at the same tile — surface the best entry in this progress bucket.
        if cached is None:
            cached = self.cache.best_in_bucket(obs.level_key, obs.progress)
        if cached is not None and self.hooks.stale_cache(obs, cached):
            return None
        if on_ground:
            # Inside a hazard section band, keep local keys — reachback would skip a taught
            # crossing keyed at x≈1040 while replaying a long chain from x≈640.
            in_section = (self.sections is not None
                          and self.sections._match(obs) is not None)
            if not in_section:
                # Reachback fires on a MISS and also on a WEAK hit: a local hop to 518 must not
                # shadow a human demo to 550 keyed a few tiles ahead/behind.
                better = self._reachback(obs, floor=cached.reach_after if cached else 0)
                if better is not None:
                    cached = better
        return cached

    def _reachback(self, obs: Observation, floor: int = 0):
        """Clone-verify a HIGH-reach entry banked near here (behind, same tile, or slightly ahead).

        The killer case: a human demo reaching past a wall sits at x≈361, but Billy's decision
        cadence lands at 294 with a weak search hop to 518 — the demo never fires because
        exact-key miss + REACHBACK_MIN_GAIN over the local floor rejects a +32 improvement.
        Reachback finds nearby high-reach entries and VERIFIES from the live state; only a
        proven survive+advance is replayed. Failed verifies are blacklisted per attempt."""
        if obs.progress < 16:
            return None
        bkey = bucket_of(obs.level_key, obs.progress, obs.elevation)
        if bkey in self._reachback_miss:
            return None
        # When we already have a local hit, any strictly better reach is worth verifying
        # (human demos often only beat a thrashing search by tens of px, not 200+).
        # On a pure miss, require the usual min-gain from current progress.
        if floor > obs.progress:
            min_gain = self._MIN_PROGRESS_PX
            cand = self.cache.nearby_reaching(obs.level_key, obs.progress, min_gain=min_gain)
            if cand is None or cand.reach_after <= floor:
                return None
        else:
            cand = self.cache.nearby_reaching(obs.level_key, obs.progress,
                                              min_gain=config.REACHBACK_MIN_GAIN)
            if cand is None:
                return None
        survived, reach = self._evaluate(cand.plan, settle=config.SEARCH_HORIZON_FRAMES)
        need = obs.progress + max(self._MIN_PROGRESS_PX, config.REACHBACK_MIN_GAIN // 4)
        if survived and reach > max(need, floor):
            print(f'  🎯 reachback: verified a banked long solution from '
                  f'{obs.level_label}@{obs.progress} (reach {reach}) — replaying')
            return cand
        if bkey not in self._reachback_reported:   # once per spot per session, not per attempt
            print(f'  🎯 reachback: candidate (banked reach {cand.reach_after}) failed verify '
                  f'from {obs.level_label}@{obs.progress} '
                  f'({"died" if not survived else f"stalled at {reach}"}) — blacklisted here')
            self._reachback_reported.add(bkey)
        self._reachback_miss.add(bkey)
        return None

    def _candidates(self, obs: Observation) -> list[Plan]:
        """Search candidate set on a cache MISS: banked nearby plans FIRST (Remix demos), then
        the reflex spread + transferable Skills. Skills/demos only widen the *search* set
        (never blind-replay) — micro-search still verifies on a clone before banking."""
        cands: list[Plan] = []
        # Prefer human/taught lines near this progress — try before cold reflex candidates.
        cands.extend(self.cache.nearby_plans(
            obs.level_key, obs.progress, min_gain=self._MIN_PROGRESS_PX))
        cands.extend(self.reflex.danger_candidates(obs))
        cands.extend(self.hooks.extra_candidates(obs))
        profile = getattr(self.reflex, "p", None)
        if profile is not None and len(self.skills):
            cands.extend(self.skills.candidates(
                obs.raw, profile, obs.summary,
                console=getattr(self.game.system, "name", "")))
        if self.guide is not None:
            cands.extend(self.guide.direction_candidates(self.game.guide_query(obs),
                                                         self.controller))
        return cands

    def _learn_from_death(self, safe_history, death_x: int,
                          level_label: str = "", level_key: tuple = ()) -> int | None:
        """After a death, look back through recent snapshots for one with enough RUNWAY before the
        death spot, then search there for a sequence that gets PAST it and cache it keyed to that
        spot. This is what advances the frontier: next attempt replays the survivor instead of
        walking into the same death. Runs on a clone in search_mode (invisible). Returns the x it
        learned to pass, or None. Trying several start points gives a stomp/clear room to set up."""
        special_death = self.hooks.is_special_death(level_label, death_x)
        learn_horizon = (self.hooks.learn_horizon_frames(level_label, death_x)
                         or config.LEARN_HORIZON_FRAMES)

        starts: list = []
        short_runway: list = []
        for x, y, lk, snap in reversed(safe_history):   # nearest the death first
            if level_key and lk[:2] != level_key[:2]:
                continue                                # same x on another level is not this death
            runway = death_x - x
            if runway <= 0:
                continue
            if runway < config.MIN_RUNWAY_PX:
                short_runway.append((x, y, lk, snap))   # too close to set up — last resort only
                continue
            if runway > learn_horizon:
                action = self.hooks.learn_runway_action(
                    level_label, death_x, runway, config.LEARN_HORIZON_FRAMES)
                if action == "continue":
                    continue
                break                                   # further back is out of rollout reach
            starts.append((x, y, lk, snap))
        if not starts:
            # No snapshot with proper runway (airborne arcs leave holes in the trail right
            # before a hazard). A tight-runway start costs only rollouts — better a cramped
            # search than learning nothing and walking into the same death forever.
            starts = short_runway[:1]

        for x, y, lk, snap in starts:
            runway = death_x - x
            bkey = bucket_of(lk, x, y)
            if bkey in self._learned_buckets:
                continue

            # Lift gap: section RL → frame timing → macro (each banks only if cacheable).
            if special_death:
                self.session.restore(snap)
                snap_obs = self._observe()
                if self.hooks.in_special_zone(snap_obs) and self.sections is not None:
                    seg = self.sections.cross(snap_obs, self.session, self._observe)
                    if seg is not None:
                        plan, reach = seg
                        if self.hooks.learn_cacheable(level_label, death_x, reach):
                            self._bank_solution(lk, x, plan, reach, y=y, source="learn_section",
                                                summary=snap_obs.summary,
                                                level_label=snap_obs.level_label)
                            self._learned_buckets.add(bkey)
                            return x
                pit_plan, pit_reach = self.hooks.try_pit_search(
                    self.session, self._observe, snap_obs, death_x=death_x,
                    min_gain=self._MIN_PROGRESS_PX)
                if pit_plan is not None and self.hooks.learn_cacheable(
                        level_label, death_x, pit_reach):
                    self._bank_solution(lk, x, pit_plan, pit_reach, y=y, source="learn_pit",
                                        summary=snap_obs.summary,
                                        level_label=snap_obs.level_label)
                    self._learned_buckets.add(bkey)
                    return x
                frame_plan, frame_reach, _ = self.hooks.try_frame_search(
                    self.session, self._observe, snap_obs, deep=True,
                    min_gain=self._MIN_PROGRESS_PX)
                if frame_plan is not None and self.hooks.learn_cacheable(
                        level_label, death_x, frame_reach):
                    self._bank_solution(lk, x, frame_plan, frame_reach, y=y,
                                        source="learn_bootstrap",
                                        summary=snap_obs.summary,
                                        level_label=snap_obs.level_label)
                    self._learned_buckets.add(bkey)
                    return x

            # Focused spread first; if nothing survives past the death, fall back to the dense grid
            # (the same brute-force set that cracks low-ceiling/enemy-ledge walls the spread misses).
            cand_sets = [self._candidates_from(snap)]
            expand = getattr(self.reflex, "expanded_candidates", None)
            if expand is not None and config.EXPANDED_FALLBACK:
                self.session.restore(snap)
                cand_sets.append(expand(self._observe()))
            best_plan, best_reach = None, x
            with self.session.search_mode():
                for candidates in cand_sets:
                    for plan in candidates:
                        self.session.restore(snap)
                        self._observe()
                        lived, reached, *_ = self._rollout(plan, settle=learn_horizon)
                        if not lived or reached <= death_x or reached <= best_reach:
                            continue
                        if not self.hooks.learn_cacheable(level_label, death_x, reached):
                            continue
                        best_plan, best_reach = plan, reached
                    if best_plan is not None:
                        break   # cracked it with this set — don't pay for the next (denser) one
            self.session.restore(snap)
            snap_here = self._observe()   # the plan-start situation (for skill distillation)
            if best_plan is not None:
                self._bank_solution(lk, x, best_plan, best_reach, y=y, source="learn_macro",
                                    summary=snap_here.summary,
                                    level_label=snap_here.level_label)
                self._learned_buckets.add(bkey)
                return x
        return None

    def _candidates_from(self, snap) -> list[Plan]:
        """Candidate escapes generated from a snapshot's observation (restores it to read state)."""
        self.session.restore(snap)
        return self._candidates(self._observe())

    _TAKEOVER_MAX_FRAMES = 9000   # ~150s cap on one live human segment

    def _human_takeover(self, obs: Observation, safe_history,
                        attempt: int) -> tuple[Observation, int]:
        """Live demo: the human pressed T in the watch window — hand them the controller from
        the EXACT live state, record their input, and on hand-back (ENTER/T) bank the segment
        if it survived and advanced. The live run IS the verification: the input just executed
        on the real deterministic state, so survive+advance needs no re-simulation. A demo that
        crosses a SCREEN boundary (Zelda border, SMB pipe) banks that segment and keeps the
        human driving — only a true level clear hands back automatically. ESC rewinds to the
        current segment's start and discards it (earlier segments stay banked); a death falls
        through to learn-from-death as usual.
        """
        from .teleop import TeleopRecorder

        session = self.session
        if not session.ensure_viewer():
            return obs, 0
        start_state = session.clone_state()
        start = obs
        if getattr(obs.raw, "on_ground", True):
            safe_history.append((obs.progress, obs.elevation, obs.level_key, start_state))
        print(f'  [attempt {attempt}] 🎮→🧑 HUMAN TAKEOVER at {obs.level_label}@{obs.progress} '
              f'— play through it; ENTER hands back (banks if it advanced), ESC discards')
        session.set_overlay(["YOU HAVE THE CONTROLLER",
                             "play through the tough spot",
                             "ENTER = hand back & bank   ESC = discard & rewind"])
        session.teleop_reset()
        rec = TeleopRecorder()
        frames = 0
        aborted = False
        cur = obs
        seg_hi = start.progress
        while frames < self._TAKEOVER_MAX_FRAMES:
            mask, fin, ab = session.teleop_poll()
            if fin or session.takeover_requested():
                break
            if ab:
                aborted = True
                break
            session.teleop_step(mask)
            rec.record(mask, 1)
            frames += 1
            cur = self._observe()
            if cur.dead:
                break
            if self.game.level_cleared(start.level_key, cur.level_key):
                break   # cleared into the next level — hand back at the natural boundary
            if self.game.screen_changed(start.level_key, cur.level_key):
                # New screen/area mid-demo (Zelda border, SMB pipe). Bank the finished segment,
                # roll the tape over to the new screen, and KEEP the human driving — handing
                # back here would drop Billy exactly at the hazard the demo is teaching (a
                # Zelda demo could otherwise never span the screen with the fight on it).
                self._bank_takeover_segment(start, rec.plan(), crossed=True,
                                            end_progress=seg_hi, attempt=attempt)
                self._tape_finish_level(seg_hi, cleared=True)
                self._tape_key = cur.level_key
                self._tape_record = []
                # Demo-discovered topology counts: the human just proved this transition.
                self.routes.record(start.level_key, cur.level_key, "screen",
                                   at=seg_hi, dst_label=cur.level_label)
                start, start_state = cur, session.clone_state()
                rec = TeleopRecorder()
                seg_hi = cur.progress
                continue
            seg_hi = max(seg_hi, cur.progress)
        session.set_overlay(None)
        session.teleop_reset()
        plan = rec.plan()

        if aborted:
            session.restore(start_state)
            cur = self._observe()
            print(f'  [attempt {attempt}]   takeover discarded — rewound to '
                  f'{cur.level_label}@{cur.progress}, Billy resumes')
            return cur, frames
        if cur.dead:
            print(f'  [attempt {attempt}]   human segment died at {cur.progress} — '
                  f'nothing banked, Billy learns from it as usual')
            return cur, frames

        crossed = cur.level_key != start.level_key
        self._bank_takeover_segment(start, plan, crossed=crossed,
                                    end_progress=seg_hi if crossed else cur.progress,
                                    attempt=attempt)
        if getattr(cur.raw, "on_ground", True):
            safe_history.append((cur.progress, cur.elevation, cur.level_key,
                                 session.clone_state()))
        return cur, frames

    def _bank_takeover_segment(self, seg_start: Observation, plan: Plan, *, crossed: bool,
                               end_progress: int, attempt: int) -> None:
        """Bank one finished human segment. The live run IS the verification: this input just
        executed on the real deterministic state, so survive+advance needs no re-simulation."""
        if plan:
            # Keep the level tape contiguous: the human's frames ARE committed live input.
            append_plan(self._tape_record, plan)
        gained = end_progress - seg_start.progress
        if plan and (crossed or gained > self._MIN_PROGRESS_PX):
            reach = (seg_start.progress + self._TRANSITION_BONUS if crossed
                     else end_progress)
            self._bank_solution(seg_start.level_key, seg_start.progress, plan, reach,
                                y=seg_start.elevation, force=True, source="demo_live",
                                summary=seg_start.summary, level_label=seg_start.level_label)
            print(f'  [attempt {attempt}] 🧑✓ human segment BANKED '
                  f'{seg_start.level_label}@{seg_start.progress} (+{gained}px'
                  f'{", crossed" if crossed else ""}) — replays forever')
        else:
            print(f'  [attempt {attempt}]   human segment made no progress '
                  f'(+{gained}px) — nothing banked')

    def _tape_begin_level(self, obs: Observation) -> Observation:
        """Start recording a level trajectory; try a verified tape replay if one exists.

        Returns the current observation — which DIFFERS from the argument when an anchored
        tape restored its entry savestate, so callers must use the return value."""
        self._tape_key = obs.level_key
        self._tape_record = []
        self._tape_mode = False
        self._tape_replay = None
        entry = self.tapes.get(obs.level_key)
        if entry:
            # Entry-state anchor: a whole-level tape only reproduces a MOVING hazard (1-3's
            # lift) if it starts from the exact state that set the hazard's phase. Restore that
            # savestate before verify/replay — it's the level start, so the snap is imperceptible
            # (Mario is at the entry either way) and it makes the lift deterministic.
            if entry.entry_state is not None:
                self.session.restore(entry.entry_state)
                self.session.save_state(0)   # respawns this life align to the anchored entry
                obs = self._observe()
                self._tape_key = obs.level_key
            if self._verify_tape(entry.plan, obs.level_key, entry.frontier,
                                 expect_clear=entry.clears_level):
                self._tape_replay = [Step(s.frames, s.buttons) for s in entry.plan]
                self._tape_mode = True
                self.tapes.record_hit(obs.level_key)
            elif entry.entry_state is None:
                # A tape that keeps failing verify no longer matches reality — drop it after
                # FAIL_LIMIT misses so an honest new recording can take its slot (corrupt or
                # drifted tapes with inflated frontiers must not squat forever). An ANCHORED tape
                # is never dropped on a verify miss: it replays from its own saved entry, so a
                # miss means the live approach differed, not that the tape is wrong.
                self.tapes.record_fail(obs.level_key)
        # Remember the entry state so a self-recorded clear of this level anchors future replays.
        self._tape_entry_state = self.session.clone_state()
        return obs

    def _log_objective(self, obs: Observation) -> None:
        """Print the strategist's current objective when entering a level (once per entry).
        A WARP objective means the known map has a discovered skip-ahead from here."""
        try:
            objective = self.strategist.objective(obs.level_key, obs.level_label)
        except Exception:
            return   # strategy is advice; never let it break the run
        if objective.via_warp:
            print(f"  [director] {objective.line()}")

    def _verify_tape(self, plan: Plan, level_key: tuple, min_frontier: int,
                     *, expect_clear: bool = True) -> bool:
        """Clone-check a stored tape before zero-search replay."""
        if not plan:
            return False
        snap = self.session.clone_state()
        start_level = level_key[:2]
        start_full = tuple(level_key)

        def advanced(obs: Observation) -> bool:
            # A tape "succeeds by transition" on a LEVEL clear or an in-level AREA/screen change
            # (pipe warp, Zelda screen cross) — screen-segment tapes end exactly at the boundary.
            return (obs.level_key[:2] > start_level
                    or self.game.search_area_advance(start_full, obs.level_key))

        with self.session.search_mode():
            self.session.send_plan(plan)
            obs = self._observe()
            coasted = 0
            # The idle coast exists to observe the clear/warp transition fire; a PARTIAL tape ends
            # mid-level at a frontier, where idling next to a hazard could kill an otherwise good
            # tape — its gate is position-only, so skip the coast.
            while expect_clear and coasted < 180 and not obs.dead and not advanced(obs):
                self.session.send_plan([Step(4, 0)])
                obs = self._observe()
                coasted += 4
        ok = (not obs.dead
              and (advanced(obs)
                   or obs.progress >= min_frontier - config.CACHE_BUCKET_PX))
        self.session.restore(snap)
        self._observe()
        return ok

    def _tape_consume(self) -> Plan | None:
        """Next committed chunk from the active level tape (search-free replay)."""
        if not self._tape_replay:
            self._tape_mode = False
            return None
        chunk_frames = 0
        chunk: list[Step] = []
        while self._tape_replay and chunk_frames < 24:
            step = self._tape_replay[0]
            take = min(step.frames, 24 - chunk_frames)
            chunk.append(Step(take, step.buttons))
            chunk_frames += take
            if take >= step.frames:
                self._tape_replay.pop(0)
            else:
                self._tape_replay[0] = Step(step.frames - take, step.buttons)
        return chunk or None

    def _try_taught_demo(self, obs: Observation) -> Plan | None:
        """Handoff a Remix/teleop BC demo: warp to its entry savestate and return its plan.

        Mid-level human demos cannot be position-cache replayed when moving hazards have
        drifted phase — the plan only reproduces from the exact taught state (same carrier
        as entry-anchored tapes). When Billy is on-ground near a banked demo's start x, we
        restore that state (one deterministic warp) and commit the verified plan. One use
        per demo start-x per attempt so a failure cannot loop."""
        if not getattr(obs.raw, "on_ground", True) or obs.progress < 16:
            return None
        game_id = self._game_id()
        demo_dir = config.DATA_DIR / "rl" / "demos" / game_id
        if not demo_dir.is_dir():
            return None
        slug = str(obs.level_label).replace("-", "_")
        # Prefer the nearest demo whose start is within approach range of live progress.
        best: tuple[int, Path, Path, int] | None = None  # (|dx|, demo, state, demo_x)
        for demo_path in demo_dir.glob(f"{slug}_x*.demo.json"):
            stem = demo_path.name.replace(".demo.json", "")
            try:
                demo_x = int(stem.rsplit("_x", 1)[1])
            except (IndexError, ValueError):
                continue
            state_path = demo_path.with_name(stem + ".state")
            if not state_path.is_file():
                continue
            if demo_x in self._taught_demo_used:
                continue
            # Window: a bit before the teach point through slightly past it.
            if not (demo_x - 100 <= obs.progress <= demo_x + 32):
                continue
            dx = abs(obs.progress - demo_x)
            if best is None or dx < best[0]:
                best = (dx, demo_path, state_path, demo_x)
        if best is None:
            return None
        _, demo_path, state_path, demo_x = best
        try:
            import json as _json
            plan = [Step(int(f), int(b)) for f, b in _json.loads(demo_path.read_text())["steps"]]
            start_state = state_path.read_bytes()
        except (OSError, KeyError, TypeError, ValueError):
            return None
        if not plan:
            return None
        from .teleop import verify_demo
        # Verify from the taught state (not live) — phase-accurate gate.
        result = verify_demo(self.session, self.game, start_state, plan,
                             min_progress=self._MIN_PROGRESS_PX)
        if not result.bankable:
            self._taught_demo_used.add(demo_x)  # don't thrash a broken seed
            return None
        # LIVE warp to the taught moment, then the main loop commits the plan.
        self.session.restore(start_state)
        self._taught_demo_used.add(demo_x)
        print(f'  🎬 taught-demo: {obs.level_label}@x{demo_x} — warping to your line '
              f'and replaying (reach {result.end_progress})')
        return plan

    def _tape_finish_level(self, frontier: int, *, cleared: bool,
                           anchor: bool = False) -> None:
        """Persist a recorded trajectory when a level ends well.

        `anchor` (only at a true world/stage clear) stores the entry savestate with the tape so
        its replay reproduces a moving hazard (the lift) deterministically. Screen/area-change
        tapes pass anchor=False — restoring their entry mid-level would snap the player."""
        if not self._tape_record:
            return
        entry_state = getattr(self, "_tape_entry_state", None) if (cleared and anchor) else None
        self.tapes.put(self._tape_key, self._tape_record, frontier,
                       clears_level=cleared, entry_state=entry_state)
        self._tape_record = []

    # --- boot ---------------------------------------------------------------------------
    def boot(self) -> Observation:
        start = self.game.boot(self.session)   # game reaches a playable state, returns start obs
        self.session.save_state(0)             # checkpoint the level start
        self._observe()                        # consume the post-save republished frame
        self.cur_level = start.level_key
        print(f"[director] in play at {start.level_label}, progress={start.progress}. "
              f"Billy has taken the controller.")
        if getattr(self.session, "_viewer", None) is not None:
            print("[director] watching? Press T in the game window anytime to TAKE THE "
                  "CONTROLLER — your segment banks if it advances (ENTER hands back, "
                  "ESC discards).")
        print(f'  🎤 Billy: "{self.commentator.event_line("start")}"')
        from .knowledge.demo_seed import seed_demos
        n = seed_demos(self._game_id(), self.cache, self.game, session=self.session)
        if n:
            print(f"[director] seeded {n} verified demo(s) into {self._game_id()} cache")
        return start

    # --- cross-session frontier: persist the furthest level-start checkpoint -------------
    def _checkpoint_ready(self, obs: Observation) -> bool:
        """On solid ground near the level start. The window is wide enough to catch a
        PIPE-entered level (Mario emerges at x≈110+ and the first commit can land past 120 —
        1-3 was never checkpointed, so every death respawned levels back) while staying near
        the start so tapes still verify from the respawn state."""
        return getattr(obs.raw, "on_ground", True) and 16 < obs.progress < 240

    def _checkpoint_now(self) -> Observation:
        """Save slot 0 (the respawn/attempt start) here and persist the cross-session copy."""
        self.session.save_state(0)
        obs = self._observe()
        print(f"  [director] checkpoint at {obs.level_label} ({obs.progress})")
        self._persist_checkpoint(obs)
        return obs

    def _checkpoint_paths(self) -> tuple:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self._game_id())
        base = config.CHECKPOINTS_DIR / safe
        return base, base / "furthest.json"

    def _persist_checkpoint(self, obs: Observation) -> None:
        """Save the furthest level-start to disk so the NEXT SESSION can resume the march
        toward game completion instead of replaying every solved level from the top.
        Only ordinal level keys ratchet (SMB world/stage); non-comparable keys are skipped."""
        import json
        base, meta_path = self._checkpoint_paths()
        try:
            key = list(obs.level_key)
            if meta_path.exists():
                prev = json.loads(meta_path.read_text())
                try:
                    if not key > list(prev.get("key", [])):
                        return   # not beyond the recorded frontier
                except TypeError:
                    return       # mixed/non-ordinal keys — no meaningful "furthest"
            base.mkdir(parents=True, exist_ok=True)
            state_path = base / f"{obs.level_label.replace('-', '_').replace(' ', '_')}.state"
            state_path.write_bytes(self.session.clone_state())
            meta_path.write_text(json.dumps({"key": key, "label": obs.level_label,
                                             "state": str(state_path),
                                             "progress": obs.progress}))
            print(f"  [director] 🚩 frontier checkpoint saved: {obs.level_label} "
                  f"(next session can --resume here)")
        except Exception as e:   # persistence must never kill the run
            print(f"  [director] checkpoint persist failed: {e}")

    def resume_from_checkpoint(self) -> str | None:
        """Load the persisted furthest level-start into slot 0 (call after boot). Returns the
        resumed level label, or None when there is nothing to resume."""
        import json
        _, meta_path = self._checkpoint_paths()
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
            snap = Path(meta["state"]).read_bytes()
        except Exception as e:
            print(f"[director] resume checkpoint unreadable ({e}) — starting fresh")
            return None
        self.session.restore(snap)
        self.session.save_state(0)      # attempts (and respawns) now start here
        obs = self._observe()
        self.cur_level = obs.level_key
        print(f"[director] ⏩ resumed at the recorded frontier: {obs.level_label} "
              f"(progress {obs.progress})")
        return obs.level_label

    # --- one attempt (a continuous playthrough across levels) ---------------------------
    def capture_savestate_from(self, state_path: str, level_label: str, x_min: int, x_max: int,
                               out_path: str, max_frames: int = 200_000) -> bool:
        """Like capture_savestate but restores `state_path` instead of booting from slot 0."""
        with open(state_path, "rb") as f:
            self.session.reset()
            self.session.env.em.set_state(f.read())
            self.session._refresh_ram()
        obs = self._observe()
        self.reflex.reset(obs)
        self.commentator.reset(obs.raw)
        self.recent.clear()
        self.cur_level = obs.level_key
        return self._capture_loop(level_label, x_min, x_max, out_path, max_frames,
                                  respawn_path=state_path)

    def capture_savestate(self, level_label: str, x_min: int, x_max: int, out_path: str,
                          max_frames: int = 200_000) -> bool:
        """Run the full play loop until Mario is on-ground in [x_min, x_max] on `level_label`."""
        self.session.load_state(0)
        obs = self._observe()
        self.reflex.reset(obs)
        self.commentator.reset(obs.raw)
        self.recent.clear()
        self.cur_level = obs.level_key
        return self._capture_loop(level_label, x_min, x_max, out_path, max_frames)

    def _capture_loop(self, level_label: str, x_min: int, x_max: int, out_path: str,
                      max_frames: int, respawn_path: str | None = None) -> bool:
        obs = self._observe()
        cur_level = obs.level_key
        respawns = config.RESPAWNS_PER_ATTEMPT
        bucket_visits: dict = {}
        frames = 0
        while frames < max_frames:
            on_ground = getattr(obs.raw, "on_ground", True)
            from .games.smb.capture_util import save_snapshot
            if save_snapshot(self.session, self._observe, out_path,
                             x_min=x_min, x_max=x_max, level_label=level_label):
                return True

            if obs.dead:
                if respawns <= 0:
                    break
                respawns -= 1
                if respawn_path:
                    with open(respawn_path, "rb") as f:
                        self.session.reset()
                        self.session.env.em.set_state(f.read())
                        self.session._refresh_ram()
                else:
                    self.session.load_state(0)
                obs = self._observe()
                self.reflex.reset(obs)
                self.commentator.reset(obs.raw)
                continue

            if respawn_path is None and obs.level_key[:2] > cur_level[:2]:
                cur_level = obs.level_key
                self.reflex.note_level_advance(obs)
                self.commentator.reset(obs.raw)
                self.session.send_plan(_IDLE)
                obs = self._observe()
                frames += 2
                continue

            decision = self.reflex.step(obs)
            danger = decision.needs_billy and (
                "enemy" in decision.note or "pit" in decision.note or "stuck" in decision.note)
            lk, ey = obs.level_key, obs.elevation
            in_special = self.hooks.in_special_zone(obs)
            cached = self._cached_at(obs, on_ground=on_ground, in_special=in_special)

            if (cached is not None or danger or in_special) and obs.progress >= 16:
                if not self.hooks.stall_break_exempt(obs):
                    bkey = bucket_of(lk, obs.progress, ey)
                    bucket_visits[bkey] = bucket_visits.get(bkey, 0) + 1
                    if bucket_visits[bkey] > config.MAX_BUCKET_VISITS:
                        self.cache.mark_dead(lk, obs.progress, ey)
                        self.session.send_plan(decision.plan or _IDLE)
                        obs = self._observe()
                        frames += 2
                        continue

            sec_match = (self.sections._match(obs)
                         if (self.sections is not None and on_ground) else None)
            plan = None
            if cached is not None and on_ground:
                stale = (sec_match is not None
                         and cached.reach_after < sec_match[0].goal_x)
                if not stale:
                    plan = cached.plan
            if plan is None and (cached is not None or danger or in_special
                                 or sec_match is not None):
                seg = (self.sections.cross(obs, self.session, self._observe)
                       if (self.sections is not None and (on_ground or in_special))
                       else None)
                if seg is not None:
                    plan, _ = seg
                else:
                    best_plan, progressed, _reach = self._micro_search(
                        self._candidates(obs), obs.progress, level_key=lk)
                    expand = getattr(self.reflex, "expanded_candidates", None)
                    if not progressed and expand is not None and config.EXPANDED_FALLBACK:
                        best_plan, progressed, _reach = self._micro_search(
                            expand(obs), obs.progress, early_exit=True,
                            settle=config.SEARCH_HORIZON_FRAMES * 3, level_key=lk)
                    plan = best_plan
            if plan is None:
                plan = decision.plan

            self.session.send_plan(plan)
            obs = self._observe()
            frames += sum(s.frames for s in plan) if plan else 1

        print(f"[capture] miss — last {obs.level_label} x={obs.progress} "
              f"ground={getattr(obs.raw, 'on_ground', True)}")
        return False

    _EVOLVE_MAX_STEPS = 400   # cap a tape's length (frames = steps * reflex chunk)

    def _tape_fitness(self, anchor, tape: Plan) -> int:
        """Roll a whole tape from the boot anchor on a clone; return progress at death/end.
        Deterministic (same anchor + same inputs => same run), invisible, live untouched."""
        self.session.restore(anchor)
        obs = self._observe()
        with self.session.search_mode():
            for step in tape:
                self.session.send_plan([step])
                obs = self._observe()
                if obs.dead:
                    break
        fit = obs.progress
        self.session.restore(anchor)
        self._observe()
        return fit

    def _evolve_seed(self) -> list[Step]:
        """The tape to evolve FROM on the very first attempt: the SIMPLEST viable trajectory —
        the game's first move (always-fire for a shmup) held for a short stretch. Deliberately
        NOT the hand reflex: the point of evolution is to LEARN the reactive policy from search,
        not inherit it. From this dumb seed the hill-climb discovers the whole dodge."""
        moves = self.game.tape_moves()
        return [Step(self._SNAP_CHUNK, moves[0]) for _ in range(20)]

    def _run_attempt_evolve(self, n: int) -> metrics.AttemptResult:
        """Reactive-game attempt: EVOLVE a whole input tape (search over trajectories) instead
        of position-keyed local search. The best tape is banked and becomes next attempt's base,
        so survival/score compounds with no position key — the answer for a game whose death is
        a positioning problem local search can't reach (a shmup, a moving-enemy gauntlet)."""
        from .knowledge.tape_evolve import TapeEvolver

        t0 = time.monotonic()
        self.session.load_state(0)                      # boot anchor (deterministic origin)
        anchor = self.session.clone_state()
        obs = self._observe()
        level_key = obs.level_key
        if self._evolver is None:
            self._evolver = TapeEvolver(self.game.tape_moves(), slot=self._SNAP_CHUNK,
                                        max_steps=self._EVOLVE_MAX_STEPS)

        entry = self.tapes.get(level_key)
        replayed_prior = entry is not None and bool(entry.plan)
        base = ([Step(s.frames, s.buttons) for s in entry.plan] if replayed_prior
                else self._evolve_seed())
        base_fit = self._tape_fitness(anchor, base)

        rounds = int(os.environ.get("BILLY_EVOLVE_ROUNDS", "5"))
        mutants = int(os.environ.get("BILLY_EVOLVE_MUTANTS", "10"))
        best, best_fit, evals = self._evolver.evolve(
            base, lambda t: self._tape_fitness(anchor, t), rounds=rounds, mutants=mutants)

        improved = best_fit > base_fit or not replayed_prior
        if improved:
            self.tapes.put(level_key, best, best_fit, clears_level=False)

        # Commit the best tape LIVE (visible, real outcome) from the anchor.
        self.session.restore(anchor)
        obs = self._observe()
        self.commentator.reset(obs.raw)
        committed = 0
        for step in best:
            self.session.send_plan([step])
            committed += step.frames
            obs = self._observe()
            if obs.dead:
                break
        outcome = "game_over" if obs.dead else "timeout"
        gained = best_fit - base_fit
        print(f'  [attempt {n}] 🧬 evolved tape: survived {base_fit}→{best_fit} '
              f'(+{gained}) over {evals} rollouts — {"banked" if improved else "no gain"}')
        result = metrics.AttemptResult(
            attempt=n, outcome=outcome, max_x=best_fit, frames=max(committed, 1),
            billy_calls=0, world_stage=obs.level_label, levels_cleared=0, score=obs.score,
            fastest_clear_frames=0, duration_s=round(time.monotonic() - t0, 2),
            search_calls=evals, replay_calls=1 if replayed_prior else 0, tape_frames=committed,
            frontier_x=best_fit, frames_to_frontier=0,
            banks=1 if improved else 0, drops=0, learns=1 if gained > 0 else 0,
            level_frontier=best_fit)
        metrics.record(result)
        print(f"  [attempt {n}] {outcome.upper()} — reached {obs.level_label}, "
              f"survived {best_fit}, score {obs.score}")
        return result

    def run_attempt(self, n: int) -> metrics.AttemptResult:
        if self.game.evolves_tapes:
            return self._run_attempt_evolve(n)
        t0 = time.monotonic()
        self.ledger.set_attempt_num(n)
        self.ledger.begin_attempt(self.cache)
        self.session.load_state(0)
        obs = self._observe()
        self.reflex.reset(obs)
        self.commentator.reset(obs.raw)
        self.recent.clear()
        if self.memory is not None:
            self.memory.reset()
        self.cur_level = obs.level_key
        start_level = obs.level_key   # for the frontier metric (cur_level moves on clear)

        trajectory: list[coach.TrajectoryStep] = []
        billy_calls = frames = levels_cleared = fastest_in_attempt = 0
        search_calls = replay_calls = tape_frames = 0   # compounding-curve telemetry
        frames_to_frontier = 0                    # frames to re-reach last attempt's furthest x
        bucket_visits: dict = {}                  # per-spot visit counter (stall breaker)
        respawns = config.RESPAWNS_PER_ATTEMPT
        seg_best = obs.progress
        self._seg_level_key = obs.level_key[:2]
        final_score = obs.score
        level_start_frame = 0
        need_checkpoint = False
        outcome = "timeout"
        furthest = obs.level_label
        # A short trail of recent (progress, level_key, savestate) snapshots. On death we search
        # from one with enough RUNWAY before the death spot (a stomp/clear needs room to set up),
        # not the frame flush against it. Throttled to ~one snapshot per tile.
        safe_history: deque = deque(maxlen=48)
        self._approach_trail: deque = deque(maxlen=20)
        safe_history.append((obs.progress, obs.elevation, obs.level_key, self.session.clone_state()))
        last_snap_x = obs.progress
        self._learned_buckets.clear()
        self._reachback_miss.clear()
        self._taught_demo_used.clear()
        obs = self._tape_begin_level(obs)
        tape_frontier = obs.progress
        self._log_objective(obs)

        while frames <= config.MAX_ATTEMPT_FRAMES:
            # --- death: LEARN from it (search the approach for a survivor), then respawn -----
            if obs.dead:
                death_at = obs.progress
                if self.hooks.pit_death(obs.level_label, death_at):
                    self._last_pit_death_x = death_at
                frontier = self.cache.solved_frontier(obs.level_key)
                self.stuck.note_death(self._game_id(), obs.level_label, death_at, frontier)
                self._capture_death_approach(safe_history, obs.level_label, death_at,
                                             self._approach_trail, level_key=obs.level_key)
                # The key to advancing the frontier: search from a recent safe spot (with runway)
                # for a sequence that gets PAST where we just died, and remember it keyed there.
                self._tape_mode = False
                self._tape_replay = None
                learned_x = self._learn_from_death(
                    safe_history, death_at, obs.level_label, level_key=obs.level_key)
                if learned_x is not None:
                    search_calls += 1
                    self.ledger.learn(obs.level_key, learned_x, death_at, "death_search")
                    print(f'  [attempt {n}] 🧠 learned to pass {obs.level_label}@{learned_x} '
                          f'(died at {death_at}) — banked for next pass')
                self._reflect(trajectory, "death", obs.level_label)
                trajectory = []
                if respawns <= 0:
                    outcome = "game_over"
                    break
                respawns -= 1
                print(f'  [attempt {n}] 💀 down at {obs.level_label} ({death_at}) — Billy: '
                      f'"{self.commentator.death_quip(obs.raw)}" ({respawns} retries left)')
                self.session.load_state(0)      # respawn at the level-start checkpoint
                obs = self._observe()
                self.reflex.reset(obs)
                self.commentator.reset(obs.raw)
                self.recent.clear()
                if self.memory is not None:
                    self.memory.reset()
                seg_best, need_checkpoint = obs.progress, False
                # Respawn is a REWIND, not a transition: sync cur_level so the loop doesn't see
                # a phantom screen change (which printed a bogus 🗺️ line and polluted the route
                # graph with checkpoint→level edges that aren't topology).
                self.cur_level = obs.level_key
                # Re-base the snapshot throttle: progress just RESET to the checkpoint. Leaving it
                # at the pre-death value starves safe_history of any snapshot before the death spot
                # on this life — learn-from-death would then have no same-level runway to search.
                last_snap_x = obs.progress
                safe_history.append((obs.progress, obs.elevation, obs.level_key,
                                     self.session.clone_state()))
                self._learned_buckets.clear()
                self._taught_demo_used.clear()
                self._last_pit_death_x = 0
                obs = self._tape_begin_level(obs)
                tape_frontier = obs.progress
                continue

            # --- level clear: keep going into the next level, then checkpoint it ----------
            # Compare only (world, stage): the cache key carries AREA too (so pipe warps get their
            # own bucket region), but an internal area boundary inside a level — e.g. 1-2's joined
            # areas — is NOT a level clear and must not over-count or trip the per-attempt cap.
            if self.game.level_cleared(self.cur_level, obs.level_key):
                self._tape_finish_level(max(tape_frontier, seg_best), cleared=True, anchor=True)
                self.routes.record(self.cur_level, obs.level_key, "clear",
                                   at=max(tape_frontier, seg_best), dst_label=obs.level_label)
                levels_cleared += 1
                furthest = obs.level_label
                clear_frames = max(1, frames - level_start_frame)
                level_start_frame = frames
                fastest_in_attempt = (clear_frames if not fastest_in_attempt
                                      else min(fastest_in_attempt, clear_frames))
                record = ""
                if self.fastest_clear_frames is None or clear_frames < self.fastest_clear_frames:
                    self.fastest_clear_frames = clear_frames
                    record = " ⏱ FASTEST YET!"
                print(f'  [attempt {n}] 🏁 CLEARED in {clear_frames/60:.1f}s (score {obs.score})'
                      f'{record} — Billy: "{self.commentator.event_line("clear")}"')
                self._reflect(trajectory, "clear", obs.level_label)
                trajectory = []
                if self.repeat_level:
                    outcome = "clear"   # eval mode: one clear per attempt -> next attempt restarts
                    break
                self.reflex.note_level_advance(obs)
                self.commentator.reset(obs.raw)
                self.cur_level = obs.level_key
                self._seg_level_key = obs.level_key[:2]
                seg_best, need_checkpoint = obs.progress, True
                last_snap_x = obs.progress   # x re-based in the new level — re-arm snapshots
                obs = self._tape_begin_level(obs)
                tape_frontier = obs.progress
                self._log_objective(obs)
                if levels_cleared >= config.MAX_LEVELS_PER_ATTEMPT:
                    outcome = "clear"
                    self.session.send_plan(_IDLE)
                    self._observe()
                    break
                self.session.send_plan(_IDLE)   # coast through the level-end cutscene
                frames += 2
                obs = self._observe()
                continue

            # --- screen / sub-area change (SMB pipe warp, Zelda new screen, etc.) ----------
            if self.game.screen_changed(self.cur_level, obs.level_key):
                self.reflex.note_level_advance(obs)
                # Crossing INTO the next screen completes the previous screen's tape — persist it
                # (cleared=True: it ends in a verified transition) so screen-segment tapes chain
                # into a whole-level, then whole-game, fast-forward.
                self._tape_finish_level(tape_frontier, cleared=True)
                self.routes.record(self.cur_level, obs.level_key, "screen",
                                   at=max(tape_frontier, seg_best), dst_label=obs.level_label)
                self.cur_level = obs.level_key
                obs = self._tape_begin_level(obs)
                tape_frontier = obs.progress
                last_snap_x = obs.progress   # x re-based in the new area — re-arm snapshots
                print(f'  [attempt {n}] 🗺️ screen → {obs.level_label} '
                      f'progress={obs.progress}')

            # --- live human takeover: T in the watch window hands the human the controller.
            # Their segment banks like any verified solution (it ran live on the real state).
            if self.session.takeover_requested():
                self._tape_mode = False          # human input supersedes tape playback
                self._tape_replay = None
                obs, used = self._human_takeover(obs, safe_history, n)
                frames += used
                seg_best = max(seg_best, obs.progress)
                tape_frontier = max(tape_frontier, obs.progress)
                final_score = max(final_score, obs.score)
                continue

            # --- tape replay (zero-search level trajectory) ------------------------------
            if self._tape_mode:
                tape_plan = self._tape_consume()
                if tape_plan:
                    replay_calls += 1
                    tape_frames += plan_frames(tape_plan)
                    self.ledger.replay()
                    # record_tape stays ON: the consumed chunks re-seed _tape_record, so when the
                    # tape exhausts mid-level the continuing live play EXTENDS the tape (prefix +
                    # new suffix) instead of replacing it with just the suffix — the frontier on a
                    # level's tape only ever grows, until the tape clears the level outright.
                    obs, last_snap_x = self._commit(tape_plan, safe_history, last_snap_x)
                    frames += plan_frames(tape_plan)
                    seg_best = max(seg_best, obs.progress)
                    tape_frontier = max(tape_frontier, obs.progress)
                    final_score = max(final_score, obs.score)
                    # Tape-carried level entries must checkpoint too: with whole levels riding
                    # tapes, this branch `continue`s every iteration — without the gate here a
                    # taped level NEVER checkpoints and deaths respawn levels back.
                    if need_checkpoint and self._checkpoint_ready(obs):
                        obs = self._checkpoint_now()
                        need_checkpoint = False
                        seg_best = obs.progress
                        furthest = obs.level_label
                    continue
                self._tape_mode = False

            # --- normal play -------------------------------------------------------------
            if obs.level_key[:2] != self._seg_level_key:
                self._seg_level_key = obs.level_key[:2]
                seg_best = obs.progress
            decision = self.reflex.step(obs)
            self.recent.append(decision.note)
            if self.memory is not None:
                self.memory.note(decision.note)
                if self.memory.should_rollup():
                    self.memory.rollup()
            # Engage the search (and stall-breaker) not just at enemy/pit hazards but whenever the
            # reflex is STUCK — a wall/dead-end with no enemy or pit (e.g. 1-2's top-right ledge)
            # otherwise never triggers search, so Billy reflex-loops at it forever without ever
            # trying a retreat/reroute. "stuck" routes it into micro-search instead.
            danger = decision.needs_billy and (
                "enemy" in decision.note or "pit" in decision.note or "stuck" in decision.note
                or "pipe" in decision.note)
            at_pipe = getattr(obs.raw, "pipe_entry_spot", lambda: None)() is not None
            lk = obs.level_key   # game-agnostic level identity; the cache key needs nothing else

            # POSITION-KEYED POLICY, consulted every ON-GROUND step (not just on danger): if we've
            # banked a verified solution for this spot — discovered live or learned-from-death —
            # replay it verbatim. This is what advances the frontier: a solution learned at a death
            # spot gets used on the next pass even though the reflex sees no danger flag there. We
            # only replay on-ground (where solutions are keyed) so the deterministic state matches.
            # Replay only ON-GROUND: a cached plan keyed by (level, x) reproduces only if the full
            # state matches, and airborne states at the same x vary with the approach (velocity/
            # height), so replaying them diverges into deaths. On-ground spots are reproducible.
            # (This caps how much compounds — airborne hazards re-search — but keeps replays safe.)
            on_ground = getattr(obs.raw, "on_ground", True)
            ey = obs.elevation
            in_special = self.hooks.in_special_zone(obs)
            cached = self._cached_at(obs, on_ground=on_ground, in_special=in_special)
            # x≈0 is a transition artifact (the pipe-entry animation reads x=0 for many frames while
            # uncontrollable), NOT a real spot — don't let it trip the stall-breaker. Level starts at
            # x≈40, so a tiny-x guard is safe and stops the false "stuck at @0" after taking an exit.
            if (cached is not None or danger or in_special or at_pipe) and obs.progress >= 16:
                # Stall breaker: keep arriving at the same spot without getting past it -> give up.
                # Special zones + pipe mouths are exempt — timing stalls x without being stuck.
                if not self.hooks.stall_break_exempt(obs):
                    bkey = bucket_of(lk, obs.progress, ey)
                    bucket_visits[bkey] = bucket_visits.get(bkey, 0) + 1
                    if bucket_visits[bkey] > config.MAX_BUCKET_VISITS:
                        # Route-awareness: remember this node is a DEAD-END so future searches avoid
                        # re-entering it (deprioritised in _micro_search) instead of re-discovering it.
                        self.cache.mark_dead(lk, obs.progress, ey)
                        # Phase 2a — propagate the dead mark BACK along the approach (the high road that
                        # led here) and DROP those cached steps, so next pass replays nothing on this
                        # branch and the re-search routes the other way (the low road stays open: distinct
                        # y band). This is what turns "give up at the dead-end" into "take the other path".
                        for px, py, plk, _ in list(safe_history)[-config.DEADEND_BACKTRACK:]:
                            self.cache.mark_dead(plk, px, py)
                            self._drop_solution(plk, px, y=py, reason="dead_end")
                        print(f'  [attempt {n}] {obs.progress} 🧱 dead-end at {obs.level_label}@{obs.progress} '
                              f'— marked the approach, will reroute next pass')
                        outcome = "stuck"
                        break

            # Cached solution here -> REPLAY IT VERBATIM (deterministic, no search). We deliberately
            # do NOT verify-then-research by default: the emulator is deterministic, so replaying the
            # same plans from the same checkpoint reproduces the exact trajectory (every enemy in the
            # same phase) — verify→research is what INJECTS timing drift (a re-search can pick a
            # different-duration plan, shifting everything downstream and snowballing). If a replay
            # genuinely dies, record_fail (below) drops it and it re-searches once: self-healing.
            # Set BILLY_VERIFY_REPLAY=1 to gate replays on a clone-check instead.
            sec_match = (self.sections._match(obs)
                         if (self.sections is not None and on_ground) else None)
            # Remix BC demos (mid-level teaches) win over position-cache: moving hazards only
            # reproduce from the taught entry savestate, so we warp then replay.
            taught_plan = self._try_taught_demo(obs) if on_ground else None
            do_replay = (cached is not None and on_ground
                         and (not config.VERIFY_REPLAY
                              or self._verify(cached.plan,
                                              max(plan_frames(cached.plan) + 24,
                                                  config.SEARCH_HORIZON_FRAMES))))
            # A registered section band (e.g. smb_lost 1-1@1040) outranks replaying a cache that
            # never clears the section goal — otherwise the RL sub-policy never fires.
            if taught_plan is None and sec_match is not None and cached is not None:
                sec, _model = sec_match
                if cached.reach_after < sec.goal_x:
                    do_replay = False
                elif do_replay:
                    # Stale high-reach entries (learned from a death, wrong ROM era) claim
                    # reach≈1123 but die at 1062 — verify before replay in the hazard band.
                    horizon = max(plan_frames(cached.plan) + 24,
                                  config.SEARCH_HORIZON_FRAMES)
                    if not self._verify(cached.plan, horizon):
                        self._drop_solution(lk, obs.progress, y=ey, reason="section_stale")
                        do_replay = False
                        cached = self._cached_at(obs, on_ground=on_ground,
                                                 in_special=in_special)
            routine = False   # True for reflex plans only — safe to cut at a cached bucket
            if taught_plan is not None:
                plan = taught_plan
                do_replay = True          # non-interruptible commit of the taught line
                replay_calls += 1
                self.ledger.replay()
                action_note = f"taught-demo {self._label(plan)}"
                obs = self._observe()     # after warp to demo entry state
                lk, ey = obs.level_key, obs.elevation
            elif do_replay:
                plan = cached.plan
                replay_calls += 1
                self.cache.record_hit(lk, obs.progress, ey)
                self.ledger.replay()
                action_note = f"replay {self._label(plan)}"
            elif cached is not None or danger or sec_match is not None:
                # HAZARD-SCOPED RL: if Billy is at a registered section, let its verified sub-policy
                # own the crossing — roll it out on a clone and, if it gets through alive, commit and
                # bank THAT directly (no death-watch scorer, which would penalize the crossing for a
                # death at the NEXT hazard further down the level). Falls through to search otherwise.
                seg = (self.sections.cross(obs, self.session, self._observe)
                       if (self.sections is not None and (on_ground or in_special))
                       else None)
                if seg is not None:
                    plan, reach = seg
                    search_calls += 1
                    if self.hooks.section_bankable(obs, reach):
                        self._bank_solution(lk, obs.progress, plan, reach, y=ey,
                                            force=cached is not None, source="section",
                                            summary=obs.summary, level_label=obs.level_label)
                        action_note = f"section✓ {self._label(plan)}"
                        print(f'  [attempt {n}] {obs.progress} 🤖 section-policy crossed '
                              f'(reach {reach}) — remembered')
                    else:
                        action_note = f"section {self._label(plan)}"
                    # fall through to the normal chunked commit so the crossing snapshots its trail
                    # (the post-section hazard then has a runway for learn-from-death).
                else:
                    # MISS or STALE cache: search (invisibly, on a clone of the LIVE state, so moving
                    # enemies are where they actually are now) for a surviving, FORWARD-PROGRESSING
                    # sequence and REMEMBER it. A non-progress escape is a stall — never cached.
                    progressed = False
                    reach = obs.progress
                    plan: Plan = [Step(2, 0)]
                    zone_plan, zone_reach, zone_cross = self.hooks.try_frame_search(
                        self.session, self._observe, obs, deep=False,
                        min_gain=self._MIN_PROGRESS_PX)
                    pit_plan, pit_reach = self.hooks.try_pit_search(
                        self.session, self._observe, obs,
                        death_x=self._last_pit_death_x,
                        min_gain=self._MIN_PROGRESS_PX)
                    special_plan = zone_plan or pit_plan
                    if special_plan is not None:
                        plan = special_plan
                        progressed = True
                        reach = zone_reach if zone_plan else pit_reach
                        search_calls += 1
                        src = "lift_frame" if zone_plan else "pit_frame"
                        self._bank_solution(lk, obs.progress, plan, reach, y=ey,
                                            force=cached is not None, source=src,
                                            summary=obs.summary, level_label=obs.level_label)
                        tag = "lift-frame✓" if zone_plan else "pit-frame✓"
                        action_note = f"{tag} {self._label(plan)}"
                        reach_str = (f"reach {reach}" if not zone_cross
                                     else f"cross {reach}")
                        print(f'  [attempt {n}] {obs.progress} {"🛗" if zone_plan else "🕳️"} '
                              f'{tag} solved ({reach_str}) — remembered')
                    if not progressed:
                        best_plan, progressed, reach = self._micro_search(
                            self._candidates(obs), obs.progress, level_key=lk)
                        plan = best_plan
                        search_calls += 1
                    # Hard wall: the focused spread couldn't progress -> optionally try a DENSE
                    # brute-force grid before the LLM (deterministic, model-free, but ~40 candidates
                    # so it's slow — opt in with BILLY_EXPANDED_SEARCH=1).
                    if not progressed:
                        expand = getattr(self.reflex, "expanded_candidates", None)
                        if expand is not None and config.EXPANDED_FALLBACK:
                            best_plan, progressed, reach = self._micro_search(
                                expand(obs), obs.progress, early_exit=True,
                                settle=config.LEARN_HORIZON_FRAMES, level_key=lk)
                            plan = best_plan
                            search_calls += 1
                    if progressed and special_plan is None:
                        if not self.hooks.cacheable_reach(obs, reach):
                            progressed = False
                    if progressed and special_plan is None:
                        # force-overwrite when refreshing a stale entry so the freshest WORKING plan
                        # replaces the one that just failed verify (else we'd re-search it forever).
                        src = "research" if cached is not None else "search"
                        self._bank_solution(lk, obs.progress, plan, reach, y=ey,
                                            force=cached is not None, source=src,
                                            summary=obs.summary, level_label=obs.level_label)
                        tag = "research✓" if cached is not None else "search✓"
                        action_note = f"{tag} {self._label(plan)}"
                        reach_str = "→ pipe/area" if reach >= self._TRANSITION_BONUS else f"reach {reach}"
                        print(f'  [attempt {n}] {obs.progress} 🔍 solved ({reach_str}) — remembered')
                    elif not progressed and self.use_llm:
                        # Genuinely hard: let Billy improvise. Then VERIFY his plan on a clone and, if
                        # it survives AND progresses, BANK it like a search win — so the next pass
                        # replays it instead of re-asking the (slow, inconsistent) LLM. This is what
                        # lets an LLM-cracked wall (e.g. early 1-2) actually compound.
                        mem = self.memory.prompt_section() if self.memory else ""
                        if self.guide is not None:
                            mem += self.guide.prompt_section(self.game.guide_query(obs))
                        mem += self.strategist.prompt_section(obs.level_key)
                        bd = billy.decide(obs, self.kb.retrieve(obs.summary,
                                                                game=self._game_id()),
                                          list(self.recent), self.controller, memory=mem)
                        plan = bd.plan
                        billy_calls += 1
                        survived, reach = self._evaluate(plan)
                        if survived and reach > obs.progress + self._MIN_PROGRESS_PX:
                            self._bank_solution(lk, obs.progress, plan, reach, y=ey,
                                                force=cached is not None, source="llm",
                                                summary=obs.summary,
                                                level_label=obs.level_label)
                            action_note = f"BILLY✓ {self._label(plan)}"
                            print(f'  [attempt {n}] {obs.progress} 🎮 Billy: "{bd.trash_talk}" — '
                                  f'cracked it (reach {reach}), remembered')
                        else:
                            action_note = f"BILLY {self._label(plan)}"
                            print(f'  [attempt {n}] {obs.progress} 🎮 Billy: "{bd.trash_talk}"')
                    elif not progressed:
                        action_note = f"search✗ {self._label(plan)}"   # best-effort, not cached
            elif decision.needs_billy and self.use_llm:
                # Non-danger escalation (stuck, etc.): consult Billy with KB lessons + guide.
                mem = self.memory.prompt_section() if self.memory else ""
                if self.guide is not None:
                    mem += self.guide.prompt_section(self.game.guide_query(obs))
                mem += self.strategist.prompt_section(obs.level_key)
                bd = billy.decide(obs, self.kb.retrieve(obs.summary, game=self._game_id()),
                                  list(self.recent), self.controller, memory=mem)
                plan = bd.plan
                billy_calls += 1
                action_note = f"BILLY {self._label(plan)}"
                print(f'  [attempt {n}] {obs.progress} 🎮 Billy: "{bd.trash_talk}"')
            elif decision.needs_billy:
                plan = list(decision.plan)   # reflex's own fallback (e.g. a recovery jump)
                action_note = f"reflex {decision.note}"
                routine = True
            else:
                plan = list(decision.plan)
                action_note = f"{decision.note} {self._label(plan)}"
                routine = True

            # Commit the plan in small same-button chunks, snapshotting the trail as we go. Chunking
            # is behaviour-preserving (identical per-frame input) but lets us capture snapshots
            # DURING a long jump — so learn-from-death always has a runway snapshot before a death,
            # even when a single ballistic plan skips many tiles at once.
            replay_x, replay_y = obs.progress, obs.elevation
            # Replays stop at intermediate cache buckets (e.g. a taught pit crossing at x≈1039)
            # so a long reachback behind doesn't blow past a hazard-specific demo.
            # Search is interruptible too: a weak hop must not fly through a Remix demo key.
            # Replays stay non-cut so a verified banked trajectory runs to completion.
            obs, last_snap_x = self._commit(plan, safe_history, last_snap_x,
                                            interruptible=(not do_replay))
            if obs.dead:
                if "replay" in action_note:
                    self._drop_solution(lk, replay_x, y=replay_y, reason="replay_fail")
                elif any(t in action_note for t in
                         ("search✓", "research✓", "section✓", "lift-frame✓",
                          "pit-frame✓", "learn_pit", "BILLY✓")):
                    reason = self.hooks.replay_death_drop_reason(
                        obs.level_label, replay_x, obs.progress)
                    if reason:
                        self._drop_solution(lk, replay_x, y=replay_y, reason=reason)
            trajectory.append(coach.TrajectoryStep(
                x=replay_x, summary=obs.summary, action=action_note, event=decision.note))
            frames += plan_frames(plan)
            seg_best = max(seg_best, obs.progress)
            tape_frontier = max(tape_frontier, obs.progress)
            final_score = max(final_score, obs.score)
            if frames_to_frontier == 0 and self._prev_best_x > 0 and obs.progress >= self._prev_best_x:
                frames_to_frontier = frames   # re-reached last attempt's furthest point

            if need_checkpoint and self._checkpoint_ready(obs):
                obs = self._checkpoint_now()
                need_checkpoint = False
                seg_best = obs.progress
                furthest = obs.level_label

            quip = self.commentator.observe(obs.raw)
            if quip:
                print(f'  🎤 Billy: "{quip}"')

        # Frontier march: an attempt that timed out ALIVE still recorded a valid trajectory to its
        # frontier — persist it as a partial (non-clearing) tape so the next attempt fast-forwards
        # to the frontier with zero search and spends its whole budget on NEW ground. (Deaths and
        # dead-end "stuck" endings are not banked: their trails end badly / were just rerouted.)
        if outcome == "timeout" and not obs.dead:
            self._tape_finish_level(tape_frontier, cleared=False)

        hi = ""
        if final_score > self.best_score:
            self.best_score = final_score
            hi = " 🏆 NEW HIGH SCORE!"

        self._prev_best_x = max(self._prev_best_x, seg_best)
        level_key = self.cur_level if self.cur_level else start_level
        learn = self.ledger.finish_attempt(self.cache, level_key)
        result = metrics.AttemptResult(
            attempt=n, outcome=outcome, max_x=seg_best, frames=frames, billy_calls=billy_calls,
            world_stage=furthest, levels_cleared=levels_cleared, score=final_score,
            fastest_clear_frames=fastest_in_attempt, duration_s=round(time.monotonic() - t0, 2),
            search_calls=search_calls, replay_calls=replay_calls, tape_frames=tape_frames,
            frontier_x=self.cache.solved_frontier(start_level), frames_to_frontier=frames_to_frontier,
            banks=learn.banks, drops=learn.drops, learns=learn.learns,
            level_frontier=learn.level_frontier)
        metrics.record(result)
        print(f"  [attempt {n}] {outcome.upper()} — reached {furthest}, "
              f"cleared {levels_cleared} level(s), score {final_score}{hi}")
        print(f"      ↳ search={search_calls} replay={replay_calls} | "
              f"{format_learning_line(learn, furthest)}")
        return result

    def _capture_death_approach(self, safe_history, level_label: str, death_x: int,
                                approach_trail: deque | None = None,
                                *, level_key: tuple = ()) -> None:
        """Auto-capture phase-accurate approach savestates for the stuck trainer."""
        band = self.hooks.approach_capture_band(level_label, death_x)
        if band is None:
            return
        x_lo, x_hi = band
        best: tuple[int, bytes] | None = None
        stage = level_key[:2] if level_key else ()
        for x, _y, lk, snap in (approach_trail or []):
            if stage and lk[:2] != stage:
                continue   # same x-band on ANOTHER level is not this death's approach
            if x_lo <= x <= x_hi and (best is None or x > best[0]):
                best = (x, snap)
        for x, _y, lk, snap in safe_history:
            if stage and lk[:2] != stage:
                continue
            if x_lo <= x <= x_hi and (best is None or x > best[0]):
                best = (x, snap)
        if best is None:
            print(f"[stuck] no trail snapshot in [{x_lo},{x_hi}] for "
                  f"{level_label} death@{death_x}")
            return
        approach_x, snap = best
        out = auto_state_path(level_label, death_x, approach_x)
        if os.path.isfile(out):
            self.stuck.note_capture(self._game_id(), level_label, death_x, out)
            return
        if write_trail_snapshot(snap, out, level_label=level_label, approach_x=approach_x):
            self.stuck.note_capture(self._game_id(), level_label, death_x, out)

    def _maybe_auto_train(self, after_attempt: int) -> None:
        """Between attempts: extended offline search when deaths cluster at a hazard."""
        if not config.AUTO_TRAIN:
            return
        pending: list[tuple] = []
        game_id = self._game_id()
        for rec in self.stuck.records.values():
            if rec.game != game_id:
                continue
            stuck = self.stuck.stuck_at(
                game_id, rec.level_label, rec.last_death_x,
                threshold=self.game.stuck_death_threshold())
            if stuck is not None:
                remedy = self.hooks.stuck_remedy(rec.level_label, rec.last_death_x)
                if remedy is not None:
                    pending.append((remedy, stuck))
        if not pending:
            return
        pending.sort(key=lambda t: t[1].deaths, reverse=True)
        remedy, record = pending[0]
        print(f"\n  [auto-train] after attempt {after_attempt}: "
              f"{remedy.level_label} stuck ({record.deaths} deaths @≈{record.last_death_x})")

        def _bank(lk, x, plan, reach, *, y=0, source="auto_stuck"):
            self._bank_solution(lk, x, plan, reach, y=y, force=True, source=source)
            self.ledger.train(lk, x, reach, source)

        result = remediate(
            self.session, self._observe, self.cache, self.hooks,
            remedy, record, bank_fn=_bank)
        if result.success:
            self.stuck.mark_remediated(game_id, remedy.level_label, remedy.death_x)
            if result.trained and self.sections is not None:
                from .rl.section_policy import SectionController, sections_for_game
                self.sections = SectionController(sections_for_game(self._game_id()),
                                                   game_id=self._game_id())
                print("  [auto-train] reloaded section sub-policies")
        else:
            print("  [auto-train] no verified crossing yet — will retry after more deaths")
            # Every autonomous tier missed here — the FINAL remedy is a human demo. Pull-based:
            # the request fires only at a wall that search + self-training could not crack.
            from .stuck_trainer import collect_auto_states, request_demo
            request_demo(getattr(self.game, "cli_name", "smb"), remedy, record,
                         collect_auto_states(remedy.level_label, record))

    def _reflect(self, trajectory, outcome: str, level_label: str) -> None:
        if not self.use_llm:
            return
        mem = self.memory.prompt_section() if self.memory else ""
        lesson = coach.reflect(trajectory, outcome, level_label, memory=mem)
        if lesson:
            self.kb.add(lesson.situation, lesson.tactic, lesson.outcome, level_label,
                        game=self._game_id())
            print(f"  [coach] lesson: {lesson.situation} -> {lesson.tactic}")

    def _label(self, plan: Plan) -> str:
        parts = ["+".join(self.controller.names_from_mask(s.buttons)) or "idle" for s in plan]
        return "[" + ", ".join(f"{p} x{s.frames}" for p, s in zip(parts, plan)) + "]"

    # --- the session --------------------------------------------------------------------
    def run_session(self, attempts: int, resume: bool = False) -> list[metrics.AttemptResult]:
        self.session.reset()
        self.session.wait_until_live()
        self.boot()
        if resume:
            self.resume_from_checkpoint()
        results = []
        try:
            for n in range(1, attempts + 1):
                results.append(self.run_attempt(n))
                if n < attempts:
                    self._maybe_auto_train(n)
        finally:
            if self.pool is not None:
                self.pool.close()
        metrics.print_curve(results)
        return results

