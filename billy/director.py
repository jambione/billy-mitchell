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

from . import config, metrics
from .abstractions import Game, Observation, Plan, Step, plan_frames
from .agents import billy, coach
from .commentary import Commentator
from .hazard_hooks import HazardHooks
from .knowledge import KnowledgeBase, SolutionCache, SkillLibrary, TapeLibrary
from .knowledge.cache import bucket_of
from .knowledge.tape import append_plan
from .learning import LearningLedger, format_learning_line
from .stuck_trainer import (
    StuckTracker, auto_state_path, remediate, write_trail_snapshot,
)

_IDLE: Plan = [Step(2, 0)]   # neutral input — to consume a pending frame or coast a cutscene


class Director:
    def __init__(self, game: Game, kb: KnowledgeBase, use_llm: bool = True,
                 cache: SolutionCache | None = None, skills: SkillLibrary | None = None,
                 tapes: TapeLibrary | None = None, sections=None) -> None:
        self.game = game
        self.hooks: HazardHooks = game.hazard_hooks()
        self.session = game.system.connect()
        self.controller = game.system.controller
        self.reflex = game.make_reflex()
        self.kb = kb
        self.cache = cache if cache is not None else SolutionCache()  # the compounding policy
        self.tapes = tapes if tapes is not None else TapeLibrary()
        self.skills = skills if skills is not None else SkillLibrary()  # cross-game transferable tactics
        self._tape_record: list = []
        self._tape_key: tuple = ()
        self._tape_replay: list | None = None
        self._tape_mode = False
        self._learned_buckets: set = set()
        self._last_pit_death_x: int = 0
        # Optional hazard-scoped RL sub-policies: at a registered section they SEED micro-search with
        # a learned crossing candidate (verified+banked like any solution). None = pure reflex/search.
        self.sections = sections
        self.use_llm = use_llm
        # Eval mode: end each attempt at the FIRST level clear so every attempt is a fresh run of
        # the same starting level. This exposes the compounding curve (search-per-clear falls and
        # clear-time drops as the cache fills) instead of the checkpoint marching forward.
        self.repeat_level = os.environ.get("BILLY_REPEAT_LEVEL", "0") == "1"
        self.recent: deque[str] = deque(maxlen=12)
        self.commentator = Commentator()
        self.best_score = 0
        self.fastest_clear_frames: int | None = None
        self.cur_level: tuple = ()
        self._prev_best_x = 0   # furthest x reached so far (for frames-to-frontier metric)
        self.ledger = LearningLedger()
        self.stuck = StuckTracker()

    def _bank_solution(self, lk, x: int, plan: Plan, reach: int, *, y: int = 0,
                       force: bool = False, source: str = "search") -> None:
        prev = self.cache.get(lk, x, y)
        prev_reach = prev.reach_after if prev else -1
        self.cache.put(lk, x, plan, reach, y=y, force=force)
        entry = self.cache.get(lk, x, y)
        if entry and (prev is None or entry.reach_after > prev_reach):
            self.ledger.bank(lk, x, reach, source)

    def _drop_solution(self, lk, x: int, *, y: int = 0, reason: str = "fail") -> None:
        self.cache.record_fail(lk, x, y)
        self.ledger.drop(lk, x, reason)

    # --- lock-step helper ---------------------------------------------------------------
    def _observe(self) -> Observation:
        st = self.session.read_state()
        return self.game.observe(st.frame, st.ram, getattr(st, "rgb", None))

    # --- micro-search: evaluate candidates on a CLONE so the live run never visibly rewinds ----
    _TRANSITION_BONUS = 100_000   # a level/area advance (e.g. entering a pipe) dominates any reach

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
        it; this is what lets Billy take 1-2's mandatory exit pipe instead of running at it forever."""
        start = self._observe()
        start_level, base_x = start.level_key, start.progress
        self.session.send_plan(plan)
        obs = self._observe()
        reached = obs.progress
        advanced = self.game.search_area_advance(start_level, obs.level_key)
        # A candidate that ended BEHIND its start is a detour (retreat/drop) that needs the long
        # horizon to recover; anything else is forward and only needs the death-watch window. This
        # is position-based (no button knowledge), so it stays game-agnostic.
        detour = obs.progress < base_x
        coasted = 0
        while coasted < settle and not obs.dead and not advanced:
            coast = self.reflex.advance_plan(obs)
            self.session.send_plan(coast)
            obs = self._observe()
            advanced = self.game.search_area_advance(start_level, obs.level_key)
            reached = max(reached, obs.progress)
            coasted += max(1, plan_frames(coast))
            # SPEED: cap forward candidates at SEARCH_HORIZON (their original death-watch window) even
            # if they didn't progress — a forward stall stays a stall, the long horizon can't save it.
            # Only a detour that hasn't recovered yet keeps coasting to the full `settle`. This is the
            # bulk of the cost at a dead-end, where most candidates survive-but-don't-progress.
            if coasted >= config.SEARCH_HORIZON_FRAMES and (
                    not detour or reached > base_x + self._MIN_PROGRESS_PX):
                break
        if advanced:
            reached = base_x + self._TRANSITION_BONUS
        # also report where the candidate ENDS UP (its route node) for dead-end / elevation scoring
        return (not obs.dead), reached, obs.progress, obs.elevation

    _MIN_PROGRESS_PX = 8   # a "solution" must actually advance, else it's a stall, not an escape

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
        with self.session.search_mode():
            for plan in candidates:
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
                self.session.restore(snap)
                if early_exit and survived and not dead and reached > start_x + self._MIN_PROGRESS_PX:
                    break   # good-enough survivor found — stop the expensive grid here
        self.session.restore(snap)   # back to the live pre-search position, nothing shown
        self._observe()
        made_progress = best_score >= 0 and best_reach > start_x + self._MIN_PROGRESS_PX
        return best_plan, made_progress, best_reach

    _SNAP_CHUNK = 6   # commit live plans in chunks this many frames long (dense snapshot trail)

    def _commit(self, plan: Plan, safe_history, last_snap_x: int,
                *, record_tape: bool = True) -> tuple[Observation, int]:
        """Execute a committed (live) plan in small same-button chunks, appending a snapshot to the
        trail every ~tile of new progress. Splitting a Step into equal sub-steps of the same button
        mask is identical input frame-for-frame, so behaviour is unchanged — but the dense trail
        guarantees learn-from-death has a runway snapshot before the next death, even mid-jump."""
        obs = self._observe()
        for step in plan:
            remaining = step.frames
            while remaining > 0:
                chunk = min(self.hooks.commit_chunk_size(obs, self._SNAP_CHUNK), remaining)
                self.session.send_plan([Step(chunk, step.buttons)])
                remaining -= chunk
                obs = self._observe()
                if obs.dead:
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
        if record_tape and not self._tape_mode and plan:
            append_plan(self._tape_record, plan)
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

    def _candidates(self, obs: Observation) -> list[Plan]:
        """Search candidate set on a cache MISS: the reflex's hand-picked spread, PLUS the
        instantiated plans of the situationally-relevant transferable Skills. Skills only widen the
        *search* set (never blind-replay), so on a new game with an empty cache Billy starts from
        sensible carried-forward tactics instead of a cold search. Requires a PhysicsProfile-bearing
        reflex (the shared platformer policy); otherwise just the reflex spread."""
        cands = list(self.reflex.danger_candidates(obs))
        cands.extend(self.hooks.extra_candidates(obs))
        profile = getattr(self.reflex, "p", None)
        if profile is not None and len(self.skills):
            cands.extend(self.skills.candidates(obs.raw, profile, obs.summary))
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

        for x, y, lk, snap in reversed(safe_history):   # nearest the death first
            if level_key and lk[:2] != level_key[:2]:
                continue                                # same x on another level is not this death
            runway = death_x - x
            if runway < config.MIN_RUNWAY_PX:
                continue                                # too close to set up an escape
            if runway > learn_horizon:
                action = self.hooks.learn_runway_action(
                    level_label, death_x, runway, config.LEARN_HORIZON_FRAMES)
                if action == "continue":
                    continue
                break                                   # further back is out of rollout reach

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
                            self._bank_solution(lk, x, plan, reach, y=y, source="learn_section")
                            self._learned_buckets.add(bkey)
                            return x
                pit_plan, pit_reach = self.hooks.try_pit_search(
                    self.session, self._observe, snap_obs, death_x=death_x,
                    min_gain=self._MIN_PROGRESS_PX)
                if pit_plan is not None and self.hooks.learn_cacheable(
                        level_label, death_x, pit_reach):
                    self._bank_solution(lk, x, pit_plan, pit_reach, y=y, source="learn_pit")
                    self._learned_buckets.add(bkey)
                    return x
                frame_plan, frame_reach, _ = self.hooks.try_frame_search(
                    self.session, self._observe, snap_obs, deep=True,
                    min_gain=self._MIN_PROGRESS_PX)
                if frame_plan is not None and self.hooks.learn_cacheable(
                        level_label, death_x, frame_reach):
                    self._bank_solution(lk, x, frame_plan, frame_reach, y=y,
                                        source="learn_bootstrap")
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
            self._observe()
            if best_plan is not None:
                self._bank_solution(lk, x, best_plan, best_reach, y=y, source="learn_macro")
                self._learned_buckets.add(bkey)
                return x
        return None

    def _candidates_from(self, snap) -> list[Plan]:
        """Candidate escapes generated from a snapshot's observation (restores it to read state)."""
        self.session.restore(snap)
        return self._candidates(self._observe())

    def _tape_begin_level(self, obs: Observation) -> None:
        """Start recording a level trajectory; try a verified tape replay if one exists."""
        self._tape_key = obs.level_key
        self._tape_record = []
        self._tape_mode = False
        self._tape_replay = None
        entry = self.tapes.get(obs.level_key)
        if entry and self._verify_tape(entry.plan, obs.level_key, entry.frontier):
            self._tape_replay = [Step(s.frames, s.buttons) for s in entry.plan]
            self._tape_mode = True
            self.tapes.record_hit(obs.level_key)

    def _verify_tape(self, plan: Plan, level_key: tuple, min_frontier: int) -> bool:
        """Clone-check a stored tape before zero-search replay."""
        if not plan:
            return False
        snap = self.session.clone_state()
        start_level = level_key[:2]
        with self.session.search_mode():
            self.session.send_plan(plan)
            obs = self._observe()
            coasted = 0
            while coasted < 180 and not obs.dead and obs.level_key[:2] <= start_level:
                self.session.send_plan([Step(4, 0)])
                obs = self._observe()
                coasted += 4
        ok = (not obs.dead
              and (obs.level_key[:2] > start_level
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

    def _tape_finish_level(self, frontier: int, *, cleared: bool) -> None:
        """Persist a recorded trajectory when a level ends well."""
        if not self._tape_record:
            return
        self.tapes.put(self._tape_key, self._tape_record, frontier, clears_level=cleared)
        self._tape_record = []

    # --- boot ---------------------------------------------------------------------------
    def boot(self) -> Observation:
        start = self.game.boot(self.session)   # game reaches a playable state, returns start obs
        self.session.save_state(0)             # checkpoint the level start
        self._observe()                        # consume the post-save republished frame
        self.cur_level = start.level_key
        print(f"[director] in play at {start.level_label}, progress={start.progress}. "
              f"Billy has taken the controller.")
        print(f'  🎤 Billy: "{self.commentator.event_line("start")}"')
        return start

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
            cached = (self.cache.get(lk, obs.progress, ey)
                      if (on_ground or in_special) else None)

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

            if cached is not None and on_ground and not self.hooks.stale_cache(obs, cached):
                plan = cached.plan
            elif cached is not None or danger or in_special:
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
            else:
                plan = decision.plan

            self.session.send_plan(plan)
            obs = self._observe()
            frames += sum(s.frames for s in plan) if plan else 1

        print(f"[capture] miss — last {obs.level_label} x={obs.progress} "
              f"ground={getattr(obs.raw, 'on_ground', True)}")
        return False

    def run_attempt(self, n: int) -> metrics.AttemptResult:
        t0 = time.monotonic()
        self.ledger.set_attempt_num(n)
        self.ledger.begin_attempt(self.cache)
        self.session.load_state(0)
        obs = self._observe()
        self.reflex.reset(obs)
        self.commentator.reset(obs.raw)
        self.recent.clear()
        self.cur_level = obs.level_key
        start_level = obs.level_key   # for the frontier metric (cur_level moves on clear)

        trajectory: list[coach.TrajectoryStep] = []
        billy_calls = frames = levels_cleared = fastest_in_attempt = 0
        search_calls = replay_calls = 0          # compounding-curve telemetry
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
        self._tape_begin_level(obs)
        tape_frontier = obs.progress

        while frames <= config.MAX_ATTEMPT_FRAMES:
            # --- death: LEARN from it (search the approach for a survivor), then respawn -----
            if obs.dead:
                death_at = obs.progress
                if self.hooks.pit_death(obs.level_label, death_at):
                    self._last_pit_death_x = death_at
                frontier = self.cache.solved_frontier(obs.level_key)
                self.stuck.note_death(obs.level_label, death_at, frontier)
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
                seg_best, need_checkpoint = obs.progress, False
                self._learned_buckets.clear()
                self._last_pit_death_x = 0
                self._tape_begin_level(obs)
                tape_frontier = obs.progress
                continue

            # --- level clear: keep going into the next level, then checkpoint it ----------
            # Compare only (world, stage): the cache key carries AREA too (so pipe warps get their
            # own bucket region), but an internal area boundary inside a level — e.g. 1-2's joined
            # areas — is NOT a level clear and must not over-count or trip the per-attempt cap.
            if self.game.level_cleared(self.cur_level, obs.level_key):
                self._tape_finish_level(max(tape_frontier, seg_best), cleared=True)
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
                self._tape_begin_level(obs)
                tape_frontier = obs.progress
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
                self.cur_level = obs.level_key
                self._tape_begin_level(obs)
                tape_frontier = obs.progress
                print(f'  [attempt {n}] 🗺️ screen → {obs.level_label} '
                      f'progress={obs.progress}')

            # --- tape replay (zero-search level trajectory) ------------------------------
            if self._tape_mode:
                tape_plan = self._tape_consume()
                if tape_plan:
                    replay_calls += 1
                    self.ledger.replay()
                    obs, last_snap_x = self._commit(tape_plan, safe_history, last_snap_x,
                                                    record_tape=False)
                    frames += plan_frames(tape_plan)
                    seg_best = max(seg_best, obs.progress)
                    tape_frontier = max(tape_frontier, obs.progress)
                    final_score = max(final_score, obs.score)
                    continue
                self._tape_mode = False

            # --- normal play -------------------------------------------------------------
            if obs.level_key[:2] != self._seg_level_key:
                self._seg_level_key = obs.level_key[:2]
                seg_best = obs.progress
            decision = self.reflex.step(obs)
            self.recent.append(decision.note)
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
            cached = (self.cache.get(lk, obs.progress, ey)
                      if (on_ground or in_special) else None)
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
            do_replay = (cached is not None and on_ground
                         and not self.hooks.stale_cache(obs, cached)
                         and (not config.VERIFY_REPLAY
                              or self._verify(cached.plan,
                                              max(plan_frames(cached.plan) + 24,
                                                  config.SEARCH_HORIZON_FRAMES))))
            if do_replay:
                plan = cached.plan
                replay_calls += 1
                self.cache.record_hit(lk, obs.progress, ey)
                self.ledger.replay()
                action_note = f"replay {self._label(plan)}"
            elif cached is not None or danger:
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
                                            force=cached is not None, source="section")
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
                                            force=cached is not None, source=src)
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
                                            force=cached is not None, source=src)
                        tag = "research✓" if cached is not None else "search✓"
                        action_note = f"{tag} {self._label(plan)}"
                        reach_str = "→ pipe/area" if reach >= self._TRANSITION_BONUS else f"reach {reach}"
                        print(f'  [attempt {n}] {obs.progress} 🔍 solved ({reach_str}) — remembered')
                    elif not progressed and self.use_llm:
                        # Genuinely hard: let Billy improvise. Then VERIFY his plan on a clone and, if
                        # it survives AND progresses, BANK it like a search win — so the next pass
                        # replays it instead of re-asking the (slow, inconsistent) LLM. This is what
                        # lets an LLM-cracked wall (e.g. early 1-2) actually compound.
                        bd = billy.decide(obs, self.kb.retrieve(obs.summary),
                                          list(self.recent), self.controller)
                        plan = bd.plan
                        billy_calls += 1
                        survived, reach = self._evaluate(plan)
                        if survived and reach > obs.progress + self._MIN_PROGRESS_PX:
                            self._bank_solution(lk, obs.progress, plan, reach, y=ey,
                                                force=cached is not None, source="llm")
                            action_note = f"BILLY✓ {self._label(plan)}"
                            print(f'  [attempt {n}] {obs.progress} 🎮 Billy: "{bd.trash_talk}" — '
                                  f'cracked it (reach {reach}), remembered')
                        else:
                            action_note = f"BILLY {self._label(plan)}"
                            print(f'  [attempt {n}] {obs.progress} 🎮 Billy: "{bd.trash_talk}"')
                    elif not progressed:
                        action_note = f"search✗ {self._label(plan)}"   # best-effort, not cached
            elif decision.needs_billy and self.use_llm:
                # Non-danger escalation (stuck, etc.): consult Billy with KB lessons.
                bd = billy.decide(obs, self.kb.retrieve(obs.summary), list(self.recent), self.controller)
                plan = bd.plan
                billy_calls += 1
                action_note = f"BILLY {self._label(plan)}"
                print(f'  [attempt {n}] {obs.progress} 🎮 Billy: "{bd.trash_talk}"')
            elif decision.needs_billy:
                plan = list(decision.plan)   # reflex's own fallback (e.g. a recovery jump)
                action_note = f"reflex {decision.note}"
            else:
                plan = list(decision.plan)
                action_note = f"{decision.note} {self._label(plan)}"

            # Commit the plan in small same-button chunks, snapshotting the trail as we go. Chunking
            # is behaviour-preserving (identical per-frame input) but lets us capture snapshots
            # DURING a long jump — so learn-from-death always has a runway snapshot before a death,
            # even when a single ballistic plan skips many tiles at once.
            replay_x, replay_y = obs.progress, obs.elevation
            obs, last_snap_x = self._commit(plan, safe_history, last_snap_x)
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
            final_score = max(final_score, obs.score)
            if frames_to_frontier == 0 and self._prev_best_x > 0 and obs.progress >= self._prev_best_x:
                frames_to_frontier = frames   # re-reached last attempt's furthest point

            if need_checkpoint and getattr(obs.raw, "on_ground", True) and 16 < obs.progress < 120:
                self.session.save_state(0)
                obs = self._observe()
                need_checkpoint = False
                seg_best = obs.progress
                furthest = obs.level_label
                print(f"  [director] checkpoint at {obs.level_label} ({obs.progress})")

            quip = self.commentator.observe(obs.raw)
            if quip:
                print(f'  🎤 Billy: "{quip}"')

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
            search_calls=search_calls, replay_calls=replay_calls,
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
        for x, _y, _lk, snap in (approach_trail or []):
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
            self.stuck.note_capture(level_label, death_x, out)
            return
        if write_trail_snapshot(snap, out, level_label=level_label, approach_x=approach_x):
            self.stuck.note_capture(level_label, death_x, out)

    def _maybe_auto_train(self, after_attempt: int) -> None:
        """Between attempts: extended offline search when deaths cluster at a hazard."""
        if not config.AUTO_TRAIN:
            return
        pending: list[tuple] = []
        for (_label, _bucket), rec in self.stuck.records.items():
            stuck = self.stuck.stuck_at(rec.level_label, rec.last_death_x)
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
            self.stuck.mark_remediated(remedy.level_label, remedy.death_x)
            if result.trained and self.sections is not None:
                from .rl.section_policy import SectionController, default_smb_sections
                self.sections = SectionController(default_smb_sections())
                print("  [auto-train] reloaded section sub-policies")
        else:
            print("  [auto-train] no verified crossing yet — will retry after more deaths")

    def _reflect(self, trajectory, outcome: str, level_label: str) -> None:
        if not self.use_llm:
            return
        lesson = coach.reflect(trajectory, outcome, level_label)
        if lesson:
            self.kb.add(lesson.situation, lesson.tactic, lesson.outcome, level_label)
            print(f"  [coach] lesson: {lesson.situation} -> {lesson.tactic}")

    def _label(self, plan: Plan) -> str:
        parts = ["+".join(self.controller.names_from_mask(s.buttons)) or "idle" for s in plan]
        return "[" + ", ".join(f"{p} x{s.frames}" for p, s in zip(parts, plan)) + "]"

    # --- the session --------------------------------------------------------------------
    def run_session(self, attempts: int) -> list[metrics.AttemptResult]:
        self.session.reset()
        self.session.wait_until_live()
        self.boot()
        results = []
        for n in range(1, attempts + 1):
            results.append(self.run_attempt(n))
            if n < attempts:
                self._maybe_auto_train(n)
        metrics.print_curve(results)
        return results

