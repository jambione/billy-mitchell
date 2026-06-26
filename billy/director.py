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
from .knowledge import KnowledgeBase, SolutionCache, SkillLibrary
from .knowledge.cache import bucket_of

_IDLE: Plan = [Step(2, 0)]   # neutral input — to consume a pending frame or coast a cutscene


class Director:
    def __init__(self, game: Game, kb: KnowledgeBase, use_llm: bool = True,
                 cache: SolutionCache | None = None, skills: SkillLibrary | None = None) -> None:
        self.game = game
        self.session = game.system.connect()
        self.controller = game.system.controller
        self.reflex = game.make_reflex()
        self.kb = kb
        self.cache = cache if cache is not None else SolutionCache()  # the compounding policy
        self.skills = skills if skills is not None else SkillLibrary()  # cross-game transferable tactics
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

    # --- lock-step helper ---------------------------------------------------------------
    def _observe(self) -> Observation:
        st = self.session.read_state()
        return self.game.observe(st.frame, st.ram)

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
        advanced = obs.level_key > start_level
        coasted = 0
        while coasted < settle and not obs.dead and not advanced:
            coast = self.reflex.advance_plan(obs)
            self.session.send_plan(coast)
            obs = self._observe()
            advanced = obs.level_key > start_level
            reached = max(reached, obs.progress)
            coasted += max(1, plan_frames(coast))
        if advanced:
            reached = base_x + self._TRANSITION_BONUS
        return (not obs.dead), reached

    _MIN_PROGRESS_PX = 8   # a "solution" must actually advance, else it's a stall, not an escape

    def _micro_search(self, candidates: list[Plan], start_x: int, early_exit: bool = False,
                      settle: int = config.SEARCH_HORIZON_FRAMES) -> tuple[Plan, bool, int]:
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
                survived, reached = self._rollout(plan, settle)
                score = reached if survived else reached - 100_000  # death ≫ worse than short
                if score > best_score:
                    best_score, best_plan, best_reach = score, plan, reached
                self.session.restore(snap)
                if early_exit and survived and reached > start_x + self._MIN_PROGRESS_PX:
                    break   # good-enough survivor found — stop the expensive grid here
        self.session.restore(snap)   # back to the live pre-search position, nothing shown
        self._observe()
        made_progress = best_score >= 0 and best_reach > start_x + self._MIN_PROGRESS_PX
        return best_plan, made_progress, best_reach

    _SNAP_CHUNK = 6   # commit live plans in chunks this many frames long (dense snapshot trail)

    def _commit(self, plan: Plan, safe_history, last_snap_x: int) -> tuple[Observation, int]:
        """Execute a committed (live) plan in small same-button chunks, appending a snapshot to the
        trail every ~tile of new progress. Splitting a Step into equal sub-steps of the same button
        mask is identical input frame-for-frame, so behaviour is unchanged — but the dense trail
        guarantees learn-from-death has a runway snapshot before the next death, even mid-jump."""
        obs = self._observe()
        for step in plan:
            remaining = step.frames
            while remaining > 0:
                chunk = min(self._SNAP_CHUNK, remaining)
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
                    safe_history.append((obs.progress, obs.level_key, self.session.clone_state()))
                    last_snap_x = obs.progress
        return obs, last_snap_x

    def _verify(self, plan: Plan, horizon: int = config.LEARN_HORIZON_FRAMES) -> bool:
        """Does this plan still survive from the CURRENT live state? Checked on a clone (invisible).
        Because the clone is the exact live state, a plan that survives here will reproduce
        identically when committed — so a verified replay is deterministic. A cached solution that
        no longer survives (e.g. a moving enemy has shifted phase since it was learned) fails here,
        and the caller live-searches fresh with the enemy where it ACTUALLY is now."""
        snap = self.session.clone_state()
        with self.session.search_mode():
            lived, _ = self._rollout(plan, horizon)
        self.session.restore(snap)
        self._observe()
        return lived

    def _evaluate(self, plan: Plan, settle: int = config.LEARN_HORIZON_FRAMES) -> tuple[bool, int]:
        """Run a plan on a clone of the CURRENT live state and report (survived, reach) — for
        scoring an LLM-improvised plan before trusting/caching it. Invisible; leaves live untouched."""
        snap = self.session.clone_state()
        with self.session.search_mode():
            survived, reach = self._rollout(plan, settle)
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
        profile = getattr(self.reflex, "p", None)
        if profile is not None and len(self.skills):
            cands.extend(self.skills.candidates(obs.raw, profile, obs.summary))
        return cands

    def _learn_from_death(self, safe_history, death_x: int) -> int | None:
        """After a death, look back through recent snapshots for one with enough RUNWAY before the
        death spot, then search there for a sequence that gets PAST it and cache it keyed to that
        spot. This is what advances the frontier: next attempt replays the survivor instead of
        walking into the same death. Runs on a clone in search_mode (invisible). Returns the x it
        learned to pass, or None. Trying several start points gives a stomp/clear room to set up."""
        for x, lk, snap in reversed(safe_history):      # nearest the death first
            runway = death_x - x
            if runway < config.MIN_RUNWAY_PX:
                continue                                # too close to set up an escape
            if runway > config.LEARN_HORIZON_FRAMES:
                break                                   # further back is out of rollout reach
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
                        lived, reached = self._rollout(plan, settle=config.LEARN_HORIZON_FRAMES)
                        if lived and reached > death_x and reached > best_reach:
                            best_plan, best_reach = plan, reached
                    if best_plan is not None:
                        break   # cracked it with this set — don't pay for the next (denser) one
            self.session.restore(snap)
            self._observe()
            if best_plan is not None:
                self.cache.put(lk, x, best_plan, best_reach)
                return x
        return None

    def _candidates_from(self, snap) -> list[Plan]:
        """Candidate escapes generated from a snapshot's observation (restores it to read state)."""
        self.session.restore(snap)
        return self._candidates(self._observe())

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
    def run_attempt(self, n: int) -> metrics.AttemptResult:
        t0 = time.monotonic()
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
        final_score = obs.score
        level_start_frame = 0
        need_checkpoint = False
        outcome = "timeout"
        furthest = obs.level_label
        # A short trail of recent (progress, level_key, savestate) snapshots. On death we search
        # from one with enough RUNWAY before the death spot (a stomp/clear needs room to set up),
        # not the frame flush against it. Throttled to ~one snapshot per tile.
        safe_history: deque = deque(maxlen=24)
        safe_history.append((obs.progress, obs.level_key, self.session.clone_state()))
        last_snap_x = obs.progress

        while frames <= config.MAX_ATTEMPT_FRAMES:
            # --- death: LEARN from it (search the approach for a survivor), then respawn -----
            if obs.dead:
                death_at = max(seg_best, obs.progress)
                # The key to advancing the frontier: search from a recent safe spot (with runway)
                # for a sequence that gets PAST where we just died, and remember it keyed there.
                learned_x = self._learn_from_death(safe_history, death_at)
                if learned_x is not None:
                    search_calls += 1
                    print(f'  [attempt {n}] 🧠 learned to pass {obs.level_label}@{learned_x} '
                          f'(died at {death_at})')
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
                continue

            # --- level clear: keep going into the next level, then checkpoint it ----------
            # Compare only (world, stage): the cache key carries AREA too (so pipe warps get their
            # own bucket region), but an internal area boundary inside a level — e.g. 1-2's joined
            # areas — is NOT a level clear and must not over-count or trip the per-attempt cap.
            if obs.level_key[:2] > self.cur_level[:2]:
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
                seg_best, need_checkpoint = obs.progress, True
                if levels_cleared >= config.MAX_LEVELS_PER_ATTEMPT:
                    outcome = "clear"
                    self.session.send_plan(_IDLE)
                    self._observe()
                    break
                self.session.send_plan(_IDLE)   # coast through the level-end cutscene
                frames += 2
                obs = self._observe()
                continue

            # --- normal play -------------------------------------------------------------
            decision = self.reflex.step(obs)
            self.recent.append(decision.note)
            danger = decision.needs_billy and ("enemy" in decision.note or "pit" in decision.note)
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
            cached = self.cache.get(lk, obs.progress) if on_ground else None
            if cached is not None or danger:
                # Stall breaker: keep arriving at the same spot without getting past it -> give up.
                bkey = bucket_of(lk, obs.progress)
                bucket_visits[bkey] = bucket_visits.get(bkey, 0) + 1
                if bucket_visits[bkey] > config.MAX_BUCKET_VISITS:
                    print(f'  [attempt {n}] {obs.progress} 🧱 stuck at {obs.level_label}@{obs.progress} — giving up this run')
                    outcome = "stuck"
                    break

            # Cached solution here -> REPLAY IT VERBATIM (deterministic, no search). We deliberately
            # do NOT verify-then-research by default: the emulator is deterministic, so replaying the
            # same plans from the same checkpoint reproduces the exact trajectory (every enemy in the
            # same phase) — verify→research is what INJECTS timing drift (a re-search can pick a
            # different-duration plan, shifting everything downstream and snowballing). If a replay
            # genuinely dies, record_fail (below) drops it and it re-searches once: self-healing.
            # Set BILLY_VERIFY_REPLAY=1 to gate replays on a clone-check instead.
            do_replay = cached is not None and (
                not config.VERIFY_REPLAY
                or self._verify(cached.plan,
                                max(plan_frames(cached.plan) + 24, config.SEARCH_HORIZON_FRAMES)))
            if do_replay:
                plan = cached.plan
                replay_calls += 1
                self.cache.record_hit(lk, obs.progress)
                action_note = f"replay {self._label(plan)}"
            elif cached is not None or danger:
                # MISS or STALE cache: search (invisibly, on a clone of the LIVE state, so moving
                # enemies are where they actually are now) for a surviving, FORWARD-PROGRESSING
                # sequence and REMEMBER it. A non-progress escape is a stall — never cached.
                best_plan, progressed, reach = self._micro_search(self._candidates(obs), obs.progress)
                # Hard wall: the focused spread couldn't progress -> optionally try a DENSE
                # brute-force grid before the LLM (deterministic, model-free, but ~40 candidates so
                # it's slow — opt in with BILLY_EXPANDED_SEARCH=1).
                expand = getattr(self.reflex, "expanded_candidates", None)
                if not progressed and expand is not None and config.EXPANDED_FALLBACK:
                    # Longer settle so a retreat/drop detour off a dead-end ledge can recover and
                    # show net forward progress (a short horizon only sees the leftward dip).
                    best_plan, progressed, reach = self._micro_search(
                        expand(obs), obs.progress, early_exit=True,
                        settle=config.LEARN_HORIZON_FRAMES)
                plan = best_plan
                search_calls += 1
                if progressed:
                    # force-overwrite when refreshing a stale entry so the freshest WORKING plan
                    # replaces the one that just failed verify (else we'd re-search it forever).
                    self.cache.put(lk, obs.progress, plan, reach, force=cached is not None)
                    tag = "research✓" if cached is not None else "search✓"
                    action_note = f"{tag} {self._label(plan)}"
                    reach_str = "→ pipe/area" if reach >= self._TRANSITION_BONUS else f"reach {reach}"
                    print(f'  [attempt {n}] {obs.progress} 🔍 solved ({reach_str}) — remembered')
                elif self.use_llm:
                    # Genuinely hard: let Billy improvise. Then VERIFY his plan on a clone and, if it
                    # survives AND progresses, BANK it like a search win — so the next pass replays
                    # it instead of re-asking the (slow, inconsistent) LLM. This is what lets an
                    # LLM-cracked wall (e.g. early 1-2) actually compound.
                    bd = billy.decide(obs, self.kb.retrieve(obs.summary),
                                      list(self.recent), self.controller)
                    plan = bd.plan
                    billy_calls += 1
                    survived, reach = self._evaluate(plan)
                    if survived and reach > obs.progress + self._MIN_PROGRESS_PX:
                        self.cache.put(lk, obs.progress, plan, reach, force=cached is not None)
                        action_note = f"BILLY✓ {self._label(plan)}"
                        print(f'  [attempt {n}] {obs.progress} 🎮 Billy: "{bd.trash_talk}" — '
                              f'cracked it (reach {reach}), remembered')
                    else:
                        action_note = f"BILLY {self._label(plan)}"
                        print(f'  [attempt {n}] {obs.progress} 🎮 Billy: "{bd.trash_talk}"')
                else:
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
            replay_x = obs.progress
            obs, last_snap_x = self._commit(plan, safe_history, last_snap_x)
            if "replay" in action_note and obs.dead:
                self.cache.record_fail(lk, replay_x)   # a banked solution drifted -> re-search it
            trajectory.append(coach.TrajectoryStep(
                x=replay_x, summary=obs.summary, action=action_note, event=decision.note))
            frames += plan_frames(plan)
            seg_best = max(seg_best, obs.progress)
            final_score = max(final_score, obs.score)
            if frames_to_frontier == 0 and self._prev_best_x > 0 and obs.progress >= self._prev_best_x:
                frames_to_frontier = frames   # re-reached last attempt's furthest point

            if need_checkpoint and getattr(obs.raw, "on_ground", True) and 16 < obs.progress < 120:
                self.session.save_state(0)
                self._observe()
                need_checkpoint = False
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
        result = metrics.AttemptResult(
            attempt=n, outcome=outcome, max_x=seg_best, frames=frames, billy_calls=billy_calls,
            world_stage=furthest, levels_cleared=levels_cleared, score=final_score,
            fastest_clear_frames=fastest_in_attempt, duration_s=round(time.monotonic() - t0, 2),
            search_calls=search_calls, replay_calls=replay_calls,
            frontier_x=self.cache.solved_frontier(start_level), frames_to_frontier=frames_to_frontier)
        metrics.record(result)
        print(f"  [attempt {n}] {outcome.upper()} — reached {furthest}, "
              f"cleared {levels_cleared} level(s), score {final_score}{hi}")
        print(f"      ↳ search={search_calls} replay={replay_calls} "
              f"frontier_x={result.frontier_x} cache={len(self.cache)}")
        return result

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
        results = [self.run_attempt(n) for n in range(1, attempts + 1)]
        metrics.print_curve(results)
        return results

