"""The Director — the game-agnostic engine loop.

Drives any Game through the abstract contracts: observe/act, boot, then a continuous
playthrough — reflex for routine play, and at each hazard a **cache-first policy**: replay the
exact verified solution if we've solved this spot before, else **search on a cloned state**
(invisible to the live run) for a surviving sequence and **remember it**. The LLM is consulted
only when search finds nothing. This is what makes the learning compound across attempts.
It never references a specific game or system.
"""
from __future__ import annotations

import time
from collections import deque

from . import config, metrics
from .abstractions import Game, Observation, Plan, Step, plan_frames
from .agents import billy, coach
from .commentary import Commentator
from .knowledge import KnowledgeBase, SolutionCache

_IDLE: Plan = [Step(2, 0)]   # neutral input — to consume a pending frame or coast a cutscene


class Director:
    def __init__(self, game: Game, kb: KnowledgeBase, use_llm: bool = True,
                 cache: SolutionCache | None = None) -> None:
        self.game = game
        self.session = game.system.connect()
        self.controller = game.system.controller
        self.reflex = game.make_reflex()
        self.kb = kb
        self.cache = cache if cache is not None else SolutionCache()  # the compounding policy
        self.use_llm = use_llm
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
    def _rollout(self, plan: Plan) -> tuple[bool, int]:
        """Simulate a candidate for the FULL horizon (so a delayed hit — e.g. a Koopa reaching
        Mario a few frames after he lands — still counts as death). Returns (survived, farthest)."""
        self.session.send_plan(plan)
        obs = self._observe()
        reached = obs.progress
        used = plan_frames(plan)
        while used < config.SEARCH_HORIZON_FRAMES and not obs.dead:
            self.session.send_plan(_IDLE)
            obs = self._observe()
            reached = max(reached, obs.progress)
            used += 2
        return (not obs.dead), reached

    def _micro_search(self, candidates: list[Plan]) -> tuple[Plan, bool, int]:
        """Try each candidate on a cloned state; return (best_plan, survived, reach_after).

        Runs inside the session's search_mode + a state clone, so none of the candidate frames
        are displayed and the live game is left exactly where it started (invisible search)."""
        snap = self.session.clone_state()
        best_plan, best_score, best_reach = candidates[0], -10 ** 9, 0
        with self.session.search_mode():
            for plan in candidates:
                survived, reached = self._rollout(plan)
                score = reached if survived else reached - 100_000  # death ≫ worse than short
                if score > best_score:
                    best_score, best_plan, best_reach = score, plan, reached
                self.session.restore(snap)
        self.session.restore(snap)   # back to the live pre-search position, nothing shown
        self._observe()
        return best_plan, best_score >= 0, best_reach

    def _candidates(self, obs: Observation) -> list[Plan]:
        """Diverse escape sequences for search: the reflex's hand-picked spread, plus — when the
        spot is genuinely hard — a few LLM-proposed sequences from Billy."""
        cands = list(self.reflex.danger_candidates(obs))
        return cands

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

        trajectory: list[coach.TrajectoryStep] = []
        billy_calls = frames = levels_cleared = fastest_in_attempt = 0
        search_calls = replay_calls = 0          # compounding-curve telemetry
        frames_to_frontier = 0                    # frames to re-reach last attempt's furthest x
        respawns = config.RESPAWNS_PER_ATTEMPT
        seg_best = obs.progress
        final_score = obs.score
        level_start_frame = 0
        need_checkpoint = False
        outcome = "timeout"
        furthest = obs.level_label

        while frames <= config.MAX_ATTEMPT_FRAMES:
            # --- death: reflect, then respawn at the checkpoint (or end) ------------------
            if obs.dead:
                death_at = max(seg_best, obs.progress)
                self._reflect(trajectory, "death", obs.level_label)
                trajectory = []
                self.session.send_plan(_IDLE)   # consume the pending frame
                self._observe()
                if respawns <= 0:
                    outcome = "game_over"
                    break
                respawns -= 1
                print(f'  [attempt {n}] 💀 down at {obs.level_label} ({death_at}) — Billy: '
                      f'"{self.commentator.death_quip(obs.raw)}" ({respawns} retries left)')
                self.session.load_state(0)
                obs = self._observe()
                self.reflex.reset(obs)
                self.commentator.reset(obs.raw)
                self.recent.clear()
                seg_best, need_checkpoint = obs.progress, False
                continue

            # --- level clear: keep going into the next level, then checkpoint it ----------
            if obs.level_key > self.cur_level:
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

            if danger:
                # CACHE-FIRST: if we've solved this exact spot before, replay it verbatim —
                # deterministic, no search, no LLM. This is the compounding fast-path.
                cached = self.cache.get(lk, obs.progress)
                if cached is not None:
                    plan = cached.plan
                    replay_calls += 1
                    self.cache.record_hit(lk, obs.progress)
                    action_note = f"replay {self._label(plan)}"
                    print(f'  [attempt {n}] {obs.progress} ⚡ replay (solved {obs.level_label}@{obs.progress})')
                else:
                    # MISS: search (invisibly, on a clone) for a surviving sequence and REMEMBER it.
                    best_plan, survived, reach = self._micro_search(self._candidates(obs))
                    plan = best_plan
                    search_calls += 1
                    if survived:
                        self.cache.put(lk, obs.progress, plan, reach)
                        action_note = f"search✓ {self._label(plan)}"
                        print(f'  [attempt {n}] {obs.progress} 🔍 solved (reach {reach}) — remembered')
                    elif self.use_llm:
                        # Genuinely hard: let Billy improvise (out of the routine path).
                        bd = billy.decide(obs, self.kb.retrieve(obs.summary),
                                          list(self.recent), self.controller)
                        plan = bd.plan
                        billy_calls += 1
                        action_note = f"BILLY {self._label(plan)}"
                        print(f'  [attempt {n}] {obs.progress} 🎮 Billy: "{bd.trash_talk}"')
                    else:
                        action_note = f"search✗ {self._label(plan)}"
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

            self.session.send_plan(plan)
            # If a replayed solution didn't survive (context drifted), drop it so search refreshes it.
            if danger and "replay" in action_note and self._observe().dead:
                self.cache.record_fail(lk, obs.progress)
            trajectory.append(coach.TrajectoryStep(
                x=obs.progress, summary=obs.summary, action=action_note, event=decision.note))
            frames += plan_frames(plan)
            obs = self._observe()
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
            frontier_x=self.cache.solved_frontier(obs.level_key), frames_to_frontier=frames_to_frontier)
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

