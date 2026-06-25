"""The Director — the game-agnostic engine loop.

Drives any Game through the abstract contracts: lock-step observe/act, boot, then per attempt
a continuous playthrough — reflex policy for routine play, Billy (LLM) at decision points,
Coach + knowledge base after each segment, danger-zone micro-search, checkpoints, respawns,
and record tracking (score + fastest clear). It never references a specific game or system.
"""
from __future__ import annotations

import time
from collections import deque

from . import config, metrics
from .abstractions import Game, Observation, Plan, Step, plan_frames
from .agents import billy, coach
from .commentary import Commentator
from .knowledge import KnowledgeBase

_IDLE: Plan = [Step(2, 0)]   # neutral input — to consume a pending frame or coast a cutscene


class Director:
    def __init__(self, game: Game, kb: KnowledgeBase, use_llm: bool = True) -> None:
        self.game = game
        self.session = game.system.connect()
        self.controller = game.system.controller
        self.reflex = game.make_reflex()
        self.kb = kb
        self.use_llm = use_llm
        self.recent: deque[str] = deque(maxlen=12)
        self.commentator = Commentator()
        self.danger_zones: set[int] = set()
        self.best_score = 0
        self.fastest_clear_frames: int | None = None
        self.cur_level: tuple = ()

    # --- lock-step helper ---------------------------------------------------------------
    def _observe(self) -> Observation:
        st = self.session.read_state()
        return self.game.observe(st.frame, st.ram)

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
        respawns = config.RESPAWNS_PER_ATTEMPT
        seg_best = obs.progress
        final_score = obs.score
        level_start_frame = 0
        consulted_zone: int | None = None
        need_checkpoint = False
        outcome = "timeout"
        furthest = obs.level_label

        while frames <= config.MAX_ATTEMPT_FRAMES:
            # --- death: reflect, then respawn at the checkpoint (or end) ------------------
            if obs.dead:
                death_at = max(seg_best, obs.progress)
                self.danger_zones.add(round(death_at / config.DANGER_BUCKET) * config.DANGER_BUCKET)
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
                seg_best, consulted_zone, need_checkpoint = obs.progress, None, False
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
                self.danger_zones.clear()
                self.commentator.reset(obs.raw)
                self.cur_level = obs.level_key
                seg_best, consulted_zone, need_checkpoint = obs.progress, None, True
                if levels_cleared >= config.MAX_LEVELS_PER_ATTEMPT:
                    outcome = "clear"
                    self.session.send_plan(_IDLE)
                    self._observe()
                    break
                self.session.send_plan(_IDLE)   # coast through the level-end cutscene
                frames += 2
                obs = self._observe()
                continue

            # --- normal play: danger-zone consult -> Billy / reflex / micro-search --------
            decision = self.reflex.step(obs)
            self.recent.append(decision.note)
            zone = self._danger_zone_near(obs.progress)
            ask_billy = decision.needs_billy
            if self.use_llm and zone is not None and zone != consulted_zone:
                ask_billy = True
                consulted_zone = zone
                print(f"  [attempt {n}] {obs.progress} ⚠️  danger zone (died here before) "
                      f"— Billy takes manual control")
            elif zone is None:
                consulted_zone = None

            if ask_billy and self.use_llm:
                bd = billy.decide(obs, self.kb.retrieve(obs.summary), list(self.recent),
                                  self.controller)
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
                # Micro-search only at spots where Billy has DIED before (worth the rewind).
                if (config.MICRO_SEARCH and zone is not None and decision.search_candidates):
                    best, tried = self._micro_search(decision.search_candidates)
                    plan = best
                    action_note = f"search/{tried} -> {self._label(best)}"
                    print(f"  [attempt {n}] {obs.progress} 🔎 Billy tries {tried} options "
                          f"at the deadly spot, keeps the cleanest")

            self.session.send_plan(plan)
            trajectory.append(coach.TrajectoryStep(
                x=obs.progress, summary=obs.summary, action=action_note, event=decision.note))
            frames += plan_frames(plan)
            obs = self._observe()
            seg_best = max(seg_best, obs.progress)
            final_score = max(final_score, obs.score)

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
        result = metrics.AttemptResult(
            attempt=n, outcome=outcome, max_x=seg_best, frames=frames, billy_calls=billy_calls,
            world_stage=furthest, levels_cleared=levels_cleared, score=final_score,
            fastest_clear_frames=fastest_in_attempt, duration_s=round(time.monotonic() - t0, 2))
        metrics.record(result)
        print(f"  [attempt {n}] {outcome.upper()} — reached {furthest}, "
              f"cleared {levels_cleared} level(s), score {final_score}{hi}")
        return result

    # --- savestate micro-search (game provides the candidate actions) -------------------
    def _rollout(self, plan: Plan) -> tuple[bool, int]:
        """Run a candidate from the search checkpoint until it lands/dies/horizon; return
        (survived, farthest_progress). Coasts on neutral input — physics carries momentum."""
        self.session.send_plan(plan)
        obs = self._observe()
        reached = obs.progress
        used = plan_frames(plan)
        while used < config.SEARCH_HORIZON_FRAMES and not obs.dead \
                and not getattr(obs.raw, "on_ground", True):
            self.session.send_plan(_IDLE)
            obs = self._observe()
            reached = max(reached, obs.progress)
            used += 2
        return (not obs.dead), reached

    def _micro_search(self, candidates: list[Plan]) -> tuple[Plan, int]:
        self.session.save_state(config.SEARCH_SLOT)
        self._observe()
        best_plan, best_score = candidates[0], -10 ** 9
        for plan in candidates:
            survived, reached = self._rollout(plan)
            score = reached if survived else reached - 100_000
            if score > best_score:
                best_score, best_plan = score, plan
            self.session.load_state(config.SEARCH_SLOT)
            self._observe()
        return best_plan, len(candidates)

    def _danger_zone_near(self, progress: int) -> int | None:
        for z in self.danger_zones:
            if z - config.DANGER_RADIUS <= progress <= z + config.DANGER_BUCKET:
                return z
        return None

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
        print("[director] waiting for the emulator bridge…")
        self.session.wait_until_live()
        self.boot()
        results = [self.run_attempt(n) for n in range(1, attempts + 1)]
        metrics.print_curve(results)
        return results
