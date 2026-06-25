"""The Director — the game-agnostic engine loop.

Drives any Game through the abstract contracts: lock-step observe/act, boot, then per attempt
a continuous playthrough — reflex policy for routine play, Billy (LLM) at decision points,
Coach + knowledge base after each segment, danger-zone micro-search, checkpoints, respawns,
and record tracking (score + fastest clear). It never references a specific game or system.
"""
from __future__ import annotations

import json
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
        self.zone_escapes: dict[int, Plan] = self._load_escapes()  # persisted across sessions
        self.best_score = 0
        self.fastest_clear_frames: int | None = None
        self.cur_level: tuple = ()

    # --- zone-escape persistence --------------------------------------------------------
    def _load_escapes(self) -> dict[int, Plan]:
        try:
            raw = json.loads(config.ESCAPES_FILE.read_text())
            escapes = {int(z): [Step(s[0], s[1]) for s in steps] for z, steps in raw.items()}
            if escapes:
                print(f"[director] loaded {len(escapes)} remembered escape(s) from disk 🧠")
            return escapes
        except FileNotFoundError:
            return {}
        except Exception as exc:
            print(f"[director] could not load escapes: {exc}")
            return {}

    def _save_escapes(self) -> None:
        try:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            payload = {str(z): [[s.frames, s.buttons] for s in steps]
                       for z, steps in self.zone_escapes.items()}
            config.ESCAPES_FILE.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            print(f"[director] could not save escapes: {exc}")

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
        n_mem = len(self.zone_escapes)
        mem_note = f" ({n_mem} escape(s) remembered)" if n_mem else ""
        print(f"[director] in play at {start.level_label}, progress={start.progress}. "
              f"Billy has taken the controller.{mem_note}")
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

            # --- normal play -------------------------------------------------------------
            decision = self.reflex.step(obs)
            self.recent.append(decision.note)
            zone = self._danger_zone_near(obs.progress)

            # Learn-from-death: at a spot Mario has died before, SEARCH for a survivor (try a
            # spread of escapes, keep the one that lives) and REMEMBER it so the next attempt
            # skips straight to it. Reliable and fast — no dependence on the flaky local LLM.
            if zone is not None and zone != consulted_zone and config.MICRO_SEARCH:
                consulted_zone = zone
                plan = self.zone_escapes.get(zone)
                if plan is not None:
                    action_note = "learned-escape"
                    print(f"  [attempt {n}] {obs.progress} 🧠 Billy remembers the escape here")
                else:
                    best, tried, survived = self._micro_search(self.reflex.danger_candidates(obs))
                    plan = best if survived else None
                    if survived:
                        self.zone_escapes[zone] = best
                        self._save_escapes()
                        action_note = f"danger-search/{tried}"
                        print(f"  [attempt {n}] {obs.progress} 🔎 died here before — tried "
                              f"{tried} escapes and LEARNED the one that survives")
                if plan is not None:
                    self.session.send_plan(plan)
                    trajectory.append(coach.TrajectoryStep(
                        x=obs.progress, summary=obs.summary, action=action_note, event="danger"))
                    frames += plan_frames(plan)
                    obs = self._observe()
                    seg_best = max(seg_best, obs.progress)
                    final_score = max(final_score, obs.score)
                    continue
            if zone is None:
                consulted_zone = None

            # Billy on a hard stall; otherwise the reflex's plan.
            if decision.needs_billy and self.use_llm:
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
        """Run a candidate from the search checkpoint and keep simulating for the FULL horizon
        (not just until it lands) so a delayed hit — e.g. a Koopa reaching Mario a few frames
        after he touches down — correctly counts as a death. Returns (survived, farthest)."""
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

    def _micro_search(self, candidates: list[Plan]) -> tuple[Plan, int, bool]:
        """Try each candidate from a checkpoint; return (best, num_tried, best_survived)."""
        self.session.save_state(config.SEARCH_SLOT)
        self._observe()
        best_plan, best_score = candidates[0], -10 ** 9
        for plan in candidates:
            survived, reached = self._rollout(plan)
            score = reached if survived else reached - 100_000  # death is far worse than short
            if score > best_score:
                best_score, best_plan = score, plan
            self.session.load_state(config.SEARCH_SLOT)
            self._observe()
        return best_plan, len(candidates), best_score >= 0

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
