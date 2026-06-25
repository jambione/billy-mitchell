"""SMB Tier-1 reflex policy — the fast, no-LLM layer for Super Mario Bros.

Implements the engine's ReflexPolicy: each exchange it returns a controller plan (run, hop a
gap, stomp, steer mid-air, bonk a block…) or flags `needs_billy` when truly stuck. Terminal
events (death / level clear) are detected by the engine from the Observation, not here.
Billy can append learned *reflex rules* that fire instantly (no LLM round-trip next time).
"""
from __future__ import annotations

from typing import Callable

from ...abstractions import Decision, Observation, Plan, ReflexPolicy, Step
from ...systems.nes import controller
from . import tuning
from .perception import Scene

ReflexRule = Callable[[Scene], "Plan | None"]


class SmbReflex(ReflexPolicy):
    def __init__(self) -> None:
        self.reflex_rules: list[ReflexRule] = []
        self._best = 0
        self._frames_stuck = 0
        self._last_scene: Scene | None = None

    def reset(self, obs: Observation) -> None:
        self._best = obs.progress
        self._frames_stuck = 0
        self._last_scene = None

    def note_level_advance(self, obs: Observation) -> None:
        self._best = obs.progress
        self._frames_stuck = 0

    def add_reflex_rule(self, rule: ReflexRule) -> None:
        self.reflex_rules.append(rule)

    def _detect_scene_change(self, scene: Scene) -> str | None:
        """Return a change description if something truly pivotal happened.
        Triggered on major events only, so Billy gets called with strategy context."""
        if self._last_scene is None:
            return None
        last = self._last_scene

        # Enemy count changed (new enemy appeared on screen or one left)
        if len(scene.enemies) != len(last.enemies):
            change = "appeared" if len(scene.enemies) > len(last.enemies) else "left"
            return f"enemy_{change}"

        # Enemy entered reaction range — fire EARLY (72px, ~1s of runway) so micro-search has room
        # to find a stomp/dodge BEFORE contact, instead of reacting once it's already touching.
        near_now = scene.nearest_enemy(within=72)
        near_before = last.nearest_enemy(within=72)
        if (near_now is not None) != (near_before is not None):
            return "enemy_close" if near_now is not None else "enemy_far"

        # Gap detection state changed (pit detected/cleared)
        gap_now = scene.gap_ahead()
        gap_before = last.gap_ahead()
        if gap_now != gap_before:
            return "pit_ahead" if gap_now else "pit_clear"

        # Size/powerup changed (hit a powerup or lost it)
        if scene.size != last.size:
            return "powerup_hit"

        return None

    # --- mid-air steering toward a landing target ---------------------------------------
    def _air_steer(self, scene: Scene) -> "Plan | None":
        target = scene.air_landing_target()
        if target is None:
            return None
        dx = target - scene.mario_x
        f = tuning.AIR_STEER_FRAMES
        if dx > 12:
            return [Step(f, controller.mask(controller.RIGHT, controller.B))]   # chase
        if dx > 3:
            return [Step(f, controller.RIGHT)]                                  # ease in
        if dx < -12:
            return [Step(f, controller.mask(controller.LEFT, controller.B))]    # pull back
        if dx < -3:
            return [Step(f, controller.LEFT)]
        return [Step(f, controller.NEUTRAL)]                                    # drop on it

    def advance_plan(self, obs: Observation) -> Plan:
        """Forward coast for micro-search rollouts: keep running right (no jump) so a candidate is
        scored while Mario keeps moving through the hazard zone, not standing on neutral."""
        return controller.run_right(tuning.REFLEX_STEP_FRAMES, sprint=True)

    def danger_candidates(self, obs: Observation) -> list[Plan]:
        """A focused spread of escapes — covers pits, enemies AND tall obstacles (pipes).
        Includes PATIENCE (let a Koopa come into range, then stomp) and a RUNNING START + a
        BACK-UP-and-run-jump so a flush-against-a-pipe spot can actually be cleared (otherwise a
        standing jump stalls there forever)."""
        c = controller
        L = controller.LEFT
        return [
            c.jump_right(jump_frames=28),                       # higher jump (clear a pit)
            c.jump_right(jump_frames=34),                       # max jump
            c.idle(16) + c.jump_right(jump_frames=14),          # wait a beat, then stomp
            c.idle(42) + c.jump_right(jump_frames=14),          # let the Koopa come to you
            c.jump_right(run_frames=14, jump_frames=24),        # running jump (clear a pipe)
            c.jump_right(run_frames=24, jump_frames=30),        # long running jump (wide pit)
            [Step(10, L)] + c.jump_right(run_frames=18, jump_frames=30),  # back up, then run-jump
        ]

    def _jump_candidates(self, base: int, width: int) -> list[Plan]:
        """Variants the engine's micro-search tries at a deadly pit (vary A-hold + launch)."""
        holds = sorted({max(tuning.JUMP_MIN_FRAMES, min(base + d, tuning.JUMP_MAX_FRAMES))
                        for d in (-8, -4, 0, 4, 8)})
        cands: list[Plan] = [controller.jump_right(jump_frames=h) for h in holds]
        cands.append(controller.jump_right(run_frames=4, jump_frames=base))
        return cands

    # --- the per-exchange decision ------------------------------------------------------
    def step(self, obs: Observation) -> Decision:
        scene: Scene = obs.raw
        if obs.progress > self._best:
            self._frames_stuck = 0
            self._best = obs.progress
        else:
            self._frames_stuck += tuning.REFLEX_STEP_FRAMES

        # Scene change: let Billy react in the moment.
        change = self._detect_scene_change(scene)
        if change is not None:
            self._last_scene = scene
            return Decision([], needs_billy=True, note=f"scene-change: {change}")

        # Stuck: reflexes have failed — escalate to Billy.
        if self._frames_stuck >= tuning.STUCK_FRAMES:
            self._last_scene = scene
            return Decision([], needs_billy=True, note=f"stuck {self._frames_stuck}f")

        # Billy-taught reflex rules get first crack.
        for rule in self.reflex_rules:
            plan = rule(scene)
            if plan:
                self._last_scene = scene
                return Decision(plan, note="reflex-rule")

        # Airborne: steer toward a landing target (D-pad + boost) instead of a blind arc.
        if not scene.on_ground:
            steer = self._air_steer(scene)
            self._last_scene = scene
            if steer is not None:
                return Decision(steer, note="air-steer")
            return Decision(controller.run_right(tuning.AIRBORNE_STEP_FRAMES), note="airborne carry")

        # Bump jump: brief on-ground stall => hop (catches blocks the geometry missed).
        if tuning.BUMP_FRAMES <= self._frames_stuck < tuning.STUCK_FRAMES:
            self._last_scene = scene
            return Decision(controller.jump_right(jump_frames=28), note="bump jump")

        gap = scene.gap_info()
        imminent_pit = gap is not None and gap[0] <= tuning.JUMP_TRIGGER_PX

        # Stomp a close enemy — unless a pit is imminent (clear the pit first).
        near = scene.nearest_enemy(tuning.STOMP_RANGE)
        if near is not None and not imminent_pit:
            dx, dy = near
            self._last_scene = scene
            if dy > 24:  # enemy on lower ground: hop off the ledge to FALL onto it
                return Decision(controller.jump_right(jump_frames=6), note=f"drop-stomp @{dx}px")
            return Decision(controller.jump_right(jump_frames=tuning.STOMP_HOLD_FRAMES),
                            note=f"stomp @{dx}px")

        # Gap-aware jump (launch near the edge, A-hold scaled to width; micro-searchable).
        if gap is not None:
            dist_px, width = gap
            self._last_scene = scene
            if dist_px <= tuning.JUMP_TRIGGER_PX:
                jf = tuning.JUMP_BASE_FRAMES + width * tuning.JUMP_PER_TILE_FRAMES
                jf = max(tuning.JUMP_MIN_FRAMES, min(jf, tuning.JUMP_MAX_FRAMES))
                return Decision(controller.jump_right(jump_frames=jf), note=f"jump pit w={width}",
                                search_candidates=self._jump_candidates(jf, width))
            return Decision(controller.run_right(tuning.REFLEX_STEP_FRAMES),
                            note=f"approach pit ({dist_px}px)")

        # Wall / pipe / stair: run up and jump over it (hold scaled to height).
        obstacle = scene.obstacle_ahead()
        if obstacle is not None:
            dist_px, height = obstacle
            self._last_scene = scene
            if dist_px <= tuning.OBSTACLE_TRIGGER_PX:
                jf = tuning.OBSTACLE_BASE_FRAMES + height * tuning.OBSTACLE_PER_HEIGHT_FRAMES
                jf = max(tuning.JUMP_MIN_FRAMES, min(jf, tuning.JUMP_MAX_FRAMES))
                return Decision(controller.jump_right(jump_frames=jf), note=f"jump wall h={height}")
            return Decision(controller.run_right(tuning.REFLEX_STEP_FRAMES),
                            note=f"approach wall ({dist_px}px)")

        if scene.enemy_ahead():
            self._last_scene = scene
            return Decision(controller.jump_right(jump_frames=20), note="hop enemy")

        # Coins / power-ups: bonk a block — but survival first, so ONLY when it's safe (no
        # enemy or pit nearby). Never risk a hit/fall for a coin.
        bonk = scene.block_above_ahead()
        safe_to_bonk = gap is None and scene.nearest_enemy(96) is None
        if bonk is not None and bonk <= tuning.BONK_TRIGGER_PX and safe_to_bonk:
            self._last_scene = scene
            return Decision(controller.jump_right(jump_frames=tuning.BONK_HOLD_FRAMES),
                            note="bonk block (coin/powerup)")

        self._last_scene = scene
        return Decision(controller.run_right(tuning.REFLEX_STEP_FRAMES), note="cruise")
