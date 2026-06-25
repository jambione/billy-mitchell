"""Shared NES-platformer reflex — the game-neutral Tier-1 policy.

The whole "play a side-scrolling platformer" behaviour (run right, hop gaps scaled to width,
stomp enemies, jump walls/pipes scaled to height, steer mid-air, bonk blocks, escalate when
stuck) is identical across NES platformers; only the *physics feel* and the *perception probes*
differ per game. So we factor the policy here, parameterised by:

  • a `PhysicsProfile` — the per-game tuning constants, and
  • a `PlatformerView` — the perception probes the policy reads (a game's Scene satisfies this
    structurally; no adapter glue needed).

A concrete game's reflex becomes a one-liner: `PlatformerReflex(THAT_GAMES_PROFILE)`. This is the
reflex-level half of cross-game transfer (the embedding Skill layer is the other half). Both target
games are NES, so we build plans with the NES controller directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from ...abstractions import Decision, Observation, Plan, ReflexPolicy, Step
from ...systems.nes import controller

ReflexRule = Callable[[object], "Plan | None"]


@dataclass(frozen=True)
class PhysicsProfile:
    """Per-game platformer feel. Defaults are SMB1's; another title overrides what differs."""
    reflex_step_frames: int = 4      # frames advanced per routine reflex exchange
    bump_frames: int = 14            # brief on-ground stall => hop (catch unseen blocks)
    stuck_frames: int = 80           # prolonged no-progress => escalate to Billy
    jump_trigger_px: int = 28        # launch a pit jump when this close to the edge
    jump_base_frames: int = 18
    jump_per_tile_frames: int = 4    # extra A-hold per tile of pit width
    jump_min_frames: int = 16
    jump_max_frames: int = 34
    airborne_step_frames: int = 6    # carry momentum right when no landing target
    air_steer_frames: int = 3        # short steps while steering toward a landing spot
    obstacle_trigger_px: int = 24    # jump a wall/pipe when this close
    obstacle_base_frames: int = 24
    obstacle_per_height_frames: int = 5
    stomp_range: int = 60            # commit early so the avatar is above the enemy on contact
    stomp_hold_frames: int = 16
    bonk_trigger_px: int = 14        # jump to bonk a ? block / brick
    bonk_hold_frames: int = 22
    enemy_react_px: int = 72         # scene-change fires when an enemy enters this range (~1s runway)


@runtime_checkable
class PlatformerView(Protocol):
    """The perception surface the platformer policy reads. A game's Scene implements these (most
    are already present on SMB's Scene), so it satisfies this Protocol structurally."""
    mario_x: int
    size: int
    enemies: list
    @property
    def on_ground(self) -> bool: ...
    def gap_info(self, max_tiles: int = ...) -> "tuple[int, int] | None": ...
    def nearest_enemy(self, within: int = ...) -> "tuple[int, int] | None": ...
    def obstacle_ahead(self, max_tiles: int = ...) -> "tuple[int, int] | None": ...
    def block_above_ahead(self, max_tiles: int = ...) -> "int | None": ...
    def air_landing_target(self) -> "int | None": ...
    def enemy_ahead(self, within: int = ...) -> bool: ...


# --- candidate builders (also reused by the Skill layer to instantiate transferable tactics) ----
def run_advance(profile: PhysicsProfile) -> Plan:
    return controller.run_right(profile.reflex_step_frames, sprint=True)


def gap_jumper(width: int, profile: PhysicsProfile) -> list[Plan]:
    """Variants for a deadly pit: vary A-hold around the width-scaled base + a short-runup."""
    base = max(profile.jump_min_frames,
               min(profile.jump_base_frames + width * profile.jump_per_tile_frames,
                   profile.jump_max_frames))
    holds = sorted({max(profile.jump_min_frames, min(base + d, profile.jump_max_frames))
                    for d in (-8, -4, 0, 4, 8)})
    cands: list[Plan] = [controller.jump_right(jump_frames=h) for h in holds]
    cands.append(controller.jump_right(run_frames=4, jump_frames=base))
    return cands


def enemy_stomper() -> list[Plan]:
    """Patience + commit: let an enemy come into range, then stomp from above."""
    c = controller
    return [c.idle(16) + c.jump_right(jump_frames=14), c.idle(42) + c.jump_right(jump_frames=14)]


def wall_jumper() -> list[Plan]:
    """Clear a tall obstacle (pipe/stair): running jumps + a back-up-then-run-jump for flush spots."""
    c = controller
    return [c.jump_right(run_frames=14, jump_frames=24), c.jump_right(run_frames=24, jump_frames=30),
            [Step(10, controller.LEFT)] + c.jump_right(run_frames=18, jump_frames=30)]


class PlatformerReflex(ReflexPolicy):
    """Game-neutral platformer Tier-1 policy. Behaviour is identical to the original SMB reflex
    when given SMB's PhysicsProfile (the 1-1 clear is the regression guard)."""

    def __init__(self, profile: PhysicsProfile) -> None:
        self.p = profile
        self.reflex_rules: list[ReflexRule] = []
        self._best = 0
        self._frames_stuck = 0
        self._last_scene = None

    def reset(self, obs: Observation) -> None:
        self._best = obs.progress
        self._frames_stuck = 0
        self._last_scene = None

    def note_level_advance(self, obs: Observation) -> None:
        self._best = obs.progress
        self._frames_stuck = 0

    def add_reflex_rule(self, rule: ReflexRule) -> None:
        self.reflex_rules.append(rule)

    # --- scene-change detection (escalate to Billy / trigger search at pivotal moments) ----
    def _detect_scene_change(self, scene) -> "str | None":
        if self._last_scene is None:
            return None
        last = self._last_scene
        if len(scene.enemies) != len(last.enemies):
            return "enemy_appeared" if len(scene.enemies) > len(last.enemies) else "enemy_left"
        near_now = scene.nearest_enemy(within=self.p.enemy_react_px)
        near_before = last.nearest_enemy(within=self.p.enemy_react_px)
        if (near_now is not None) != (near_before is not None):
            return "enemy_close" if near_now is not None else "enemy_far"
        gap_now = scene.gap_ahead()
        gap_before = last.gap_ahead()
        if gap_now != gap_before:
            return "pit_ahead" if gap_now else "pit_clear"
        if scene.size != last.size:
            return "powerup_hit"
        return None

    def _air_steer(self, scene) -> "Plan | None":
        target = scene.air_landing_target()
        if target is None:
            return None
        dx = target - scene.mario_x
        f = self.p.air_steer_frames
        if dx > 12:
            return [Step(f, controller.mask(controller.RIGHT, controller.B))]
        if dx > 3:
            return [Step(f, controller.RIGHT)]
        if dx < -12:
            return [Step(f, controller.mask(controller.LEFT, controller.B))]
        if dx < -3:
            return [Step(f, controller.LEFT)]
        return [Step(f, controller.NEUTRAL)]

    def advance_plan(self, obs: Observation) -> Plan:
        """Forward coast for micro-search rollouts: keep running right (no jump)."""
        return run_advance(self.p)

    def danger_candidates(self, obs: Observation) -> list[Plan]:
        """Focused escape spread — pits, enemies AND tall obstacles, with patience + running/back-up
        jumps so a flush-against-a-pipe spot can actually be cleared."""
        c = controller
        return [
            c.jump_right(jump_frames=28),
            c.jump_right(jump_frames=34),
            *enemy_stomper(),
            *wall_jumper(),
        ]

    def _jump_candidates(self, base: int, width: int) -> list[Plan]:
        return gap_jumper(width, self.p)

    # --- the per-exchange decision (ported verbatim from the SMB reflex) ------------------
    def step(self, obs: Observation) -> Decision:
        scene = obs.raw
        p = self.p
        if obs.progress > self._best:
            self._frames_stuck = 0
            self._best = obs.progress
        else:
            self._frames_stuck += p.reflex_step_frames

        change = self._detect_scene_change(scene)
        if change is not None:
            self._last_scene = scene
            return Decision([], needs_billy=True, note=f"scene-change: {change}")

        if self._frames_stuck >= p.stuck_frames:
            self._last_scene = scene
            return Decision([], needs_billy=True, note=f"stuck {self._frames_stuck}f")

        for rule in self.reflex_rules:
            plan = rule(scene)
            if plan:
                self._last_scene = scene
                return Decision(plan, note="reflex-rule")

        if not scene.on_ground:
            steer = self._air_steer(scene)
            self._last_scene = scene
            if steer is not None:
                return Decision(steer, note="air-steer")
            return Decision(controller.run_right(p.airborne_step_frames), note="airborne carry")

        if p.bump_frames <= self._frames_stuck < p.stuck_frames:
            self._last_scene = scene
            return Decision(controller.jump_right(jump_frames=28), note="bump jump")

        gap = scene.gap_info()
        imminent_pit = gap is not None and gap[0] <= p.jump_trigger_px

        near = scene.nearest_enemy(p.stomp_range)
        if near is not None and not imminent_pit:
            dx, dy = near
            self._last_scene = scene
            if dy > 24:  # enemy on lower ground: hop off the ledge to FALL onto it
                return Decision(controller.jump_right(jump_frames=6), note=f"drop-stomp @{dx}px")
            return Decision(controller.jump_right(jump_frames=p.stomp_hold_frames),
                            note=f"stomp @{dx}px")

        if gap is not None:
            dist_px, width = gap
            self._last_scene = scene
            if dist_px <= p.jump_trigger_px:
                jf = p.jump_base_frames + width * p.jump_per_tile_frames
                jf = max(p.jump_min_frames, min(jf, p.jump_max_frames))
                return Decision(controller.jump_right(jump_frames=jf), note=f"jump pit w={width}",
                                search_candidates=self._jump_candidates(jf, width))
            return Decision(controller.run_right(p.reflex_step_frames),
                            note=f"approach pit ({dist_px}px)")

        obstacle = scene.obstacle_ahead()
        if obstacle is not None:
            dist_px, height = obstacle
            self._last_scene = scene
            if dist_px <= p.obstacle_trigger_px:
                jf = p.obstacle_base_frames + height * p.obstacle_per_height_frames
                jf = max(p.jump_min_frames, min(jf, p.jump_max_frames))
                return Decision(controller.jump_right(jump_frames=jf), note=f"jump wall h={height}")
            return Decision(controller.run_right(p.reflex_step_frames),
                            note=f"approach wall ({dist_px}px)")

        if scene.enemy_ahead():
            self._last_scene = scene
            return Decision(controller.jump_right(jump_frames=20), note="hop enemy")

        bonk = scene.block_above_ahead()
        safe_to_bonk = gap is None and scene.nearest_enemy(96) is None
        if bonk is not None and bonk <= p.bonk_trigger_px and safe_to_bonk:
            self._last_scene = scene
            return Decision(controller.jump_right(jump_frames=p.bonk_hold_frames),
                            note="bonk block (coin/powerup)")

        self._last_scene = scene
        return Decision(controller.run_right(p.reflex_step_frames), note="cruise")
