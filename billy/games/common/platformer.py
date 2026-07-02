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
    def nearest_powerup(self, within: int = ...) -> "tuple[int, int] | None": ...   # optional
    def pipe_entry_spot(self, max_tiles: int = ...) -> "int | None": ...           # optional


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


def pipe_entry_candidates() -> list[Plan]:
    """Centre on a pipe mouth and hold DOWN (1-2 exit pipe, warp zone → worlds 4/5/6)."""
    c = controller
    enter = [(c.run_right(n, sprint=False) if n else []) + [Step(h, controller.DOWN)]
             for n in (0, 6, 12, 18, 24) for h in (24, 40, 56)]
    enter += [c.run_left(n, sprint=False) + [Step(48, controller.DOWN)] for n in (6, 12, 20)]
    return enter


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
        cands = [
            c.jump_right(jump_frames=28),
            c.jump_right(jump_frames=34),
            *enemy_stomper(),
            *wall_jumper(),
        ]
        spot = getattr(obs.raw, "pipe_entry_spot", lambda: None)()
        if spot is not None:
            cands.extend(pipe_entry_candidates())
        return cands

    def _jump_candidates(self, base: int, width: int) -> list[Plan]:
        return gap_jumper(width, self.p)

    def expanded_candidates(self, obs: Observation) -> list[Plan]:
        """A DENSE brute-force grid for hard walls the focused spread can't crack. Beyond the
        run-up × jump-hold sweep it carries the move *types* a pure-jump grid lacks — the ones that
        crack 1-2's low-ceiling + enemy-ledge walls (x=908/978):
          • SHORT HOPS — clear a single enemy/step without gaining the height that bonks a low
            ceiling (and drops you back onto the hazard);
          • WALK-THROUGH (no jump) — run/walk under a low ceiling or off a ledge without leaping
            into it;
          • PATIENCE — idle out a moving enemy's phase, then a short hop or walk.
        When a low ceiling is detected ahead, the ceiling-safe moves are tried FIRST (early-exit then
        banks the first survivor fast). Tried when the normal search fails, before the (slow) LLM."""
        c = controller
        scene = obs.raw
        runs = (0, 6, 10, 14, 18, 24, 30)
        holds = (16, 20, 24, 28, 31, 34)
        big_jumps: list[Plan] = [c.jump_right(run_frames=r, jump_frames=h) for r in runs for h in holds]
        short_hops = [c.jump_right(run_frames=r, jump_frames=h) for r in (0, 8, 16) for h in (6, 9, 12)]
        walk_through = [c.run_right(n, sprint=s) for n in (8, 14, 22) for s in (True, False)]
        patience = [c.idle(d) + tail for d in (12, 28, 48, 70)
                    for tail in (c.jump_right(jump_frames=9), c.jump_right(jump_frames=24),
                                 c.run_right(12))]
        # BACK-UP then run-jump: Billy is right-biased, but some walls only clear with a real run-up.
        # Step left first (varying distance), then run-jump right — the search discovers when backing
        # up unblocks a jump that no standing/short run-up could make.
        backup = [c.run_left(b, sprint=False) + c.jump_right(run_frames=r, jump_frames=h)
                  for b in (8, 16, 24) for r in (14, 24) for h in (28, 34)]
        enter_pipe = pipe_entry_candidates()
        # RETREAT / drop off a dead-end: when a wall ahead reaches the ceiling (impassable at this
        # level), the way on is BELOW — go back left and off the ledge so the lower path can be
        # found. These travel left a real distance; the search's long post-coast (run-right) then
        # recovers along whatever path the drop lands on, crediting net forward progress. Only
        # offered when truly walled in, so they don't tempt Billy backward in open play.
        tall_wall = (scene.obstacle_ahead() or (0, 0))[1] >= 3
        retreat = ([c.run_left(n, sprint=True) for n in (24, 40, 60)] +
                   [c.run_left(n, sprint=True) + c.jump_right(run_frames=24, jump_frames=h)
                    for n in (24, 40) for h in (16, 30)]) if tall_wall else []

        low_ceiling = scene.block_above_ahead() is not None
        if low_ceiling:   # height is a liability here — lead with hops/walks that won't bonk
            return short_hops + walk_through + enter_pipe + patience + big_jumps + backup + retreat
        return big_jumps + short_hops + walk_through + enter_pipe + patience + backup + retreat

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

        # PIPE ENTRY: 1-2's exit pipe and warp-zone pipes (→ 4/5/6) need DOWN, not forward x.
        # x stalls while ducking — that is progress, not stuck.
        pipe_align = getattr(scene, "pipe_entry_spot", lambda: None)()
        if pipe_align is not None and gap is None:
            self._last_scene = scene
            self._frames_stuck = 0
            if pipe_align > 10:
                return Decision(controller.run_right(p.reflex_step_frames, sprint=False),
                                note=f"align pipe {pipe_align}px")
            return Decision([Step(48, controller.DOWN)], note="enter pipe")

        # BALANCED power-up handling (survival-first: every hazard above is resolved before this).
        # 1) If a power-up is already out, COLLECT it (don't sprint past the slow-emerging mushroom).
        pu = scene.nearest_powerup() if hasattr(scene, "nearest_powerup") else None
        if pu is not None:
            dx, dy = pu
            self._last_scene = scene
            if dy < -16:        # above / still emerging -> a small hop to meet it
                return Decision(controller.jump_right(jump_frames=10), note=f"grab powerup ^{dx}px")
            if dx < 0:          # bounced behind him -> wait a beat for it to come back
                return Decision(controller.idle(6), note="wait for powerup")
            return Decision(controller.run_right(p.reflex_step_frames, sprint=False),  # walk, no overshoot
                            note=f"grab powerup {dx}px")

        # 2) SEEK-AND-BONK: while small (wants the mushroom), line up under a reachable floating block
        # and bump it from BELOW with a STRAIGHT-UP jump. A jump_right would carry Billy up ONTO the
        # block row instead of bumping it, so the power-up never pops — that was the bug. Gated to
        # small + safe, so once Big/Fire he stops detouring and just cruises (keeps it "balanced").
        bonk = scene.block_above_ahead(max_tiles=4)
        safe_to_bonk = gap is None and scene.nearest_enemy(52) is None
        if bonk is not None and safe_to_bonk and scene.size == 0:
            self._last_scene = scene
            if bonk > 8:        # not lined up yet -> shuffle right under the block (no sprint)
                return Decision(controller.run_right(p.reflex_step_frames, sprint=False),
                                note=f"line up block {bonk}px")
            return Decision([Step(p.bonk_hold_frames, controller.A)],   # straight up: bump from below
                            note="bonk for powerup")

        self._last_scene = scene
        return Decision(controller.run_right(p.reflex_step_frames), note="cruise")
