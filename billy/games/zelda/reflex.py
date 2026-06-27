"""Top-down reflex policy for The Legend of Zelda (NES).

Unlike the shared platformer reflex, Zelda needs four-way movement, sword swings, and
screen-edge transitions. The director's cache / micro-search / learn-from-death loop is
unchanged — this tier supplies `step`, `advance_plan`, and `danger_candidates`.
"""
from __future__ import annotations

from collections import deque

from ...abstractions import Decision, Observation, Plan, ReflexPolicy, Step
from ...systems.nes import controller as c
from .curiosity import (
    cave_approach_button,
    needs_cave_approach,
    requires_start_cave_inspection,
    start_cave_inspection_action,
)
from .explore import pick_explore_direction
from .perception import Scene
from .start_cave import (
    cave_attempt_exhausted,
    cave_quest_active,
    has_wooden_sword,
    interior_phase,
    phase_plan,
)
from .tuning import (
    ATTACK_RANGE,
    BACKTRACK_MEMORY,
    REFLEX_FRAMES,
    RETREAT_HEALTH,
    STUCK_FRAMES,
)

ITEM_PICKUP_RANGE = 12
CAVE_MACRO_STUCK_FRAMES = 180


def _walk(button: int, frames: int = REFLEX_FRAMES) -> Plan:
    return [Step(frames, button)]


def _sword(button: int, frames: int = REFLEX_FRAMES) -> Plan:
    return [Step(frames, c.mask(button, c.B))]


def _retreat_from(dx: int, dy: int) -> Plan:
    if abs(dx) >= abs(dy):
        btn = c.LEFT if dx >= 0 else c.RIGHT
    else:
        btn = c.UP if dy >= 0 else c.DOWN
    return _walk(btn, REFLEX_FRAMES * 2)


def combat_candidates(obs: Observation) -> list[Plan]:
    """Expanded sword + spacing set for micro-search at enemy hazards."""
    scene: Scene = obs.raw
    near = scene.nearest_enemy(within=ATTACK_RANGE + 24)
    base = [
        _sword(c.RIGHT, 14),
        _sword(c.LEFT, 14),
        _sword(c.UP, 14),
        _sword(c.DOWN, 14),
        _walk(c.RIGHT, 12) + _sword(c.RIGHT, 14),
        _walk(c.LEFT, 12) + _sword(c.LEFT, 14),
        _walk(c.DOWN, 12) + _sword(c.DOWN, 14),
        _walk(c.UP, 12) + _sword(c.UP, 14),
        _walk(c.RIGHT, 20),
        _walk(c.LEFT, 16),
        c.idle(16),
        c.idle(32),
    ]
    if near:
        dx, dy = near
        if abs(dx) >= abs(dy):
            btn = c.RIGHT if dx >= 0 else c.LEFT
        else:
            btn = c.DOWN if dy >= 0 else c.UP
        base.insert(0, _walk(btn, 8) + _sword(btn, 16))
        base.insert(1, _retreat_from(dx, dy))
    return base


class ZeldaReflex(ReflexPolicy):
    def __init__(self) -> None:
        self._best = 0
        self._frames_stuck = 0
        self._last_health = 3
        self._recent_screens: deque[int] = deque(maxlen=BACKTRACK_MEMORY)
        self._commit_btn: int | None = None
        self._commit_label: str = ""
        self._last_scene: Scene | None = None
        self._last_sword_level = 0
        self._cave_frames = 0
        self._cave_gave_up = False
        self._macro_phase = ""
        self._macro_idx = 0
        self._macro_stuck_frames = 0
        self._macro_last_progress = 0

    def reset(self, obs: Observation) -> None:
        self._best = obs.progress
        self._frames_stuck = 0
        self._last_health = obs.raw.health
        self._recent_screens.clear()
        self._recent_screens.append(obs.raw.map_location)
        self._commit_btn = None
        self._last_scene = None
        self._last_sword_level = obs.raw.sword_level
        self._cave_frames = 0
        self._cave_gave_up = False
        self._macro_phase = ""
        self._macro_idx = 0
        self._macro_stuck_frames = 0
        self._macro_last_progress = obs.progress

    def note_level_advance(self, obs: Observation) -> None:
        self._best = obs.progress
        self._frames_stuck = 0
        self._recent_screens.append(obs.raw.map_location)
        self._commit_btn = None

    def advance_plan(self, obs: Observation) -> Plan:
        scene: Scene = obs.raw
        if scene.in_cave and not self._cave_gave_up:
            phase = interior_phase(scene)
            plan = phase_plan(phase)
            if plan:
                idx = min(self._macro_idx, len(plan) - 1)
                return [plan[idx]]
        visited = set(scene.visited_screens)
        btn, _, _ = pick_explore_direction(
            scene.map_location, visited, recent=self._recent_screens,
            sword_level=scene.sword_level, max_hearts=scene.max_hearts,
            in_cave=scene.in_cave, link_x=scene.link_x, link_y=scene.link_y,
            cave_gave_up=self._cave_gave_up, cave_frames=self._cave_frames)
        return _walk(btn, REFLEX_FRAMES)

    def danger_candidates(self, obs: Observation) -> list[Plan]:
        return combat_candidates(obs)

    def expanded_candidates(self, obs: Observation) -> list[Plan]:
        """Dense combat grid for learn-from-death / hard walls."""
        out = combat_candidates(obs)
        for wait in (8, 16, 24):
            out.append(c.idle(wait) + combat_candidates(obs)[0])
        return out

    def _sword_toward(self, dx: int, dy: int) -> Plan:
        if abs(dx) >= abs(dy):
            btn = c.RIGHT if dx >= 0 else c.LEFT
        else:
            btn = c.DOWN if dy >= 0 else c.UP
        return _sword(btn, REFLEX_FRAMES)

    def _edge_transition(self, btn: int, scene: Scene, label: str) -> Decision | None:
        hold = 48   # screen scroll needs sustained input
        if btn == c.RIGHT and scene.at_right_edge:
            return Decision(_walk(c.RIGHT, hold), note=f"edge-transition {label}")
        if btn == c.LEFT and scene.at_left_edge:
            return Decision(_walk(c.LEFT, hold), note=f"edge-transition {label}")
        if btn == c.DOWN and scene.at_bottom_edge:
            return Decision(_walk(c.DOWN, hold), note=f"edge-transition {label}")
        if btn == c.UP and scene.at_top_edge:
            return Decision(_walk(c.UP, hold), note=f"edge-transition {label}")
        return None

    def _track_cave_timeout(self, scene: Scene, obs: Observation) -> None:
        if cave_quest_active(
                scene.map_location, scene.sword_level, in_cave=scene.in_cave,
                cave_frames=self._cave_frames, cave_gave_up=self._cave_gave_up):
            self._cave_frames += REFLEX_FRAMES
            if cave_attempt_exhausted(self._cave_frames, scene.sword_level):
                self._cave_gave_up = True
                self._macro_phase = ""
                self._macro_idx = 0

    def _cave_interior_step(self, scene: Scene, obs: Observation) -> Decision | None:
        """ROM-verified start cave macro (start_cave.py): text → climb → pickup → exit."""
        if not scene.in_cave:
            self._macro_phase = ""
            self._macro_idx = 0
            self._macro_stuck_frames = 0
            return None

        if self._cave_gave_up:
            return None

        self._track_cave_timeout(scene, obs)

        if has_wooden_sword(scene.sword_level):
            self._last_sword_level = scene.sword_level

        phase = interior_phase(scene)
        if phase == "overworld":
            return None

        plan = phase_plan(phase)
        if phase != self._macro_phase:
            self._macro_phase = phase
            self._macro_idx = 0
            self._macro_stuck_frames = 0
            self._macro_last_progress = obs.progress

        if obs.progress <= self._macro_last_progress:
            self._macro_stuck_frames += REFLEX_FRAMES
        else:
            self._macro_stuck_frames = 0
            self._macro_last_progress = obs.progress

        if self._macro_stuck_frames >= CAVE_MACRO_STUCK_FRAMES:
            return Decision([], needs_billy=True, note="cave-macro-stuck")

        if self._macro_idx >= len(plan):
            if phase == "exit":
                if scene.in_cave:
                    return Decision([Step(16, c.DOWN)], note="cave-macro-exit-continue")
                return None
            self._macro_idx = 0

        step = plan[self._macro_idx]
        self._macro_idx += 1
        return Decision([step], note=f"cave-macro-{phase}")

    def step(self, obs: Observation) -> Decision:
        scene: Scene = obs.raw

        if obs.progress > self._best:
            self._frames_stuck = 0
            self._best = obs.progress
        else:
            self._frames_stuck += REFLEX_FRAMES

        if scene.health < self._last_health:
            self._last_health = scene.health
            self._last_scene = scene
            return Decision([], needs_billy=True, note=f"enemy damage ({scene.health} hearts)")

        cave_step = self._cave_interior_step(scene, obs)
        if cave_step is not None:
            self._last_scene = scene
            return cave_step

        item = scene.nearest_ground_item(within=64)
        if item is not None and scene.enemy_count() == 0:
            dx, dy, ground = item
            if abs(dx) + abs(dy) <= ITEM_PICKUP_RANGE:
                return Decision(_walk(c.NEUTRAL, REFLEX_FRAMES), note=f"pickup-item@{ground.x},{ground.y}")
            btn = walk_toward(dx, dy)
            self._last_scene = scene
            return Decision(_walk(btn, REFLEX_FRAMES), note=f"walk-item ({dx},{dy})")

        visited = set(scene.visited_screens)
        mouths = scene.cave_mouths
        inspecting_cave = requires_start_cave_inspection(
            scene.map_location, visited, self._recent_screens,
            sword_level=scene.sword_level, in_cave=scene.in_cave,
            cave_gave_up=self._cave_gave_up, cave_frames=self._cave_frames)

        if inspecting_cave:
            if needs_cave_approach(scene.map_location, scene.link_x, scene.link_y,
                                   cave_mouths=mouths):
                approach = cave_approach_button(
                    scene.map_location, scene.link_x, scene.link_y, cave_mouths=mouths)
                if approach is not None:
                    btn, note = approach
                    self._last_scene = scene
                    return Decision(_walk(btn, REFLEX_FRAMES), note=note)
            btn, label, dest = start_cave_inspection_action(
                scene.link_x, scene.link_y, cave_mouths=mouths)
            self._commit_btn, self._commit_label = btn, label
            if btn == c.UP and scene.at_top_edge:
                btn = c.mask(c.UP, c.LEFT)
                label = "walkthrough-enter-nw-cave"
            self._last_scene = scene
            return Decision(_walk(btn, REFLEX_FRAMES * 2), note=label)

        near = scene.nearest_enemy(within=ATTACK_RANGE)
        if near is not None:
            dx, dy = near
            self._last_scene = scene
            if scene.health <= RETREAT_HEALTH and abs(dx) + abs(dy) < ATTACK_RANGE:
                return Decision(_retreat_from(dx, dy), note=f"retreat enemy ({dx},{dy})")
            return Decision(self._sword_toward(dx, dy), note=f"sword enemy ({dx},{dy})")

        if self._frames_stuck >= 48:
            self._commit_btn = None

        if self._frames_stuck >= STUCK_FRAMES:
            self._last_scene = scene
            return Decision([], needs_billy=True, note=f"stuck {self._frames_stuck}f")

        if needs_cave_approach(scene.map_location, scene.link_x, scene.link_y,
                               cave_mouths=mouths):
            approach = cave_approach_button(
                scene.map_location, scene.link_x, scene.link_y, cave_mouths=mouths)
            if approach is not None:
                btn, note = approach
                self._last_scene = scene
                return Decision(_walk(btn, REFLEX_FRAMES), note=note)

        prefer_dungeons = len(visited) >= 2
        if self._commit_btn is None:
            btn, label, dest = pick_explore_direction(
                scene.map_location, visited,
                recent=self._recent_screens, prefer_dungeons=prefer_dungeons,
                cave_mouths=mouths, sword_level=scene.sword_level,
                max_hearts=scene.max_hearts, in_cave=scene.in_cave,
                link_x=scene.link_x, link_y=scene.link_y,
                cave_gave_up=self._cave_gave_up, cave_frames=self._cave_frames)
            self._commit_btn, self._commit_label = btn, label
        else:
            btn, label, dest = self._commit_btn, self._commit_label, scene.map_location

        if not inspecting_cave:
            edge = self._edge_transition(btn, scene, label)
            if edge is not None:
                self._last_scene = scene
                return edge

        self._last_scene = scene
        return Decision(_walk(btn, REFLEX_FRAMES), note=f"explore {label} →#{dest}")