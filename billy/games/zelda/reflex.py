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
from .items import walk_toward
from .perception import Scene
from .tuning import (
    ATTACK_RANGE,
    BACKTRACK_MEMORY,
    REFLEX_FRAMES,
    RETREAT_HEALTH,
    STUCK_FRAMES,
)

CAVE_TEXT_FRAMES = 120
ITEM_PICKUP_RANGE = 12


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
        self._cave_text_frames = 0
        self._last_sword_level = 0
        self._cave_stuck_pos: tuple[int, int] | None = None
        self._cave_stuck_tries = 0

    def reset(self, obs: Observation) -> None:
        self._best = obs.progress
        self._frames_stuck = 0
        self._last_health = obs.raw.health
        self._recent_screens.clear()
        self._recent_screens.append(obs.raw.map_location)
        self._commit_btn = None
        self._last_scene = None
        self._cave_text_frames = 0
        self._last_sword_level = obs.raw.sword_level
        self._cave_stuck_pos = None
        self._cave_stuck_tries = 0

    def note_level_advance(self, obs: Observation) -> None:
        self._best = obs.progress
        self._frames_stuck = 0
        self._recent_screens.append(obs.raw.map_location)
        self._commit_btn = None

    def advance_plan(self, obs: Observation) -> Plan:
        scene: Scene = obs.raw
        visited = set(scene.visited_screens)
        btn, _, _ = pick_explore_direction(
            scene.map_location, visited, recent=self._recent_screens,
            sword_level=scene.sword_level, max_hearts=scene.max_hearts,
            in_cave=scene.in_cave, link_x=scene.link_x, link_y=scene.link_y)
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

    def _cave_interior_step(self, scene: Scene) -> Decision | None:
        """FAQ start cave (screen 119 / mode 11).

        Sequence: mash A until text done (drop type ≥ 2) → climb UP → walk toward
        ground items. Known ROM ceiling at link_y≈141 blocks sword pickup today;
        see STATUS.md P0.
        """
        if not scene.in_cave:
            self._cave_text_frames = 0
            return None

        if scene.link_y >= 180:
            self._cave_text_frames += REFLEX_FRAMES
            if self._cave_text_frames < CAVE_TEXT_FRAMES:
                return Decision(_walk(c.A, REFLEX_FRAMES), note="cave-dismiss-text")
            self._cave_text_frames = CAVE_TEXT_FRAMES

        item = scene.nearest_ground_item(within=96)
        if item is not None:
            dx, dy, ground = item
            if abs(dx) + abs(dy) <= ITEM_PICKUP_RANGE:
                return Decision(_walk(c.NEUTRAL, REFLEX_FRAMES), note=f"cave-loot@{ground.x},{ground.y}")
            pos = (scene.link_x, scene.link_y)
            if self._cave_stuck_pos == pos:
                self._cave_stuck_tries += 1
            else:
                self._cave_stuck_pos = pos
                self._cave_stuck_tries = 0
            if self._cave_stuck_tries >= 3 and scene.link_y > ground.y:
                btn = c.RIGHT if dx > 0 else c.LEFT
                self._cave_stuck_tries = 0
            elif scene.link_y > ground.y + 8 and abs(dx) >= 4:
                btn = c.RIGHT if dx > 0 else c.LEFT
            else:
                btn = walk_toward(dx, dy)
                if btn == c.DOWN:
                    btn = c.RIGHT if dx > 0 else c.LEFT
            return Decision(_walk(btn, REFLEX_FRAMES), note=f"cave-walk-item ({dx},{dy})")

        if scene.sword_level > self._last_sword_level:
            self._last_sword_level = scene.sword_level
            return Decision(_walk(c.UP, REFLEX_FRAMES * 4), note="cave-exit-after-sword")

        if scene.link_y > 150:
            return Decision(_walk(c.UP, REFLEX_FRAMES), note="cave-walk-up")

        return Decision(_walk(c.UP, REFLEX_FRAMES), note="cave-explore")

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

        cave_step = self._cave_interior_step(scene)
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
            sword_level=scene.sword_level, in_cave=scene.in_cave)

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
                link_x=scene.link_x, link_y=scene.link_y)
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