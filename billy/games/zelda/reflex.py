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
from .dungeon_nav import dungeon_combat_decision, dungeon_explore_decision, dungeon_key_decision
from .east_march import (
    EAST_SCROLL_SETTLE_FRAMES,
    east_march_active,
    east_march_approach_decision,
    east_march_at_lip,

    east_march_combat_decision,
    east_march_cross_decision,
    east_march_decision,
    east_march_lane_decision,
    east_march_needs_cross,
    east_march_on_west_entry,
    east_march_entry_guard_decision,
    east_march_post_settle_macro,
    east_march_route_commit,
    east_march_scroll_settle_decision,
)
from .explore import pick_explore_direction
from .perception import Scene
from .items import walk_toward
from .start_cave import (
    ENTER_PLAN,
    POST_CAVE_REPOSITION_PLAN,
    cave_attempt_exhausted,
    cave_quest_active,
    has_wooden_sword,
    interior_phase,
    remaining_interior_plan,
)
from .tuning import START_SCREEN
from .tuning import (
    SCREEN_EDGE_HI,
    ATTACK_RANGE,
    BACKTRACK_MEMORY,
    REFLEX_FRAMES,
    RETREAT_HEALTH,
    STUCK_FRAMES,
)

ITEM_PICKUP_RANGE = 12
CAVE_MACRO_STUCK_FRAMES = 180
_BEAM_RANGE = 9999   # full-health sword beam reaches across the whole screen (engage any enemy)

# Incoming-projectile dodge: a high-slot object moving this fast (px/frame) is a shot, not a
# walking enemy; dodge when it's on Link's row/column band and closing within this range.
_SHOT_SPEED_MIN = 2
_SHOT_SPEED_MAX = 12
_SHOT_LANE = 12        # within this many px of Link's row/col => it will connect
_SHOT_DODGE_RANGE = 56  # start sidestepping once the shot is this close


def _walk(button: int, frames: int = REFLEX_FRAMES) -> Plan:
    return [Step(frames, button)]


def _sword(button: int, frames: int = REFLEX_FRAMES) -> Plan:
    # The sword is the A button on the NES pad; B is the (often empty) secondary-item slot.
    # Earlier this pressed B, so every "swing" fired the item slot and no enemy ever took a hit.
    return [Step(frames, c.mask(button, c.A))]


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
        self._cave_repositioned = False
        self._dungeon_visited: set[int] = set()
        self._east_cross_tick = 0
        self._east_scroll_settle = 0
        self._east_screen_frames = 0
        self._prev_objects: dict[int, tuple[int, int]] = {}   # last frame's high-slot objects

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
        self._cave_repositioned = False
        self._dungeon_visited.clear()
        self._east_cross_tick = 0
        self._east_scroll_settle = 0
        self._east_screen_frames = 0
        self._prev_objects = {}

    def _incoming_dodge(self, scene: Scene, cur: dict[int, tuple[int, int]],
                        prev: dict[int, tuple[int, int]]) -> tuple[int, tuple[int, int]] | None:
        """If a projectile is about to hit Link, return (perpendicular dodge button, shot pos).

        A high-slot object that moved fast+straight last frame, sits on Link's row/column band,
        and is closing in is an incoming shot — sidestep OUT of its lane. Enemies (slow) and
        static objects fail the speed gate, so this only fires on real shots."""
        lx, ly = scene.link_x, scene.link_y
        best: tuple[int, tuple[int, int]] | None = None
        best_dist = _SHOT_DODGE_RANGE + 1
        for slot, (x, y) in cur.items():
            if slot not in prev:
                continue
            px, py = prev[slot]
            vx, vy = x - px, y - py
            speed = abs(vx) + abs(vy)
            if speed < _SHOT_SPEED_MIN or speed > _SHOT_SPEED_MAX:
                continue
            dist = abs(x - lx) + abs(y - ly)
            if dist > _SHOT_DODGE_RANGE:
                continue
            if abs(vx) >= abs(vy):            # horizontal shot — threatens Link's ROW
                if abs(y - ly) > _SHOT_LANE or (lx - x) * vx <= 0:
                    continue
                dodge = c.DOWN if y <= ly else c.UP
            else:                             # vertical shot — threatens Link's COLUMN
                if abs(x - lx) > _SHOT_LANE or (ly - y) * vy <= 0:
                    continue
                dodge = c.RIGHT if x <= lx else c.LEFT
            if dist < best_dist:
                best_dist, best = dist, (dodge, (x, y))
        return best

    def note_level_advance(self, obs: Observation) -> None:
        self._best = obs.progress
        self._frames_stuck = 0
        self._recent_screens.append(obs.raw.map_location)
        self._commit_btn = None
        self._east_cross_tick = 0
        if east_march_active(obs.raw, visited=set(obs.raw.visited_screens)):
            loc = obs.raw.map_location
            settle = EAST_SCROLL_SETTLE_FRAMES
            if loc >= START_SCREEN + 5:
                settle += 32
            elif loc >= START_SCREEN + 4:
                settle += 16
            self._east_scroll_settle = settle
            self._east_screen_frames = 0
        if obs.raw.in_dungeon:
            self._dungeon_visited.add(obs.raw.map_location)

    def advance_plan(self, obs: Observation) -> Plan:
        scene: Scene = obs.raw
        if scene.in_cave and not self._cave_gave_up:
            phase = interior_phase(scene)
            plan = remaining_interior_plan(phase)
            if plan:
                return list(plan)
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

        plan = remaining_interior_plan(phase)
        if not plan:
            return None

        if (phase == "exit" and scene.in_cave and has_wooden_sword(scene.sword_level)
                and self._macro_idx >= len(plan)):
            # EXIT_PLAN's long DOWN may need a short nudge before overworld mode clears.
            return Decision([Step(16, c.DOWN)], note="cave-macro-exit-continue")

        self._macro_idx = len(plan)
        return Decision(list(plan), note=f"cave-macro-{phase}-full")

    def step(self, obs: Observation) -> Decision:
        scene: Scene = obs.raw

        if has_wooden_sword(scene.sword_level) and not has_wooden_sword(self._last_sword_level):
            self._commit_btn = None
            self._commit_label = ""
            self._last_sword_level = scene.sword_level

        if (has_wooden_sword(scene.sword_level) and not scene.in_cave
                and scene.map_location == START_SCREEN and not self._cave_repositioned
                and scene.link_y >= 180):
            self._cave_repositioned = True
            self._last_scene = scene
            return Decision(list(POST_CAVE_REPOSITION_PLAN), note="post-cave-reposition-full")

        visited = set(scene.visited_screens)
        marching_east = east_march_active(scene, visited=visited)

        # Track high-slot objects every frame so we can spot an incoming shot by its velocity.
        # Dodge applies on the overworld (incl. the east-march sea route, where octoroks pepper
        # Link) and in dungeons; only cave interiors are skipped (their macros own movement). A
        # sidestep preempts the east-march crossing for that frame; the lane logic re-centers Link
        # on row 8 next tick, so the shot is cleared without losing the march.
        cur_objects = scene.object_positions()
        dodge = (None if scene.in_cave
                 else self._incoming_dodge(scene, cur_objects, self._prev_objects))
        self._prev_objects = cur_objects

        if obs.progress > self._best:
            self._frames_stuck = 0
            self._best = obs.progress
        elif not (marching_east and self._east_scroll_settle > 0):
            self._frames_stuck += REFLEX_FRAMES

        if scene.health < self._last_health:
            self._last_health = scene.health
            self._last_scene = scene
            if marching_east:
                invuln = [Step(24, c.NEUTRAL)]
                if (scene.map_location >= START_SCREEN + 4
                        and east_march_needs_cross(scene)):
                    cross = east_march_cross_decision(
                        scene, tick=self._east_cross_tick,
                        screen_frames=self._east_screen_frames)
                    if cross is not None:
                        self._east_cross_tick += max(1, len(cross.plan))
                        return Decision(
                            invuln + list(cross.plan),
                            note=f"east-march-after-hit-cross ({scene.health} hearts)",
                        )
                fight = east_march_combat_decision(scene)
                if fight is not None:
                    return Decision(invuln + list(fight.plan),
                                    note=f"east-march-after-hit ({scene.health} hearts)")
                return Decision(
                    invuln + _walk(c.RIGHT, REFLEX_FRAMES * 2),
                    note=f"east-march-after-hit ({scene.health} hearts)",
                )
            return Decision([], needs_billy=True, note=f"enemy damage ({scene.health} hearts)")

        cave_step = self._cave_interior_step(scene, obs)
        if cave_step is not None:
            self._last_scene = scene
            return cave_step

        # Dodge an incoming shot before anything else on-foot — don't walk to an item or an enemy
        # into a rock. The sidestep clears the lane; next tick, with the shot past, combat resumes.
        if dodge is not None:
            self._last_scene = scene
            return Decision(_walk(dodge[0], REFLEX_FRAMES), note=f"dodge shot @{dodge[1]}")

        item = scene.nearest_ground_item(within=64)
        if item is not None and scene.enemy_count() == 0:
            dx, dy, ground = item
            if abs(dx) + abs(dy) <= ITEM_PICKUP_RANGE:
                return Decision(_walk(c.NEUTRAL, REFLEX_FRAMES), note=f"pickup-item@{ground.x},{ground.y}")
            btn = walk_toward(dx, dy)
            self._last_scene = scene
            return Decision(_walk(btn, REFLEX_FRAMES), note=f"walk-item ({dx},{dy})")

        mouths = scene.cave_mouths
        inspecting_cave = requires_start_cave_inspection(
            scene.map_location, visited, self._recent_screens,
            sword_level=scene.sword_level, in_cave=scene.in_cave,
            cave_gave_up=self._cave_gave_up, cave_frames=self._cave_frames)

        if inspecting_cave:
            if needs_cave_approach(scene.map_location, scene.link_x, scene.link_y,
                                   cave_mouths=mouths, sword_level=scene.sword_level,
                                   cave_gave_up=self._cave_gave_up, cave_frames=self._cave_frames):
                approach = cave_approach_button(
                    scene.map_location, scene.link_x, scene.link_y, cave_mouths=mouths)
                if approach is not None:
                    btn, note = approach
                    self._last_scene = scene
                    return Decision(_walk(btn, REFLEX_FRAMES), note=note)
            btn, label, dest = start_cave_inspection_action(
                scene.link_x, scene.link_y, cave_mouths=mouths)
            self._commit_btn, self._commit_label = btn, label
            if label == "walkthrough-enter-nw-cave":
                self._last_scene = scene
                return Decision(list(ENTER_PLAN), note="cave-enter-full")
            self._last_scene = scene
            return Decision(_walk(btn, REFLEX_FRAMES * 2), note=label)

        if scene.in_dungeon:
            key_step = dungeon_key_decision(scene)
            if key_step is not None:
                self._last_scene = scene
                return key_step
            fight = dungeon_combat_decision(scene)
            if fight is not None:
                self._last_scene = scene
                return fight
            self._dungeon_visited.add(scene.map_location)
            if scene.enemy_count() == 0:
                dung_step = dungeon_explore_decision(
                    scene, self._dungeon_visited, self._recent_screens)
                if dung_step is not None:
                    self._last_scene = scene
                    return dung_step

        # Sword beam: at FULL health Link's stab fires a projectile the length of the screen, so
        # engage ANY on-screen enemy — stab toward it and the beam travels (holding the direction
        # also closes toward alignment). Below full, the sword only reaches melee (its own length),
        # so only the enemies already within reach are engaged; the rest are left to the march.
        full_health = scene.full_health
        near = scene.nearest_enemy(within=_BEAM_RANGE if full_health else ATTACK_RANGE)
        if near is not None and not marching_east:
            dx, dy = near
            self._last_scene = scene
            melee = abs(dx) + abs(dy) <= ATTACK_RANGE
            if scene.health <= RETREAT_HEALTH and melee:
                return Decision(_retreat_from(dx, dy), note=f"retreat enemy ({dx},{dy})")
            return Decision(self._sword_toward(dx, dy),
                            note=f"sword enemy ({dx},{dy})" + ("" if melee else " [beam]"))

        if self._frames_stuck >= 48:
            self._commit_btn = None

        if self._frames_stuck >= STUCK_FRAMES:
            self._last_scene = scene
            return Decision([], needs_billy=True, note=f"stuck {self._frames_stuck}f")

        if needs_cave_approach(scene.map_location, scene.link_x, scene.link_y,
                               cave_mouths=mouths, sword_level=scene.sword_level,
                               cave_gave_up=self._cave_gave_up, cave_frames=self._cave_frames):
            approach = cave_approach_button(
                scene.map_location, scene.link_x, scene.link_y, cave_mouths=mouths)
            if approach is not None:
                btn, note = approach
                self._last_scene = scene
                return Decision(_walk(btn, REFLEX_FRAMES), note=note)

        prefer_dungeons = len(visited) >= 2
        if marching_east:
            btn, label = east_march_route_commit()
            dest = scene.map_location + 1
            self._commit_btn, self._commit_label = btn, label
        elif self._commit_btn is None:
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

        if not inspecting_cave and marching_east:
            screen_frames = self._east_screen_frames
            settle = east_march_scroll_settle_decision(
                scene, frames_left=self._east_scroll_settle)
            if settle is not None:
                self._east_scroll_settle = max(0, self._east_scroll_settle - REFLEX_FRAMES * 2)
                self._east_screen_frames += REFLEX_FRAMES * 2
                self._last_scene = scene
                return settle
            lane = east_march_lane_decision(scene, btn)
            if lane is not None:
                self._east_screen_frames += REFLEX_FRAMES * 2
                self._last_scene = scene
                return lane
            guard = east_march_entry_guard_decision(scene, screen_frames=screen_frames)
            if guard is not None:
                self._east_screen_frames += sum(s.frames for s in guard.plan)
                self._last_scene = scene
                return guard
            macro = east_march_post_settle_macro(scene, screen_frames=screen_frames)
            if macro is not None:
                self._east_screen_frames += 400
                self._last_scene = scene
                return macro
            crossing = (not east_march_at_lip(scene, screen_frames=screen_frames)
                        and (east_march_needs_cross(scene)
                             or (scene.enemy_count() > 0
                                 and scene.link_x < SCREEN_EDGE_HI)))
            if crossing:
                cross = east_march_cross_decision(
                    scene, tick=self._east_cross_tick, screen_frames=screen_frames)
                if cross is not None:
                    self._east_cross_tick += max(1, len(cross.plan))
                    self._east_screen_frames += sum(s.frames for s in cross.plan)
                    self._last_scene = scene
                    return cross
            skip_fight = (scene.map_location >= START_SCREEN + 4
                          and east_march_needs_cross(scene))
            if scene.enemy_count() > 0 and not skip_fight:
                fight = east_march_combat_decision(scene)
                if fight is not None:
                    self._east_screen_frames += REFLEX_FRAMES * 4
                    self._last_scene = scene
                    return fight
            approach = east_march_approach_decision(scene, btn)
            if approach is not None:
                self._east_screen_frames += REFLEX_FRAMES * 2
                self._last_scene = scene
                return approach
            hop = east_march_decision(scene, btn, label, screen_frames=screen_frames)
            if hop is not None:
                self._east_screen_frames += REFLEX_FRAMES * 12
                self._last_scene = scene
                return hop
            self._last_scene = scene
            self._east_screen_frames += REFLEX_FRAMES
            if east_march_at_lip(scene, screen_frames=screen_frames):
                return Decision(_walk(c.RIGHT, REFLEX_FRAMES), note="east-march-lip-walk")
            return Decision(_walk(c.RIGHT, REFLEX_FRAMES), note=f"east-march-walk {label}")

        if not inspecting_cave and not marching_east:
            edge = self._edge_transition(btn, scene, label)
            if edge is not None:
                self._last_scene = scene
                return edge

        self._last_scene = scene
        return Decision(_walk(btn, REFLEX_FRAMES), note=f"explore {label} →#{dest}")