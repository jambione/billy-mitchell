"""FAQ east-to-sea screen hops (row 8) — sustained RIGHT holds at the lip."""
from __future__ import annotations

from ...abstractions import Decision, Plan, Step
from ...systems.nes import controller as c
from .tuning import ATTACK_RANGE, REFLEX_FRAMES, SCREEN_EDGE_HI, START_SCREEN
from .walkthrough import SEA_EAST_SCREEN, current_phase

# ROM-tuned: 70f RIGHT from x≈199 scrolls 119→120; 80f was overshooting lip entry.
EAST_HOP_FRAMES = 70
EAST_EDGE_FRAMES = 48   # sustained RIGHT at east lip to scroll out
EAST_APPROACH_X = 192   # mid-lip — start 70f hop from here on prior screen
EAST_HOP_MAX_X = 200    # above this, use edge hold — 70f hop wraps on same screen
EAST_MARCH_LANE_Y = 141
EAST_MARCH_LANE_LO = 128
EAST_MARCH_LANE_HI = 152
# Engage fights before enemies close on a blind 80f RIGHT commit.
EAST_MARCH_COMBAT_RANGE = ATTACK_RANGE + 48
# Block hop when any enemy could reach Link during the hold.
EAST_HOP_THREAT_RANGE = ATTACK_RANGE + 32

# ROM: 48f edge scroll only from true east lip (link_x≥220); mid-lip commits wrap in-place.
EAST_SCROLL_SETTLE_FRAMES = 48

EAST_HOP_PLAN = [Step(EAST_HOP_FRAMES, c.RIGHT)]
EAST_EDGE_PLAN = [Step(EAST_EDGE_FRAMES, c.RIGHT)]
EAST_MARCH_TRANSITION_PLANS = (EAST_HOP_PLAN, EAST_EDGE_PLAN)


def is_east_march_plan(plan) -> bool:
    return list(plan) in [list(p) for p in EAST_MARCH_TRANSITION_PLANS]


def is_east_march_cross_plan(plan) -> bool:
    """Alternating RIGHT / RIGHT+B screen-cross macros banked for replay."""
    if not plan or is_east_march_plan(plan):
        return False
    if len(plan) < 4:
        return False
    for step in plan:
        if step.buttons & (c.UP | c.DOWN | c.LEFT):
            return False
        if not (step.buttons & c.RIGHT):
            return False
    return True


def is_east_march_combat_plan(plan) -> bool:
    """Walk+sword sequences seeded for row-8 combat screens."""
    if not plan or is_east_march_plan(plan) or is_east_march_cross_plan(plan):
        return False
    has_b = any(step.buttons & c.B for step in plan)
    has_vert = any(step.buttons & (c.UP | c.DOWN) for step in plan)
    return has_b and not has_vert


def east_march_active(scene, *, visited: set[int]) -> bool:
    if scene.in_cave or scene.in_dungeon or scene.sword_level < 1:
        return False
    return current_phase(
        map_location=scene.map_location,
        sword_level=scene.sword_level,
        max_hearts=scene.max_hearts,
        visited=visited,
        in_cave=scene.in_cave,
    ) == "east_to_sea" and scene.map_location < SEA_EAST_SCREEN


def east_march_threatened(scene) -> bool:
    """True when a nearby enemy can reach Link during an 80f RIGHT commit."""
    if scene.enemy_count() == 0:
        return False
    near = scene.nearest_enemy(within=EAST_HOP_THREAT_RANGE)
    if near is None:
        return False
    dx, dy = near
    return dx >= -24


def east_march_screen_clear(scene) -> bool:
    """Full screen transition only after the current screen has no live enemies."""
    return scene.enemy_count() == 0


def east_march_lane_ok(scene) -> bool:
    return EAST_MARCH_LANE_LO <= scene.link_y <= EAST_MARCH_LANE_HI


def _fight_button(dx: int, dy: int, *, row_lock: bool = False) -> int:
    # FAQ row 8: never LEFT (west screen hop) or north/south chase.
    if row_lock:
        return c.RIGHT
    if abs(dx) >= abs(dy):
        return c.RIGHT if dx >= 0 else c.LEFT
    return c.DOWN if dy >= 0 else c.UP


def east_march_combat_plan(dx: int, dy: int, *, low_health: bool = False) -> Plan:
    """Walk-then-swing sequences micro-search can verify on combat screens."""
    if low_health:
        return [
            Step(14, c.NEUTRAL),
            Step(16, c.mask(c.RIGHT, c.B)),
            Step(12, c.mask(c.RIGHT, c.B)),
        ]
    if dy < -8:
        return [
            Step(14, c.mask(c.RIGHT, c.B)),
            Step(10, c.RIGHT),
            Step(16, c.mask(c.RIGHT, c.B)),
            Step(10, c.RIGHT),
            Step(14, c.mask(c.RIGHT, c.B)),
        ]
    return [
        Step(8, c.RIGHT),
        Step(16, c.mask(c.RIGHT, c.B)),
        Step(8, c.RIGHT),
        Step(14, c.mask(c.RIGHT, c.B)),
        Step(10, c.mask(c.RIGHT, c.B)),
    ]


def east_march_nearest_enemy(scene) -> tuple[int, int] | None:
    """Closest on-screen enemy — full screen when clearing before a hop."""
    best: tuple[int, int] | None = None
    best_dist = 999
    for enemy in scene.enemies:
        dx = enemy.x - scene.link_x
        dy = enemy.y - scene.link_y
        dist = abs(dx) + abs(dy)
        if dist < best_dist:
            best_dist = dist
            best = (dx, dy)
    return best


def east_march_combat_decision(scene) -> Decision | None:
    """Clear overworld mobs on row 8 before committing a screen hop."""
    if scene.enemy_count() == 0:
        return None
    near = east_march_nearest_enemy(scene)
    if near is None:
        return None
    dx, dy = near
    if dx < -32:
        return None
    if abs(dx) + abs(dy) > EAST_MARCH_COMBAT_RANGE:
        btn = _fight_button(dx, dy, row_lock=True)
        return Decision(
            [Step(REFLEX_FRAMES * 2, btn)],
            note=f"east-march-approach ({dx},{dy})",
        )
    low = scene.health <= 1
    return Decision(
        east_march_combat_plan(dx, dy, low_health=low),
        note=f"east-march-fight ({dx},{dy})",
    )


def east_march_blocks_edge(btn: int) -> bool:
    """FAQ row-8 march only crosses east — block north/south/west screen hops."""
    return btn != c.RIGHT


def east_march_on_west_entry(scene) -> bool:
    """Fresh screen entries land on the west — must march east before hopping out."""
    return scene.link_x < EAST_APPROACH_X


def east_march_entry_unstable(scene, *, screen_frames: int) -> bool:
    """Block lip edge until Link has marched to the east half of the screen."""
    if scene.link_x < EAST_APPROACH_X:
        return True
    if screen_frames <= EAST_SCROLL_SETTLE_FRAMES and scene.link_x >= 160:
        return True
    return False


def east_march_at_lip(scene, *, screen_frames: int = 999) -> bool:
    """True at the east scroll-out tile after coordinates have settled."""
    return scene.at_right_edge and not east_march_entry_unstable(scene, screen_frames=screen_frames)


def east_march_deep_screen(map_location: int) -> bool:
    """Screens #121+ — denser octorok packs, longer cross runs."""
    return map_location >= START_SCREEN + 2


def east_march_needs_cross(scene) -> bool:
    """True when Link must slash/march east before a lip edge on this screen."""
    if east_march_deep_screen(scene.map_location):
        return scene.link_x < EAST_HOP_MAX_X
    return east_march_on_west_entry(scene)


def east_march_cross_reps(map_location: int) -> int:
    """Later row-8 screens need more cross steps before the lip edge."""
    if map_location <= START_SCREEN:
        return 35
    if map_location < START_SCREEN + 4:
        return 35 + (map_location - START_SCREEN) * 5
    return min(85, 60 + (map_location - START_SCREEN - 4) * 6)


def east_march_screen_cross_plan(reps: int | None = None, *, map_location: int = 0) -> Plan:
    """ROM-verified west→east march (reaches next screen without clearing all mobs)."""
    n = reps if reps is not None else east_march_cross_reps(map_location)
    plan: Plan = []
    for i in range(n):
        if i % 2:
            plan.append(Step(12, c.RIGHT))
        else:
            plan.append(Step(16, c.mask(c.RIGHT, c.B)))
    return plan


def east_march_screen_transition_macro(reps: int | None = None, *, map_location: int = 0) -> Plan:
    """Bankable west→east cross plus lip edge scroll for Director replay."""
    loc = map_location or START_SCREEN
    return east_march_screen_cross_plan(reps, map_location=loc) + list(EAST_EDGE_PLAN)


def east_march_cross_step(tick: int, *, map_location: int = 0) -> Step:
    if map_location >= START_SCREEN + 4:
        # Walk-heavy on #123+ — frequent RIGHT+B lifts Link into north octorok shots.
        if tick % 4 == 0:
            return Step(16, c.mask(c.RIGHT, c.B))
        return Step(14, c.RIGHT)
    return Step(16, c.mask(c.RIGHT, c.B)) if tick % 2 else Step(12, c.RIGHT)


def east_march_cross_decision(scene, *, tick: int, screen_frames: int = 999) -> Decision | None:
    """Alternate RIGHT / RIGHT+B while holding row 8 — crosses octorok screens."""
    if east_march_at_lip(scene, screen_frames=screen_frames):
        return None
    lane = east_march_lane_decision(scene, c.RIGHT)
    if lane is not None:
        return lane
    loc = scene.map_location
    if scene.health <= 2 and loc >= START_SCREEN + 4:
        return Decision(
            [Step(18, c.RIGHT) for _ in range(8)],
            note="east-march-survival-walk",
        )
    if loc >= START_SCREEN + 4:
        plan = [east_march_cross_step(tick + j, map_location=loc) for j in range(4)]
        return Decision(plan, note="east-march-cross-chunk")
    return Decision([east_march_cross_step(tick, map_location=loc)], note="east-march-cross")


def east_march_lane_decision(scene, btn: int) -> Decision | None:
    """Stay on FAQ row 8 between fights and screen hops."""
    if btn != c.RIGHT:
        return None
    if scene.link_y < EAST_MARCH_LANE_LO:
        hold = REFLEX_FRAMES * 2 if scene.link_y >= EAST_MARCH_LANE_LO - 12 else REFLEX_FRAMES
        return Decision([Step(hold, c.DOWN)], note="east-march-lane-down")
    if scene.link_y > EAST_MARCH_LANE_HI:
        hold = REFLEX_FRAMES * 3 if scene.link_y > EAST_MARCH_LANE_HI + 8 else REFLEX_FRAMES * 2
        return Decision([Step(hold, c.UP)], note="east-march-lane-up")
    if scene.link_y > EAST_MARCH_LANE_Y + 4:
        return Decision([Step(REFLEX_FRAMES * 2, c.UP)], note="east-march-lane-nudge-up")
    return None


def east_march_post_settle_macro(scene, *, screen_frames: int) -> Decision | None:
    """After hop settle, commit cross(+edge on #122) to punch through octoroks."""
    if scene.map_location > SEA_EAST_SCREEN:
        return None
    if screen_frames > EAST_SCROLL_SETTLE_FRAMES + 8:
        return None
    if not east_march_needs_cross(scene) or not east_march_lane_ok(scene):
        return None
    loc = scene.map_location
    if loc == START_SCREEN + 3:
        plan = east_march_lip_walk_plan(map_location=loc)
        return Decision(plan, note=f"east-march-lip-walk-#{loc}")
    if loc >= START_SCREEN + 4:
        plan = east_march_entry_burst_plan(map_location=loc)
        return Decision(plan, note=f"east-march-entry-burst-#{loc}")
    return None


def is_east_march_macro_plan(plan) -> bool:
    """Full cross+edge screen transition (long RIGHT-only plan)."""
    if not is_east_march_cross_plan(plan):
        return False
    return len(plan) > 40


def east_march_scroll_settle_decision(scene, *, frames_left: int) -> Decision | None:
    """Hold neutral while link_x settles after a screen hop (RAM scroll bit stays set)."""
    if frames_left <= 0:
        return None
    hold = min(frames_left, REFLEX_FRAMES * 2)
    return Decision([Step(hold, c.NEUTRAL)], note="east-march-scroll-settle")


def east_march_route_commit() -> tuple[int, str]:
    """Pinned FAQ direction while marching row 8 east."""
    return c.RIGHT, "walkthrough-east-to-sea"


def east_march_transition_plan(scene, *, screen_frames: int = 999) -> Plan | None:
    """ROM: 70f hop between screens (clear) or 48f lip scroll (enemies OK)."""
    if not east_march_lane_ok(scene):
        return None
    at_lip = east_march_at_lip(scene, screen_frames=screen_frames)
    if scene.health <= 1 and not at_lip:
        return None
    if east_march_entry_unstable(scene, screen_frames=screen_frames):
        return None
    if at_lip:
        return list(EAST_EDGE_PLAN)
    if not east_march_screen_clear(scene):
        return None
    if east_march_threatened(scene):
        return None
    if scene.link_x >= EAST_HOP_MAX_X:
        return None
    if EAST_APPROACH_X <= scene.link_x < EAST_HOP_MAX_X:
        return list(EAST_HOP_PLAN)
    return None


def east_march_approach_decision(scene, btn: int) -> Decision | None:
    """Walk the FAQ lip before a verified hop (screen clear, x < approach line)."""
    if btn != c.RIGHT or not east_march_screen_clear(scene):
        return None
    if scene.link_x >= EAST_APPROACH_X or not east_march_lane_ok(scene):
        return None
    return Decision([Step(REFLEX_FRAMES * 2, c.RIGHT)], note="east-march-approach-lip")


def east_march_walk_cross_plan(reps: int | None = None, *, map_location: int = 0) -> Plan:
    """Sword-free east march — avoids lifting Link into north octorok shots."""
    n = reps if reps is not None else east_march_cross_reps(map_location)
    return [Step(14, c.RIGHT) for _ in range(n)]


def is_east_march_walk_cross_plan(plan) -> bool:
    """Pure RIGHT holds — safe to bank/replay on #123+ without heart burn."""
    if not plan or len(plan) < 4:
        return False
    for step in plan:
        if step.buttons != c.RIGHT:
            return False
    return True


def east_march_lip_walk_plan(*, map_location: int) -> Plan:
    """Short walk-only burst from west entry — stops at lip (no screen scroll)."""
    n = max(28, east_march_cross_reps(map_location) - 20)
    return [Step(14, c.RIGHT) for _ in range(n)]


def east_march_entry_burst_plan(*, map_location: int) -> Plan:
    """Fresh-screen entry on #123+ — walk east before sword swings."""
    return [Step(14, c.RIGHT) for _ in range(20)]


def east_march_bank_candidates(scene) -> list[Plan]:
    """Director search order: walk-first on deep screens (preserve hearts)."""
    loc = scene.map_location
    out: list[Plan] = []
    if loc >= START_SCREEN + 4:
        out.append(east_march_lip_walk_plan(map_location=loc))
        out.append(east_march_walk_cross_plan(map_location=loc))
        out.append(east_march_entry_burst_plan(map_location=loc))
        out.extend(east_march_combat_candidates(scene))
        return out
    return east_march_combat_candidates(scene)


def east_march_entry_guard_decision(scene, *, screen_frames: int) -> Decision | None:
    """West entry on #124+ — neutral then walk before any replay tail."""
    if scene.map_location < START_SCREEN + 5:
        return None
    if not east_march_needs_cross(scene):
        return None
    if screen_frames > EAST_SCROLL_SETTLE_FRAMES + 24:
        return None
    if screen_frames <= 40:
        return Decision([Step(48, c.NEUTRAL)], note="east-march-entry-invuln")
    return Decision([Step(14, c.RIGHT) for _ in range(6)], note="east-march-entry-walk")


def east_march_combat_candidates(scene) -> list[Plan]:
    """Combat seeds for Director search on east-march screens."""
    loc = scene.map_location
    out: list[Plan] = [
        east_march_screen_cross_plan(map_location=loc),
        east_march_screen_transition_macro(map_location=loc),
        east_march_walk_cross_plan(map_location=loc),
    ]
    if loc >= START_SCREEN + 4:
        out.append(
            east_march_walk_cross_plan(map_location=loc) + list(EAST_EDGE_PLAN)
        )
    if scene.enemy_count() == 0:
        return out
    near = east_march_nearest_enemy(scene)
    if near is None:
        return []
    dx, dy = near
    btn = _fight_button(dx, dy, row_lock=True)
    out.extend([
        east_march_combat_plan(dx, dy),
        [Step(12, btn), Step(18, c.mask(btn, c.B))],
        [Step(16, c.mask(btn, c.B))],
        [Step(12, c.RIGHT), Step(16, c.mask(c.RIGHT, c.B))],
    ])
    return out


def east_march_decision(scene, btn: int, label: str, *, screen_frames: int = 999) -> Decision | None:
    """Commit a verified screen transition along FAQ row 8."""
    if btn != c.RIGHT or not east_march_lane_ok(scene):
        return None
    plan = east_march_transition_plan(scene, screen_frames=screen_frames)
    if plan is None:
        return None
    frames = plan[0].frames
    kind = "edge" if frames == EAST_EDGE_FRAMES else "hop"
    return Decision(plan, note=f"east-march-{kind} {label}")