"""Zelda hazard hooks — combat zones, damage learning, screen-transition stalls."""
from __future__ import annotations

from ...abstractions import Observation, Plan



class ZeldaHazardHooks:
    """Top-down combat + exploration hooks for the game-agnostic Director."""

    def commit_chunk_size(self, obs: Observation, default: int) -> int:
        scene = obs.raw
        if scene.enemy_count() > 0:
            return 2   # finer commits during fights for learn-from-death runway
        return default

    def in_special_zone(self, obs: Observation) -> bool:
        scene = obs.raw
        if scene.enemy_count() > 0 or scene.scrolling:
            return True
        if scene.in_cave:
            from .start_cave import has_wooden_sword, interior_phase
            if not has_wooden_sword(scene.sword_level):
                phase = interior_phase(scene)
                if phase in ("text", "climb", "pickup", "exit"):
                    return False
        return scene.in_cave

    def stall_break_exempt(self, obs: Observation) -> bool:
        scene = obs.raw
        # Screen scrolls and edge transitions stall progress without being dead-ends.
        if scene.scrolling:
            return True
        if (scene.at_right_edge or scene.at_left_edge
                or scene.at_top_edge or scene.at_bottom_edge):
            return True
        # Early overworld: Link may walk a long path to a cave mouth or screen lip.
        if not scene.in_dungeon and len(scene.visited_screens) < 12:
            return True
        from .curiosity import needs_cave_approach
        if needs_cave_approach(scene.map_location, scene.link_x, scene.link_y,
                               cave_mouths=scene.cave_mouths):
            return True
        if scene.in_cave or scene.item_count() > 0:
            return True
        return False

    def stale_cache(self, obs: Observation, cached) -> bool:
        if cached is None:
            return False
        from .start_cave import has_wooden_sword
        from .tuning import START_SCREEN
        scene = obs.raw
        if scene.map_location != START_SCREEN:
            return False
        # Cave interior + pre-sword overworld: reflex macro owns the route; cached
        # item-walk / partial interior plans from earlier runs loop forever.
        if scene.in_cave or not has_wooden_sword(scene.sword_level):
            return True
        return False

    def pit_death(self, level_label: str, death_x: int) -> bool:
        return False

    def is_special_death(self, level_label: str, death_x: int) -> bool:
        return "overworld" in level_label or "dungeon" in level_label

    def learn_horizon_frames(self, level_label: str, death_x: int) -> int | None:
        return 120

    def learn_runway_action(self, level_label: str, death_x: int, runway: int,
                            default_horizon: int) -> str | None:
        return None

    def learn_cacheable(self, level_label: str, death_x: int, reach: int) -> bool:
        return reach > death_x + 8

    def try_frame_search(self, session, observe, obs: Observation, *, deep: bool,
                         min_gain: int) -> tuple[Plan | None, int, bool]:
        return None, obs.progress, False

    def try_pit_search(self, session, observe, obs: Observation, *,
                       death_x: int, min_gain: int) -> tuple[Plan | None, int]:
        return None, obs.progress

    def cacheable_reach(self, obs: Observation, reach: int, *, crossed: bool = False) -> bool:
        return reach > obs.progress + 8

    def section_bankable(self, obs: Observation, reach: int) -> bool:
        return True

    def replay_death_drop_reason(self, level_label: str, replay_x: int,
                                 death_x: int) -> str | None:
        return "combat_replay_fail"

    def extra_candidates(self, obs: Observation) -> list[Plan]:
        from ...systems.nes import controller as c
        from .reflex import combat_candidates, _walk
        from .start_cave import has_wooden_sword, macro_candidates
        scene = obs.raw
        out: list[Plan] = []
        if scene.in_cave and not has_wooden_sword(scene.sword_level):
            out.extend(macro_candidates())
        out.extend(combat_candidates(obs))
        if scene.in_cave or scene.item_count() > 0:
            out.extend([
                _walk(c.UP, 24),
                _walk(c.RIGHT, 16) + _walk(c.UP, 24),
                _walk(c.A, 30) + _walk(c.UP, 32),
                _walk(c.LEFT, 12) + _walk(c.UP, 32),
            ])
        return out

    def approach_capture_band(self, level_label: str, death_x: int) -> tuple[int, int] | None:
        return None

    def approach_snapshot_band(self, obs: Observation) -> tuple[int, int] | None:
        return None

    def stuck_remedy(self, level_label: str, death_x: int):
        return None