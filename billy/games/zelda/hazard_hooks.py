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
        if scene.in_dungeon:
            return True
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
                               cave_mouths=scene.cave_mouths,
                               sword_level=scene.sword_level):
            return True
        if scene.in_cave or scene.item_count() > 0:
            return True
        if scene.in_dungeon:
            return True
        from .east_march import east_march_active
        if east_march_active(scene, visited=set(scene.visited_screens)):
            if scene.enemy_count() > 0 or scene.scrolling:
                return True
        return False

    def stale_cache(self, obs: Observation, cached) -> bool:
        if cached is None:
            return False
        from ...abstractions import plan_frames
        from ...systems.nes import controller as c
        from .east_march import (
            is_east_march_combat_plan,
            is_east_march_cross_plan,
            is_east_march_macro_plan,
            is_east_march_plan,
            is_east_march_walk_cross_plan,
        )
        from .start_cave import has_wooden_sword, is_cave_macro_plan
        from .tuning import START_SCREEN
        from .walkthrough import SEA_EAST_SCREEN, current_phase, screen_to_grid

        scene = obs.raw
        if is_cave_macro_plan(cached.plan):
            if scene.in_cave:
                return False
            return True   # never replay interior macros on overworld
        from .east_march import EAST_EDGE_PLAN
        from .tuning import SCREEN_EDGE_HI
        if (is_east_march_plan(cached.plan) or is_east_march_cross_plan(cached.plan)
                or is_east_march_macro_plan(cached.plan)):
            if has_wooden_sword(scene.sword_level):
                if (not scene.in_cave and not scene.in_dungeon
                        and START_SCREEN <= scene.map_location <= SEA_EAST_SCREEN):
                    # 48f lip-edge only replays at true east tile — mid-screen causes ping-pong.
                    if (list(cached.plan) == list(EAST_EDGE_PLAN)
                            and scene.link_x < SCREEN_EDGE_HI):
                        return True
                    # Full cross+edge macros must replay from west entry, not mid-lip.
                    if (is_east_march_macro_plan(cached.plan)
                            and scene.link_x >= SCREEN_EDGE_HI):
                        return True
                    # #122: cached transition macros scroll through #123 into #124.
                    if (scene.map_location == START_SCREEN + 3
                            and (is_east_march_macro_plan(cached.plan)
                                 or (is_east_march_cross_plan(cached.plan)
                                     and len(cached.plan) > 30))):
                        return True
                    # West entry on #124+: always reflex (entry guard), never replay tail.
                    if (scene.map_location >= START_SCREEN + 5
                            and scene.link_x < 140
                            and (is_east_march_cross_plan(cached.plan)
                                 or is_east_march_macro_plan(cached.plan))):
                        return True
                    # Long walk replays on #123+ scroll into the next screen — lip plans only.
                    if (scene.map_location >= START_SCREEN + 4
                            and is_east_march_walk_cross_plan(cached.plan)
                            and len(cached.plan) > 44):
                        return True
                    # Sword-cross replays on #123+ burn hearts — prefer walk plans.
                    if (scene.map_location >= START_SCREEN + 4
                            and is_east_march_cross_plan(cached.plan)
                            and not is_east_march_walk_cross_plan(cached.plan)):
                        return True
                    # Post-damage on #124+: long walk replays from carryover kill.
                    if (scene.health <= 2
                            and scene.map_location >= START_SCREEN + 5
                            and is_east_march_walk_cross_plan(cached.plan)
                            and len(cached.plan) > 24):
                        return True
                    # Post-damage retry: macros learned at full hearts often kill mid-screen.
                    if (scene.health < scene.max_hearts
                            and scene.map_location <= START_SCREEN + 1
                            and scene.link_x < 192
                            and (is_east_march_cross_plan(cached.plan)
                                 or is_east_march_macro_plan(cached.plan))):
                        return True
                    return False

        # Pre-sword: stale partial wander on start screen / in-cave.
        if not has_wooden_sword(scene.sword_level):
            if scene.map_location == START_SCREEN:
                return True
            return False

        # In-cave post-sword: only verified macros replay.
        if scene.in_cave:
            return True

        # Post-sword overworld east march — prevent #119↔#120 ping-pong from short edge cache.
        if (not scene.in_dungeon
                and START_SCREEN <= scene.map_location <= SEA_EAST_SCREEN):
            pf = plan_frames(cached.plan)
            # Short lip hops only — not combat/learn-from-death survivors on #121+.
            if is_east_march_plan(cached.plan) and pf < 56:
                return True
            if (pf < 56
                    and not is_east_march_combat_plan(cached.plan)
                    and scene.map_location <= START_SCREEN + 1):
                return True
            buttons = 0
            for step in cached.plan:
                buttons |= step.buttons
            if scene.map_location > START_SCREEN and (buttons & c.LEFT) and scene.at_left_edge:
                return True
            visited = set(scene.visited_screens)
            phase = current_phase(
                map_location=scene.map_location,
                sword_level=scene.sword_level,
                max_hearts=scene.max_hearts,
                visited=visited,
                in_cave=scene.in_cave,
            )
            _, gy = screen_to_grid(scene.map_location)
            if (phase == "east_to_sea" and gy == 8
                    and not is_east_march_plan(cached.plan)
                    and not is_east_march_cross_plan(cached.plan)
                    and not is_east_march_combat_plan(cached.plan)):
                if buttons & (c.UP | c.DOWN):
                    return True
                if (buttons & c.LEFT) and not scene.at_left_edge:
                    return True
                if (scene.enemy_count() > 0 and pf < 48
                        and not is_east_march_combat_plan(cached.plan)):
                    return True
            return False

        return False

    def pit_death(self, level_label: str, death_x: int) -> bool:
        return False

    def is_special_death(self, level_label: str, death_x: int) -> bool:
        return "overworld" in level_label or "dungeon" in level_label

    def learn_horizon_frames(self, level_label: str, death_x: int) -> int | None:
        if "dungeon" in level_label:
            return 160
        return 240 if "overworld" in level_label else 120

    def learn_runway_action(self, level_label: str, death_x: int, runway: int,
                            default_horizon: int) -> str | None:
        return None

    def learn_cacheable(self, level_label: str, death_x: int, reach: int) -> bool:
        if "overworld" in level_label:
            return reach > death_x + 8
        return reach > death_x + 8

    def try_frame_search(self, session, observe, obs: Observation, *, deep: bool,
                         min_gain: int) -> tuple[Plan | None, int, bool]:
        from .east_march import east_march_active, east_march_bank_candidates
        from .tuning import START_SCREEN

        scene = obs.raw
        if not east_march_active(scene, visited=set(scene.visited_screens)):
            return None, obs.progress, False
        deep_screen = scene.map_location >= START_SCREEN + 4
        if deep_screen:
            min_gain = min(min_gain, 8)
        if scene.enemy_count() == 0 and scene.link_x >= 200:
            return None, obs.progress, False

        snap = session.clone_state()
        start = obs.progress
        start_screen = scene.map_location
        from .east_march import east_march_lip_walk_plan
        candidates = east_march_bank_candidates(scene)
        if deep_screen:
            candidates = [east_march_lip_walk_plan(map_location=scene.map_location)] + candidates
        if deep and not deep_screen:
            from .reflex import combat_candidates
            candidates = candidates + combat_candidates(obs)

        best_plan, best_reach = None, start
        best_hearts = scene.health
        best_score: tuple | None = None
        with session.search_mode():
            for plan in candidates:
                session.restore(snap)
                observe()
                session.send_plan(plan)
                trial = observe()
                if trial.dead:
                    continue
                reach = trial.progress
                hearts = trial.raw.health
                if reach < start + min_gain:
                    continue
                if deep_screen:
                    advanced = trial.raw.map_location > start_screen
                    if advanced and hearts < 2:
                        continue
                    if advanced and len(plan) > 44:
                        continue
                    score = (0 if advanced else 1, hearts, reach)
                    if best_score is None or score > best_score:
                        best_plan, best_reach, best_hearts = list(plan), reach, hearts
                        best_score = score
                elif reach > best_reach:
                    best_plan, best_reach, best_hearts = list(plan), reach, hearts

        session.restore(snap)
        observe()
        if best_plan is not None:
            return best_plan, best_reach, False
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
        from .east_march import (
            EAST_EDGE_PLAN,
            EAST_HOP_PLAN,
            east_march_active,
            east_march_bank_candidates,
            east_march_lip_walk_plan,
            east_march_screen_cross_plan,
            east_march_screen_transition_macro,
            east_march_walk_cross_plan,
        )
        from .reflex import combat_candidates, _walk
        from .start_cave import has_wooden_sword, macro_candidates
        scene = obs.raw
        out: list[Plan] = []
        if scene.in_cave and not has_wooden_sword(scene.sword_level):
            out.extend(macro_candidates())
        if east_march_active(scene, visited=set(scene.visited_screens)):
            out.append(EAST_HOP_PLAN)
            out.append(EAST_EDGE_PLAN)
            loc = scene.map_location
            from .tuning import START_SCREEN as _SS
            if loc >= _SS + 4:
                out.append(east_march_lip_walk_plan(map_location=loc))
                out.append(east_march_walk_cross_plan(map_location=loc))
            out.append(east_march_screen_cross_plan(map_location=loc))
            out.append(east_march_screen_transition_macro(map_location=loc))
            out.extend(east_march_bank_candidates(scene))
        out.extend(combat_candidates(obs))
        if scene.in_dungeon:
            from .dungeon_nav import dungeon_combat_decision
            from .reflex import _walk, _sword
            step = dungeon_combat_decision(scene)
            if step is not None:
                out.append(list(step.plan))
            out.extend([
                _sword(c.RIGHT, 14),
                _walk(c.RIGHT, 16) + _sword(c.RIGHT, 14),
                _walk(c.LEFT, 12) + _sword(c.LEFT, 14),
            ])
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
        from .east_march import east_march_active
        from .tuning import START_SCREEN

        scene = obs.raw
        if east_march_active(scene, visited=set(scene.visited_screens)):
            if scene.map_location >= START_SCREEN + 4:
                return (obs.progress - 48, obs.progress + 96)
        if scene.in_dungeon:
            return (obs.progress - 32, obs.progress + 64)
        return None

    def stuck_remedy(self, level_label: str, death_x: int):
        return None