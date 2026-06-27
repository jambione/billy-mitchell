"""Game-defined hazard hooks — keeps the Director game-agnostic.

Each title may supply a `HazardHooks` implementation for special zones (SMB's lift gap,
pit approaches, etc.). The engine calls these instead of importing game modules directly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .abstractions import Observation, Plan, Step

if TYPE_CHECKING:
    from .knowledge.cache import CacheEntry
    from .stuck_trainer import StuckRemedy


@runtime_checkable
class HazardHooks(Protocol):
    """Optional per-game callbacks for hazard-scoped behaviour."""

    def commit_chunk_size(self, obs: Observation, default: int) -> int:
        """Frames per live-commit chunk (1 = frame-granular for timing zones)."""
        ...

    def in_special_zone(self, obs: Observation) -> bool:
        """True when the agent is in a zone that needs search/replay off-ground."""
        ...

    def stall_break_exempt(self, obs: Observation) -> bool:
        """True to skip the stall-breaker (timing stalls are not dead-ends)."""
        ...

    def stale_cache(self, obs: Observation, cached: "CacheEntry | None") -> bool:
        """True when a cached replay must be discarded and re-searched."""
        ...

    def pit_death(self, level_label: str, death_x: int) -> bool:
        """Death at a pit lip (drives pit search goal after a fall)."""
        ...

    def is_special_death(self, level_label: str, death_x: int) -> bool:
        """Death in a zone that needs extended learn-from-death."""
        ...

    def learn_horizon_frames(self, level_label: str, death_x: int) -> int | None:
        """Post-candidate coast budget for learn-from-death; None = engine default."""
        ...

    def learn_runway_action(self, level_label: str, death_x: int, runway: int,
                            default_horizon: int) -> str | None:
        """'continue' | 'break' | None — how to scan the snapshot trail."""
        ...

    def learn_cacheable(self, level_label: str, death_x: int, reach: int) -> bool:
        """Whether a learn-from-death survivor may be banked."""
        ...

    def try_frame_search(self, session, observe, obs: Observation, *, deep: bool,
                         min_gain: int) -> tuple[Plan | None, int, bool]:
        """Zone-specific frame search (e.g. lift gap). Returns (plan, reach, crossed)."""
        ...

    def try_pit_search(self, session, observe, obs: Observation, *,
                       death_x: int, min_gain: int) -> tuple[Plan | None, int]:
        """Pit-approach gap-jump search. Returns (plan, reach)."""
        ...

    def cacheable_reach(self, obs: Observation, reach: int, *, crossed: bool = False) -> bool:
        """Whether a live-search survivor at this spot may be banked."""
        ...

    def section_bankable(self, obs: Observation, reach: int) -> bool:
        """Whether a section-policy crossing may be banked."""
        ...

    def replay_death_drop_reason(self, level_label: str, replay_x: int,
                                 death_x: int) -> str | None:
        """Drop reason when a committed replay dies; None = default handling."""
        ...

    def extra_candidates(self, obs: Observation) -> list[Plan]:
        """Additional micro-search candidates for the current zone."""
        ...

    def stuck_remedy(self, level_label: str, death_x: int) -> "StuckRemedy | None":
        """Recipe for auto-stuck remediation when deaths cluster at this hazard."""
        ...

    def approach_capture_band(self, level_label: str, death_x: int) -> tuple[int, int] | None:
        """(x_lo, x_hi) for auto-capturing an approach savestate after a death; None = skip."""
        ...

    def approach_snapshot_band(self, obs: Observation) -> tuple[int, int] | None:
        """(x_lo, x_hi) while live-playing — retained in a dedicated approach trail."""
        ...


class NullHazardHooks:
    """Default no-op hooks — used by games without special hazard logic."""

    def commit_chunk_size(self, obs: Observation, default: int) -> int:
        return default

    def in_special_zone(self, obs: Observation) -> bool:
        return False

    def stall_break_exempt(self, obs: Observation) -> bool:
        return False

    def stale_cache(self, obs: Observation, cached: "CacheEntry | None") -> bool:
        return False

    def pit_death(self, level_label: str, death_x: int) -> bool:
        return False

    def is_special_death(self, level_label: str, death_x: int) -> bool:
        return False

    def learn_horizon_frames(self, level_label: str, death_x: int) -> int | None:
        return None

    def learn_runway_action(self, level_label: str, death_x: int, runway: int,
                            default_horizon: int) -> str | None:
        return None

    def learn_cacheable(self, level_label: str, death_x: int, reach: int) -> bool:
        return True

    def try_frame_search(self, session, observe, obs: Observation, *, deep: bool,
                         min_gain: int) -> tuple[Plan | None, int, bool]:
        return None, obs.progress, False

    def try_pit_search(self, session, observe, obs: Observation, *,
                       death_x: int, min_gain: int) -> tuple[Plan | None, int]:
        return None, obs.progress

    def cacheable_reach(self, obs: Observation, reach: int, *, crossed: bool = False) -> bool:
        return True

    def section_bankable(self, obs: Observation, reach: int) -> bool:
        return True

    def replay_death_drop_reason(self, level_label: str, replay_x: int,
                                 death_x: int) -> str | None:
        return None

    def extra_candidates(self, obs: Observation) -> list[Plan]:
        return []

    def stuck_remedy(self, level_label: str, death_x: int):
        return None

    def approach_capture_band(self, level_label: str, death_x: int) -> tuple[int, int] | None:
        return None

    def approach_snapshot_band(self, obs: Observation) -> tuple[int, int] | None:
        return None


_NULL = NullHazardHooks()


def null_hooks() -> HazardHooks:
    return _NULL