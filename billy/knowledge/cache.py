"""The Solution Cache — Billy's compounding memory (the real "learning"), game-agnostic.

Deterministic games (SMB and most retro titles) replay identically from the same state, so the
moment micro-search discovers a button sequence that *verifiably survives* a hazard, we store
that exact sequence keyed to **where it happened**. On any later pass at that spot we **replay
the stored sequence directly** — no LLM, no re-search. Each hazard solved once is solved
forever, so attempt N+1 replays all of attempt N's solutions for free and only searches the new
frontier. That is what makes the learning curve compound instead of flat-line.

Cross-game by construction: the key is `(level_key, progress_bucket)` built from the engine's
generic `Observation.level_key` (any tuple a game defines) and `Observation.progress` (any
monotonic progress a game defines) — there is NOTHING SMB-specific here. A brand-new game gets
the entire discover-once / replay-forever capability for free just by implementing the existing
`Observation` contract; only the per-spot *solutions* are game-specific (as they must be). The
transferable cross-game knowledge (abstract tactics that generalize, e.g. "wait for the enemy,
then jump") lives separately in the embedding KB (knowledge/store.py), which seeds search and
biases the LLM when a new game's cache is still empty.

Entries are tiny (a few button steps), so the whole policy persists to a small `solutions.jsonl`
— no 768-dim embeddings, no megabytes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..abstractions import Plan, Step

LevelKey = tuple  # whatever a game uses to identify a level/area (game-agnostic)


def bucket_of(level_key: LevelKey, x: int, y: int = 0,
              bucket_px: int = config.CACHE_BUCKET_PX,
              y_px: int = config.CACHE_Y_BAND_PX) -> tuple:
    """Quantize a game-agnostic position (level_key, progress, elevation) into a route-node key.
    One x-bucket ≈ one NES tile (16px); the y-band disambiguates a high road from a low road at the
    same x (so the engine can tell 'I'm on the dead-end ledge' from 'I'm on the main path')."""
    return (tuple(level_key), x // bucket_px, y // y_px)


@dataclass
class CacheEntry:
    plan: list[Step]      # the exact verified-surviving button sequence
    reach_after: int      # progress reached after executing it (for picking the better of two)
    hits: int = 0         # times replayed (telemetry: shows compounding)
    fails: int = 0        # times a replay later failed (context drifted -> re-search)


@dataclass
class SolutionCache:
    """Position-keyed store of verified surviving action sequences. Persisted to solutions.jsonl."""
    path: Path = field(default_factory=lambda: config.SOLUTIONS_FILE)
    entries: dict[tuple, CacheEntry] = field(default_factory=dict)
    dead_ends: set = field(default_factory=set)   # route nodes proven to lead nowhere (in-memory)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._load()

    # --- lookup / update ----------------------------------------------------------------
    def get(self, level_key: LevelKey, x: int, y: int = 0) -> CacheEntry | None:
        return self.entries.get(bucket_of(level_key, x, y))

    def put(self, level_key: LevelKey, x: int, plan: Plan, reach_after: int,
            y: int = 0, force: bool = False) -> CacheEntry:
        """Store the solution for this route node. By default keep whichever reaches further; pass
        force=True to overwrite regardless — used when a cached plan went stale (failed verify) and
        a FRESH survivor was found: the fresh one works from the current state and must replace the
        stale one even at equal reach, otherwise the stale plan is re-searched on every pass and the
        learning never stabilises."""
        key = bucket_of(level_key, x, y)
        steps = [Step(s.frames, s.buttons) for s in plan]
        existing = self.entries.get(key)
        if existing is None or force or reach_after > existing.reach_after:
            self.entries[key] = CacheEntry(plan=steps, reach_after=reach_after,
                                           hits=existing.hits if existing else 0,
                                           fails=existing.fails if existing else 0)
            self._save()
        return self.entries[key]

    def record_hit(self, level_key: LevelKey, x: int, y: int = 0) -> None:
        e = self.get(level_key, x, y)
        if e:
            e.hits += 1
            self._save()

    def record_fail(self, level_key: LevelKey, x: int, y: int = 0) -> None:
        """A replayed solution didn't survive (context drifted). Drop it so search refreshes it."""
        key = bucket_of(level_key, x, y)
        e = self.entries.get(key)
        if e:
            e.fails += 1
            del self.entries[key]
            self._save()

    # --- dead-end memory (route-awareness): a node the stall-breaker proved leads nowhere ------
    def mark_dead(self, level_key: LevelKey, x: int, y: int = 0) -> None:
        self.dead_ends.add(bucket_of(level_key, x, y))

    def is_dead(self, level_key: LevelKey, x: int, y: int = 0) -> bool:
        return bucket_of(level_key, x, y) in self.dead_ends

    def solved_frontier(self, level_key: LevelKey) -> int:
        """Highest solved progress-bucket (in px) on this level — how far the policy reaches."""
        lk = tuple(level_key)
        xs = [xb * config.CACHE_BUCKET_PX for (k, xb, yb) in self.entries if k == lk]
        return max(xs) if xs else 0

    def __len__(self) -> int:
        return len(self.entries)

    # --- persistence (compact: level + bucket + button steps; no embeddings) ------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = (tuple(d["level"]), d["bucket"], d.get("yband", 0))
            plan = [Step(f, b) for f, b in d["plan"]]
            self.entries[key] = CacheEntry(plan=plan, reach_after=d["reach_after"],
                                           hits=d.get("hits", 0), fails=d.get("fails", 0))

    def _save(self) -> None:
        config.ensure_dirs()
        with self.path.open("w") as f:
            for (level_key, bucket, yband), e in sorted(
                    self.entries.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
                f.write(json.dumps({
                    "level": list(level_key), "bucket": bucket, "yband": yband,
                    "plan": [[s.frames, s.buttons] for s in e.plan],
                    "reach_after": e.reach_after, "hits": e.hits, "fails": e.fails,
                }) + "\n")
