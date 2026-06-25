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


def bucket_of(level_key: LevelKey, x: int,
              bucket_px: int = config.CACHE_BUCKET_PX) -> tuple:
    """Quantize a game-agnostic position (level_key, progress) into a cache key.
    One bucket ≈ one NES tile (16px) by default."""
    return (tuple(level_key), x // bucket_px)


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

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._load()

    # --- lookup / update ----------------------------------------------------------------
    def get(self, level_key: LevelKey, x: int) -> CacheEntry | None:
        return self.entries.get(bucket_of(level_key, x))

    def put(self, level_key: LevelKey, x: int, plan: Plan, reach_after: int) -> CacheEntry:
        """Store (or improve) the solution for this bucket. Keep whichever reaches further."""
        key = bucket_of(level_key, x)
        steps = [Step(s.frames, s.buttons) for s in plan]
        existing = self.entries.get(key)
        if existing is None or reach_after > existing.reach_after:
            self.entries[key] = CacheEntry(plan=steps, reach_after=reach_after,
                                           hits=existing.hits if existing else 0,
                                           fails=existing.fails if existing else 0)
            self._save()
        return self.entries[key]

    def record_hit(self, level_key: LevelKey, x: int) -> None:
        e = self.get(level_key, x)
        if e:
            e.hits += 1
            self._save()

    def record_fail(self, level_key: LevelKey, x: int) -> None:
        """A replayed solution didn't survive (context drifted). Drop it so search refreshes it."""
        key = bucket_of(level_key, x)
        e = self.entries.get(key)
        if e:
            e.fails += 1
            del self.entries[key]
            self._save()

    def solved_frontier(self, level_key: LevelKey) -> int:
        """Highest solved progress-bucket (in px) on this level — how far the policy reaches."""
        lk = tuple(level_key)
        xs = [b * config.CACHE_BUCKET_PX for (k, b) in self.entries if k == lk]
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
            key = (tuple(d["level"]), d["bucket"])
            plan = [Step(f, b) for f, b in d["plan"]]
            self.entries[key] = CacheEntry(plan=plan, reach_after=d["reach_after"],
                                           hits=d.get("hits", 0), fails=d.get("fails", 0))

    def _save(self) -> None:
        config.ensure_dirs()
        with self.path.open("w") as f:
            for (level_key, bucket), e in sorted(self.entries.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
                f.write(json.dumps({
                    "level": list(level_key), "bucket": bucket,
                    "plan": [[s.frames, s.buttons] for s in e.plan],
                    "reach_after": e.reach_after, "hits": e.hits, "fails": e.fails,
                }) + "\n")
