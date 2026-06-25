"""The Solution Cache — Billy's compounding memory (the real "learning").

SMB is deterministic: the same exact inputs from the same state always produce the same result.
So the moment micro-search discovers a button sequence that *verifiably survives* a hazard, we
store that exact sequence keyed to **where it happened** — `(world, stage, x_bucket)`. On any
later pass, if Billy reaches that bucket again we **replay the stored sequence directly** — no
LLM, no re-search. Each hazard solved once is solved forever, so attempt N+1 replays all of
attempt N's solutions for free and only searches the new frontier. That is what makes the
learning curve compound instead of flat-line.

This deliberately replaces the old prose-lesson + embedding + LLM-re-interpretation pipeline as
the *execution* policy: that pipeline threw the exact solution away and re-guessed it stochastically
every time, which is why learning never compounded. Entries are tiny (a few button steps), so the
whole policy persists to a small `solutions.jsonl` — no 768-dim embeddings, no megabytes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..abstractions import Plan, Step


def bucket_of(world: int, stage: int, x: int,
              bucket_px: int = config.CACHE_BUCKET_PX) -> tuple[int, int, int]:
    """Quantize a position into a cache key. One bucket ≈ one NES tile (16px) by default."""
    return (world, stage, x // bucket_px)


@dataclass
class CacheEntry:
    plan: list[Step]      # the exact verified-surviving button sequence
    reach_after: int      # world-x reached after executing it (for picking the better of two)
    hits: int = 0         # times replayed (telemetry: shows compounding)
    fails: int = 0        # times a replay later failed (context drifted -> re-search)


@dataclass
class SolutionCache:
    """Position-keyed store of verified surviving action sequences. Persisted to solutions.jsonl."""
    path: Path = field(default_factory=lambda: config.SOLUTIONS_FILE)
    entries: dict[tuple[int, int, int], CacheEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._load()

    # --- lookup / update ----------------------------------------------------------------
    def get(self, world: int, stage: int, x: int) -> CacheEntry | None:
        return self.entries.get(bucket_of(world, stage, x))

    def put(self, world: int, stage: int, x: int, plan: Plan, reach_after: int) -> CacheEntry:
        """Store (or improve) the solution for this bucket. Keep whichever reaches further."""
        key = bucket_of(world, stage, x)
        steps = [Step(s.frames, s.buttons) for s in plan]
        existing = self.entries.get(key)
        if existing is None or reach_after > existing.reach_after:
            self.entries[key] = CacheEntry(plan=steps, reach_after=reach_after,
                                           hits=existing.hits if existing else 0,
                                           fails=existing.fails if existing else 0)
            self._save()
        return self.entries[key]

    def record_hit(self, world: int, stage: int, x: int) -> None:
        e = self.get(world, stage, x)
        if e:
            e.hits += 1
            self._save()

    def record_fail(self, world: int, stage: int, x: int) -> None:
        """A replayed solution didn't survive (context drifted). Drop it so search refreshes it."""
        key = bucket_of(world, stage, x)
        e = self.entries.get(key)
        if e:
            e.fails += 1
            del self.entries[key]
            self._save()

    def solved_frontier(self, world: int, stage: int) -> int:
        """Highest solved x-bucket (in px) on this level — how far the cached policy reaches."""
        xs = [b * config.CACHE_BUCKET_PX for (w, s, b) in self.entries if w == world and s == stage]
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
            key = (d["world"], d["stage"], d["bucket"])
            plan = [Step(f, b) for f, b in d["plan"]]
            self.entries[key] = CacheEntry(plan=plan, reach_after=d["reach_after"],
                                           hits=d.get("hits", 0), fails=d.get("fails", 0))

    def _save(self) -> None:
        config.ensure_dirs()
        with self.path.open("w") as f:
            for (world, stage, bucket), e in sorted(self.entries.items()):
                f.write(json.dumps({
                    "world": world, "stage": stage, "bucket": bucket,
                    "plan": [[s.frames, s.buttons] for s in e.plan],
                    "reach_after": e.reach_after, "hits": e.hits, "fails": e.fails,
                }) + "\n")
