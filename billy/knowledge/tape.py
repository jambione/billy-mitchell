"""Whole-trajectory tapes — frame-perfect level replays from a checkpoint.

The emulator is deterministic: replaying the exact committed input stream from a level-entry
savestate reproduces the run. Tapes make repeat clears search-free (the headline compounding win).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..abstractions import Plan, Step, plan_frames

LevelKey = tuple


@dataclass
class TapeEntry:
    level_key: LevelKey
    plan: list[Step]
    frontier: int          # max progress reached on this tape
    clears_level: bool     # advanced world/stage when recorded
    hits: int = 0


@dataclass
class TapeLibrary:
    """Per-level input tapes persisted to JSONL."""
    path: Path = field(default_factory=lambda: config.TAPES_FILE)
    entries: dict[LevelKey, TapeEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._load()

    def get(self, level_key: LevelKey) -> TapeEntry | None:
        return self.entries.get(tuple(level_key))

    def put(self, level_key: LevelKey, plan: Plan, frontier: int,
            *, clears_level: bool) -> TapeEntry:
        key = tuple(level_key)
        steps = [Step(s.frames, s.buttons) for s in plan]
        existing = self.entries.get(key)
        if existing is not None and existing.clears_level and not clears_level:
            replace = False   # a partial (frontier) tape never displaces a full clear
        else:
            replace = (existing is None
                       or clears_level and not existing.clears_level
                       or frontier > existing.frontier
                       or (clears_level and frontier >= existing.frontier))
        if replace:
            self.entries[key] = TapeEntry(
                level_key=key, plan=steps, frontier=frontier,
                clears_level=clears_level,
                hits=existing.hits if existing else 0)
            self._save()
        return self.entries[key]

    def record_hit(self, level_key: LevelKey) -> None:
        e = self.get(level_key)
        if e:
            e.hits += 1
            self._save()

    def __len__(self) -> int:
        return len(self.entries)

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = tuple(d["level_key"])
            plan = [Step(f, b) for f, b in d["plan"]]
            self.entries[key] = TapeEntry(
                level_key=key, plan=plan, frontier=d["frontier"],
                clears_level=d.get("clears_level", False), hits=d.get("hits", 0))

    def _save(self) -> None:
        config.ensure_dirs()
        with self.path.open("w") as f:
            for key in sorted(self.entries, key=str):
                e = self.entries[key]
                f.write(json.dumps({
                    "level_key": list(e.level_key),
                    "plan": [[s.frames, s.buttons] for s in e.plan],
                    "frontier": e.frontier,
                    "clears_level": e.clears_level,
                    "hits": e.hits,
                }) + "\n")


def append_plan(record: list[Step], plan: Plan) -> None:
    """Extend a tape recording with a committed plan, merging adjacent same-button steps so a
    tape replayed in small chunks re-records to the same compact stream (no step-count bloat)."""
    for s in plan:
        if record and record[-1].buttons == s.buttons:
            record[-1] = Step(record[-1].frames + s.frames, s.buttons)
        else:
            record.append(Step(s.frames, s.buttons))