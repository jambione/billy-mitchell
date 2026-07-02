"""Whole-trajectory tapes — frame-perfect level replays from a checkpoint.

The emulator is deterministic: replaying the exact committed input stream from a level-entry
savestate reproduces the run. Tapes make repeat clears search-free (the headline compounding win).

A clearing tape may carry an ENTRY-STATE ANCHOR (`entry_state`, stored as a sidecar .state
file): the exact savestate it was recorded from. For a level with a MOVING hazard (1-3's lift,
whose phase is set at level load) the input stream only reproduces from that precise state, so
the Director restores the anchor at level begin before replaying — imperceptible (the player is
at the level start regardless) and it makes the whole level deterministic.
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
    fails: int = 0         # consecutive clone-verify failures (drops at FAIL_LIMIT)
    entry_state: bytes | None = None   # savestate the tape was recorded from (anchors replay)


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
            *, clears_level: bool, entry_state: bytes | None = None) -> TapeEntry:
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
            # Carry forward an existing anchor if this bank didn't supply one (a self-recorded
            # extension keeps the demo's entry state — the thing that makes the lift reproduce).
            anchor = entry_state if entry_state is not None else (
                existing.entry_state if existing else None)
            self.entries[key] = TapeEntry(
                level_key=key, plan=steps, frontier=frontier,
                clears_level=clears_level,
                hits=existing.hits if existing else 0,
                entry_state=anchor)
            self._save()
        return self.entries[key]

    FAIL_LIMIT = 2   # verify failures before a tape is dropped (mirrors cache record_fail)

    def record_hit(self, level_key: LevelKey) -> None:
        e = self.get(level_key)
        if e:
            e.hits += 1
            e.fails = 0
            self._save()

    def record_fail(self, level_key: LevelKey) -> None:
        """A stored tape failed its clone-verify. Repeated failures mean the tape no longer
        matches reality (drifted checkpoint, or a corrupt legacy entry whose inflated frontier
        would otherwise BLOCK honest replacements forever) — drop it so the next good run
        stores fresh. Self-healing, same contract as SolutionCache.record_fail."""
        e = self.get(level_key)
        if e is None:
            return
        e.fails += 1
        if e.fails >= self.FAIL_LIMIT:
            del self.entries[tuple(level_key)]
        self._save()

    def __len__(self) -> int:
        return len(self.entries)

    def _state_path(self, key: LevelKey) -> Path:
        """Sidecar file for a tape's entry savestate (kept out of the JSONL — it's ~13KB)."""
        safe = "_".join(str(p) for p in key)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in safe)
        return self.path.parent / "tape_states" / f"{safe}.state"

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
            entry_state = None
            if d.get("entry_state"):
                sp = Path(d["entry_state"])
                if sp.exists():
                    entry_state = sp.read_bytes()
            self.entries[key] = TapeEntry(
                level_key=key, plan=plan, frontier=d["frontier"],
                clears_level=d.get("clears_level", False), hits=d.get("hits", 0),
                fails=d.get("fails", 0), entry_state=entry_state)

    def _save(self) -> None:
        config.ensure_dirs()
        with self.path.open("w") as f:
            for key in sorted(self.entries, key=str):
                e = self.entries[key]
                rec = {
                    "level_key": list(e.level_key),
                    "plan": [[s.frames, s.buttons] for s in e.plan],
                    "frontier": e.frontier,
                    "clears_level": e.clears_level,
                    "hits": e.hits,
                    "fails": e.fails,
                }
                if e.entry_state is not None:
                    sp = self._state_path(key)
                    sp.parent.mkdir(parents=True, exist_ok=True)
                    sp.write_bytes(e.entry_state)
                    rec["entry_state"] = str(sp)
                f.write(json.dumps(rec) + "\n")


def append_plan(record: list[Step], plan: Plan) -> None:
    """Extend a tape recording with a committed plan, merging adjacent same-button steps so a
    tape replayed in small chunks re-records to the same compact stream (no step-count bloat)."""
    for s in plan:
        if record and record[-1].buttons == s.buttons:
            record[-1] = Step(record[-1].frames + s.frames, s.buttons)
        else:
            record.append(Step(s.frames, s.buttons))