"""Learning ledger — makes Billy's compounding memory visible attempt-to-attempt."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config
from .knowledge.cache import SolutionCache, LevelKey


@dataclass
class AttemptLearning:
    """Per-attempt learning deltas (the proof exponential learning is happening)."""
    banks: int = 0          # new verified solutions stored
    drops: int = 0          # stale solutions removed (self-healing)
    learns: int = 0         # learn-from-death successes
    trains: int = 0         # auto-stuck remediation successes
    replay_hits: int = 0    # free cache replays
    cache_size: int = 0     # entries at end of attempt
    level_frontier: int = 0 # solved px frontier on the furthest level reached


@dataclass
class LearningLedger:
    """Counts learning events during a session; persists a compact event log."""
    path: Path = field(default_factory=lambda: config.DATA_DIR / "learning.jsonl")
    attempt: AttemptLearning = field(default_factory=AttemptLearning)
    _cache_size_start: int = 0

    def begin_attempt(self, cache: SolutionCache) -> None:
        self.attempt = AttemptLearning()
        self._cache_size_start = len(cache)

    def bank(self, level_key: LevelKey, x: int, reach: int, source: str) -> None:
        self.attempt.banks += 1
        self._event("bank", level_key=level_key, x=x, reach=reach, source=source)

    def drop(self, level_key: LevelKey, x: int, reason: str) -> None:
        self.attempt.drops += 1
        self._event("drop", level_key=level_key, x=x, reason=reason)

    def learn(self, level_key: LevelKey, x: int, death_x: int, source: str) -> None:
        self.attempt.learns += 1
        self._event("learn", level_key=level_key, x=x, death_x=death_x, source=source)

    def train(self, level_key: LevelKey, x: int, reach: int, source: str) -> None:
        self.attempt.trains += 1
        self._event("train", level_key=level_key, x=x, reach=reach, source=source)

    def replay(self) -> None:
        self.attempt.replay_hits += 1

    def finish_attempt(self, cache: SolutionCache, level_key: LevelKey) -> AttemptLearning:
        self.attempt.cache_size = len(cache)
        self.attempt.level_frontier = cache.solved_frontier(level_key)
        self._event("attempt_end", **asdict(self.attempt))
        return self.attempt

    def _event(self, kind: str, **fields) -> None:
        config.ensure_dirs()
        row = {"kind": kind, "attempt": getattr(self, "_n", 0), **fields}
        # Level keys are tuples — JSON needs lists
        if "level_key" in row and isinstance(row["level_key"], tuple):
            row["level_key"] = list(row["level_key"])
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def set_attempt_num(self, n: int) -> None:
        self._n = n


def format_learning_line(learn: AttemptLearning, level_label: str) -> str:
    """One-line learning summary for the attempt footer."""
    net = learn.banks - learn.drops
    net_str = f"+{net}" if net >= 0 else str(net)
    train_str = f", {learn.trains} auto-trained" if learn.trains else ""
    return (f"learning: {learn.banks} banked, {learn.drops} dropped ({net_str} net), "
            f"{learn.learns} learned{train_str}, {learn.replay_hits} replays | "
            f"{level_label} frontier {learn.level_frontier}px | cache {learn.cache_size}")


def print_session_learning(results: list) -> None:
    """Show attempt-to-attempt learning trend (search↓ replay↑ is the compounding signature)."""
    if len(results) < 2:
        return
    print("\n=== Learning momentum (attempt → attempt) ===")
    for i in range(1, len(results)):
        a, b = results[i - 1], results[i]
        ds = b.search_calls - a.search_calls
        dr = b.replay_calls - a.replay_calls
        df = b.level_frontier - a.level_frontier if hasattr(b, "level_frontier") else 0
        db = getattr(b, "banks", 0) - getattr(a, "banks", 0)
        trend = "✅" if dr >= 0 and ds <= 0 else "↗️" if df > 0 or db > 0 else "—"
        print(f"  {a.attempt}→{b.attempt}: search {ds:+d}, replay {dr:+d}, "
              f"frontier {df:+d}px, banked {getattr(b, 'banks', 0)} {trend}")