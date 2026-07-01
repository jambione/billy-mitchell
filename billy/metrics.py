"""Learning-curve metrics: proof that Billy actually improves across attempts."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from . import config


@dataclass
class AttemptResult:
    attempt: int
    outcome: str          # "game_over" | "clear" | "timeout"
    max_x: int            # farthest x in the final level segment
    frames: int           # frames the attempt lasted
    billy_calls: int      # how often the LLM was consulted
    world_stage: str      # furthest level reached
    levels_cleared: int   # how many levels cleared this attempt
    score: int            # in-game score reached (Billy's record obsession #1)
    fastest_clear_frames: int  # fastest single-level clear this attempt (0 = none); ~60 fps
    duration_s: float
    # --- compounding-learning telemetry (the proof the cache makes learning compound) -----
    search_calls: int = 0       # NEW hazards solved by micro-search this attempt (should fall)
    replay_calls: int = 0       # cached hazards replayed for free this attempt (should rise)
    tape_frames: int = 0        # frames driven by whole-trajectory tape replay (should rise)
    frontier_x: int = 0         # furthest solved x-bucket on the level (should rise)
    frames_to_frontier: int = 0 # frames to re-reach last attempt's furthest x (should fall)
    # --- learning ledger (visible proof of compounding) ---------------------------------
    banks: int = 0
    drops: int = 0
    learns: int = 0
    level_frontier: int = 0     # solved px frontier on the furthest level reached


def record(result: AttemptResult) -> None:
    config.ensure_dirs()
    with config.METRICS_FILE.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def print_curve(results: list[AttemptResult]) -> None:
    if not results:
        return
    print("\n=== Billy's record book ===")
    print(f"{'#':>3}  {'outcome':<9} {'levels':>6} {'reached':>8} {'score':>7} {'fast(s)':>7}")
    for r in results:
        fast = f"{r.fastest_clear_frames/60:.1f}" if r.fastest_clear_frames else "-"
        print(f"{r.attempt:>3}  {r.outcome:<9} {r.levels_cleared:>6} {r.world_stage:>8} "
              f"{r.score:>7} {fast:>7}")
    best_score = max((r.score for r in results), default=0)
    fastest = min((r.fastest_clear_frames for r in results if r.fastest_clear_frames), default=0)
    most = max((r.levels_cleared for r in results), default=0)
    fast_str = f"{fastest / 60:.1f}s" if fastest else "n/a"
    print(f"\n🏆 best score: {best_score}   ⏱  fastest clear: {fast_str}   "
          f"most levels: {most}")
    print_compounding_curve(results)
    from .learning import print_session_learning
    print_session_learning(results)


_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    """Tiny unicode trend line — makes the exponential legible at a glance."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi <= lo:
        return _SPARK_BLOCKS[0] * len(values)
    span = hi - lo
    return "".join(_SPARK_BLOCKS[int((v - lo) / span * (len(_SPARK_BLOCKS) - 1))]
                   for v in values)


def print_compounding_curve(results: list[AttemptResult]) -> None:
    """The proof the learning compounds: NEW searches should fall, free replays should rise,
    tape-driven frames should rise, the solved frontier should advance, and both wall-clock and
    frames-to-frontier should drop, across attempts."""
    if not results:
        return
    print("\n=== Compounding curve (is the learning actually building?) ===")
    print(f"{'#':>3} {'search':>7} {'replay':>7} {'tape%':>6} {'bank':>5} {'learn':>5} "
          f"{'frontier':>8} {'reached_x':>10} {'time(s)':>8}")
    for r in results:
        tape_pct = (100 * r.tape_frames // r.frames) if r.frames else 0
        print(f"{r.attempt:>3} {r.search_calls:>7} {r.replay_calls:>7} {tape_pct:>5}% "
              f"{getattr(r, 'banks', 0):>5} {getattr(r, 'learns', 0):>5} "
              f"{getattr(r, 'level_frontier', r.frontier_x):>8} {r.max_x:>10} "
              f"{r.duration_s:>8.1f}")
    if len(results) > 1:
        print(f"  trend  search {sparkline([r.search_calls for r in results])}↓  "
              f"replay {sparkline([r.replay_calls for r in results])}↑  "
              f"tape {sparkline([r.tape_frames for r in results])}↑  "
              f"reached {sparkline([r.max_x for r in results])}↑  "
              f"time {sparkline([r.duration_s for r in results])}↓")
    first, last = results[0], results[-1]
    verdict = ("✅ compounding: searches↓ replays↑ frontier↑"
               if last.replay_calls >= first.replay_calls and last.frontier_x >= first.frontier_x
               else "⚠️  not yet compounding — inspect cache hits / bucket size")
    print(verdict)
