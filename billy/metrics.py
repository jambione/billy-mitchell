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
