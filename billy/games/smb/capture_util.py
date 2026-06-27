"""Helpers for hazard savestate capture (settle velocity before snapshot)."""
from __future__ import annotations

from ...abstractions import Step, Session
from ...systems.nes import controller as C


def settle_mario(session: Session, observe, *, max_frames: int = 120,
                 allow_left: bool = True) -> tuple[bool, int]:
    """Bleed horizontal momentum until Mario is on-ground with ~zero x-speed."""
    obs = observe()
    gap = obs.raw.gap_info() if hasattr(obs.raw, "gap_info") else None
    near_pit = gap is not None and gap[0] <= 32
    for _ in range(max_frames // 4):
        if obs.dead:
            return False, obs.progress
        if obs.raw.on_ground and abs(obs.raw.x_speed) <= 1:
            return True, obs.progress
        if allow_left and not near_pit and obs.raw.x_speed > 2:
            session.send_plan([Step(4, C.LEFT)])
        else:
            session.send_plan([Step(4, C.NEUTRAL)])
        obs = observe()
    return obs.raw.on_ground and not obs.dead, obs.progress


def near_pit(obs) -> bool:
    gap = obs.raw.gap_info() if hasattr(obs.raw, "gap_info") else None
    return gap is not None and gap[0] <= 32


def capture_ready(obs, *, x_min: int, x_max: int, level_label: str) -> bool:
    """On-ground in range with low x-speed — safe to snapshot (settle optional)."""
    return (obs.level_label == level_label and obs.raw.on_ground
            and x_min <= obs.progress <= x_max and abs(obs.raw.x_speed) <= 2)


def save_snapshot(session: Session, observe, out_path: str, *,
                  x_min: int, x_max: int, level_label: str,
                  max_vx: int = 2) -> bool:
    """Write a pit-edge savestate; skip settle when near the lip (settle slides off)."""
    from pathlib import Path

    obs = observe()
    vx_cap = 40 if near_pit(obs) else max_vx
    if not (obs.level_label == level_label and obs.raw.on_ground
            and x_min <= obs.progress <= x_max and abs(obs.raw.x_speed) <= vx_cap):
        return False
    if not near_pit(obs):
        ok, _ = settle_mario(session, observe, allow_left=False)
        obs = observe()
        if not ok or not capture_ready(obs, x_min=x_min, x_max=x_max,
                                       level_label=level_label):
            return False
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(session.clone_state())
    print(f"[capture] saved {obs.level_label} x={obs.progress} y={obs.elevation} "
          f"vx={obs.raw.x_speed} gap={obs.raw.gap_info() if hasattr(obs.raw, 'gap_info') else None} "
          f"-> {out_path}")
    return True


def zero_x_speed(session: Session) -> None:
    """Best-effort: hold neutral briefly so x-speed bleeds off before a snapshot."""
    for _ in range(6):
        session.send_plan([Step(4, C.NEUTRAL)])