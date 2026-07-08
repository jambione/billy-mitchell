"""Auto-Stuck Trainer — closed-loop self-improvement when Billy hits the same wall repeatedly.

When deaths cluster at a hazard bucket without frontier advance, this module:
  1. Auto-captures approach savestates from live play (phase-accurate for timing hazards)
  2. Runs extended offline search from those + known bootstrap states
  3. Banks only verified survivors (lift_cacheable / pit_cacheable gates)
  4. Optionally quick-trains a section sub-policy when search finds partial progress

This closes the loop from "stuck for 2 days" → "deaths trigger training → banks → compounds"
without manual train_section.py / savestate capture scripts.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from . import config
from .abstractions import Plan
from .knowledge.cache import SolutionCache

RemedyKind = Literal["frame_search", "pit_search", "section_train"]

def _stuck_file() -> Path:
    """Live path — always follows config.DATA_DIR (tests monkeypatch that)."""
    return config.DATA_DIR / "stuck.json"


def _auto_state_dir() -> Path:
    return config.DATA_DIR / "rl" / "states" / "auto"


def _demo_requests_file() -> Path:
    return config.DATA_DIR / "demo_requests.jsonl"


# Module-level names kept for importers; prefer the helpers above when DATA_DIR may change.
_STUCK_FILE = config.DATA_DIR / "stuck.json"
_AUTO_STATE_DIR = config.DATA_DIR / "rl" / "states" / "auto"
_DEMO_REQUESTS_FILE = config.DATA_DIR / "demo_requests.jsonl"


@dataclass(frozen=True)
class StuckRemedy:
    """Game-defined recipe for breaking a recurring hazard stall."""
    kind: RemedyKind
    level_label: str
    death_x: int
    goal_x: int
    savestate_paths: tuple[str, ...] = ()
    bank_x_lo: int = 0          # only bank if start x is in [lo, hi]
    bank_x_hi: int = 9999
    section_out: str = ""       # for section_train
    section_timesteps: int = 100_000


@dataclass
class StuckRecord:
    level_label: str
    death_bucket: int
    game: str = ""              # cli_name scope — smb vs smb_lost no longer collide on "1-1"
    deaths: int = 0
    last_death_x: int = 0
    frontier_at_first: int = 0
    remediated: bool = False
    captured_states: list[str] = field(default_factory=list)


class StuckTracker:
    """Cross-attempt death clustering — persists so progress compounds session-to-session."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else _stuck_file()
        self.records: dict[tuple[str, str, int], StuckRecord] = {}
        self._load()

    @staticmethod
    def death_bucket(death_x: int) -> int:
        return death_x // config.CACHE_BUCKET_PX

    def _key(self, game: str, level_label: str, death_x: int) -> tuple[str, str, int]:
        return game, level_label, self.death_bucket(death_x)

    def note_death(self, game: str, level_label: str, death_x: int, frontier: int) -> StuckRecord:
        key = self._key(game, level_label, death_x)
        rec = self.records.get(key)
        if rec is None:
            rec = StuckRecord(level_label=level_label, death_bucket=key[2],
                              game=game, frontier_at_first=frontier)
            self.records[key] = rec
        if rec.remediated:
            # Fresh failure after a teach / auto-remedy — start a new death streak.
            rec.remediated = False
            rec.deaths = 1
            rec.frontier_at_first = frontier
        else:
            rec.deaths += 1
        rec.last_death_x = death_x
        if frontier > rec.frontier_at_first + config.CACHE_BUCKET_PX * 2:
            rec.deaths = 0
            rec.remediated = False
        self._save()
        return rec

    def note_capture(self, game: str, level_label: str, death_x: int, state_path: str) -> None:
        key = self._key(game, level_label, death_x)
        rec = self.records.get(key)
        if rec is None:
            return
        if state_path not in rec.captured_states:
            rec.captured_states.append(state_path)
            self._save()

    def stuck_at(self, game: str, level_label: str, death_x: int, *,
                 threshold: int | None = None) -> StuckRecord | None:
        key = self._key(game, level_label, death_x)
        rec = self.records.get(key)
        if rec is None:
            return None
        cap = threshold if threshold is not None else config.STUCK_DEATH_THRESHOLD
        if rec.deaths >= cap and not rec.remediated:
            return rec
        return None

    def mark_remediated(self, game: str, level_label: str, death_x: int) -> None:
        key = self._key(game, level_label, death_x)
        rec = self.records.get(key)
        if rec:
            rec.remediated = True
            rec.deaths = 0
            self._save()

    def mark_level_remediated(self, game: str, level_label: str) -> int:
        """Mark every death-bucket on this level as taught/fixed (Remix success).

        Zeros death counts so discovery does not re-queue from stale history. New deaths after
        this call clear `remediated` via note_death and can re-open the wall. Returns how many
        records were updated. Also backfills empty `game` on legacy rows that match the label."""
        n = 0
        for rec in self.records.values():
            if rec.level_label != level_label:
                continue
            if rec.game and rec.game != game:
                continue
            rec.remediated = True
            rec.deaths = 0
            if not rec.game:
                rec.game = game
            n += 1
        if n:
            self._save()
        return n

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            rows = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for row in rows:
            game = row.get("game", "")
            key = (game, row["level_label"], row["death_bucket"])
            self.records[key] = StuckRecord(**{k: row[k] for k in StuckRecord.__dataclass_fields__
                                               if k in row})

    def _save(self) -> None:
        config.ensure_dirs()
        rows = []
        for rec in self.records.values():
            rows.append({
                "game": rec.game,
                "level_label": rec.level_label,
                "death_bucket": rec.death_bucket,
                "deaths": rec.deaths,
                "last_death_x": rec.last_death_x,
                "frontier_at_first": rec.frontier_at_first,
                "remediated": rec.remediated,
                "captured_states": rec.captured_states,
            })
        self.path.write_text(json.dumps(rows, indent=0))


def write_trail_snapshot(snap: bytes, out_path: str, *, level_label: str,
                         approach_x: int) -> bool:
    """Persist an on-ground trail snapshot — no re-validation (vx may be >2 mid-run)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(snap)
    print(f"[stuck] captured approach {level_label} x={approach_x} -> {out_path}")
    return True


def capture_approach_snapshot(session, observe, snap: bytes, out_path: str,
                              *, level_label: str, x_lo: int, x_hi: int) -> bool:
    """Save a clone snapshot if it's on-ground in the approach band (manual capture path)."""
    from .games.smb.capture_util import capture_ready, settle_mario, near_pit

    session.restore(snap)
    obs = observe()
    if obs.level_label != level_label:
        return False
    if not (x_lo <= obs.progress <= x_hi and getattr(obs.raw, "on_ground", True)):
        return False
    if not near_pit(obs):
        ok, _ = settle_mario(session, observe, allow_left=False)
        obs = observe()
        if not ok:
            return False
    if not capture_ready(obs, x_min=x_lo, x_max=x_hi, level_label=level_label):
        return False
    return write_trail_snapshot(session.clone_state(), out_path,
                                level_label=level_label, approach_x=obs.progress)


def collect_auto_states(level_label: str, record: StuckRecord) -> list[str]:
    """All auto-captured + recorded savestates for a stuck hazard."""
    paths: list[str] = []
    seen: set[str] = set()
    for p in record.captured_states:
        if os.path.isfile(p) and p not in seen:
            paths.append(p)
            seen.add(p)
    label = level_label.replace("-", "_")
    if _AUTO_STATE_DIR.is_dir():
        for p in sorted(_AUTO_STATE_DIR.glob(f"{label}_d*.state")):
            s = str(p)
            if s not in seen:
                paths.append(s)
                seen.add(s)
    return paths


def _search_from_state(session, observe, state_path: str, remedy: StuckRemedy,
                       hooks, min_gain: int = 8) -> tuple[Plan | None, int, int, tuple]:
    """Run remedy-specific offline search from one savestate. Returns (plan, start_x, reach, lk)."""
    with open(state_path, "rb") as f:
        state_bytes = f.read()
    session.reset()
    session.env.em.set_state(state_bytes)
    session._refresh_ram()
    obs = observe()
    if obs.level_label != remedy.level_label:
        return None, 0, 0, obs.level_key
    start_x = obs.progress
    lk = obs.level_key

    if remedy.kind == "frame_search":
        os.environ.setdefault("BILLY_LIFT_FRAME_SEARCH", "1")
        os.environ["BILLY_LIFT_SEARCH_DEPTH"] = os.environ.get(
            "BILLY_STUCK_SEARCH_DEPTH", "8")
        os.environ["BILLY_LIFT_SEARCH_BEAM"] = os.environ.get(
            "BILLY_STUCK_SEARCH_BEAM", "48")
        os.environ["BILLY_LIFT_IDLE_MAX"] = os.environ.get(
            "BILLY_STUCK_IDLE_MAX", "240")
        plan, reach, crossed = hooks.try_frame_search(
            session, observe, obs, deep=True, min_gain=min_gain)
        if plan and hooks.learn_cacheable(remedy.level_label, remedy.death_x, reach):
            return plan, start_x, reach, lk
        return None, start_x, reach, lk

    if remedy.kind == "pit_search":
        plan, reach = hooks.try_pit_search(
            session, observe, obs, death_x=remedy.death_x, min_gain=min_gain)
        if plan and hooks.learn_cacheable(remedy.level_label, remedy.death_x, reach):
            return plan, start_x, reach, lk
        return None, start_x, reach, lk

    return None, start_x, obs.progress, lk


def _quick_section_train(remedy: StuckRemedy, state_paths: list[str]) -> bool:
    """Fast section PPO train when offline search only gets partial progress."""
    if not remedy.section_out or not state_paths:
        return False
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
        from .rl.section_env import SectionEnv
    except ImportError:
        print("[stuck] section train skipped (stable-baselines3 not installed)")
        return False

    milestones = ((760, 25.0), (800, 45.0), (850, 70.0), (900, 100.0))
    n_envs = min(4, len(state_paths))

    def _make_env(state: str, seed: int):
        def _init():
            env = SectionEnv(
                state, goal_x=remedy.goal_x, landing_waits=2, randomize_frames=24,
                max_steps=220, start_x=remedy.bank_x_lo,
                back_x=max(0, remedy.bank_x_lo - 80), milestones=milestones)
            env.reset(seed=seed)
            return env
        return _init

    fns = [_make_env(state_paths[i % len(state_paths)], i) for i in range(n_envs)]
    vec = DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns, start_method="spawn")
    vec = VecMonitor(vec)
    resume = remedy.section_out
    if os.path.isfile(f"{resume}.zip"):
        model = PPO.load(resume, env=vec, device="cpu")
        print(f"[stuck] resuming section train from {resume}.zip")
    else:
        model = PPO("MlpPolicy", vec, device="cpu", verbose=0, n_steps=512,
                    batch_size=256, gae_lambda=0.95, gamma=0.99, ent_coef=0.03)
    model.learn(total_timesteps=remedy.section_timesteps, progress_bar=False)
    os.makedirs(os.path.dirname(remedy.section_out) or ".", exist_ok=True)
    model.save(remedy.section_out)
    vec.close()
    print(f"[stuck] section train saved -> {remedy.section_out}.zip")
    return True


@dataclass
class RemediationResult:
    success: bool
    plan: Plan | None = None
    bank_x: int = 0
    reach: int = 0
    source: str = ""
    trained: bool = False


def remediate(session, observe, cache: SolutionCache, hooks,
              remedy: StuckRemedy, record: StuckRecord,
              *, bank_fn: Callable | None = None) -> RemediationResult:
    """Extended offline search (+ optional section train) for a stuck hazard."""
    if not config.AUTO_TRAIN:
        return RemediationResult(success=False)

    state_paths: list[str] = []
    seen: set[str] = set()
    for p in remedy.savestate_paths:
        if os.path.isfile(p) and p not in seen:
            state_paths.append(p)
            seen.add(p)
    for p in collect_auto_states(remedy.level_label, record):
        if p not in seen:
            state_paths.append(p)
            seen.add(p)

    if not state_paths:
        print(f"[stuck] no savestates for {remedy.level_label} — capture on next death")
        return RemediationResult(success=False)

    print(f"[stuck] remediating {remedy.level_label} death≈{remedy.death_x} "
          f"from {len(state_paths)} state(s)...")

    best: tuple[Plan, int, int, tuple] | None = None
    best_reach = 0
    for path in state_paths:
        plan, start_x, reach, lk = _search_from_state(
            session, observe, path, remedy, hooks)
        if plan and reach > best_reach:
            if remedy.bank_x_lo <= start_x <= remedy.bank_x_hi:
                best = (plan, start_x, reach, lk)
                best_reach = reach

    if best is not None:
        plan, bank_x, reach, lk = best
        y = 0
        if bank_fn is not None:
            bank_fn(lk, bank_x, plan, reach, y=y, source="auto_stuck")
        else:
            cache.put(lk, bank_x, plan, reach, y=y, force=True)
        try:
            from .games.smb.lift_search import persist_lift_plan
            if remedy.kind == "frame_search":
                persist_lift_plan(plan, reach)
        except ImportError:
            pass
        print(f"[stuck] ✅ banked crossing @{remedy.level_label} x={bank_x} reach={reach}")
        return RemediationResult(success=True, plan=plan, bank_x=bank_x,
                               reach=reach, source="auto_stuck_search")

    if (config.STUCK_SECTION_TRAIN and remedy.kind == "frame_search"
            and remedy.section_out and best_reach >= remedy.death_x - 120):
        trained = _quick_section_train(remedy, state_paths)
        if trained:
            for path in state_paths:
                plan, start_x, reach, lk = _search_from_state(
                    session, observe, path, remedy, hooks)
                if plan and reach > best_reach and remedy.bank_x_lo <= start_x <= remedy.bank_x_hi:
                    if bank_fn is not None:
                        bank_fn(lk, start_x, plan, reach, y=0, source="auto_stuck_train")
                    else:
                        cache.put(lk, start_x, plan, reach, force=True)
                    print(f"[stuck] ✅ post-train bank @{remedy.level_label} x={start_x} "
                          f"reach={reach}")
                    return RemediationResult(success=True, plan=plan, bank_x=start_x,
                                           reach=reach, source="auto_stuck_train",
                                           trained=True)

    print(f"[stuck] remediation miss (best_reach={best_reach}, goal={remedy.goal_x})")
    return RemediationResult(success=False)


def auto_state_path(level_label: str, death_x: int, approach_x: int) -> str:
    """Deterministic path for an auto-captured approach savestate."""
    label = level_label.replace("-", "_")
    bucket = death_x // config.CACHE_BUCKET_PX
    return str(_AUTO_STATE_DIR / f"{label}_d{bucket}_x{approach_x}.state")


def notify_remix_wall(game_name: str, remedy: StuckRemedy, record: StuckRecord) -> None:
    """Surface a filed wall for the human: one-line inbox + macOS notification."""
    config.ensure_dirs()
    inbox = config.DATA_DIR / "remix_inbox.txt"
    line = (f"Billy needs you at {game_name} {remedy.level_label} "
            f"(x≈{remedy.death_x}, {record.deaths} deaths)")
    inbox.write_text(line + "\n")
    if platform.system() != "Darwin":
        return
    msg = f"Billy needs you at {game_name} {remedy.level_label}"
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{msg}" with title "Billy Mitchell"'],
            check=False, timeout=5, capture_output=True)
    except (OSError, subprocess.TimeoutExpired):
        pass


def request_demo(game_name: str, remedy: StuckRemedy, record: StuckRecord,
                 state_paths: list[str], *, requests_file: Path | None = None) -> str | None:
    """The final remedy: every autonomous tier failed here — ask the human for ONE demo.

    Pull-based teaching: Billy only requests a demo where search, pit/frame search, and section
    training have ALL missed, so the human's time is spent exactly where it multiplies. Appends a
    request to data/demo_requests.jsonl (deduped per hazard) and prints the ready-to-run teleop
    command. A banked demo flows through the same verify gate as any search survivor.
    """
    if not state_paths:
        return None
    # The furthest approach state gives the human the shortest runway to play.
    def _approach_x(p: str) -> int:
        stem = Path(p).stem
        try:
            return int(stem.rsplit("_x", 1)[1])
        except (IndexError, ValueError):
            return 0
    state = max(state_paths, key=_approach_x)

    requests_file = requests_file or _DEMO_REQUESTS_FILE
    key = {"game": game_name, "level_label": remedy.level_label,
           "death_bucket": StuckTracker.death_bucket(remedy.death_x)}
    existing = []
    if requests_file.is_file():
        for line in requests_file.read_text().splitlines():
            if line.strip():
                existing.append(json.loads(line))
    if any(all(e.get(k) == v for k, v in key.items()) for e in existing):
        return state   # already requested — don't spam
    config.ensure_dirs()
    with requests_file.open("a") as f:
        f.write(json.dumps({**key, "death_x": remedy.death_x, "deaths": record.deaths,
                            "state": state}) + "\n")
    notify_remix_wall(game_name, remedy, record)
    cmd = (f".venv/bin/python teleop.py play --game {game_name} "
           f"--from-state {state} --bank")
    print(f"\n  [stuck] 🙋 BILLY REQUESTS A DEMO — {remedy.level_label} x≈{remedy.death_x}: "
          f"{record.deaths} deaths and self-training hasn't cracked it.")
    print(f"  [stuck]    Teach him once (a window opens; play past the hazard, it auto-banks):")
    print(f"  [stuck]    {cmd}\n")
    return state