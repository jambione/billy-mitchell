#!/usr/bin/env python3
"""Capture named Zelda savestates for fixtures / hazard-scoped RL (B1).

Named states land in data/zelda/states/<name>.state with a manifest row in
data/zelda/states/manifest.json. Use the stable-retro Integration UI to refine
RAM maps, then capture checkpoints here for replay verification.

Examples:
    # Snapshot current emulator state (after manual play or probe):
    BILLY_HEADLESS=1 .venv/bin/python capture_zelda_state.py snap \\
        --name cave_interior_text --note "link_y~212 text phase"

    # Drive reflex to a milestone, then snap:
    BILLY_HEADLESS=1 .venv/bin/python capture_zelda_state.py play \\
        --plan billy.games.zelda.start_cave:ENTER_PLAN \\
        --name cave_after_enter --max-frames 300

    # List catalog:
    .venv/bin/python capture_zelda_state.py list
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("BILLY_HEADLESS", "1")

STATES_DIR = Path("data/zelda/states")
MANIFEST = STATES_DIR / "manifest.json"


def _load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    return json.loads(MANIFEST.read_text())


def _save_manifest(rows: list[dict]) -> None:
    STATES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n")


def _connect(from_state: str | None):
    from billy.games.zelda.game import ZeldaGame

    game = ZeldaGame()
    session = game.system.connect()
    session.wait_until_live()
    if from_state:
        with open(from_state, "rb") as f:
            session.reset()
            session.env.em.set_state(f.read())
            session._refresh_ram()
        obs = game.observe(session.read_state().frame, session.read_state().ram)
    else:
        session.reset()
        obs = game.boot(session)
    return game, session, obs


def _scene_row(obs) -> dict:
    from billy.games.zelda.path_probe import scene_record
    return scene_record(obs.raw, note="capture")


def cmd_snap(args: argparse.Namespace) -> int:
    _, session, obs = _connect(args.from_state)
    out = STATES_DIR / f"{args.name}.state"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(session.clone_state())
    row = {
        "name": args.name,
        "path": str(out),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "note": args.note,
        "scene": _scene_row(obs),
    }
    rows = [r for r in _load_manifest() if r.get("name") != args.name]
    rows.append(row)
    _save_manifest(rows)
    print(f"[capture] {obs.level_label} link=({obs.raw.link_x},{obs.raw.link_y}) "
          f"sword={obs.raw.sword_level} -> {out}")
    session.close()
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    from billy.abstractions import plan_frames
    from billy.games.zelda.path_probe import load_plan

    game, session, obs = _connect(args.from_state)
    plan = load_plan(args.plan)
    session.send_plan(plan)
    obs = game.observe(session.read_state().frame, session.read_state().ram)
    print(f"[capture] after plan ({plan_frames(plan)}f): {obs.level_label} "
          f"link=({obs.raw.link_x},{obs.raw.link_y}) sword={obs.raw.sword_level}")
    if args.name:
        out = STATES_DIR / f"{args.name}.state"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(session.clone_state())
        row = {
            "name": args.name,
            "path": str(out),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "note": args.note or f"after plan {args.plan}",
            "scene": _scene_row(obs),
        }
        rows = [r for r in _load_manifest() if r.get("name") != args.name]
        rows.append(row)
        _save_manifest(rows)
        print(f"[capture] saved -> {out}")
    session.close()
    return 0


def cmd_drive_east(args: argparse.Namespace) -> int:
    """Run Director with persisted cache, then snap reached east-march milestones."""
    from billy.director import Director
    from billy.games.zelda.fixtures import STATES_DIR
    from billy.games.zelda.game import ZeldaGame
    from billy.games.zelda.walkthrough import SEA_EAST_SCREEN
    from billy.knowledge import KnowledgeBase

    targets = {int(t) for t in args.targets.split(",")} if args.targets else {
        123, 124, SEA_EAST_SCREEN}
    game = ZeldaGame()
    director = Director(game, KnowledgeBase(), use_llm=False)
    best_screen = 119
    director.run_session(max(1, args.attempts))
    obs = director._observe()
    session = director.session
    rows = _load_manifest()
    STATES_DIR.mkdir(parents=True, exist_ok=True)
    screen = obs.raw.map_location
    best_screen = max(best_screen, screen)
    snapped = []
    for snap_screen in sorted(targets & {best_screen, screen}):
        name = f"east_row8_screen_{snap_screen}"
        path = STATES_DIR / f"{name}.state"
        path.write_bytes(session.clone_state())
        rows[:] = [r for r in rows if r.get("name") != name]
        rows.append({
            "name": name,
            "path": str(path),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "note": f"drive-east reached #{snap_screen}",
            "scene": _scene_row(obs),
        })
        snapped.append(snap_screen)
        print(f"[capture] milestone screen #{snap_screen} -> {path}")
    if obs.raw.sword_level >= 1:
        name = "east_row8_sword"
        path = STATES_DIR / f"{name}.state"
        path.write_bytes(session.clone_state())
        rows[:] = [r for r in rows if r.get("name") != name]
        rows.append({
            "name": name,
            "path": str(path),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "note": "post-sword overworld",
            "scene": _scene_row(obs),
        })
    _save_manifest(rows)
    print(f"[capture] done screen=#{screen} snapped={snapped}")
    session.close()
    return 0 if screen in targets or screen >= SEA_EAST_SCREEN else 1


def cmd_list(_args: argparse.Namespace) -> int:
    rows = _load_manifest()
    if not rows:
        print("[capture] no named states — use snap or play --name")
        return 0
    for r in rows:
        sc = r.get("scene", {})
        print(f"  {r['name']:24s} screen=#{sc.get('screen','?')} "
              f"link={sc.get('link','?')} sword={sc.get('sword','?')}  {r.get('note','')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Capture named Zelda savestates.")
    sub = p.add_subparsers(dest="cmd", required=True)

    snap = sub.add_parser("snap", help="Save current emulator state under a name.")
    snap.add_argument("--name", required=True)
    snap.add_argument("--note", default="")
    snap.add_argument("--from-state", default="")

    play = sub.add_parser("play", help="Run a plan then optionally snap.")
    play.add_argument("--plan", required=True)
    play.add_argument("--name", default="")
    play.add_argument("--note", default="")
    play.add_argument("--from-state", default="")
    play.add_argument("--max-frames", type=int, default=8000)

    drive = sub.add_parser("drive-east", help="Play east march and snap milestone states.")
    drive.add_argument("--targets", default="123,124,127",
                       help="comma-separated screen ids to capture")
    drive.add_argument("--attempts", type=int, default=3)

    sub.add_parser("list", help="List named states in the manifest.")

    args = p.parse_args(argv)
    if args.cmd == "snap":
        return cmd_snap(args)
    if args.cmd == "play":
        return cmd_play(args)
    if args.cmd == "drive-east":
        return cmd_drive_east(args)
    if args.cmd == "list":
        return cmd_list(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())