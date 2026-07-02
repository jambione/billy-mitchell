#!/usr/bin/env python3
"""Capture Zelda scene/coords to discover and verify paths Billy should take.

Examples:
    # Log coords while the ROM-verified start-cave macro runs:
    BILLY_HEADLESS=1 .venv/bin/python probe_zelda_path.py record \\
        --plan billy.games.zelda.start_cave:FULL_FROM_APPROACH \\
        --out data/zelda/paths/cave_full.jsonl

    # Log while the live reflex plays (no plan — Billy decides):
    BILLY_HEADLESS=1 .venv/bin/python probe_zelda_path.py record \\
        --reflex --max-frames 3000 --out data/zelda/paths/reflex.jsonl

    # Replay a plan and assert milestones:
    BILLY_HEADLESS=1 .venv/bin/python probe_zelda_path.py replay \\
        --plan billy.games.zelda.start_cave:FULL_FROM_APPROACH \\
        --expect-sword 1 --expect-screen 119

    # Custom script (button holds):
    BILLY_HEADLESS=1 .venv/bin/python probe_zelda_path.py record \\
        --plan "left:35,up:25,left:15,up:15" --out data/zelda/paths/approach.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("BILLY_HEADLESS", "1")


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


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _save_keyframe(session, state_dir: Path, tag: str, frame: int) -> str:
    state_dir.mkdir(parents=True, exist_ok=True)
    safe = tag.replace("/", "_")
    out = state_dir / f"f{frame:06d}_{safe}.state"
    out.write_bytes(session.clone_state())
    return str(out)


def cmd_record(args: argparse.Namespace) -> int:
    from billy.abstractions import plan_frames
    from billy.games.zelda.path_probe import keyframe_reason, load_plan, scene_record
    from billy.games.zelda.reflex import ZeldaReflex

    out = Path(args.out)
    if out.exists() and not args.append:
        out.unlink()
    state_dir = Path(args.state_dir) if args.state_dir else out.with_suffix("")

    game, session, obs = _connect(args.from_state)
    reflex = ZeldaReflex()
    reflex.reset(obs)

    plan = None if args.reflex else load_plan(args.plan)
    plan_iter = iter(plan) if plan else None
    prev_row: dict | None = None
    t = 0
    frames = 0
    max_frames = args.max_frames

    def log(buttons: int = 0, note: str = "") -> dict:
        nonlocal prev_row, t
        scene = obs.raw
        row = scene_record(scene, t=t, buttons=buttons, note=note)
        tag = None
        if args.keyframe_every == "screen":
            tag = keyframe_reason(prev_row, row)
        elif args.keyframe_every == "phase":
            if prev_row is None or row["faq_phase"] != prev_row["faq_phase"] or (
                    row["interior_phase"] != prev_row["interior_phase"]):
                tag = keyframe_reason(prev_row, row) or (
                    f"faq_{row['faq_phase']}" if prev_row else "start")
        elif args.keyframe_every.isdigit():
            tag = keyframe_reason(prev_row, row, every_n=int(args.keyframe_every))
        if tag:
            row["keyframe"] = True
            row["state_path"] = _save_keyframe(session, state_dir, tag, scene.frame)
        _append_jsonl(out, row)
        if args.verbose or row.get("keyframe"):
            print(f"[record] t={t:5d} screen=#{row['screen']} link={row['link']} "
                  f"phase={row['faq_phase']}/{row['interior_phase']} sword={row['sword']} "
                  f"note={note or '-'}")
        prev_row = row
        t += 1
        return row

    log(note="boot" if not args.from_state else f"restore:{args.from_state}")

    while frames < max_frames and not obs.dead:
        if args.reflex:
            decision = reflex.step(obs)
            steps = list(decision.plan)
            note = decision.note
        else:
            assert plan_iter is not None
            try:
                step = next(plan_iter)
            except StopIteration:
                break
            steps = [step]
            note = f"plan:{c_names(step.buttons)}:{step.frames}"

        if not steps:
            if args.reflex and note:
                log(note=note)
            break

        for step in steps:
            if frames >= max_frames or obs.dead:
                break
            session.send_plan([step])
            frames += step.frames
            obs = game.observe(session.read_state().frame, session.read_state().ram)
            if args.sample_every <= 1 or frames % args.sample_every == 0:
                log(buttons=step.buttons, note=note)

    log(note="end")
    print(f"[record] {t} rows -> {out} ({frames} frames, screen=#{obs.raw.map_location}, "
          f"sword={obs.raw.sword_level})")
    session.close()
    return 0


def c_names(mask: int) -> str:
    from billy.systems.nes import controller as c
    names = c.names_from_mask(mask)
    return "+".join(names) if names else "idle"


def cmd_replay(args: argparse.Namespace) -> int:
    from billy.abstractions import plan_frames
    from billy.games.zelda.path_probe import check_milestones, load_plan, scene_record

    plan = load_plan(args.plan)
    game, session, obs = _connect(args.from_state)

    print(f"[replay] {len(plan)} steps, {plan_frames(plan)} frames "
          f"from {obs.level_label} progress={obs.progress}")

    if args.log:
        out = Path(args.log)
        if out.exists() and not args.append:
            out.unlink()

    frames = 0
    for i, step in enumerate(plan):
        session.send_plan([step])
        frames += step.frames
        obs = game.observe(session.read_state().frame, session.read_state().ram)
        if args.verbose and i % max(1, args.sample_every) == 0:
            s = obs.raw
            print(f"  step {i:3d} {c_names(step.buttons)}:{step.frames}f "
                  f"screen=#{s.map_location} link=({s.link_x},{s.link_y}) sword={s.sword_level}")
        if args.log and frames % args.sample_every == 0:
            row = scene_record(obs.raw, t=i, buttons=step.buttons, note=f"step:{i}")
            _append_jsonl(Path(args.log), row)
        if obs.dead:
            print(f"[replay] died at step {i} ({obs.level_label})")
            session.close()
            return 1

    fails = check_milestones(
        obs,
        expect_screen=args.expect_screen,
        expect_sword=args.expect_sword,
        expect_progress=args.expect_progress,
    )
    scene = obs.raw
    print(f"[replay] done — screen=#{scene.map_location} link=({scene.link_x},{scene.link_y}) "
          f"sword={scene.sword_level} progress={obs.progress} dead={obs.dead}")
    if fails:
        for msg in fails:
            print(f"[replay] FAIL: {msg}")
        session.close()
        return 1
    print("[replay] OK — all milestones met")
    session.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Probe Zelda paths: record coords or replay plans.")
    sub = p.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="Log scene/coords while a plan or reflex runs.")
    rec.add_argument("--out", default="data/zelda/paths/record.jsonl")
    rec.add_argument("--state-dir", default="", help="savestates on keyframes (default: <out> stem)")
    rec.add_argument("--from-state", default="", help="restore this savestate before recording")
    rec.add_argument("--plan", default="",
                     help="module:ATTR, .json plan, or script (RIGHT:120,...) — omit with --reflex")
    rec.add_argument("--reflex", action="store_true", help="drive with ZeldaReflex instead of a plan")
    rec.add_argument("--max-frames", type=int, default=8000)
    rec.add_argument("--sample-every", type=int, default=1,
                     help="log every N emulator frames (default 1)")
    rec.add_argument("--keyframe-every", default="screen",
                     help="save state on: screen | phase | N (frame ticks)")
    rec.add_argument("--append", action="store_true", help="append to existing jsonl")
    rec.add_argument("-v", "--verbose", action="store_true")

    rep = sub.add_parser("replay", help="Run a plan and check milestones.")
    rep.add_argument("--plan", required=True, help="module:ATTR, .json, or script")
    rep.add_argument("--from-state", default="")
    rep.add_argument("--expect-screen", type=int, default=None, help="min map_location at end")
    rep.add_argument("--expect-sword", type=int, default=None, help="min sword_level at end")
    rep.add_argument("--expect-progress", type=int, default=None, help="min objective_score at end")
    rep.add_argument("--log", default="", help="optional jsonl path while replaying")
    rep.add_argument("--sample-every", type=int, default=8)
    rep.add_argument("--append", action="store_true")
    rep.add_argument("-v", "--verbose", action="store_true")

    args = p.parse_args(argv)
    if args.cmd == "record":
        if not args.reflex and not args.plan:
            print("record requires --plan or --reflex", file=sys.stderr)
            return 2
        return cmd_record(args)
    if args.cmd == "replay":
        return cmd_replay(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())