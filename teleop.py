#!/usr/bin/env python3
"""Human-in-the-loop teleop for ANY Billy game: teach Billy past a wall, once.

A captured demo is an exact button sequence — Billy's native currency. Play through a spot Billy
is stuck on; it verifies your run survives AND advances progress on a clone, then banks it to the
SolutionCache. Every later autonomous run replays it for free. Game-agnostic (the core lives in
billy/teleop.py); this CLI just selects the game and drives to the start spot.

Two steps (separate processes — one emulator per process):

    # 1. Capture the start state (headless; drives Billy there with the learned cache):
    BILLY_HEADLESS=1 .venv/bin/python teleop.py capture --game smb --until-level 1-2 \\
        --out data/states/smb_1_2.state

    # 2. Play through it in a window, then bank your demo (NOT headless — needs the window):
    .venv/bin/python teleop.py play --game smb --from-state data/states/smb_1_2.state --bank

Controls (window must have focus):  arrows = move · Z = A (jump/sword) · X = B (run/attack)
    Tab = Start · RShift = Select · ENTER = finish & verify · ESC = abort
Auto-finish: the moment you reach a new level/screen alive (level_key[:2] changes), the demo is
captured automatically — no need to press ENTER in time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _game(name: str):
    from run import GAMES
    if name not in GAMES:
        raise SystemExit(f"unknown game '{name}' — choose from {sorted(GAMES)}")
    return GAMES[name]()


def cmd_capture(args: argparse.Namespace) -> int:
    """Drive Billy (Director + persisted cache) to a level/screen, save the first state there."""
    os.environ.setdefault("BILLY_HEADLESS", "1")
    os.environ.setdefault("BILLY_TURBO", "1")
    os.environ.setdefault("BILLY_MAX_FRAMES", str(args.max_frames))

    from billy.director import Director
    from billy.knowledge import KnowledgeBase

    game = _game(args.game)
    director = Director(game, KnowledgeBase(), use_llm=False)
    grabbed = {}
    orig = director._observe
    out = Path(args.out)

    def reached(obs) -> bool:
        if not obs.raw or not getattr(obs.raw, "in_play", True):
            return False
        if args.until_level is not None and obs.level_label != args.until_level:
            return False
        if args.until_progress is not None and obs.progress < args.until_progress:
            return False
        # Hand the human a stable, grounded stance — never a mid-jump/airborne frame (which
        # spawns Mario already falling into a pit). Games without the concept default to True.
        if not getattr(obs.raw, "on_ground", True):
            return False
        return True

    class _Grabbed(Exception):
        pass

    def hooked():
        obs = orig()
        if not grabbed and reached(obs):
            grabbed["state"] = director.session.clone_state()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(grabbed["state"])
            print(f"[capture] {obs.level_label} progress={obs.progress} -> saved {out}")
            raise _Grabbed   # stop the run immediately; no need to play out the whole session
        return obs

    director._observe = hooked
    try:
        director.run_session(max(1, args.attempts))
    except _Grabbed:
        pass
    if not grabbed:
        tgt = args.until_level or f"progress>={args.until_progress}"
        print(f"[capture] never reached {tgt} — raise --attempts/--max-frames")
        return 1
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    """Windowed teleop: restore a state, let the human play, verify the demo, optionally bank."""
    os.environ["BILLY_HEADLESS"] = "0"   # we need the watch window for keyboard focus
    os.environ.setdefault("BILLY_TURBO", "1")

    from billy.config import SOLUTIONS_FILE
    from billy.knowledge.cache import SolutionCache
    from billy.teleop import TeleopRecorder, bank_demo, verify_demo

    game = _game(args.game)
    session = game.system.connect()
    session.wait_until_live()

    start_state = Path(args.from_state).read_bytes()
    session.restore(start_state)

    def observe():
        st = session.read_state()
        return game.observe(st.frame, st.ram, getattr(st, "rgb", None))

    start_obs = observe()
    if not session.ensure_viewer():
        print("[teleop] no window available (display attached? BILLY_HEADLESS=1?)")
        return 2
    session.teleop_reset()

    start_key2 = tuple(start_obs.level_key[:2])
    print(f"[teleop] take control at {start_obs.level_label} progress={start_obs.progress}")
    print("[teleop] arrows=move  Z=A  X=B  Tab=Start  RShift=Select  ENTER=finish  ESC=abort")

    rec = TeleopRecorder()
    obs = start_obs
    frames = 0
    aborted = False
    while frames < args.max_frames:
        mask, finish, abort = session.teleop_poll()
        if finish:
            break
        if abort:
            aborted = True
            break
        session.teleop_step(mask)
        rec.record(mask, 1)
        frames += 1
        obs = observe()
        if obs.dead:
            print(f"[teleop] died at progress={obs.progress} after {frames}f")
            break
        # Auto-finish the instant you reach a new level/screen alive (level_key[:2] changes):
        # that's the success condition, so capture it without an ENTER-timing race.
        if (args.auto_finish and getattr(obs.raw, "in_play", True)
                and tuple(obs.level_key[:2]) != start_key2):
            print(f"[teleop] auto-finish: reached {obs.level_label} alive at {frames}f")
            break

    if aborted:
        print("[teleop] aborted — nothing banked")
        session.close()
        return 1

    plan = rec.plan()
    print(f"[teleop] captured {len(plan)} steps / {rec.frame_count()} frames")
    demo_path = Path(args.from_state).with_suffix(".demo.json")
    demo_path.write_text(json.dumps({"steps": [[s.frames, s.buttons] for s in plan]}))
    print(f"[teleop] demo saved -> {demo_path}")

    # Verify with a FRESH observer: the live game carries a monotonic progress high-water mark
    # that would mask the demo's gain after rewinding to the start state.
    result = verify_demo(session, _game(args.game), start_state, plan, min_progress=args.min_progress)
    print(f"[teleop] verify: {result.summary()}")

    if not result.bankable:
        print("[teleop] not bankable (must survive AND advance) — nothing banked")
        session.close()
        return 1

    if args.bank:
        cache = SolutionCache(path=SOLUTIONS_FILE)
        key = bank_demo(cache, start_obs, plan, result.end_progress)
        print(f"[teleop] BANKED to {SOLUTIONS_FILE} at key {key} (reach {result.end_progress}). "
              f"Next autonomous run will replay it.")
    else:
        print("[teleop] dry run (no --bank) — verified bankable but not stored")
    session.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Human-in-the-loop teleop demo capture (any game).")
    sub = p.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="Drive Billy to a level/progress and save the state.")
    cap.add_argument("--game", default="smb")
    cap.add_argument("--until-level", default=None, help="stop when level_label matches (e.g. 1-2)")
    cap.add_argument("--until-progress", type=int, default=None, help="stop at this progress")
    cap.add_argument("--out", required=True, help="path to write the .state file")
    cap.add_argument("--attempts", type=int, default=1)
    cap.add_argument("--max-frames", type=int, default=40000)

    pl = sub.add_parser("play", help="Windowed teleop from a state; verify and optionally bank.")
    pl.add_argument("--game", default="smb")
    pl.add_argument("--from-state", required=True)
    pl.add_argument("--bank", action="store_true", help="bank the demo if it verifies")
    pl.add_argument("--max-frames", type=int, default=9000, help="max teleop frames (~150s at 60fps)")
    pl.add_argument("--min-progress", type=int, default=8)
    pl.add_argument("--no-auto-finish", dest="auto_finish", action="store_false",
                    help="don't auto-stop on level/screen change (press ENTER yourself)")

    args = p.parse_args(argv)
    if args.cmd == "capture":
        if args.until_level is None and args.until_progress is None:
            print("capture needs --until-level or --until-progress", file=sys.stderr)
            return 2
        return cmd_capture(args)
    if args.cmd == "play":
        return cmd_play(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
