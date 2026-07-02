#!/usr/bin/env python3
"""Human-in-the-loop teleop for Zelda: teach Billy past a wall, once.

A captured demo is an exact button sequence — Billy's native currency. This tool lets you play
through a spot Billy is stuck on; it verifies your run survives AND advances progress on a clone,
then banks it to the SolutionCache. Every later autonomous run replays it for free.

Two steps (separate processes — one emulator per process):

    # 1. Capture the stuck-spot state (headless; drives Billy there with the learned cache):
    BILLY_HEADLESS=1 .venv/bin/python teleop_zelda.py capture --screen 121 \\
        --out data/zelda/states/teleop_121.state

    # 2. Play through it in a window, then bank your demo (NOT headless — needs the window):
    .venv/bin/python teleop_zelda.py play --from-state data/zelda/states/teleop_121.state --bank

Controls (window must have focus):  arrows = move · Z = A (sword/use) · X = B
    Tab = Start · RShift = Select · ENTER = finish & verify demo · ESC = abort
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_capture(args: argparse.Namespace) -> int:
    """Drive Billy (Director + persisted cache) to a screen and save the first clone_state there."""
    os.environ.setdefault("BILLY_HEADLESS", "1")
    os.environ.setdefault("BILLY_TURBO", "1")
    os.environ.setdefault("BILLY_MAX_FRAMES", str(args.max_frames))

    from billy.director import Director
    from billy.games.zelda.game import ZeldaGame
    from billy.knowledge import KnowledgeBase

    game = ZeldaGame()
    director = Director(game, KnowledgeBase(), use_llm=False)
    grabbed = {}
    orig = director._observe

    def hooked():
        obs = orig()
        if (not grabbed and obs.raw.in_play and obs.raw.map_location == args.screen):
            grabbed["state"] = director.session.clone_state()
            grabbed["obs"] = obs
            print(f"[capture] #{args.screen}: link=({obs.raw.link_x},{obs.raw.link_y}) "
                  f"prog={obs.progress} hearts={obs.raw.health}")
        return obs

    director._observe = hooked
    director.run_session(max(1, args.attempts))
    if not grabbed:
        print(f"[capture] never reached screen #{args.screen} — raise --attempts/--max-frames")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(grabbed["state"])
    print(f"[capture] saved -> {out}")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    """Windowed teleop: restore a state, let the human play, verify the demo, optionally bank."""
    os.environ["BILLY_HEADLESS"] = "0"   # we need the watch window for keyboard focus
    os.environ.setdefault("BILLY_TURBO", "1")

    from billy.config import SOLUTIONS_FILE
    from billy.games.zelda.game import ZeldaGame
    from billy.knowledge.cache import SolutionCache
    from billy.systems.nes import controller as c
    from billy.teleop import TeleopRecorder, bank_demo, verify_demo

    game = ZeldaGame()
    session = game.system.connect()
    session.wait_until_live()

    start_state = Path(args.from_state).read_bytes()
    session.restore(start_state)

    def observe():
        st = session.read_state()
        return game.observe(st.frame, st.ram, getattr(st, "rgb", None))

    start_obs = observe()
    if not session.ensure_viewer():
        print("[teleop] no window available (is a display attached? is BILLY_HEADLESS=1?)")
        return 2
    session.teleop_reset()

    s = start_obs.raw
    print(f"[teleop] take control at {start_obs.level_label} link=({s.link_x},{s.link_y}) "
          f"prog={start_obs.progress} hearts={s.health}")
    print("[teleop] arrows=move  Z=A  X=B  Tab=Start  RShift=Select  ENTER=finish  ESC=abort")

    start_screen = s.map_location
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
            print(f"[teleop] died at prog={obs.progress} after {frames}f")
            break
        # Auto-finish the instant you reach the next screen east, ALIVE — no need to press ENTER
        # in time. This is the success condition (crossed the hazard), so capture it immediately.
        if args.auto_finish and obs.raw.in_play and obs.raw.map_location > start_screen:
            print(f"[teleop] auto-finish: crossed to #{obs.raw.map_location} alive at {frames}f")
            break

    if aborted:
        print("[teleop] aborted — nothing banked")
        session.close()
        return 1

    plan = rec.plan()
    print(f"[teleop] captured {len(plan)} steps / {rec.frame_count()} frames")
    # Always persist the demo so a good run is never lost (e.g. to a measurement glitch).
    demo_path = Path(args.from_state).with_suffix(".demo.json")
    import json
    demo_path.write_text(json.dumps({"steps": [[s.frames, s.buttons] for s in plan]}))
    print(f"[teleop] demo saved -> {demo_path}")
    # Verify with a FRESH observer: the live `game` carries a monotonic progress high-water mark
    # from this play session, which would mask the demo's gain after we rewind to the start state.
    result = verify_demo(session, ZeldaGame(), start_state, plan, min_progress=args.min_progress)
    print(f"[teleop] verify: {result.summary()}")

    if not result.bankable:
        print("[teleop] not bankable (must survive AND advance) — nothing banked")
        session.close()
        return 1

    if args.bank:
        cache = SolutionCache(path=SOLUTIONS_FILE)
        key = bank_demo(cache, start_obs, plan, result.end_progress)
        print(f"[teleop] BANKED to {SOLUTIONS_FILE} at key {key} "
              f"(reach {result.end_progress}). Next autonomous run will replay it.")
    else:
        print("[teleop] dry run (no --bank) — verified bankable but not stored")
    session.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Human-in-the-loop teleop demo capture for Zelda.")
    sub = p.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="Drive Billy to a screen and save the state there.")
    cap.add_argument("--screen", type=int, default=121, help="overworld screen id to stop at")
    cap.add_argument("--out", required=True, help="path to write the .state file")
    cap.add_argument("--attempts", type=int, default=1)
    cap.add_argument("--max-frames", type=int, default=20000)

    pl = sub.add_parser("play", help="Windowed teleop from a state; verify and optionally bank.")
    pl.add_argument("--from-state", required=True, help=".state file to restore and play from")
    pl.add_argument("--bank", action="store_true", help="bank the demo if it verifies")
    pl.add_argument("--max-frames", type=int, default=3600, help="max teleop frames (~60s at 60fps)")
    pl.add_argument("--min-progress", type=int, default=8, help="min progress gain to count as advance")
    pl.add_argument("--no-auto-finish", dest="auto_finish", action="store_false",
                    help="don't auto-stop when you cross to the next screen (press ENTER yourself)")

    args = p.parse_args(argv)
    if args.cmd == "capture":
        return cmd_capture(args)
    if args.cmd == "play":
        return cmd_play(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
