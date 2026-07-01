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
Gamepad: calibrate once with `teleop.py calibrate` (saves data/pad_map.json — press each
    button when prompted, auto-detects stick/hat quirks, live-verifies). BILLY_PAD_* env vars
    override the saved map per-run; `pad-debug` shows the raw indices if you prefer manual.
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
        # Third artifact from the same demo: a transferable sequence Skill — the maneuver seeds
        # micro-search at SIMILAR hazards on other levels/games (verified before any commit).
        from billy.knowledge.distill import distill_solution
        from billy.knowledge.skills import SkillLibrary
        if distill_solution(SkillLibrary(), summary=start_obs.summary,
                            level_label=start_obs.level_label, plan=plan,
                            start_x=start_obs.progress, reach=result.end_progress,
                            source="demo", console=getattr(game.system, "name", "nes")):
            print("[teleop] DISTILLED into a transferable skill — this maneuver now seeds "
                  "search at similar hazards everywhere.")

    if args.tape and args.bank:
        # Second artifact from the same demo: a whole-trajectory tape for this level/screen.
        # cleared=True when the demo ended in a level/screen transition (the auto-finish case).
        from billy.knowledge.tape import TapeLibrary
        crossed = tuple(result.end_level_key) != tuple(start_obs.level_key)
        tapes = TapeLibrary()
        tapes.put(start_obs.level_key, plan, result.end_progress, clears_level=crossed)
        print(f"[teleop] TAPED for {start_obs.level_label} "
              f"(frontier {result.end_progress}, clears={crossed}) — a verified tape replays "
              f"the whole segment search-free.")

    if not args.bank:
        print("[teleop] dry run (no --bank) — verified bankable but not stored")
    session.close()
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Interactive gamepad calibration: assign each role by pressing it, auto-detect stick/hat
    behavior, save to data/pad_map.json (loaded by every later teleop), then live-verify."""
    os.environ["BILLY_HEADLESS"] = "0"
    os.environ.setdefault("BILLY_TURBO", "1")

    from billy.systems.nes.pad_map import describe, save_pad_map

    game = _game(args.game)
    session = game.system.connect()
    session.wait_until_live()
    if not session.ensure_viewer():
        print("[calibrate] no window available (display attached? BILLY_HEADLESS=1?)")
        return 2
    session.teleop_reset()
    ctrl = session.controller   # the active console's button module (NES or SNES)
    neutral = ctrl.NEUTRAL

    def pump() -> tuple:
        """One frame: pump window+HID events, keep the game alive, read raw pad state.
        Returns (state|None, finish_pressed, abort_pressed)."""
        _mask, fin, ab = session.teleop_poll()
        session.teleop_step(neutral)
        return session.pad_state(), fin, ab

    st, _, _ = pump()
    if st is None:
        print("[calibrate] no gamepad detected — is it paired/awake? (pyglet HID)")
        session.close()
        return 1
    print(f"[calibrate] gamepad: {st['name']}")
    print("[calibrate] keep the GAME WINDOW focused. ENTER (keyboard) = skip a step, "
          "ESC = abort.\n")

    # -- 1. rest analysis: find buttons that read pressed at rest + resting hat/stick --------
    print("[calibrate] hands off the pad for a moment (reading its resting state)...")
    session.set_overlay(["HANDS OFF THE PAD", "reading its resting state..."])
    rest_buttons: set[int] = set()
    rest_hats, rest_xs, rest_ys = [], [], []
    for _ in range(90):
        st, _, ab = pump()
        if ab:
            session.close()
            return 1
        if st:
            rest_buttons.update(st["buttons"])
            rest_hats.append(st["hat"])
            rest_xs.append(st["stick"][0])
            rest_ys.append(st["stick"][1])
    rest_x = sum(rest_xs) / max(1, len(rest_xs))
    rest_y = sum(rest_ys) / max(1, len(rest_ys))
    hat_rests_centered = all(h == (0, 0) for h in rest_hats)
    if rest_buttons:
        print(f"[calibrate]   note: indices {sorted(rest_buttons)} read pressed at rest "
              f"(ignored for assignment)")
    if not hat_rests_centered:
        print("[calibrate]   note: the d-pad hat does not rest centered on this pad — "
              "movement will use the analog stick")

    # -- 2. assign button roles by pressing them ---------------------------------------------
    roles: list[tuple[str, str]] = [
        ("A", "A — jump / sword"),
        ("B", "B — run / attack"),
        ("START", "START"),
        ("SELECT", "SELECT"),
        ("FINISH", "FINISH — ends a teleop demo (optional)"),
    ]
    if getattr(ctrl, "SPIN", 0):
        roles.insert(2, ("SPIN", "SPIN — SMW spin jump (optional)"))

    mapping: dict = {"use_hat": hat_rests_centered, "invert_x": False, "invert_y": False,
                     "deadzone": round(min(0.6, max(0.35, abs(rest_x) + 0.2,
                                                    abs(rest_y) + 0.2)), 2)}
    taken: set[int] = set()
    done_roles: list[str] = []
    for role, label in roles:
        print(f"[calibrate] press the pad button for  {label}   (or keyboard ENTER to skip)")
        so_far = "assigned: " + ("  ".join(done_roles) if done_roles else "(none yet)")
        session.set_overlay([f"PRESS:  {label}",
                             "on your GAMEPAD, press that button now",
                             "keyboard ENTER = skip   ESC = abort", so_far])
        session.teleop_reset()          # clear any pending finish/abort
        assigned = -1
        prev: set[int] = set(rest_buttons)
        for _ in range(60 * 30):        # up to ~30s per role
            st, fin, ab = pump()
            if ab:
                print("[calibrate] aborted — nothing saved")
                session.set_overlay(None)
                session.close()
                return 1
            if fin:
                print(f"[calibrate]   {role}: skipped")
                break
            now = set(st["buttons"]) if st else set()
            fresh = now - prev - rest_buttons - taken
            if fresh:
                assigned = min(fresh)
                taken.add(assigned)
                print(f"[calibrate]   {role} = button {assigned}")
                session.set_overlay([f"{role} = button {assigned}  OK",
                                     "release the button..."])
                while True:             # wait for release so one press can't claim two roles
                    st, _, _ = pump()
                    if not st or assigned not in st["buttons"]:
                        break
                break
            prev = now
        mapping[role] = assigned
        done_roles.append(f"{role}={assigned}" if assigned >= 0 else f"{role}=skip")

    # -- 3. movement: detect hat vs stick and the stick's up-sign ----------------------------
    def sample_direction(prompt: str, frames: int = 600):   # ~10s — reading the banner takes time
        """Prompt, then wait for a sustained hat/stick deviation; returns (hat_dx, dx, dy)."""
        print(f"[calibrate] {prompt}")
        session.set_overlay([prompt.upper().rstrip(" ."),
                             "hold it steady for a second",
                             "keyboard ENTER = skip   ESC = abort"])
        session.teleop_reset()
        streak = 0
        for _ in range(frames):
            st, fin, ab = pump()
            if ab or fin or not st:
                return None
            hat_dx = st["hat"][0] - (rest_hats[0][0] if rest_hats else 0)
            dx, dy = st["stick"][0] - rest_x, st["stick"][1] - rest_y
            if abs(hat_dx) > 0.5 or abs(dx) > 0.35 or abs(dy) > 0.35:
                streak += 1
                if streak >= 12:        # ~0.2s sustained — a hold, not a blip
                    return hat_dx, dx, dy
            else:
                streak = 0
        return None

    left = sample_direction("hold LEFT (d-pad or stick) ...")
    if left is not None:
        hat_dx, dx, _ = left
        used_hat = hat_rests_centered and abs(hat_dx) > 0.5
        if used_hat:
            mapping["use_hat"] = True
            print("[calibrate]   left/right: d-pad hat works")
            if hat_dx > 0.5:
                print("[calibrate]   ⚠ hat reads reversed (LEFT read as RIGHT) — "
                      "no auto-fix for a reversed hat; set BILLY_PAD_USE_HAT=0 to fall "
                      "back to the stick instead")
        elif abs(dx) > 0.35:
            print("[calibrate]   left/right: analog stick")
            # Engine convention (see _joy_mask): LEFT fires on x < -deadzone. If pushing LEFT
            # gives a POSITIVE x on this pad/backend, record invert_x so the sign matches —
            # same fix as invert_y below, just for the horizontal axis.
            mapping["invert_x"] = dx > 0.35
            if mapping["invert_x"]:
                print("[calibrate]   stick left reads positive on this pad — saving invert_x")
    up = sample_direction("now hold UP (stick) ...")
    if up is not None:
        _, _, dy = up
        # Engine convention (see _joy_mask): UP fires on y < -deadzone. If pushing up gives a
        # POSITIVE y on this pad/backend, record invert_y so the sign matches.
        mapping["invert_y"] = dy > 0.35
        if mapping["invert_y"]:
            print("[calibrate]   stick up reads positive on this pad — saving invert_y")

    # -- 4. save + live verify ----------------------------------------------------------------
    path = save_pad_map(mapping)
    print(f"\n[calibrate] saved -> {path}")
    print(f"[calibrate]   {describe(mapping)}")
    print("[calibrate] BILLY_PAD_* env vars still override the saved map per-run.\n")

    session.set_pad_map(mapping)
    session.teleop_reset()
    print("[calibrate] VERIFY (15s): press things — decoded buttons print below. "
          "ENTER/ESC to finish.")
    last = None
    for _ in range(60 * 15):
        mask, fin, ab = session.teleop_poll()
        session.teleop_step(neutral)
        if fin or ab:
            break
        names = "+".join(ctrl.names_from_mask(mask)) or "-"
        if names != last:
            print(f"    {names}")
            last = names
        session.set_overlay(["VERIFY: press buttons and move the stick",
                             f"reading:  {names}",
                             "keyboard ENTER = done"])
    session.set_overlay(None)
    session.close()
    print("[calibrate] done. Play with: "
          f".venv/bin/python teleop.py play --game {args.game} --from-state <state> --bank")
    return 0


def cmd_pad_debug(args: argparse.Namespace) -> int:
    """Open a window + gamepad and print live button/hat/stick state, to calibrate the mapping."""
    os.environ["BILLY_HEADLESS"] = "0"
    os.environ.setdefault("BILLY_TURBO", "1")
    from billy.systems.nes import controller as c

    game = _game(args.game)
    session = game.system.connect()
    session.wait_until_live()
    if not session.ensure_viewer():
        print("[pad-debug] no window (display attached?)")
        return 2
    st = session.pad_state()
    if st is None:
        print("[pad-debug] no gamepad detected by pyglet — is it paired/awake?")
        return 1
    print(f"[pad-debug] gamepad: {st['name']}")
    print("[pad-debug] press each button; note the index. Ctrl-C to stop.")
    print("[pad-debug] map with: BILLY_PAD_A=<jump idx> BILLY_PAD_B=<run idx> "
          "BILLY_PAD_START=<i> BILLY_PAD_SELECT=<i>  (BILLY_PAD_HATY_INV=1 if up/down flipped)")
    last = None
    frames = 0
    while frames < args.max_frames:
        session.teleop_poll()          # pump window+device events
        session.teleop_step(c.NEUTRAL) # keep the window alive, no input
        frames += 1
        st = session.pad_state()
        if not st:
            continue
        sig = (tuple(st["buttons"]), st["hat"])   # print EVERY change incl. rest (reveals resting hat)
        if sig != last:
            print(f"  buttons={st['buttons']} hat={st['hat']} stick={st['stick']}")
            last = sig
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
    pl.add_argument("--tape", action="store_true",
                    help="also bank the demo as a whole-trajectory tape for its level/screen "
                         "(use when --from-state is a level/screen ENTRY; the Director's "
                         "clone-verify gate rejects it harmlessly if the state doesn't match)")

    cal = sub.add_parser("calibrate", help="Interactive gamepad calibration → data/pad_map.json "
                                           "(press each button when prompted; auto-detects "
                                           "stick/hat quirks; ends with a live verify).")
    cal.add_argument("--game", default="smb")

    dbg = sub.add_parser("pad-debug", help="Print live gamepad button/hat indices (raw view; "
                                           "prefer `calibrate` for setup).")
    dbg.add_argument("--game", default="smb")
    dbg.add_argument("--max-frames", type=int, default=3600)

    args = p.parse_args(argv)
    if args.cmd == "calibrate":
        return cmd_calibrate(args)
    if args.cmd == "pad-debug":
        return cmd_pad_debug(args)
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
