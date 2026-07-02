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


class PadLostError(RuntimeError):
    """The gamepad or the watch window vanished mid-calibration (fail loudly, not silently)."""


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
        # Level-transition artifact guard: on the frames where the label has already flipped
        # to the next level, progress can still read the PREVIOUS level's x (e.g. 3266 while
        # "1-3" is loading). Such a state is mid-transition, not a playable spot — require the
        # transition to have settled (x reset to a small in-level value first).
        if args.until_level is not None and getattr(reached, "_settled", False) is False:
            if obs.progress < 16 or obs.progress > 400:
                return False
            reached._settled = True
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
        Returns (state, finish_pressed, abort_pressed). Raises PadLostError if the pad or
        the window dies mid-calibration — silently reading None here made every stage look
        'skipped' with a frozen window, which is worse than failing loudly."""
        _mask, fin, ab = session.teleop_poll()
        session.teleop_step(neutral)
        st = session.pad_state()
        if st is None:
            raise PadLostError(
                "gamepad/window lost mid-calibration (viewer disabled or pad disconnected) — "
                "check the [viewer] error above, then re-run calibrate")
        return st, fin, ab

    # Bluetooth pads auto-sleep; don't bail on the first miss — prompt and wait for a wake.
    st = None
    for i in range(60 * 45):            # up to ~45s of waiting for the pad
        _mask, _fin, ab = session.teleop_poll()
        session.teleop_step(neutral)
        if ab:
            session.close()
            return 1
        st = session.pad_state()
        if st is not None:
            break
        if i % 60 == 0:
            session.set_overlay(["WAKE THE GAMEPAD",
                                 "press any button on the pad (Bluetooth pads sleep)",
                                 "ESC = abort"])
            if i == 0:
                print("[calibrate] no gamepad yet — press a button on it to wake it "
                      "(waiting up to 45s)...")
            session.reopen_joystick()
    if st is None:
        print("[calibrate] no gamepad detected — pair/wake it, then re-run calibrate")
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

    # -- 3. movement: calibrate each DIRECTION by holding it -----------------------------------
    # Records whatever actually changes while the user holds the direction — a plain button
    # (many pads report the d-pad as buttons), a hat axis, or ANY of the six HID axes. Three
    # defenses against phantom sources (floating hats, axes that report late and stick high):
    #   • the baseline is re-snapshotted at EACH prompt (a stuck axis shows ~zero deviation),
    #   • the captured source must RELEASE back to baseline when the user lets go,
    #   • two directions can never share a source+sign (banned set).
    def _sources(st, base):
        """All (deviation, key, spec) candidates for the current frame vs a baseline."""
        out = []
        fresh = set(st["buttons"]) - rest_buttons - taken
        if fresh:
            out.append((1.0, ("button", min(fresh)), {"src": "button", "idx": min(fresh)}))
        # The HAT is matched as an EXACT TUPLE, not per-axis signs: scrambled/rotated hats
        # (this pad's known quirk) emit *some* stable tuple per held direction — that tuple
        # IS the direction. The baseline holds the SET of resting tuples (this hat flickers
        # at rest), so a resting value can never be captured as a press.
        hat = tuple(st["hat"])
        if hat not in base["hat_set"]:
            out.append((2.0, ("hat", hat), {"src": "hat", "value": list(hat)}))
        for n, v in st["axes"].items():
            dev = v - base["axes"].get(n, 0.0)
            if abs(dev) > 0.35:
                sign = 1 if dev > 0 else -1
                out.append((abs(dev), (f"axis_{n}", sign),
                            {"src": f"axis_{n}", "sign": sign,
                             "rest": round(base["axes"].get(n, 0.0), 3)}))
        return out

    def sample_dir_spec(name: str, banned: set, frames: int = 900):
        """Hold a direction; returns its verified spec dict, or None on skip/timeout."""
        print(f"[calibrate] press and HOLD {name} (d-pad or stick) ...")
        session.set_overlay([f"HOLD:  {name}",
                             "on the d-pad or stick - hold it steady",
                             "keyboard ENTER = skip   ESC = abort"])
        session.teleop_reset()
        # Per-prompt baseline: ~1/3s of hands-off frames. Collect the SET of resting hat
        # tuples (flickering hats produce several) and the axes' current values.
        hat_set: set = set()
        st, fin, ab = pump()
        axes0 = dict(st["axes"])
        for _ in range(20):
            st, fin, ab = pump()
            if ab or fin:
                return None
            hat_set.add(tuple(st["hat"]))
        base = {"hat_set": hat_set, "axes": axes0}
        streak, last_key, spec = 0, None, None
        for _ in range(frames):
            st, fin, ab = pump()
            if ab or fin:
                return None
            cands = [c for c in _sources(st, base) if c[1] not in banned]
            if cands:
                cands.sort(reverse=True, key=lambda c: c[0])
                _, key, cur = cands[0]
                streak = streak + 1 if key == last_key else 1
                last_key, spec = key, cur
                if streak >= 12:        # ~0.2s sustained on the SAME source — a hold, not a blip
                    session.set_overlay([f"{name}: got it - now LET GO", "(verifying release)"])
                    released = False
                    for _ in range(240):    # ~4s to release
                        st, _, ab2 = pump()
                        if ab2:
                            return None
                        still = any(k == key for _, k, _s in _sources(st, base))
                        if not still:
                            released = True
                            break
                    if released:
                        return spec
                    # Never released = a stuck/floating source, not the user's hold. Ban it
                    # and keep listening for the real one.
                    print(f"[calibrate]   ignoring {key[0]} (never releases — floating source)")
                    banned.add(key)
                    streak, last_key, spec = 0, None, None
            else:
                streak, last_key = 0, None
        return None

    def _spec_key(spec: dict) -> tuple:
        if spec["src"] == "button":
            return ("button", spec["idx"])
        if spec["src"] == "hat":
            return ("hat", tuple(spec["value"]))
        return (spec["src"], spec.get("sign"))

    def _spec_desc(spec: dict) -> str:
        if spec["src"] == "button":
            return f"d-pad button {spec['idx']}"
        if spec["src"] == "hat":
            return f"d-pad hat {tuple(spec['value'])}"
        return f"{spec['src']} {'+' if spec.get('sign', 1) > 0 else '-'}"

    dirs: dict = {}
    banned: set = set()
    for name in ("LEFT", "RIGHT", "UP", "DOWN"):
        spec = sample_dir_spec(name, banned)
        if spec is None:
            print(f"[calibrate]   {name}: skipped (falls back to stick defaults)")
            continue
        dirs[name] = spec
        banned.add(_spec_key(spec))
        print(f"[calibrate]   {name} = {_spec_desc(spec)}")
    if dirs:
        mapping["dirs"] = dirs

    # -- 4. REVIEW before saving: try the candidate mapping live, then confirm ----------------
    # Nothing is written yet — the mapping is applied to the live window only. Press every
    # button and direction and watch the decode; confirm to save, ESC to walk away clean.
    session.set_pad_map(mapping)
    session.teleop_reset()
    print("\n[calibrate] REVIEW — nothing saved yet. Candidate mapping:")
    print(f"[calibrate]   {describe(mapping)}")
    print("[calibrate] try every button/direction; decoded input prints below.")
    print("[calibrate] CONFIRM & SAVE: keyboard ENTER, or hold your mapped START ~1s. "
          "DISCARD: ESC.")
    dir_line = "  ".join(f"{d}={_spec_desc(s)}" for d, s in (dirs or {}).items()) \
        or "stick defaults"
    summary = "  ".join(x for x in done_roles if not x.endswith("=skip"))
    last = None
    confirmed = False
    start_hold = 0
    for _ in range(60 * 120):           # up to 2 minutes of review
        mask, fin, ab = session.teleop_poll()
        session.teleop_step(neutral)
        if ab:
            break
        if fin:
            confirmed = True
            break
        names = "+".join(ctrl.names_from_mask(mask)) or "-"
        if names != last:
            print(f"    {names}")
            last = names
        # Pad-only confirm: hold the freshly-mapped START for ~1s.
        start_hold = start_hold + 1 if (mask & getattr(ctrl, "START", 0)) else 0
        if mapping.get("START", -1) >= 0 and start_hold >= 60:
            confirmed = True
            break
        session.set_overlay(["REVIEW - try everything (not saved yet)",
                             f"reading:  {names}",
                             f"buttons: {summary}",
                             f"directions: {dir_line}",
                             "SAVE: keyboard ENTER or hold START   DISCARD: ESC"])
    session.set_overlay(None)

    if not confirmed:
        print("[calibrate] discarded — nothing written. Run calibrate again when ready.")
        session.close()
        return 1
    path = save_pad_map(mapping)
    print(f"\n[calibrate] confirmed — saved -> {path}")
    print(f"[calibrate]   {describe(mapping)}")
    print("[calibrate] BILLY_PAD_* env vars still override the saved map per-run.")
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
        try:
            return cmd_calibrate(args)
        except PadLostError as e:
            print(f"[calibrate] ❌ {e}")
            return 1
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
