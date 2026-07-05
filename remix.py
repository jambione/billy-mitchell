#!/usr/bin/env python3
"""Billy Mitchell REMIX — a NES-Remix-style gauntlet where YOU play, and Billy learns.

Each challenge drops YOU into control at a hard spot in a game with one short goal (cross the
lift, clear the pit, walk out of town). You play it in the window; the moment you succeed, your
run is verified and banked as a demo — a cache entry + a transferable skill — so **Billy learns
the line from you** and can do it himself forever after. Short segments, clear goals, you're the
one holding the controller.

    .venv/bin/python remix.py                 # play the whole gauntlet (needs a window)
    .venv/bin/python remix.py --only lift     # one challenge
    .venv/bin/python remix.py --list          # show the card

Controls (the game window must have focus):
    arrows = move   ·   Z = A (jump / confirm)   ·   X = B (run / cancel)
    Tab = Start   ·   ENTER = finish now   ·   ESC = skip this challenge
    Gamepad: run `teleop.py calibrate` once (saves data/pad_map.json).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from billy import config

PROGRESS_FILE = config.DATA_DIR / "remix_progress.json"
_STAR = {"gold": 3, "silver": 2, "bronze": 1, "none": 0}
_MEDAL_ICON = {"gold": "🥇", "silver": "🥈", "bronze": "🥉", "none": "⬜"}


@dataclass
class Challenge:
    id: str
    title: str
    game: str                       # run.py GAMES key
    goal_text: str                  # what YOU need to do (shown on screen)
    goal_kind: str                  # "clear" | "advance" | "psii_exit"
    time_s: int                     # budget for one try
    medals: tuple                   # (gold_s, silver_s) — seconds to finish; slower = bronze
    goal_px: int = 0                # for "advance": how far past the start counts as done
    from_state: str = ""            # savestate to drop you in at; "" = play from the game's start
    tries: int = 3
    hint: str = ""

    def reached(self, start_obs, obs) -> bool:
        """True once YOU have accomplished the goal (survive is implicit — death ends the try)."""
        if not getattr(obs.raw, "in_play", True):
            return False
        if self.goal_kind == "clear":
            return tuple(obs.level_key[:2]) != tuple(start_obs.level_key[:2])
        if self.goal_kind == "advance":
            return obs.progress >= start_obs.progress + self.goal_px
        if self.goal_kind == "psii_exit":
            sp, op = start_obs.raw.place, obs.raw.place
            return bool(op[2]) and op[:2] != sp[:2]        # a NEW outdoor place = out of Paseo
        return False

    def medal(self, seconds: float | None) -> str:
        if seconds is None:
            return "none"
        g, s = self.medals
        return "gold" if seconds <= g else "silver" if seconds <= s else "bronze"


# ---------------------------------------------------------------------------------------------
# The card. Each is a short, human-played line at a spot where teaching Billy actually matters.
# ---------------------------------------------------------------------------------------------
CHALLENGES: list[Challenge] = [
    Challenge("flagpole", "SMB 1-1 · To the Flagpole", "smb",
              "Reach the flagpole — clear World 1-1.", "clear",
              time_s=75, medals=(45, 65), hint="Run right; jump the pits and Goombas."),
    Challenge("lift", "SMB 1-3 · Ride the Lift", "smb",
              "Cross the moving-lift gap and reach solid ground.", "advance", goal_px=180,
              time_s=40, medals=(12, 22), from_state="data/states/smb_1_3_lift_approach.state",
              hint="Time the jump onto the lift, ride it, hop off before it drops."),
    Challenge("wall", "SMB 2-1 · Over the Wall", "smb",
              "Get past the wall and keep moving right.", "advance", goal_px=140,
              time_s=35, medals=(10, 20), from_state="data/states/smb_2_1_wall.state",
              hint="Back up for a run-up, then a full running jump."),
    Challenge("paseo", "Phantasy Star II · Escape Paseo", "psii",
              "Walk Rolf out of town onto the Mota overworld.", "psii_exit",
              time_s=60, medals=(25, 45),
              hint="Head for a street that leaves town — cross the edge onto open ground."),
]
_BY_ID = {c.id: c for c in CHALLENGES}


def _load_progress() -> dict:
    if PROGRESS_FILE.is_file():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_progress(prog: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(prog, indent=2) + "\n")


def _game(name: str):
    from run import GAMES
    return GAMES[name]()


def _play_once(ch: Challenge, session, game, start_state: bytes, start_obs) -> tuple[str, float]:
    """One try: YOU play; on success verify+bank the demo. Returns (outcome, seconds)."""
    from billy.teleop import TeleopRecorder

    session.restore(start_state)
    session.teleop_reset()
    rec = TeleopRecorder()
    budget = ch.time_s * 60
    frames = 0
    outcome = "timeout"

    def overlay(remaining: float, prog_note: str = ""):
        session.set_overlay([f"▶ {ch.goal_text}",
                             f"{remaining:4.0f}s   {prog_note}",
                             "controller or keys · reach the goal to win · ESC=skip"])

    overlay(ch.time_s)
    obs = start_obs
    while frames < budget:
        mask, finish, abort = session.teleop_poll()
        if abort:
            outcome = "skip"
            break
        session.teleop_step(mask)
        rec.record(mask, 1)
        frames += 1
        obs = _observe(session, game)
        if obs.dead:
            outcome = "died"
            break
        if ch.reached(start_obs, obs) or finish:
            outcome = "win"
            break
        if frames % 15 == 0:
            gained = obs.progress - start_obs.progress
            overlay((budget - frames) / 60, f"+{gained}")

    seconds = frames / 60.0
    if outcome != "win":
        session.set_overlay([f"✗ {outcome.upper()}", f"{ch.goal_text}", "…"])
        return outcome, seconds

    # You made it — turn your run into something Billy keeps.
    plan = rec.plan()
    banked = _bank(ch, session, game, start_state, start_obs, plan)
    session.set_overlay([f"✓ CLEARED in {seconds:.1f}s",
                         "Billy learned it." if banked else "(nice — run didn't verify to bank)",
                         ""])
    return "win", seconds


def _bank(ch: Challenge, session, game, start_state: bytes, start_obs, plan) -> bool:
    """Verify the human demo on the SAME session (stable-retro allows only one emulator per
    process) and, if it survives+advances, bank it for Billy: a SolutionCache entry (exact
    replay) + a distilled transferable skill. A FRESH game object does the observing so the
    live run's monotonic progress high-water mark can't mask the demo's gain (as teleop does)."""
    from billy.config import SOLUTIONS_FILE
    from billy.knowledge.cache import SolutionCache
    from billy.teleop import bank_demo, verify_demo

    verify_game = _game(ch.game)                    # fresh observer, same emulator session
    result = verify_demo(session, verify_game, start_state, plan, min_progress=8)
    print(f"     verify: {result.summary()}")
    if not result.bankable:
        return False
    cache = SolutionCache(path=SOLUTIONS_FILE)
    key = bank_demo(cache, start_obs, plan, result.end_progress)
    print(f"     ⇒ banked demo at {key} (reach {result.end_progress}) — Billy replays it now.")
    try:
        from billy.knowledge.distill import distill_solution
        from billy.knowledge.skills import SkillLibrary
        if distill_solution(SkillLibrary(), summary=start_obs.summary,
                            level_label=start_obs.level_label, plan=plan,
                            start_x=start_obs.progress, reach=result.end_progress,
                            source="demo", console=getattr(game.system, "name", "nes")):
            print("     ⇒ distilled into a transferable skill (seeds search on similar spots).")
    except Exception as e:
        print(f"     (skill distill skipped: {type(e).__name__})")
    return True


def _observe(session, game):
    st = session.read_state()
    return game.observe(st.frame, st.ram, getattr(st, "rgb", None))


def _run_challenge(ch: Challenge) -> tuple[str, float | None]:
    from billy.abstractions import BootError

    print(f"\n{'═' * 64}\n  ▶  {ch.title}\n     GOAL: {ch.goal_text}")
    if ch.hint:
        print(f"     hint: {ch.hint}")
    print(f"{'═' * 64}")
    game = _game(ch.game)
    session = game.system.connect()
    try:
        session.wait_until_live()
        if ch.from_state:
            p = Path(ch.from_state)
            if not p.is_file():
                print(f"     ⚠  start state missing ({p}) — skipping. "
                      f"(capture it with teleop.py capture)")
                return "none", None
            session.restore(p.read_bytes())
        else:
            game.boot(session)
        if not session.ensure_viewer():
            print("     ⚠  no game window (are you headless?). Remix needs a window to play in.")
            return "none", None
        # Wake a gamepad that was asleep when the window opened (else it silently falls back to
        # the keyboard). Uses your saved calibration (data/pad_map.json — teleop.py calibrate).
        if session.reopen_joystick():
            print("     🎮 controller ready.")
        else:
            print("     ⌨  no gamepad found — using the keyboard (arrows / Z / X / Tab).")
        start_state = session.clone_state()
        start_obs = _observe(session, game)

        best: float | None = None
        for t in range(1, ch.tries + 1):
            if t > 1:
                print(f"     try {t}/{ch.tries}…")
            outcome, secs = _play_once(ch, session, game, start_state, start_obs)
            if outcome == "win":
                best = secs if best is None else min(best, secs)
                print(f"     ✓ cleared in {secs:.1f}s — {_MEDAL_ICON[ch.medal(secs)]}")
                break
            if outcome == "skip":
                print("     ↷ skipped.")
                break
            print(f"     ✗ {outcome} at {secs:.1f}s.")
        return ch.medal(best), best
    except BootError as e:
        print(f"     ⚠  couldn't start ({e}).")
        return "none", None
    finally:
        session.close()


def _scoreboard(prog: dict) -> None:
    print(f"\n{'━' * 64}\n  REMIX — YOUR MEDALS  (and what Billy learned from you)\n{'━' * 64}")
    stars = 0
    for ch in CHALLENGES:
        best = prog.get(ch.id, {}).get("best")
        medal = ch.medal(best)
        stars += _STAR[medal]
        val = f"{best:.1f}s" if best is not None else "—"
        print(f"  {_MEDAL_ICON[medal]}  {ch.title:<34} {val}")
    mx = 3 * len(CHALLENGES)
    filled = round(12 * stars / mx) if mx else 0
    print(f"{'━' * 64}\n  MASTERY  [{'█'*filled}{'░'*(12-filled)}]  {stars}/{mx} stars")
    print("  Every medal you earn is a line Billy can now run himself." + ("  👑" if stars == mx else ""))
    print("━" * 64)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Billy Mitchell Remix — YOU play, Billy learns.")
    p.add_argument("--only", default="", help="comma-separated challenge ids (see --list)")
    p.add_argument("--list", action="store_true", help="show the challenge card and exit")
    args = p.parse_args(argv)

    if args.list:
        print("  Billy Mitchell Remix — challenges (you play each one):")
        for c in CHALLENGES:
            where = f"from {c.from_state}" if c.from_state else "from the start"
            print(f"    {c.id:<10} {c.title:<34} — {c.goal_text}  [{where}]")
        return 0

    if os.environ.get("BILLY_HEADLESS", "0") == "1":
        print("[remix] Remix is a HANDS-ON gauntlet — it needs a game window. "
              "Run it without BILLY_HEADLESS (e.g. the Desktop launcher).")
        return 2

    chosen = CHALLENGES
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        chosen = [c for c in CHALLENGES if c.id in want]
        if not chosen:
            print(f"[remix] no matching challenges in {sorted(want)} — see --list.")
            return 1

    config.ensure_dirs()
    prog = _load_progress()
    print("🎮  BILLY MITCHELL — REMIX\n    You take the controller. Clear each line and Billy "
          "learns it from you.\n")
    for ch in chosen:
        _, best = _run_challenge(ch)
        if best is not None:
            prev = prog.get(ch.id, {}).get("best")
            prog[ch.id] = {"best": min(best, prev) if prev is not None else best,
                           "updated": _dt.datetime.now().isoformat(timespec="seconds")}
            _save_progress(prog)
    _scoreboard(prog)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[remix] halted — Billy demands a rematch.")
        sys.exit(130)
