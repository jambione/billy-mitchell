#!/usr/bin/env python3
"""Billy Mitchell REMIX — teach Billy past his next wall, across games, until he finishes.

This is a HANDS-ON, DYNAMIC, MULTI-GAME gauntlet. It doesn't hand you a fixed card of tricks —
it surfaces the walls Billy is ACTUALLY stuck at right now, one per game, and drops you in to
teach each one. Your run banks as a demo (cache entry + skill), Billy learns the line, and next
time he gets further before hitting the NEXT wall. The goals never get artificially harder — the
only thing that moves is Billy, forward through each game, toward finishing it.

How a run goes:
  1. SCOUT (headless, quick): Billy plays each game himself from his furthest checkpoint. Where
     search + self-training still can't pass, he files a demo request — his real next wall.
  2. TEACH (you, in the window): each wall becomes a challenge. You're dropped in right before it
     with one goal: get past the spot Billy keeps dying at. Clear it and he owns the line.
  3. SCOREBOARD: how far Billy can now get in each game — his march to the ending.

    .venv/bin/python remix.py               # scout, then teach every open wall
    .venv/bin/python remix.py --only zelda  # just one game
    .venv/bin/python remix.py --no-scout    # skip scouting; teach the walls already on file
    .venv/bin/python remix.py --list        # show Billy's frontier + open walls

Controls (click the game window first):  arrows/pad = move · Z = jump/confirm · X = run/cancel
    Tab = Start · ENTER = finish now · ESC = skip this wall.  (Gamepad: teleop.py calibrate once.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from billy import config

REQUESTS_FILE = config.DATA_DIR / "demo_requests.jsonl"
CLEARED_FILE = config.DATA_DIR / "remix_cleared.jsonl"
PAST_MARGIN = 24                 # must end this far past Billy's death spot (a real crossing)

# The campaign roster: games with a completion arc worth marching. Skipped gracefully if the
# ROM/integration isn't present.
CAMPAIGN = ["smb", "smb_lost", "zelda", "psii"]
_ENDING = {                      # a game is "finished" when its furthest label reaches here
    "smb": "8-4", "smb_lost": "8-4", "zelda": "level-9", "psii": "end",
}


# ---------------------------------------------------------------------------------------------
# Billy's state on disk: his frontier per game, and the walls he's asked for help on.
# ---------------------------------------------------------------------------------------------
def _furthest(game: str) -> tuple[str, int]:
    """(furthest level label, progress) Billy has checkpointed in this game — his march front."""
    meta = config.CHECKPOINTS_DIR / game / "furthest.json"
    if meta.is_file():
        try:
            d = json.loads(meta.read_text())
            return d.get("label", "start"), int(d.get("progress", 0))
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return "start", 0


def _read_requests() -> list[dict]:
    if not REQUESTS_FILE.is_file():
        return []
    out = []
    for line in REQUESTS_FILE.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _resolve_request(req: dict) -> None:
    """Drop a taught wall from the open queue and log it as cleared (Billy owns it now)."""
    kept = []
    for r in _read_requests():
        same = (r.get("game") == req.get("game")
                and r.get("level_label") == req.get("level_label")
                and r.get("death_bucket") == req.get("death_bucket"))
        if not same:
            kept.append(r)
    REQUESTS_FILE.write_text("".join(json.dumps(r) + "\n" for r in kept))
    config.ensure_dirs()
    with CLEARED_FILE.open("a") as f:
        f.write(json.dumps(req) + "\n")


def _game(name: str):
    from run import GAMES
    return GAMES[name]()


def _observe(session, game):
    st = session.read_state()
    return game.observe(st.frame, st.ram, getattr(st, "rgb", None))


# ---------------------------------------------------------------------------------------------
# SCOUT: let Billy play each game himself from his checkpoint, so he surfaces his current wall.
# ---------------------------------------------------------------------------------------------
def _scout(game_key: str, attempts: int) -> str:
    """Run Billy headless from his furthest checkpoint; he files a demo request at any wall his
    own search + self-training can't pass. Returns the furthest level label reached."""
    os.environ["BILLY_HEADLESS"] = "1"
    os.environ.setdefault("BILLY_TURBO", "1")
    from billy.abstractions import BootError
    from billy.director import Director
    from billy.knowledge import KnowledgeBase, SkillLibrary

    try:
        game = _game(game_key)
        game.cli_name = game_key
        director = Director(game, KnowledgeBase(), use_llm=False, skills=SkillLibrary())
        director.session.reset()
        director.session.wait_until_live()
        director.boot()
        try:
            director.resume_from_checkpoint()
        except Exception:
            pass                                    # no checkpoint yet — scout from the start
        try:
            for n in range(1, attempts + 1):
                director.run_attempt(n)
        finally:
            if getattr(director, "pool", None) is not None:
                director.pool.close()
            director.session.close()
    except (BootError, FileNotFoundError) as e:
        return f"(unavailable: {type(e).__name__})"
    return _furthest(game_key)[0]


# ---------------------------------------------------------------------------------------------
# TEACH: drop the human in at a wall's state and bank their crossing as a demo Billy keeps.
# ---------------------------------------------------------------------------------------------
def _teach_wall(req: dict) -> bool:
    """One wall: restore its state, YOU play past Billy's death spot, verify + bank. True if
    banked (Billy learned it)."""
    os.environ["BILLY_HEADLESS"] = "0"
    os.environ.setdefault("BILLY_TURBO", "1")
    from billy.abstractions import BootError

    game_key = req["game"]
    death_x = int(req.get("death_x", 0))
    target = death_x + PAST_MARGIN
    label = req.get("level_label", "?")
    state_path = Path(req.get("state", ""))
    print(f"\n{'═' * 64}\n  ▶  {game_key} · {label}  —  Billy's wall at x≈{death_x} "
          f"({req.get('deaths', '?')} deaths)\n     GOAL: get past x≈{death_x} and keep going — "
          f"clear it and Billy learns the line.\n{'═' * 64}")
    if not state_path.is_file():
        print(f"     ⚠  drop-in state missing ({state_path}) — skipping.")
        return False

    game = _game(game_key)
    session = game.system.connect()
    try:
        session.wait_until_live()
        session.restore(state_path.read_bytes())
        if not session.ensure_viewer():
            print("     ⚠  no game window (headless?). The remix needs a window to play in.")
            return False
        print("     🎮 controller ready." if session.reopen_joystick()
              else "     ⌨  keyboard (arrows / Z / X / Tab).")
        start_state = session.clone_state()
        start_obs = _observe(session, game)

        for t in range(1, 4):
            if t > 1:
                print(f"     try {t}/3…")
            outcome, secs, plan = _play_wall(session, game, start_obs, target, death_x)
            if outcome == "win":
                print(f"     ✓ you cleared it in {secs:.1f}s")
                return _bank(game_key, session, game, start_state, start_obs, plan)
            if outcome == "skip":
                print("     ↷ skipped.")
                return False
            print(f"     ✗ {outcome} at {secs:.1f}s — try again.")
        print("     (out of tries — the wall stays open for next time.)")
        return False
    except BootError as e:
        print(f"     ⚠  couldn't start ({e}).")
        return False
    finally:
        session.close()


def _play_wall(session, game, start_obs, target: int, death_x: int) -> tuple[str, float, list]:
    from billy.teleop import TeleopRecorder
    session.teleop_reset()
    rec = TeleopRecorder()
    budget = 45 * 60
    frames = 0
    obs = start_obs

    def overlay(rem):
        session.set_overlay([f"▶ Get past x≈{death_x} and keep going",
                             f"{rem:4.0f}s   at x={obs.progress}  →  need {target}",
                             "controller or keys · ESC = skip"])
    overlay(45)
    while frames < budget:
        mask, finish, abort = session.teleop_poll()
        if abort:
            return "skip", frames / 60.0, []
        session.teleop_step(mask)
        rec.record(mask, 1)
        frames += 1
        obs = _observe(session, game)
        if obs.dead:
            return "died", frames / 60.0, []
        alive_past = getattr(obs.raw, "in_play", True) and obs.progress >= target
        if alive_past or finish:
            return "win", frames / 60.0, rec.plan()
        if frames % 15 == 0:
            overlay((budget - frames) / 60)
    return "timeout", frames / 60.0, []


def _bank(game_key: str, session, game, start_state: bytes, start_obs, plan) -> bool:
    """Verify the human demo on the SAME emulator (search_mode, invisible) and bank it: a
    SolutionCache entry (exact replay) + a distilled transferable skill."""
    from billy.config import SOLUTIONS_FILE
    from billy.knowledge.cache import SolutionCache
    from billy.teleop import bank_demo, verify_demo

    if not plan:
        return False
    result = verify_demo(session, _game(game_key), start_state, plan, min_progress=8)
    print(f"     verify: {result.summary()}")
    if not result.bankable:
        print("     (run didn't verify to bank — survive AND advance past the spot; try again.)")
        return False
    cache = SolutionCache(path=SOLUTIONS_FILE)
    key = bank_demo(cache, start_obs, plan, result.end_progress)
    print(f"     ⇒ 🧠 Billy learned it — banked at {key} (reach {result.end_progress}).")
    try:
        from billy.knowledge.distill import distill_solution
        from billy.knowledge.skills import SkillLibrary
        if distill_solution(SkillLibrary(), summary=start_obs.summary,
                            level_label=start_obs.level_label, plan=plan,
                            start_x=start_obs.progress, reach=result.end_progress,
                            source="demo", console=getattr(game.system, "name", "nes")):
            print("     ⇒ distilled a transferable skill (helps at similar spots, other games too).")
    except Exception as e:
        print(f"     (skill distill skipped: {type(e).__name__})")
    return True


# ---------------------------------------------------------------------------------------------
def _scoreboard(games: list[str], taught: int) -> None:
    print(f"\n{'━' * 64}\n  BILLY'S MARCH — how far he can get in each game now\n{'━' * 64}")
    for g in games:
        label, prog = _furthest(g)
        end = _ENDING.get(g, "?")
        done = "🏁 FINISHED" if label == end else f"at {label}"
        open_walls = sum(1 for r in _read_requests() if r.get("game") == g)
        wtxt = f"  ·  {open_walls} wall(s) to teach" if open_walls else ""
        print(f"  {g:<9} {done:<14} (goal {end}){wtxt}")
    print(f"{'━' * 64}\n  You taught Billy {taught} new line(s) this session. "
          f"Every one moves him closer to the ending.\n{'━' * 64}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Billy Mitchell Remix — teach his next wall, per game.")
    p.add_argument("--only", default="", help="comma-separated game keys (default: all campaign games)")
    p.add_argument("--no-scout", action="store_true", help="skip scouting; teach walls already on file")
    p.add_argument("--scout-attempts", type=int, default=2, help="attempts per game while scouting")
    p.add_argument("--list", action="store_true", help="show Billy's frontier + open walls and exit")
    args = p.parse_args(argv)

    games = [g for g in CAMPAIGN if not args.only or g in
             {s.strip() for s in args.only.split(",")}]
    config.ensure_dirs()

    if args.list:
        print("  Billy's frontier + open walls:")
        for g in games:
            label, prog = _furthest(g)
            walls = [r for r in _read_requests() if r.get("game") == g]
            wtxt = ", ".join(f"{w['level_label']}@{w['death_x']}" for w in walls) or "—"
            print(f"    {g:<9} at {label:<8} (goal {_ENDING.get(g,'?')})  "
                  f"— {len(walls)} open wall(s): {wtxt}")
        return 0

    print("🎮  BILLY MITCHELL — REMIX\n    Teach Billy past his next wall in each game. "
          "He keeps every line you show him.\n")

    # 1) SCOUT — let Billy find his current walls (headless, quick).
    if not args.no_scout:
        print("  Scouting… (Billy plays each game himself to find where he's stuck)")
        for g in games:
            before = _furthest(g)[0]
            reached = _scout(g, args.scout_attempts)
            print(f"    · {g:<9} reached {reached}"
                  + (f"  (was {before})" if reached != before else ""))

    # 2) TEACH — the human clears each open wall (needs a window).
    walls = [r for r in _read_requests() if r.get("game") in set(games)]
    if not walls:
        print("\n  No open walls right now — Billy is cruising. "
              "Run again later (or with more --scout-attempts) to find the next one.")
        _scoreboard(games, 0)
        return 0
    if os.environ.get("REMIX_SCOUT_ONLY") == "1":
        print(f"\n  {len(walls)} wall(s) waiting — run without REMIX_SCOUT_ONLY to teach them.")
        _scoreboard(games, 0)
        return 0

    print(f"\n  Billy needs help at {len(walls)} wall(s). Take the controller —")
    taught = 0
    for req in sorted(walls, key=lambda r: (r["game"], r.get("death_x", 0))):
        if _teach_wall(req):
            _resolve_request(req)
            taught += 1

    _scoreboard(games, taught)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[remix] halted — Billy demands a rematch.")
        sys.exit(130)
