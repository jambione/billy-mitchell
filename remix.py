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
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from billy import config

REQUESTS_FILE = config.DATA_DIR / "demo_requests.jsonl"
CLEARED_FILE = config.DATA_DIR / "remix_cleared.jsonl"
PAST_MARGIN = 24                 # must end this far past Billy's death spot (a real crossing)
ZELDA_MIN_PROGRESS = 48          # trivial link_x nudges must not count as teaching #121

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


def _iter_jsonl_records(path: Path) -> list[dict]:
    """Parse JSONL, tolerating accidental concatenation on one line (}{ between objects)."""
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        chunk = line.strip()
        if not chunk:
            continue
        while chunk:
            try:
                r, idx = json.JSONDecoder().raw_decode(chunk)
            except json.JSONDecodeError:
                break
            if isinstance(r, dict):
                out.append(r)
            chunk = chunk[idx:].lstrip()
    return out


def _read_cleared_keys() -> set[tuple[str, str, int]]:
    """Walls already taught via remix — don't re-queue them."""
    keys: set[tuple[str, str, int]] = set()
    for r in _iter_jsonl_records(CLEARED_FILE):
        keys.add((r.get("game", ""), r.get("level_label", ""),
                  int(r.get("death_bucket", 0))))
    return keys


def _read_cleared_levels() -> set[tuple[str, str]]:
    """Levels fully taught — one success clears the whole screen/level, not one death bucket."""
    return {(r.get("game", ""), r.get("level_label", ""))
            for r in _iter_jsonl_records(CLEARED_FILE)}


def _has_banked_solution(game: str, level_label: str) -> bool:
    """True if the solution cache already has a entry for this level (teach worked)."""
    from billy.config import SOLUTIONS_FILE
    if not SOLUTIONS_FILE.is_file():
        return False
    if game == "zelda":
        m = re.search(r"#(\d+)", level_label)
        if not m:
            return False
        level_key = json.dumps(["overworld", int(m.group(1))])
    elif game == "smb" and re.fullmatch(r"\d+-\d+", level_label):
        w, s = level_label.split("-")
        level_key = json.dumps([0, int(w), int(s)])   # SMB level_key prefix in cache
    else:
        return False
    for line in SOLUTIONS_FILE.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if json.dumps(row.get("level", [])) == level_key and row.get("plan"):
            return True
    return False


def _still_stuck_on_level(game: str, level_label: str) -> bool:
    """True if Billy is STILL dying here — re-queue even if remix_cleared says we taught it."""
    stuck_path = config.DATA_DIR / "stuck.json"
    if not stuck_path.is_file():
        return False
    try:
        rows = json.loads(stuck_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    best = 0
    for rec in rows:
        if _game_for_level(rec.get("level_label", "")) != game:
            continue
        if rec.get("level_label") != level_label or rec.get("remediated"):
            continue
        best = max(best, int(rec.get("deaths", 0)))
    return best >= config.STUCK_DEATH_THRESHOLD


def _dedupe_walls_by_level(walls: list[dict]) -> list[dict]:
    """One teach session per (game, level) — not one per x-bucket on the same Zelda screen."""
    best: dict[tuple[str, str], dict] = {}
    for w in walls:
        key = (w.get("game", ""), w.get("level_label", ""))
        if key not in best or int(w.get("deaths", 0)) > int(best[key].get("deaths", 0)):
            best[key] = w
    return list(best.values())


def _goal_blurb(game_key: str, req: dict) -> str:
    death_x = int(req.get("death_x", 0))
    label = req.get("level_label", "?")
    if game_key == "zelda":
        m = re.search(r"#(\d+)", label)
        n = int(m.group(1)) if m else 0
        return f"clear {label} alive and march east to screen #{n + 1}"
    return f"get past x≈{death_x} and keep going"


def _teach_params(game_key: str, req: dict, obs) -> dict:
    """Per-game win criteria. Zelda crosses screens; platformers use progress."""
    death_x = int(req.get("death_x", 0))
    if game_key == "zelda":
        start_screen = int(getattr(obs.raw, "map_location", 0))
        m = re.search(r"#(\d+)", req.get("level_label", ""))
        wall_screen = int(m.group(1)) if m else start_screen
        return {
            "mode": "zelda_screen",
            "start_screen": start_screen,
            "target_screen": wall_screen + 1,
            "min_progress": ZELDA_MIN_PROGRESS,
            "overlay_need": f"screen #{wall_screen} → #{wall_screen + 1}",
        }
    return {
        "mode": "progress",
        "target": death_x + PAST_MARGIN,
        "min_progress": 8,
        "overlay_need": f"x={obs.progress}  →  need {death_x + PAST_MARGIN}",
    }


def _wall_cleared(game_key: str, obs, params: dict) -> bool:
    if params["mode"] == "zelda_screen":
        return (getattr(obs.raw, "in_play", True)
                and int(getattr(obs.raw, "map_location", 0)) >= params["target_screen"])
    return getattr(obs.raw, "in_play", True) and obs.progress >= params["target"]


def _level_rank(label: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)-(\d+)", label)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _at_or_beyond_frontier(game: str, level_label: str) -> bool:
    """Skip walls on levels Billy has already marched past."""
    furthest_label, _ = _furthest(game)
    if furthest_label == "start" or not re.fullmatch(r"\d+-\d+", level_label):
        return True
    return _level_rank(level_label) >= _level_rank(furthest_label)


def _game_for_level(level_label: str) -> str | None:
    """Map a stuck-record level label to a campaign game key."""
    if level_label.startswith(("overworld", "dungeon")):
        return "zelda"
    if level_label.startswith("psii"):
        return "psii"
    if re.fullmatch(r"\d+-\d+", level_label):
        return "smb"          # smb_lost shares the format; it has no stuck history yet
    return None


def _auto_states_for_level(level_label: str) -> list[str]:
    """Auto-captured .state files for a level (uses live config.DATA_DIR)."""
    auto_dir = config.DATA_DIR / "rl" / "states" / "auto"
    slug = level_label.replace("-", "_")
    if not auto_dir.is_dir():
        return []
    return [str(p) for p in sorted(auto_dir.glob(f"{slug}_d*.state"))]


def _state_for_wall(game: str, level_label: str, record: dict) -> str | None:
    """Best drop-in savestate for teaching a stuck wall (auto-capture → teleop → checkpoint)."""
    paths = _auto_states_for_level(level_label)
    if not paths:
        from billy.stuck_trainer import StuckRecord, collect_auto_states
        stub = StuckRecord(level_label=level_label,
                           death_bucket=int(record.get("death_bucket", 0)),
                           captured_states=list(record.get("captured_states") or []))
        paths = collect_auto_states(level_label, stub)
    if paths:
        return max(paths, key=_approach_x_from_path)

    if game == "zelda":
        m = re.search(r"#(\d+)", level_label)
        if m:
            teleop = config.DATA_DIR / "zelda" / "states" / f"teleop_{m.group(1)}.state"
            if teleop.is_file():
                return str(teleop)

    ck_dir = config.CHECKPOINTS_DIR / game
    ck = ck_dir / (level_label.replace("-", "_") + ".state")
    if ck.is_file():
        return str(ck)

    furthest_label, _ = _furthest(game)
    if furthest_label != "start":
        ck2 = ck_dir / (furthest_label.replace("-", "_") + ".state")
        if ck2.is_file():
            return str(ck2)

    if ck_dir.is_dir():
        extras = sorted(p for p in ck_dir.glob("*.state"))
        if extras:
            return str(extras[-1])    # best runway we have (e.g. 2-3 wall → 2_2.state)
    return None


def _discover_walls(games: list[str]) -> list[dict]:
    """When demo_requests.jsonl is empty, surface Billy's real stuck spots from stuck.json.

    The stuck trainer only files requests for a few hard-coded SMB hazards (1-3 lift, 2-2 pit).
    Zelda's stuck_remedy is None. Most walls never reach demo_requests — remix discovers them
    here so the desktop shortcut still has something to teach."""
    stuck_path = config.DATA_DIR / "stuck.json"
    if not stuck_path.is_file():
        return []
    try:
        rows = json.loads(stuck_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    cleared_levels = _read_cleared_levels()
    existing = {(r["game"], r["level_label"]) for r in _read_requests()}
    game_set = set(games)
    candidates: list[dict] = []

    for rec in rows:
        if rec.get("remediated") or int(rec.get("deaths", 0)) < config.STUCK_DEATH_THRESHOLD:
            continue
        level_label = rec.get("level_label", "")
        game = _game_for_level(level_label)
        if game is None or game not in game_set:
            continue
        if not _at_or_beyond_frontier(game, level_label):
            continue
        level_key = (game, level_label)
        if level_key in existing:
            continue
        if level_key in cleared_levels:
            if _has_banked_solution(game, level_label):
                continue                    # teach worked — solution is in cache
            if not _still_stuck_on_level(game, level_label):
                continue                    # cleared and Billy moved past it
        state = _state_for_wall(game, level_label, rec)
        if not state:
            continue
        candidates.append({
            "game": game,
            "level_label": level_label,
            "death_bucket": int(rec.get("death_bucket", 0)),
            "death_x": int(rec.get("last_death_x", 0)),
            "deaths": int(rec.get("deaths", 0)),
            "state": state,
            "source": "stuck.json",
        })

    # One wall per game — the level Billy has died at most on (his real next wall).
    by_game: dict[str, dict] = {}
    for c in _dedupe_walls_by_level(candidates):
        g = c["game"]
        if g not in by_game or c["deaths"] > by_game[g]["deaths"]:
            by_game[g] = c
    return list(by_game.values())


def _open_walls(games: list[str], *, persist: bool = True) -> list[dict]:
    """Open teach queue: explicit demo requests, plus stuck.json discovery when empty."""
    game_set = set(games)
    reqs = _dedupe_walls_by_level(
        [r for r in _read_requests() if r.get("game") in game_set])
    if reqs:
        return reqs

    discovered = _discover_walls(games)
    if not discovered:
        return []

    if persist:
        config.ensure_dirs()
        with REQUESTS_FILE.open("a") as f:
            for r in discovered:
                f.write(json.dumps(r) + "\n")
        print(f"  [remix] queued {len(discovered)} stuck wall(s) from Billy's death log "
              f"(demo_requests.jsonl was empty).")
    return discovered


def _resolve_request(req: dict) -> None:
    """Drop a taught wall from the open queue and log it as cleared (Billy owns it now)."""
    game, label = req.get("game"), req.get("level_label")
    kept = [r for r in _read_requests()
            if not (r.get("game") == game and r.get("level_label") == label)]
    REQUESTS_FILE.write_text("".join(json.dumps(r) + "\n" for r in kept))
    config.ensure_dirs()
    if CLEARED_FILE.is_file() and CLEARED_FILE.stat().st_size:
        raw = CLEARED_FILE.read_bytes()
        if raw[-1:] != b"\n":
            with CLEARED_FILE.open("ab") as f:
                f.write(b"\n")
    with CLEARED_FILE.open("a") as f:
        f.write(json.dumps(req) + "\n")


def _game(name: str):
    from run import GAMES
    return GAMES[name]()


def _approach_x_from_path(p: str) -> int:
    stem = Path(p).stem
    try:
        return int(stem.rsplit("_x", 1)[1])
    except (IndexError, ValueError):
        return 0


def _prepare_smb_approach(req: dict) -> str | None:
    """Headless capture at the real wall (e.g. 2-3 pit) using Billy's cache from his frontier."""
    label = req.get("level_label", "")
    death_x = int(req.get("death_x", 0))
    slug = label.replace("-", "_")
    auto_dir = config.DATA_DIR / "rl" / "states" / "auto"
    matches = _auto_states_for_level(label)
    if matches:
        return max(matches, key=_approach_x_from_path)

    bucket = death_x // config.CACHE_BUCKET_PX
    capture_out = str(auto_dir / f"{slug}_d{bucket}_remix.state")
    if Path(capture_out).is_file():
        return capture_out

    x_min = max(32, death_x - 100)
    x_max = max(x_min + 16, death_x - 8)
    print(f"     … driving Billy to {label} x≈{death_x} for a proper drop-in "
          f"(headless from his frontier — may take 1–2 min)…")
    os.environ["BILLY_HEADLESS"] = "1"
    os.environ.setdefault("BILLY_TURBO", "1")
    from billy.director import Director
    from billy.knowledge import KnowledgeBase, SkillLibrary
    from billy.rl.section_policy import SectionController, default_smb_sections

    auto_dir.mkdir(parents=True, exist_ok=True)
    game = _game("smb")
    game.cli_name = "smb"
    director = Director(
        game, KnowledgeBase(), use_llm=False, skills=SkillLibrary(),
        sections=SectionController(default_smb_sections()))
    director.session.reset()
    director.session.wait_until_live()
    grabbed: dict = {}
    grabbing = False
    orig_observe = director._observe

    def _try_grab(obs) -> None:
        nonlocal grabbing
        if grabbed or grabbing or obs.level_label != label:
            return
        if not (getattr(obs.raw, "on_ground", True) and x_min <= obs.progress <= x_max):
            return
        grabbing = True
        try:
            from billy.games.smb.capture_util import settle_mario
            ok, _ = settle_mario(director.session, orig_observe, allow_left=False)
            obs2 = orig_observe()
            if ok and obs2.level_label == label and x_min <= obs2.progress <= x_max:
                grabbed["bytes"] = director.session.clone_state()
                grabbed["x"] = obs2.progress
        finally:
            grabbing = False

    def hooked_observe():
        obs = orig_observe()
        _try_grab(obs)
        return obs

    director._observe = hooked_observe
    try:
        director.boot()
        try:
            director.resume_from_checkpoint()
        except Exception:
            pass
        os.environ.setdefault("BILLY_MAX_FRAMES", "120000")
        director.run_attempt(1)
    finally:
        director._observe = orig_observe
        if getattr(director, "pool", None) is not None:
            director.pool.close()
        director.session.close()
    os.environ["BILLY_HEADLESS"] = "0"
    if grabbed:
        Path(capture_out).write_bytes(grabbed["bytes"])
        print(f"     … approach saved {label} x={grabbed['x']} ({capture_out})")
        return capture_out
    print(f"     ⚠  approach capture missed — can't teach {label} without a savestate there.")
    return None


def _is_captured_approach(path: Path) -> bool:
    """Remix/teleop/auto-captured states are meant to drop you AT the wall, not level start."""
    s = str(path)
    return "/auto/" in s or path.stem.endswith("_remix") or "teleop" in path.stem


def _dropin_state(req: dict) -> tuple[bytes | None, str]:
    """Where to drop the human in.

    Captured approach states (auto/*_remix, teleop_*) go first — Billy drove there for this
    wall. Level-start checkpoints win otherwise (stops tight approach momentum auto-winning)."""
    game, label = req.get("game", ""), req.get("level_label", "")
    p = Path(req.get("state", ""))
    if p.is_file() and _is_captured_approach(p):
        return p.read_bytes(), f"approach at {label}"
    ck = config.CHECKPOINTS_DIR / game / (label.replace("-", "_") + ".state")
    if ck.is_file():
        return ck.read_bytes(), f"start of {label}"
    if p.is_file():
        return p.read_bytes(), "approach"
    return None, ""


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
    req = dict(req)
    if game_key == "smb":
        approach = _prepare_smb_approach(req)
        if not approach:
            print(f"\n  ⚠  smb {req.get('level_label')}: no approach state at the real wall — "
                  f"skipping (won't drop you in the wrong level).")
            return False
        req["state"] = approach
    death_x = int(req.get("death_x", 0))
    label = req.get("level_label", "?")
    dropin, source = _dropin_state(req)
    goal = _goal_blurb(game_key, req)
    wall_at = (f"screen {label}" if game_key == "zelda"
               else f"x≈{death_x}")
    print(f"\n{'═' * 64}\n  ▶  {game_key} · {label}  —  Billy's wall at {wall_at} "
          f"({req.get('deaths', '?')} deaths)\n     GOAL: {goal} — "
          f"clear it and Billy learns the line.  (drop-in: {source})\n{'═' * 64}")
    if dropin is None:
        print("     ⚠  no drop-in state available — skipping.")
        return False

    game = _game(game_key)
    session = game.system.connect()
    try:
        session.wait_until_live()
        session.restore(dropin)
        drop_obs = _observe(session, game)
        if game_key == "smb" and drop_obs.level_label != label:
            print(f"     ⚠  drop-in is {drop_obs.level_label}, not {label} — skipping.")
            return False
        if game_key == "smb":
            print(f"     📍 drop-in confirmed: {drop_obs.level_label} x={drop_obs.progress}")
        if not session.ensure_viewer():
            print("     ⚠  no game window (headless?). The remix needs a window to play in.")
            return False
        pad = session.reopen_joystick()
        print("     🎮 controller ready." if pad
              else "     ⌨  keyboard (arrows / Z / X / Tab).")
        print("     👉 Click the GAME window, then press an arrow (or Z) to take control.")
        if game_key == "zelda":
            print("     🗡  Fight through the screen and walk EAST to the next screen alive.")

        for t in range(1, 4):
            if t > 1:
                print(f"     try {t}/3…")
            session.restore(dropin)
            outcome, secs, plan, start_state, start_obs = _play_wall(
                session, game, game_key, req, death_x)
            if outcome == "win":
                print(f"     ✓ you cleared it in {secs:.1f}s")
                params = _teach_params(game_key, req, start_obs)
                return _bank(game_key, session, start_state, start_obs, plan, req=req,
                               min_progress=params["min_progress"])
            if outcome == "skip":
                print("     ↷ skipped.")
                return False
            if outcome == "afk":
                print("     ⏳ no input — you didn't take the controller. Skipping this wall.")
                return False
            if outcome == "bad_dropin":
                print("     ⚠  this drop-in coasts past the wall on its own — can't teach it "
                      "cleanly. Skipping (needs an earlier start state).")
                return False
            if outcome == "early_finish":
                print(f"     ✗ ENTER pressed too soon ({secs:.1f}s) — cross the wall first, "
                      f"then ENTER (Zelda: march east to the next screen alive).")
                continue
            print(f"     ✗ {outcome} at {secs:.1f}s — try again.")
        print("     (out of tries — the wall stays open for next time.)")
        return False
    except BootError as e:
        print(f"     ⚠  couldn't start ({e}).")
        return False
    finally:
        session.close()


def _play_wall(session, game, game_key: str, req: dict, death_x: int):
    """Hand the controller to the human, then record their crossing. Returns
    (outcome, seconds, plan, armed_start_state, armed_start_obs). Nothing counts until YOU take
    control (fixes auto-completing on the drop-in's momentum): the demo begins the moment you
    first press something, re-anchored to that state."""
    from billy.teleop import TeleopRecorder
    session.teleop_reset()
    goal = _goal_blurb(game_key, req)

    # 1) Wait for you to take the controller. The window stays live via neutral steps, but the
    #    goal is NOT armed and nothing is recorded — so the drop-in's coast can't auto-win.
    session.set_overlay(["YOUR TURN — take the controller", goal,
                         "move / press a button to begin   ·   ESC = skip"])
    armed = False
    for _ in range(60 * 30):                       # up to 30s to take control
        mask, finish, abort = session.teleop_poll()
        if abort:
            return "skip", 0.0, [], None, None
        session.teleop_step(mask)                  # your input from the very first frame
        if mask:                                   # ENTER alone must not arm (was instant-win)
            armed = True
            break
    if not armed:
        return "afk", 0.0, [], None, None

    # 2) Re-anchor: the demo is what YOU do from here. If the drop-in already coasted past the
    #    wall before you took control, it can't teach the crossing — bail.
    start_state = session.clone_state()
    start_obs = _observe(session, game)
    params = _teach_params(game_key, req, start_obs)
    if params["mode"] == "zelda_screen":
        if int(getattr(start_obs.raw, "map_location", 0)) >= params["target_screen"]:
            return "bad_dropin", 0.0, [], None, None
    elif start_obs.progress >= death_x:
        return "bad_dropin", 0.0, [], None, None

    rec = TeleopRecorder()
    frames, budget, obs = 0, 90 * 60, start_obs
    while frames < budget:
        mask, finish, abort = session.teleop_poll()
        if abort:
            return "skip", frames / 60.0, [], None, None
        session.teleop_step(mask)
        rec.record(mask, 1)
        frames += 1
        obs = _observe(session, game)
        if obs.dead:
            return "died", frames / 60.0, [], None, None
        if _wall_cleared(game_key, obs, params):
            return "win", frames / 60.0, rec.plan(), start_state, start_obs
        if finish:
            return "early_finish", frames / 60.0, [], None, None
        if frames % 15 == 0:
            if params["mode"] == "zelda_screen":
                s = obs.raw
                session.set_overlay([f"▶ {goal}",
                                     f"screen #{s.map_location}  →  need #{params['target_screen']}",
                                     "cross east alive · ESC = skip"])
            else:
                session.set_overlay([f"▶ {goal}",
                                     params["overlay_need"],
                                     "controller or keys · ESC = skip"])
    return "timeout", frames / 60.0, [], None, None


def _bank(game_key: str, session, start_state: bytes, start_obs, plan, *,
          req: dict | None = None, min_progress: int = 8) -> bool:
    """Verify the human demo on the SAME emulator (search_mode, invisible) and bank it: a
    SolutionCache entry (exact replay) + a distilled transferable skill."""
    from billy.config import SOLUTIONS_FILE
    from billy.knowledge.cache import SolutionCache
    from billy.teleop import bank_demo, verify_demo

    if not plan:
        return False
    result = verify_demo(session, _game(game_key), start_state, plan,
                         min_progress=min_progress)
    print(f"     verify: {result.summary()}")
    want = (req or {}).get("level_label", "")
    if game_key == "smb" and want and result.end_label != want:
        print(f"     (demo ended on {result.end_label}, not {want} — won't bank for this wall.)")
        return False
    if not result.bankable:
        print("     (run didn't verify to bank — survive AND advance past the spot; try again.)")
        return False
    cache = SolutionCache(path=SOLUTIONS_FILE)
    key = bank_demo(cache, start_obs, plan, result.end_progress)
    print(f"     ⇒ 🧠 Billy learned it — banked at {key} (reach {result.end_progress}).")
    try:
        from billy.knowledge.distill import distill_solution
        from billy.knowledge.skills import SkillLibrary
        fresh = _game(game_key)
        if distill_solution(SkillLibrary(), summary=start_obs.summary,
                            level_label=start_obs.level_label, plan=plan,
                            start_x=start_obs.progress, reach=result.end_progress,
                            source="demo", console=getattr(fresh.system, "name", "nes")):
            print("     ⇒ distilled a transferable skill (helps at similar spots, other games too).")
    except Exception as e:
        print(f"     (skill distill skipped: {type(e).__name__}: {e})")
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
        walls = _open_walls(games, persist=False)
        for g in games:
            label, prog = _furthest(g)
            g_walls = [r for r in walls if r.get("game") == g]
            wtxt = ", ".join(f"{w['level_label']}@{w['death_x']}" for w in g_walls) or "—"
            print(f"    {g:<9} at {label:<8} (goal {_ENDING.get(g,'?')})  "
                  f"— {len(g_walls)} open wall(s): {wtxt}")
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
    walls = _open_walls(games)
    if not walls:
        print("\n  No open walls right now — Billy is cruising, or every stuck spot lacks a "
              "savestate to drop you in. Run with more --scout-attempts, or capture a state "
              "with teleop.py / teleop_zelda.py capture.")
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
