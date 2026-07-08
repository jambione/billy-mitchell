#!/usr/bin/env python3
"""Billy Mitchell REMIX — teach Billy past his next wall, across games, until he finishes.

This is a HANDS-ON, DYNAMIC, MULTI-GAME gauntlet. It doesn't hand you a fixed card of tricks —
it surfaces the walls Billy is ACTUALLY stuck at right now, one per game, and drops you in to
teach each one. Your run banks as a demo (cache entry + entry-anchored tape + skill) and then —
the payoff — Billy REPLAYS your exact line back in the window so you watch him nail the spot he
kept dying at. Next time he gets further before hitting the NEXT wall. The goals never get
artificially harder — the only thing that moves is Billy, forward through each game, to the end.

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
# Back-compat aliases — defaults live on Game.remix_past_margin() / remix_min_progress().
PAST_MARGIN = 24
ZELDA_MIN_PROGRESS = 48
MAX_WALLS_PER_GAME = 6           # per-session cap on the teach→re-scout→next-wall march

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


def _record_game(rec: dict) -> str | None:
    """Campaign game key for a stuck.json row — explicit game field wins over label heuristics."""
    if rec.get("game"):
        return rec["game"]
    return _game_for_level(rec.get("level_label", ""))


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
        if _record_game(rec) != game:
            continue
        if rec.get("level_label") != level_label or rec.get("remediated"):
            continue
        best = max(best, int(rec.get("deaths", 0)))
    return best >= _stuck_threshold(game)


def _dedupe_walls_by_level(walls: list[dict]) -> list[dict]:
    """One teach session per (game, level) — not one per x-bucket on the same Zelda screen."""
    best: dict[tuple[str, str], dict] = {}
    for w in walls:
        key = (w.get("game", ""), w.get("level_label", ""))
        if key not in best or int(w.get("deaths", 0)) > int(best[key].get("deaths", 0)):
            best[key] = w
    return list(best.values())


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
        level_label = rec.get("level_label", "")
        game = _record_game(rec)
        if game is None or game not in game_set:
            continue
        if rec.get("remediated") or int(rec.get("deaths", 0)) < _stuck_threshold(game):
            continue
        if not _at_or_beyond_frontier(game, level_label):
            continue
        level_key = (game, level_label)
        if level_key in existing:
            continue
        if level_key in cleared_levels:
            # Taught before. Only re-queue if Billy is dying here AGAIN (fresh streak after
            # remediate was cleared by new deaths). Stale historical counts must not re-offer.
            if not _still_stuck_on_level(game, level_label):
                continue
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


def _wall_is_stale(req: dict) -> bool:
    """True when this wall was already taught and Billy is not stuck there again.

    Prevents demo_requests.jsonl leftovers (or re-appends from old death history) from
    re-offering a wall the human already banked."""
    game, label = req.get("game"), req.get("level_label")
    if not game or not label:
        return False
    if (game, label) not in _read_cleared_levels():
        return False
    return not _still_stuck_on_level(game, label)


def _open_walls(games: list[str], *, persist: bool = True) -> list[dict]:
    """Open teach queue: explicit demo requests, plus stuck.json discovery per game."""
    game_set = set(games)
    raw_reqs = [r for r in _read_requests() if r.get("game") in game_set]
    stale = [r for r in raw_reqs if _wall_is_stale(r)]
    reqs = _dedupe_walls_by_level([r for r in raw_reqs if not _wall_is_stale(r)])
    if persist and stale:
        # Drop taught walls from the on-disk queue so the next launch doesn't re-offer them.
        kept = [r for r in _read_requests() if not _wall_is_stale(r)]
        REQUESTS_FILE.write_text("".join(json.dumps(r) + "\n" for r in kept))
        print(f"  [remix] dropped {len(stale)} already-taught wall(s) from the queue "
              f"(Billy owns those lines; re-queue only if he dies there again).")
    covered = {r.get("game") for r in reqs}
    need_discovery = [g for g in games if g not in covered]
    discovered = _discover_walls(need_discovery) if need_discovery else []
    walls = reqs + discovered
    if not walls:
        return []

    if persist and discovered:
        config.ensure_dirs()
        from billy.stuck_trainer import StuckRecord, StuckRemedy, notify_remix_wall
        with REQUESTS_FILE.open("a") as f:
            for r in discovered:
                f.write(json.dumps(r) + "\n")
                notify_remix_wall(
                    r["game"],
                    StuckRemedy(kind="frame_search", level_label=r["level_label"],
                                death_x=int(r["death_x"]), goal_x=int(r["death_x"]) + 100),
                    StuckRecord(level_label=r["level_label"],
                                death_bucket=int(r["death_bucket"]),
                                deaths=int(r["deaths"]), last_death_x=int(r["death_x"])))
        print(f"  [remix] queued {len(discovered)} stuck wall(s) from Billy's death log.")
    return walls


def _resolve_request(req: dict) -> None:
    """Drop a taught wall from the open queue, log it cleared, and mark stuck remediated.

    Marking the whole level remediated zeros historical death counts so discovery does not
    immediately re-queue the same wall. New deaths after this clear `remediated` and can
    re-open the wall if Billy is truly stuck again."""
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
    if game and label:
        try:
            from billy.stuck_trainer import StuckTracker
            n = StuckTracker().mark_level_remediated(str(game), str(label))
            if n:
                print(f"     ⇒ stuck log: marked {label} remediated ({n} bucket(s)) — "
                      f"won't re-queue until Billy dies there again.")
        except Exception as e:
            print(f"     (stuck remediate skipped: {type(e).__name__}: {e})")


def _game(name: str):
    from run import GAMES
    return GAMES[name]()


def _approach_x_from_path(p: str) -> int:
    stem = Path(p).stem
    try:
        return int(stem.rsplit("_x", 1)[1])
    except (IndexError, ValueError):
        return 0


def _stuck_threshold(game_key: str) -> int:
    return _game(game_key).stuck_death_threshold()


def _prepare_approach(req: dict, game) -> str | None:
    """Headless capture just-before-the-wall using Billy's cache from his frontier."""
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

    if not game.remix_needs_approach_capture(req):
        return None

    x_min, x_max = game.remix_approach_progress_window(req)
    print(f"     … driving Billy to {label} for a proper drop-in "
          f"(headless from his frontier — may take 1–2 min)…")
    os.environ["BILLY_HEADLESS"] = "1"
    os.environ.setdefault("BILLY_TURBO", "1")
    from billy.director import Director
    from billy.knowledge import KnowledgeBase, SkillLibrary

    auto_dir.mkdir(parents=True, exist_ok=True)
    sections = game.remix_director_sections()
    director = Director(
        game, KnowledgeBase(), use_llm=False, skills=SkillLibrary(),
        sections=sections) if sections else Director(
        game, KnowledgeBase(), use_llm=False, skills=SkillLibrary())
    director.session.reset()
    director.session.wait_until_live()
    grabbed: dict = {}
    grabbing = False
    orig_observe = director._observe

    def _try_grab(obs) -> None:
        nonlocal grabbing
        if grabbed or grabbing or not game.remix_dropin_level_ok(obs, req):
            return
        lo, hi = game.remix_approach_progress_window(req)
        if not (game.remix_on_ground(obs) and lo <= obs.progress <= hi):
            return
        grabbing = True
        try:
            ok, obs2 = game.remix_capture_ready(director.session, orig_observe, req)
            if ok:
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
def _close_session(session) -> None:
    """Tear down an emulator session (and its viewer). Idempotent; never raises."""
    if session is None:
        return
    try:
        session.close()
    except Exception:
        pass


def _open_teach_session(game_key: str):
    """Open ONE windowed session for teaching. Caller owns close via _close_session."""
    os.environ["BILLY_HEADLESS"] = "0"
    os.environ.setdefault("BILLY_TURBO", "1")
    game_obj = _game(game_key)
    game_obj.cli_name = game_key
    session = game_obj.system.connect()
    session.wait_until_live()
    return session


def _approach_needs_live_drive(req: dict) -> bool:
    """True only when approach capture would open a headless Director (no cache hit)."""
    game = _game(req["game"])
    if not game.remix_needs_approach_capture(req):
        return False
    state = Path(str(req.get("state", "")))
    if state.is_file() and _is_captured_approach(state):
        return False
    label = req.get("level_label", "")
    if _auto_states_for_level(label):
        return False
    death_x = int(req.get("death_x", 0))
    slug = label.replace("-", "_")
    bucket = death_x // config.CACHE_BUCKET_PX
    capture_out = config.DATA_DIR / "rl" / "states" / "auto" / f"{slug}_d{bucket}_remix.state"
    return not capture_out.is_file()


def _attach_approach_state(req: dict) -> dict | None:
    """Return a copy of `req` with approach drop-in attached when the game needs one.

    Runs headless Director capture if no cached approach exists — MUST be called with no other
    emulator open (stable-retro: one instance per process). Returns None if capture is required
    and fails."""
    out = dict(req)
    game_key = out["game"]
    game = _game(game_key)
    game.cli_name = game_key
    if not game.remix_needs_approach_capture(out):
        return out
    if Path(str(out.get("state", ""))).is_file() and _is_captured_approach(Path(out["state"])):
        return out
    approach = _prepare_approach(out, game)
    if not approach:
        print(f"\n  ⚠  {game_key} {out.get('level_label')}: no approach state at the "
              f"real wall — skipping (won't drop you in the wrong level).")
        return None
    out["state"] = approach
    return out


def _teach_wall(req: dict, *, session=None) -> bool:
    """One wall: restore its state, YOU play past Billy's death spot, verify + bank. True if
    banked (Billy learned it). Pass a live `session` to keep one game window across walls.

    When `session` is provided, approach capture must already be attached on `req` (see
    `_attach_approach_state`) — never opens a second emulator while the teach window is live."""
    os.environ["BILLY_HEADLESS"] = "0"
    os.environ.setdefault("BILLY_TURBO", "1")
    from billy.abstractions import BootError

    game_key = req["game"]
    req = dict(req)
    game = _game(game_key)
    game.cli_name = game_key
    owns_session = session is None
    if game.remix_needs_approach_capture(req):
        # Prefer caller-prepped state. Live drive only when we own the only emulator slot.
        has_approach = (Path(str(req.get("state", ""))).is_file()
                        and _is_captured_approach(Path(req["state"])))
        if not has_approach:
            if not owns_session:
                print(f"\n  ⚠  {game_key} {req.get('level_label')}: approach capture needed "
                      f"but teach window is already open — skipping.")
                return False
            approach = _prepare_approach(req, game)
            if not approach:
                print(f"\n  ⚠  {game_key} {req.get('level_label')}: no approach state at the "
                      f"real wall — skipping (won't drop you in the wrong level).")
                return False
            req["state"] = approach
    death_x = int(req.get("death_x", 0))
    label = req.get("level_label", "?")
    dropin, source = _dropin_state(req)
    goal = game.remix_goal(req)
    wall_at = game.remix_wall_at(req)
    print(f"\n{'═' * 64}\n  ▶  {game_key} · {label}  —  Billy's wall at {wall_at} "
          f"({req.get('deaths', '?')} deaths)\n     GOAL: {goal} — "
          f"clear it and Billy learns the line.  (drop-in: {source})\n{'═' * 64}")
    if dropin is None:
        print("     ⚠  no drop-in state available — skipping.")
        return False
    if owns_session:
        session = game.system.connect()
    try:
        if owns_session:
            session.wait_until_live()
        session.restore(dropin)
        drop_obs = _observe(session, game)
        if not game.remix_dropin_level_ok(drop_obs, req):
            print(f"     ⚠  drop-in is {drop_obs.level_label}, not {label} — skipping.")
            return False
        print(f"     📍 drop-in confirmed: {drop_obs.level_label} "
              f"progress={drop_obs.progress}")
        if not session.ensure_viewer():
            print("     ⚠  no game window (headless?). The remix needs a window to play in.")
            return False
        pad = session.reopen_joystick()
        print("     🎮 controller ready." if pad
              else "     ⌨  keyboard (arrows / Z / X / Tab).")
        print("     👉 Click the GAME window, then press an arrow (or Z) to take control.")
        if not game.remix_dropin_ok(drop_obs, req):
            print("     ⚠  drop-in already past the wall — skipping.")
            return False

        # Bank EVERY clean crossing in the try budget, not just the first. A second, faster or
        # further line replaces the banked one (cache force=True / tape frontier) — Billy keeps
        # your best. The take-control gate is the opt-out: once you've banked, step away and the
        # "afk" timeout ends the wall (a success, not a failure).
        banked = 0
        for t in range(1, 4):
            if t > 1:
                print(f"     try {t}/3 — land a cleaner line (Billy keeps the best), "
                      f"or step away to move on." if banked else f"     try {t}/3…")
            session.restore(dropin)
            outcome, secs, plan, start_state, start_obs = _play_wall(session, game, req)
            if outcome == "win":
                print(f"     ✓ you cleared it in {secs:.1f}s")
                if _bank(game, session, start_state, start_obs, plan, req=req, source=source):
                    banked += 1
                continue
            if outcome == "skip":
                print("     ↷ done with this wall." if banked else "     ↷ skipped.")
                break
            if outcome == "afk":
                if banked:
                    break                     # taught it, then stepped away — done, not a failure
                print("     ⏳ no input — you didn't take the controller. Skipping this wall.")
                return False
            if outcome == "bad_dropin":
                print("     ⚠  this drop-in coasts past the wall on its own — can't teach it "
                      "cleanly. Skipping (needs an earlier start state).")
                return banked > 0
            if outcome == "early_finish":
                print(f"     ✗ ENTER pressed too soon ({secs:.1f}s) — cross the wall first, "
                      f"then ENTER (Zelda: march east to the next screen alive).")
                continue
            print(f"     ✗ {outcome} at {secs:.1f}s — try again.")
        if banked:
            print(f"     🏅 banked {banked} clean crossing(s) — Billy keeps your best line.")
            return True
        print("     (out of tries — the wall stays open for next time.)")
        return False
    except BootError as e:
        print(f"     ⚠  couldn't start ({e}).")
        return False
    finally:
        if owns_session and session is not None:
            session.close()


def _play_wall(session, game, req: dict):
    """Hand the controller to the human, then record their crossing. Returns
    (outcome, seconds, plan, armed_start_state, armed_start_obs). Nothing counts until YOU take
    control (fixes auto-completing on the drop-in's momentum): the demo begins the moment you
    first press something, re-anchored to that state."""
    from billy.teleop import TeleopRecorder
    session.teleop_reset()
    goal = game.remix_goal(req)

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
    if not game.remix_dropin_ok(start_obs, req):
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
        if game.remix_win(obs, req, start_obs):
            return "win", frames / 60.0, rec.plan(), start_state, start_obs
        if finish:
            return "early_finish", frames / 60.0, [], None, None
        if frames % 15 == 0:
            session.set_overlay([f"▶ {goal}",
                                 game.remix_overlay_hint(obs, req),
                                 "controller or keys · ESC = skip"])
    return "timeout", frames / 60.0, [], None, None


def _bank(game, session, start_state: bytes, start_obs, plan, *,
          req: dict | None = None, source: str = "") -> bool:
    """Verify the human demo on the SAME emulator (search_mode, invisible) and bank it: a
    SolutionCache entry (exact replay) + an entry-anchored TAPE (deterministic whole-trajectory
    reproduction, the only carrier that beats a MOVING hazard) + a distilled transferable skill.
    Then replay the line back IN THE WINDOW so you watch Billy run what you just taught."""
    from billy.config import SOLUTIONS_FILE
    from billy.knowledge.cache import SolutionCache
    from billy.teleop import bank_demo, verify_demo

    if not plan:
        return False
    result = verify_demo(session, game, start_state, plan,
                         min_progress=game.remix_min_progress())
    print(f"     verify: {result.summary()}")
    if not game.remix_demo_end_ok(result, req or {}):
        want = (req or {}).get("level_label", "")
        print(f"     (demo ended on {result.end_label}, not {want} — won't bank for this wall.)")
        return False
    if not result.bankable:
        print("     (run didn't verify to bank — survive AND advance past the spot; try again.)")
        return False
    game_key = str(getattr(game, "cli_name", "") or (req or {}).get("game", ""))
    cache = SolutionCache(path=SOLUTIONS_FILE, game_id=game_key or "smb")
    key = bank_demo(cache, start_obs, plan, result.end_progress)
    print(f"     ⇒ 🧠 Billy learned it — banked at {key} (reach {result.end_progress}).")
    _bank_tape(game, start_obs, start_state, plan, result, source)
    _write_bc_seed(game_key, start_obs, start_state, plan, result, game)
    try:
        from billy.knowledge.distill import distill_solution
        from billy.knowledge.skills import SkillLibrary
        if distill_solution(SkillLibrary(), summary=start_obs.summary,
                            level_label=start_obs.level_label, plan=plan,
                            start_x=start_obs.progress, reach=result.end_progress,
                            source="demo", console=getattr(game.system, "name", "nes")):
            print("     ⇒ distilled a transferable skill (helps at similar spots, other games too).")
    except Exception as e:
        print(f"     (skill distill skipped: {type(e).__name__}: {e})")
    _replay_taught_line(session, start_state, plan, game.remix_goal(req or {}))
    return True


def _anchor_is_level_entry(game_key: str, source: str) -> bool:
    """Thin wrapper for tests — delegates to Game.remix_anchor_ok."""
    return _game(game_key).remix_anchor_ok(source)


def _bank_tape(game, start_obs, entry_state: bytes, plan, result, source: str) -> bool:
    """Bank the crossing as an entry-state-anchored tape when the anchor is a legitimate
    level/screen entry. This is the carrier that reproduces MOVING hazards deterministically — a
    position-keyed cache entry re-searches them each pass; the tape restores the exact state and
    replays the exact stream. Safe either way: the Director clone-VERIFIES a tape before ever
    replaying it, so a bad anchor just fails verify and self-drops. Returns True if banked."""
    if not game.remix_anchor_ok(source):
        return False
    try:
        from billy.knowledge.tape import TapeLibrary
        tapes = TapeLibrary()
        clears = tuple(result.end_level_key) != tuple(start_obs.level_key)
        tapes.put(start_obs.level_key, plan, result.end_progress,
                  clears_level=clears, entry_state=entry_state)
        print("     ⇒ 🎞  banked an entry-anchored tape — reproduces this spot deterministically "
              "(even moving hazards, where a cache entry alone re-searches).")
        return True
    except Exception as e:
        print(f"     (tape bank skipped: {type(e).__name__}: {e})")
        return False


def _replay_taught_line(session, entry_state: bytes, plan, goal: str) -> None:
    """The payoff: restore the exact state you took control from and replay YOUR inputs in the
    window, paced at 60fps (teleop_step forces real-time even under BILLY_TURBO), so you watch
    Billy run the line you just taught. Non-fatal — a windowing hiccup never fails the bank."""
    if not plan:
        return
    try:
        session.restore(entry_state)
        # verify_demo runs invisibly (search_mode); refresh the SAME teach window before replay.
        if hasattr(session, "ensure_viewer"):
            session.ensure_viewer()
        session.set_overlay(["🎬 WATCH BILLY RUN YOUR LINE", goal, "(the line you just taught him)"])
        for step in plan:
            for _ in range(step.frames):
                session.teleop_step(step.buttons)
        session.set_overlay(["🏆 Billy owns this line now", "on to the next wall…"])
        print("     🎬 replayed your line back — that's Billy running what you taught.")
    except Exception as e:
        print(f"     (replay skipped: {type(e).__name__}: {e})")


# SectionEnv (the hazard sub-policy trainer) is SMB-shaped — its action vocabulary and features
# are the platformer's, so the BC warm-start carrier applies to the SMB family.
_BC_GAMES = {"smb", "smb_lost"}


def _write_bc_seed(game_key: str, start_obs, start_state: bytes, plan, result,
                   game=None) -> str | None:
    """The 4th demo carrier: persist the crossing as a behavior-cloning warm start — the exact
    start savestate + its input stream (.demo.json), the two artifacts `train_section.py --demo`
    clones into a hazard sub-policy (one human crossing turns PPO-from-scratch into fine-tuning).
    Remix does NOT run the heavy, offline PPO here; it writes the seed and prints the command."""
    if game_key not in _BC_GAMES:
        return None
    try:
        demos = config.DATA_DIR / "rl" / "demos" / game_key
        demos.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^0-9A-Za-z]+", "_", str(start_obs.level_label)).strip("_")
        stem = f"{slug}_x{int(start_obs.progress)}"
        state_path = demos / f"{stem}.state"
        demo_path = demos / f"{stem}.demo.json"
        state_path.write_bytes(start_state)
        demo_path.write_text(json.dumps({"steps": [[s.frames, s.buttons] for s in plan]}))
        margin = game.remix_past_margin() if game else PAST_MARGIN
        goal_x = max(int(result.end_progress), int(start_obs.progress) + margin)
        print(f"     ⇒ 🤖 BC seed saved ({demo_path.name}). Fine-tune a sub-policy offline:")
        print(f"        .venv/bin/python train_section.py --state {state_path} "
              f"--demo {demo_path} --goal-x {goal_x}")
        return str(demo_path)
    except Exception as e:
        print(f"     (BC seed skipped: {type(e).__name__}: {e})")
        return None


# ---------------------------------------------------------------------------------------------
def _teach_game(game: str, *, scout_attempts: int, rescout: bool,
                teach=_teach_wall, scout=_scout, open_walls=_open_walls,
                resolve=_resolve_request, reuse_window: bool | None = None) -> int:
    """Teach this game's walls one after another — the compounding march in one sitting. After
    each taught line, re-scout so Billy (now owning that line) surfaces his NEXT wall; teach that
    too. Stops when he has no open wall left, when a wall can't be taught, or at the per-game cap.

    Emulator exclusivity (stable-retro: one instance per process):
      • One teach WINDOW is reused across consecutive walls in this game.
      • Approach capture and re-scout need their own headless session — the teach window is
        CLOSED first, then reopened for the next wall. Learning is already banked to disk, so
        releasing the window never drops a taught line.
    Deps are injectable so the loop is unit-testable without an emulator. Returns lines taught."""
    taught = 0
    session = None
    if reuse_window is None:
        reuse_window = teach is _teach_wall
    try:
        for _ in range(MAX_WALLS_PER_GAME):
            walls = [w for w in open_walls([game]) if w.get("game") == game]
            if not walls:
                break
            req = min(walls, key=lambda r: r.get("death_x", 0))
            if reuse_window:
                # Approach capture may open a headless Director — release the teach window first.
                if session is not None and _approach_needs_live_drive(req):
                    _close_session(session)
                    session = None
                prepared = _attach_approach_state(req)
                if prepared is None:
                    break
                req = prepared
                if session is None:
                    session = _open_teach_session(game)
                banked = teach(req, session=session)
            else:
                banked = teach(req)
            if not banked:
                break                   # skipped / couldn't cross — leave it open for next time
            resolve(req)
            taught += 1
            if not rescout:
                # --no-scout: keep teaching walls already on file in the SAME window; don't hunt.
                continue
            # One emulator per process: release the teach window before headless re-scout.
            _close_session(session)
            session = None
            reached = scout(game, scout_attempts)
            print(f"    · after your line, Billy now reaches {reached} in {game} — "
                  f"finding his next wall…")
    finally:
        _close_session(session)
    return taught


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

    print(f"\n  Billy needs help at {len(walls)} wall(s). Take the controller — and after each "
          f"line, he re-scouts to find his NEXT wall, so one sitting marches him forward.")
    taught = 0
    for g in games:
        taught += _teach_game(g, scout_attempts=args.scout_attempts, rescout=not args.no_scout)

    _scoreboard(games, taught)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[remix] halted — Billy demands a rematch.")
        sys.exit(130)
