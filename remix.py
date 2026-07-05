#!/usr/bin/env python3
"""Billy Mitchell REMIX — a NES-Remix-style gauntlet of short, varied challenges.

Each challenge is a tiny goal on one game ("clear 1-1", "survive the swarm", "cross the
pit", "wander out of Paseo"). Billy attempts it under a frame budget; clearing it earns a
medal by how well he did. The engine's whole learning stack runs underneath, and cache /
tapes / skills persist to disk — so **Billy gets greater as he plays**: a solution banked in
one challenge (even a skill distilled in another game) carries into the next.

    .venv/bin/python remix.py                 # run the whole gauntlet (watchable)
    .venv/bin/python remix.py --only smb_11   # one challenge
    .venv/bin/python remix.py --list          # show the card

When a challenge is run in a WINDOW and Billy can't crack it on his own, it drops into a
DEMO ROUND: press and hold **T** in the game window to take the controller and show him the
line, then let go — the takeover banks a cache entry + tape + skill, and the medal is his to
keep on the retry. (Headless runs skip the demo round.)
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass, field

from billy import config
from billy.abstractions import BootError
from billy.director import Director
from billy.knowledge import KnowledgeBase, SkillLibrary

PROGRESS_FILE = config.DATA_DIR / "remix_progress.json"
_STAR = {"gold": 3, "silver": 2, "bronze": 1, "none": 0}
_MEDAL_ICON = {"gold": "🥇", "silver": "🥈", "bronze": "🥉", "none": "⬜"}


@dataclass
class Challenge:
    id: str
    title: str
    game: str                    # run.py GAMES key
    blurb: str
    kind: str                    # "clear" (fastest time wins) | "reach" (farther/longer wins)
    medals: tuple                # clear: (gold_s, silver_s); reach: (gold, silver, bronze)
    unit: str = "px"
    attempts: int = 3
    env: dict = field(default_factory=dict)
    demo_hint: str = ""

    # -- scoring ---------------------------------------------------------------------------
    def attempt_value(self, r) -> float | None:
        """The headline number for one attempt, or None if the goal wasn't met."""
        if self.kind == "clear":
            return r.fastest_clear_frames / 60.0 if (
                r.outcome == "clear" and r.fastest_clear_frames) else None
        best = max(r.max_x, r.level_frontier)          # progress / survival
        return float(best) if best >= self.medals[-1] else None

    def better(self, a: float | None, b: float | None) -> float | None:
        """Fold two attempt values, keeping the better one (min time / max reach)."""
        vals = [v for v in (a, b) if v is not None]
        if not vals:
            return None
        return min(vals) if self.kind == "clear" else max(vals)

    def medal(self, value: float | None) -> str:
        if value is None:
            return "none"
        if self.kind == "clear":
            g, s = self.medals
            return "gold" if value <= g else "silver" if value <= s else "bronze"
        g, s, b = self.medals
        return "gold" if value >= g else "silver" if value >= s else "bronze" if value >= b else "none"

    def render_value(self, value: float | None) -> str:
        if value is None:
            return "—"
        return f"{value:.1f}s" if self.kind == "clear" else f"{int(value)}{self.unit}"


# ---------------------------------------------------------------------------------------------
# The card. Ordered easy → spicy. Every game here plays with the ROMs already imported.
# ---------------------------------------------------------------------------------------------
CHALLENGES: list[Challenge] = [
    # Thresholds are tuned so a strong solo run lands SILVER — gold is reserved for genuine
    # mastery (a record clear, crossing the wall, reaching the next area), the target a demo or
    # more banked learning unlocks.
    Challenge("smb_11", "World 1-1 Sprint", "smb",
              "Clear Super Mario Bros 1-1. The faster, the shinier the medal.",
              kind="clear", medals=(38, 48), attempts=3,
              env={"BILLY_REPEAT_LEVEL": "1", "BILLY_MAX_FRAMES": "9000"},
              demo_hint="Run right, jump the pits and Goombas, hit the flagpole."),
    Challenge("shmup_survive", "Airstriker: Hold the Line", "shmup",
              "Survive the shooter as long as you can — dodge, weave, keep firing.",
              kind="reach", medals=(320, 230, 150), unit="",
              attempts=3, env={"BILLY_MAX_FRAMES": "6000"},
              demo_hint="Strafe under the gaps in the falling fire; never stop shooting."),
    Challenge("pixel_pit", "Pixel Pioneer", "pixel",
              "Play Mario FROM PIXELS ALONE (no RAM) and push as far right as you can.",
              kind="reach", medals=(1500, 1000, 600), unit="px",
              attempts=3, env={"BILLY_MAX_FRAMES": "8000"},
              demo_hint="Time a running jump at the pit's edge (x≈1120)."),
    Challenge("smb2j_start", "Lost Levels: First Steps", "smb_lost",
              "SMB2-Japan is brutal. Get as deep a run off the line as you can.",
              kind="reach", medals=(4000, 2000, 900), unit="px",
              attempts=3, env={"BILLY_MAX_FRAMES": "9000"},
              demo_hint="Same moves as SMB, but the jumps are meaner — commit."),
    Challenge("zelda_start", "Zelda: Leave the Cave", "zelda",
              "Get Link moving through the overworld and banking ground.",
              kind="reach", medals=(5000, 2500, 1000), unit="",
              attempts=3, env={"BILLY_MAX_FRAMES": "9000"},
              demo_hint="Grab the sword, then push through a screen edge."),
    Challenge("psii_wander", "Paseo Wanderer", "psii",
              "Phantasy Star II — explore Paseo, then reach the Mota overworld.",
              kind="reach", medals=(2500, 1000, 300), unit="",
              attempts=2, env={"BILLY_MAX_FRAMES": "14000"},
              demo_hint="Walk the streets to the town edge and out onto Mota."),
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


@contextlib.contextmanager
def _env(overrides: dict):
    prev = {k: os.environ.get(k) for k in overrides}
    os.environ.update({k: str(v) for k, v in overrides.items()})
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _build_director(game_key: str, skills: SkillLibrary) -> Director:
    from run import GAMES
    game = GAMES[game_key]()
    game.cli_name = game_key
    # No LLM, no guide ingestion: keep the gauntlet fast + offline. The reflex/cache/search/
    # tape/skill stack (the parts that compound) all run regardless.
    return Director(game, KnowledgeBase(), use_llm=False, skills=skills, guide=None)


def _run_challenge(ch: Challenge, skills: SkillLibrary, allow_demo: bool) -> tuple[str, float | None]:
    print(f"\n{'═' * 68}\n  ▶  {ch.title}  ·  {ch.game}\n     {ch.blurb}\n{'═' * 68}")
    best: float | None = None
    with _env(ch.env):
        try:
            director = _build_director(ch.game, skills)
            director.session.reset()            # same bring-up run_session does before attempts
            director.session.wait_until_live()
            director.boot()
        except (BootError, FileNotFoundError) as e:
            print(f"     ⚠  unavailable ({type(e).__name__}: {e}) — skipping.")
            return "none", None

        try:
            total = ch.attempts + (1 if allow_demo else 0)
            for n in range(1, total + 1):
                demo_round = n > ch.attempts
                # only spend the demo round if he hasn't already passed on his own
                if demo_round and best is not None:
                    break
                if demo_round:
                    print(f"\n     🎬 DEMO ROUND — Billy is stuck. HOLD **T** in the game window "
                          f"to take over and show him:\n        “{ch.demo_hint}”\n        …then "
                          f"let go. He learns from what you do.")
                try:
                    r = director.run_attempt(n)
                except BootError as e:
                    print(f"     ⚠  attempt failed ({e}) — skipping.")
                    break
                val = ch.attempt_value(r)
                best = ch.better(best, val)
                tag = "✅ GOAL" if val is not None else "·"
                print(f"     attempt {n}{' (demo)' if demo_round else ''}: "
                      f"{r.outcome} · reached {r.world_stage} · best so far "
                      f"{ch.render_value(best)}  {tag}")
                if best is not None and ch.medal(best) == "gold":
                    break                    # already maxed — no need to grind further
        finally:
            if getattr(director, "pool", None) is not None:
                director.pool.close()
    return ch.medal(best), best


def _scoreboard(prog: dict) -> None:
    print(f"\n{'━' * 68}\n  BILLY MITCHELL — REMIX SCOREBOARD\n{'━' * 68}")
    print(f"  {'medal':<6} {'challenge':<26} {'game':<9} best")
    stars = 0
    for ch in CHALLENGES:
        best = prog.get(ch.id, {}).get("best")
        medal = ch.medal(best)                 # derive from best: thresholds are the source of truth
        stars += _STAR[medal]
        val = ch.render_value(best) if best is not None else "—"
        print(f"  {_MEDAL_ICON[medal]:<5} {ch.title:<26} {ch.game:<9} {val}")
    max_stars = 3 * len(CHALLENGES)
    filled = round(12 * stars / max_stars) if max_stars else 0
    bar = "█" * filled + "░" * (12 - filled)
    print(f"{'━' * 68}\n  MASTERY  [{bar}]  {stars}/{max_stars} stars")
    if stars == max_stars:
        print("  👑 PERFECT RUN. Billy is, and has always been, the greatest of all time.")
    elif stars >= max_stars * 0.6:
        print("  🔥 Billy is on a tear. A few lines left to master.")
    else:
        print("  🎮 The gauntlet awaits. Teach him the hard lines with a demo (hold T).")
    print("━" * 68)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Billy Mitchell Remix — a gauntlet of short game challenges.")
    p.add_argument("--only", default="", help="comma-separated challenge ids (see --list)")
    p.add_argument("--list", action="store_true", help="show the challenge card and exit")
    p.add_argument("--no-demo", action="store_true", help="never drop into a demo round on failure")
    args = p.parse_args(argv)

    if args.list:
        print("  Billy Mitchell Remix — challenges:")
        for c in CHALLENGES:
            print(f"    {c.id:<14} {c.title:<26} [{c.game}] — {c.blurb}")
        return 0

    chosen = CHALLENGES
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        chosen = [c for c in CHALLENGES if c.id in want]
        if not chosen:
            print(f"[remix] no matching challenges in {sorted(want)} — see --list.")
            return 1

    # A window means demos are possible; headless/turbo benchmarks skip the demo round.
    windowed = os.environ.get("BILLY_HEADLESS", "0") != "1"
    allow_demo = windowed and not args.no_demo

    config.ensure_dirs()
    skills = SkillLibrary()          # shared across the gauntlet → cross-game carry-forward
    prog = _load_progress()

    print("🎮  BILLY MITCHELL — REMIX")
    print(f"    {len(chosen)} challenge(s) · Billy keeps everything he learns.\n")

    for ch in chosen:
        medal, best = _run_challenge(ch, skills, allow_demo)
        prev = prog.get(ch.id, {})
        prev_best = prev.get("best")
        # keep the best result ever seen for this challenge
        merged = ch.better(prev_best, best) if best is not None else prev_best
        prog[ch.id] = {                        # store best only; medal is derived from thresholds
            "best": merged,
            "updated": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        _save_progress(prog)
        print(f"     ⇒ {_MEDAL_ICON[ch.medal(merged)]}  {ch.title}: {ch.render_value(merged)}")

    _scoreboard(prog)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[remix] halted — Billy demands a rematch.")
        sys.exit(130)
