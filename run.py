#!/usr/bin/env python3
"""Billy Mitchell learns to play games.

The emulator now runs in-process via stable-retro (one-time setup: ./emulator/setup_retro.sh).
    python run.py --attempts 20
    python run.py --attempts 5 --no-llm           # pure reflex run (no Billy/Coach)
    python run.py --game smb_lost --seed-skills    # SMB2-Japan, seeded with SMB transfer skills
    BILLY_HEADLESS=1 python run.py ...             # no window (fast benchmarks)
"""
from __future__ import annotations

import argparse
import sys

from billy import config, llm
from billy.abstractions import BootError
from billy.director import Director
from billy.games.smb import SmbGame
from billy.games.smb_lost import SmbLostGame
from billy.games.zelda import ZeldaGame
from billy.knowledge import KnowledgeBase, SkillLibrary

GAMES = {"smb": SmbGame, "smb_lost": SmbLostGame, "zelda": ZeldaGame}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Billy Mitchell plays games.")
    p.add_argument("--attempts", type=int, default=10, help="number of attempts to play")
    p.add_argument("--game", choices=sorted(GAMES), default="smb", help="which title to play")
    p.add_argument("--no-llm", action="store_true",
                   help="pure reflex run (no Billy/Coach LLM calls)")
    p.add_argument("--seed-skills", action="store_true",
                   help="seed the SkillLibrary with SMB's transferable tactics (cross-game carry-forward)")
    p.add_argument("--fresh", action="store_true",
                   help="wipe learned lessons, solution cache AND skills before starting")
    p.add_argument("--rl", metavar="MODEL", default="",
                   help="use a trained PPO policy (a .zip from train_rl.py) as the reflex tier; "
                        "the hand-crafted reflex remains the fallback at hazards")
    p.add_argument("--rl-sections", action="store_true",
                   help="enable hazard-scoped RL sub-policies (e.g. 1-3's platform/lift chain): "
                        "they SEED micro-search with a learned crossing candidate, which search "
                        "still verifies and the cache banks — the reflex/cache/search loop is unchanged")
    args = p.parse_args(argv)

    config.ensure_dirs()
    use_llm = not args.no_llm
    if use_llm and not llm.health():
        print(f"[warn] LM Studio unreachable at {config.LMSTUDIO_BASE_URL} — "
              f"Billy will improvise with fallbacks (load a model to fix).")
    if args.fresh:
        for f in (config.LESSONS_FILE, config.SOLUTIONS_FILE, config.SKILLS_FILE,
                  config.TAPES_FILE):
            if f.exists():
                f.unlink()
        print("[run] wiped prior lessons + solution cache + skills; Billy starts from scratch.")

    skills = SkillLibrary()
    if args.seed_skills:
        skills.seed_starter()
        print(f"[run] seeded {len(skills)} transferable skills.")

    game = GAMES[args.game]()
    sections = None
    if args.rl_sections:
        # Lazy import (torch/SB3 only needed with RL). Sub-policies seed micro-search at their
        # registered hazards; the cache/search/reflex loop is otherwise untouched.
        from billy.rl.section_policy import SectionController, default_smb_sections
        sections = SectionController(default_smb_sections())
    director = Director(game, KnowledgeBase(), use_llm=use_llm, skills=skills, sections=sections)
    if args.rl:
        # Lazy import so torch/SB3 are only needed when actually using RL. The learned policy
        # becomes the reflex tier; its fallback is the game's hand-crafted reflex, so cache +
        # micro-search + learn-from-death + LLM still own the lethal hazards.
        from billy.rl.learned_reflex import LearnedReflex
        director.reflex = LearnedReflex(args.rl, fallback=director.reflex)
    print(f"[run] {game.name} on {game.system.name} (in-process stable-retro).")
    try:
        director.run_session(args.attempts)
    except BootError as e:
        print("[error]", e)
        return 1
    except TimeoutError as e:
        print("[error]", e)
        return 1
    except KeyboardInterrupt:
        print("\n[run] interrupted — Billy demands a rematch.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
