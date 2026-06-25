#!/usr/bin/env python3
"""Billy Mitchell learns to play games.

The emulator now runs in-process via stable-retro (one-time setup: ./emulator/setup_retro.sh).
    python run.py --attempts 20
    python run.py --attempts 5 --no-llm     # pure reflex run (no Billy/Coach)
    BILLY_HEADLESS=1 python run.py ...       # no window (fast benchmarks)
"""
from __future__ import annotations

import argparse
import sys

from billy import config, llm
from billy.abstractions import BootError
from billy.director import Director
from billy.games.smb import SmbGame
from billy.knowledge import KnowledgeBase


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Billy Mitchell plays games.")
    p.add_argument("--attempts", type=int, default=10, help="number of attempts to play")
    p.add_argument("--no-llm", action="store_true",
                   help="pure reflex run (no Billy/Coach LLM calls)")
    p.add_argument("--fresh", action="store_true",
                   help="wipe learned lessons AND the solution cache before starting")
    args = p.parse_args(argv)

    config.ensure_dirs()
    use_llm = not args.no_llm
    if use_llm and not llm.health():
        print(f"[warn] LM Studio unreachable at {config.LMSTUDIO_BASE_URL} — "
              f"Billy will improvise with fallbacks (load a model to fix).")
    if args.fresh:
        for f in (config.LESSONS_FILE, config.SOLUTIONS_FILE):
            if f.exists():
                f.unlink()
        print("[run] wiped prior lessons + solution cache; Billy starts from scratch.")

    game = SmbGame()
    director = Director(game, KnowledgeBase(), use_llm=use_llm)
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
