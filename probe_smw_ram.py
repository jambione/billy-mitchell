#!/usr/bin/env python3
"""Verify the SMW WRAM map against a live ROM (bring-up step 2 in games/smw/STATUS.md).

Boots the stable-retro SuperMarioWorld-Snes integration, holds RIGHT for a spell, and checks
each mapped address behaves as documented (x increases, mode is level, ground flag sane...).
On a miss it scans WRAM for the expected signatures so the base offset can be relocated.

    BILLY_HEADLESS=1 .venv/bin/python probe_smw_ram.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("BILLY_HEADLESS", "1")
os.environ.setdefault("BILLY_TURBO", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    from billy.abstractions import Step
    from billy.games.smw import ram_map as R
    from billy.games.smw.game import SmwGame
    from billy.systems.snes import controller as C

    game = SmwGame()
    try:
        session = game.system.connect()
    except FileNotFoundError as e:
        print(f"[probe] no SMW integration/ROM yet: {e}")
        print("[probe] drop the ROM in roms/ and run: "
              ".venv/bin/python -m stable_retro.import roms/")
        return 2
    session.wait_until_live()

    def ram() -> bytes:
        return session.read_state().ram

    r0 = ram()
    checks: list[tuple[str, bool, str]] = []
    mode = r0[R.GAME_MODE]
    checks.append(("GAME_MODE==0x14 (in level)", mode == R.GAME_MODE_LEVEL,
                   f"got 0x{mode:02x}"))
    x0 = r0[R.PLAYER_X] | (r0[R.PLAYER_X + 1] << 8)
    lives = r0[R.LIVES]
    checks.append(("LIVES plausible (1..99)", 1 <= lives <= 99, f"got {lives}"))

    session.send_plan([Step(60, C.mask(C.RIGHT, C.B))])
    r1 = ram()
    x1 = r1[R.PLAYER_X] | (r1[R.PLAYER_X + 1] << 8)
    checks.append(("PLAYER_X increases holding RIGHT", x1 > x0, f"{x0} -> {x1}"))
    checks.append(("PLAYER_STATE normal (0)", r1[R.PLAYER_STATE] == 0,
                   f"got {r1[R.PLAYER_STATE]}"))

    session.send_plan([Step(20, C.mask(C.RIGHT, C.A))])   # logical A = jump (SNES B)
    r2 = ram()
    checks.append(("PLAYER_IN_AIR set mid-jump", r2[R.PLAYER_IN_AIR] != 0,
                   f"got {r2[R.PLAYER_IN_AIR]}"))

    ok = True
    for name, passed, detail in checks:
        print(f"  {'✅' if passed else '❌'} {name} ({detail})")
        ok = ok and passed

    if not ok:
        print("\n[probe] some checks failed — scanning WRAM for a 16-bit counter that "
              "increased by the RIGHT-hold distance (candidate PLAYER_X bases):")
        delta_lo, delta_hi = 20, 260
        hits = []
        for a in range(0, min(len(r0), len(r1)) - 1):
            v0 = r0[a] | (r0[a + 1] << 8)
            v1 = r1[a] | (r1[a + 1] << 8)
            if delta_lo <= v1 - v0 <= delta_hi:
                hits.append((a, v0, v1))
        for a, v0, v1 in hits[:20]:
            print(f"    0x{a:05x}: {v0} -> {v1}")
        print(f"  ({len(hits)} candidates; PLAYER_X documented at 0x{R.PLAYER_X:05x})")
    session.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
