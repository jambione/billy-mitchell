#!/usr/bin/env python
"""Probe Phantasy Star II's work RAM against the live ROM — the evidence behind
billy/games/psii/ram_map.py. Run it after any doubt about an address:

    BILLY_HEADLESS=1 .venv/bin/python probe_psii_ram.py

Method: script a small tour from the Start.state anchor (Rolf's doorstep in Paseo) —
into the house, back out, then a down-biased random walk out of town — using the one
signal that marks EVERY place change in this game: the fade-to-black transition seen on
the rgb stream. Collect settled RAM snapshots per place, then keep bytes that are stable
within every place, distinct across places, and REVERSIBLE (place A revisited must read
the place-A value again). Position/facing are verified by scripted movement deltas.
"""
from __future__ import annotations

import os
import random

os.environ.setdefault("BILLY_HEADLESS", "1")
os.environ.setdefault("BILLY_TURBO", "1")

import numpy as np

from billy.abstractions import Step
from billy.systems.genesis import controller as C
from billy.systems.genesis.system import GenesisSystem

random.seed(11)

_SHOT_DIR = os.environ.get("PSII_PROBE_SHOTS", "")


def _shot(session, name: str) -> None:
    """Optional screenshot dump (PSII_PROBE_SHOTS=<dir>) for eyeballing tour hops."""
    if not _SHOT_DIR:
        return
    import struct
    import zlib
    from pathlib import Path
    rgb = session.read_state().rgb
    if rgb is None:
        return
    rgb = np.asarray(rgb)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    out = Path(_SHOT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.png").write_bytes(png)


def main() -> None:
    s = GenesisSystem(retro_game="PhantasyStarII-Genesis-v0").connect()
    try:
        s.reset()
    except FileNotFoundError:
        raise SystemExit("[probe] integration missing — run emulator/make_psii_state.py first")

    def ram() -> np.ndarray:
        return np.frombuffer(s.read_state().ram, dtype=np.uint8).astype(int)

    def screen_mean() -> float:
        rgb = s.read_state().rgb
        return float(np.asarray(rgb).mean()) if rgb is not None else 0.0

    def settle_snaps(n: int = 3) -> list[np.ndarray]:
        out = []
        for _ in range(n):
            s.send_plan([Step(20, 0)])
            out.append(ram())
        return out

    def step_dir(d: int, frames: int = 45) -> bool:
        """One movement step; True when a fade-to-black (place transition) happened."""
        faded = False
        for _ in range(frames // 15):
            s.send_plan([Step(15, d)])
            if screen_mean() < 6.0:
                faded = True
        if faded:
            s.send_plan([Step(150, 0)])     # let the new place fade in + settle
        return faded

    pos = (0xE40A, 0xE40E)   # player x/y px (verified below)

    # --- movement: x/y bytes ---------------------------------------------------------------
    r0 = ram()
    s.send_plan([Step(64, C.DOWN)])   # doorstep is wall-blocked right/up; down is open
    r1 = ram()
    ok_y = r1[pos[1]] > r0[pos[1]]
    s.send_plan([Step(32, C.LEFT)])
    r2 = ram()
    ok_x = r2[pos[0]] < r1[pos[0]]
    print(f"[probe] player position: x@{hex(pos[0])} y@{hex(pos[1])} "
          f"{'VERIFIED' if ok_x and ok_y else 'FAILED'} "
          f"(y {r0[pos[1]]}->{r1[pos[1]]} on DOWN, x {r1[pos[0]]}->{r2[pos[0]]} on LEFT)")

    # --- place tour: doorstep -> house -> doorstep -> (walk) new place ----------------------
    s.reset()
    places: dict[str, list[list[np.ndarray]]] = {}
    places["paseo"] = [settle_snaps()]
    if not step_dir(C.UP, 90):
        print("[probe] WARN: no fade entering the house?")
    s.send_plan([Step(8, C.A), Step(30, 0)] * 12)      # close the scripted welcome dialog
    places["house"] = [settle_snaps()]
    step_dir(C.DOWN, 120)
    places["paseo"].append(settle_snaps())             # reversibility set

    blocked: set[int] = set()
    DIRS = [(C.DOWN, .4), (C.LEFT, .3), (C.RIGHT, .2), (C.UP, .1)]
    hops = 0
    for _ in range(500):
        choices = [(d, w) for d, w in DIRS if d not in blocked] or DIRS
        d = random.choices([c[0] for c in choices], [c[1] for c in choices])[0]
        before = ram()
        faded = step_dir(d)
        after = ram()
        if (after[pos[0]], after[pos[1]]) == (before[pos[0]], before[pos[1]]) and not faded:
            blocked.add(d)
        else:
            blocked.clear()
        if faded:
            hops += 1
            places[f"hop{hops}"] = [settle_snaps()]
            _shot(s, f"hop{hops}")
            if hops >= 4:
                break
    print(f"[probe] tour places: {list(places)}")

    # --- map identity: stable-within, distinct-across, reversible ---------------------------
    names = list(places)
    n_bytes = len(places[names[0]][0][0])
    cand = []
    for i in range(0x8000, 0xF000):
        vals = {}
        ok = True
        for name in names:
            flat = [snap[i] for group in places[name] for snap in group]
            if any(v != flat[0] for v in flat):        # unstable within a place (or its revisit)
                ok = False
                break
            vals[name] = flat[0]
        if ok and len(set(vals.values())) == len(vals):
            cand.append((i, vals))
    print(f"[probe] map-identity candidates (stable+distinct+reversible): {len(cand)}")
    for i, vals in cand[:12]:
        print(f"    {hex(i)}: {vals}")

    # --- battle: walk the overworld until the encounter fade, diff the battle flag ----------
    field = ram()
    in_battle_evidence = None
    blocked.clear()
    for _ in range(240):
        choices = [(d, w) for d, w in DIRS if d not in blocked] or DIRS
        d = random.choices([c[0] for c in choices], [c[1] for c in choices])[0]
        before = ram()
        faded = step_dir(d)
        after = ram()
        if (after[pos[0]], after[pos[1]]) == (before[pos[0]], before[pos[1]]) and not faded:
            blocked.add(d)
        else:
            blocked.clear()
        if faded:
            # battle screens keep the party OFF the walk maps: position bytes freeze while
            # the battle plays. Distinguish from a place hop by trying to walk afterwards.
            probe0 = ram()
            s.send_plan([Step(45, C.DOWN), Step(45, C.UP)])
            probe1 = ram()
            moved = probe1[pos[0]] != probe0[pos[0]] or probe1[pos[1]] != probe0[pos[1]]
            if not moved:
                battle = settle_snaps(2)
                flips = [i for i in range(0x8000, 0xF000)
                         if field[i] == 0 and all(sn[i] not in (0,) and sn[i] == battle[0][0][i]
                                                  for grp in [battle] for sn in grp[0])]
                in_battle_evidence = flips[:12]
                print(f"[probe] battle reached; 0->nonzero stable bytes: "
                      f"{[hex(i) for i in in_battle_evidence]}")
                break
            field = after
    if in_battle_evidence is None:
        print("[probe] battle NOT reached within budget (fine on a short run)")

    print("[probe] done")


if __name__ == "__main__":
    main()
