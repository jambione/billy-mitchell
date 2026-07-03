#!/usr/bin/env python
"""One-time bring-up: craft the PhantasyStarII-Genesis Start.state anchor.

Boots the ROM from power-on (retro.State.NONE), replays a scripted "boot dance" (frames +
buttons) through the SEGA logo / title / opening, dumping periodic PNG screenshots so the
dance can be steered empirically, then (with --save) gzips the emulator state into the
integration folder as Start.state — the deterministic in-play anchor every Billy session
(and every tape) starts from. Same lesson as the shmup boot: anchor EARLY in play.

    .venv/bin/python emulator/make_psii_state.py --shots /tmp/psii_shots        # look
    .venv/bin/python emulator/make_psii_state.py --save                         # commit anchor

The integration folder needs the ROM at rom.md (gitignored); this script copies it from
roms/ on first run, verifying the SHA1 in rom.sha.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
INTEGRATION = REPO / "emulator" / "integrations" / "PhantasyStarII-Genesis-v0"
ROM_SRC = REPO / "roms" / "Phantasy Star II (UE) (REV 02) [!].bin"
GAME = "PhantasyStarII-Genesis-v0"

# The boot dance: (frames, [physical button names]). Steered by looking at the screenshot
# dumps — expect SEGA logo, title ("PRESS START BUTTON"), save-select, and the opening
# sequence before control lands. START taps walk menus; C confirms / advances text.
DANCE: list[tuple[int, list[str]]] = (
    [(1100, [])]                          # power-on: planets intro → title by ~1100
    + [(8, ["START"]), (30, [])] * 30     # blink-proof START mash: title → data check → menu
    + [(8, ["C"]), (30, [])] * 8          # select NEW; extra Cs type "AAAA" (4-char cap)
    + [(8, ["DOWN"]), (20, [])] * 3       # name-entry grid: cursor A → bottom row
    + [(8, ["RIGHT"]), (20, [])] * 2      # ... → END
    + [(8, ["C"]), (90, [])]              # confirm name
    + [(8, ["C"]), (45, [])] * 88         # nightmare → wake → briefing → Nei scene (overshoot:
    + [(8, ["B"]), (30, [])] * 8          #   safe — B closes whatever a stray C opened in play)
    + [(60, [])]                          # settle in play (control live)
)


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal stdlib PNG writer (no PIL in the venv)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    path.write_bytes(png)


def ensure_rom() -> None:
    dst = INTEGRATION / "rom.md"
    want = (INTEGRATION / "rom.sha").read_text().split()[0]
    if dst.exists() and hashlib.sha1(dst.read_bytes()).hexdigest() == want:
        return
    if not ROM_SRC.exists():
        sys.exit(f"[psii] ROM not found at {ROM_SRC}")
    got = hashlib.sha1(ROM_SRC.read_bytes()).hexdigest()
    if got != want:
        sys.exit(f"[psii] ROM sha1 mismatch: {got} != {want}")
    shutil.copy2(ROM_SRC, dst)
    print(f"[psii] ROM copied into integration as rom.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=Path, default=None, help="dump a PNG every --every frames")
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--save", action="store_true", help="write Start.state at the dance's end")
    ap.add_argument("--extra", type=int, default=0, help="extra idle frames after the dance")
    args = ap.parse_args()

    ensure_rom()
    import stable_retro as retro
    from stable_retro.data import Integrations
    Integrations.add_custom_path(str(REPO / "emulator" / "integrations"))
    # Actions.ALL: the Genesis FILTERED action space silently strips START — the title screen
    # (and PSII's START-opened menu) is unreachable without it.
    env = retro.make(GAME, state=retro.State.NONE, inttype=Integrations.CUSTOM_ONLY,
                     render_mode="rgb_array", use_restricted_actions=retro.Actions.ALL)
    env.reset()
    idx = {name: i for i, name in enumerate(env.buttons) if name}
    if args.shots:
        args.shots.mkdir(parents=True, exist_ok=True)

    frame = 0
    for frames, names in DANCE + [(args.extra, [])] * (1 if args.extra else 0):
        action = np.zeros(len(env.buttons), dtype=np.uint8)
        for n in names:
            action[idx[n]] = 1
        for _ in range(frames):
            env.step(action)
            frame += 1
            if args.shots and frame % args.every == 0:
                write_png(args.shots / f"f{frame:06d}.png", np.asarray(env.render()))
    print(f"[psii] dance complete at frame {frame}")

    # Controllability check (attract/cutscene-proof): from the anchor, hold LEFT for 60 frames
    # in one branch and RIGHT in another. If control is live the two end-screens diverge hard
    # (opposite scrolls); in a cutscene/dialog both branches render identically. Idle drift
    # (wandering NPCs) is the noise floor.
    anchor = env.em.get_state()
    base = np.asarray(env.render()).copy()
    zero = np.zeros(len(env.buttons), dtype=np.uint8)

    def branch(button: str | None) -> np.ndarray:
        env.em.set_state(anchor)
        a = zero.copy()
        if button:
            a[idx[button]] = 1
        for _ in range(60):
            env.step(a)
        return np.asarray(env.render()).astype(int).copy()

    idle_end, left_end, right_end = branch(None), branch("LEFT"), branch("RIGHT")
    env.em.set_state(anchor)
    idle_diff = int(np.abs(idle_end - base.astype(int)).sum())
    lr_diff = int(np.abs(left_end - right_end).sum())
    if args.shots:
        write_png(args.shots / "anchor.png", base)
    print(f"[psii] controllability: idle_drift={idle_diff} left_vs_right={lr_diff} "
          f"({'LIVE' if lr_diff > max(idle_diff, 20000) else 'NOT PROVEN'})")

    if args.save:
        state = env.em.get_state()
        out = INTEGRATION / "Start.state"
        with gzip.open(out, "wb") as f:
            f.write(state)
        print(f"[psii] Start.state written ({out.stat().st_size} bytes gzipped)")
    env.close()


if __name__ == "__main__":
    main()
