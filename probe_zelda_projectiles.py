#!/usr/bin/env python3
"""Find where octorok projectiles (rocks/arrows) live in Zelda RAM — the prerequisite for
dodge-then-kill combat.

Perception today reads only the 6 enemy slots (type 0x350+, X 0x71+, Y 0x85+). Monster shots are
NOT there. This probe drives Billy (headless) until enemies fire, then AUTO-DISCOVERS the shot by
scanning the object tables for a byte-pair that moves fast, straight, and toward Link — the
kinematic signature of a projectile. It prints the candidate addresses and, on the frame a shot is
in flight, saves a savestate fixture so the mapping can be validated and regression-tested.

    BILLY_HEADLESS=1 .venv/bin/python probe_zelda_projectiles.py [--frames 6000] [--from-checkpoint]

Object tables probed (standard NES Zelda): X at 0x70+i, Y at 0x84+i for i in 0..15 (i=0 is Link;
1..6 the enemies perception already reads; 7..15 the suspected shot/misc slots).

FINDINGS (2026-07-19, validated against a captured octorok_shot_in_flight fixture):
  * Projectiles DO occupy the object position table at the HIGH slots (7..15). A captured rock
    sat in slot 14 (X 0x7E / Y 0x92) and travelled 3px/frame, dead straight, along Link's row.
  * The reliable, screen-independent detector is KINEMATIC: a nonzero high-slot object moving
    fast (~2-4px/frame) and straight toward Link is an incoming shot. Perception is stateless, so
    the reflex (which keeps _last_scene) computes the velocity and decides the dodge.
  * A per-slot type table at 0x394+i looked promising (the rock read 10) but needs more shots to
    confirm; the velocity signal is what dodge should trust.
"""
from __future__ import annotations

import argparse
import pathlib

from billy.abstractions import Step
from billy.games.zelda import ZeldaGame
from billy.games.zelda.perception import build_scene
from billy.systems.nes import controller as c

X_BASE, Y_BASE, N_SLOTS = 0x70, 0x84, 16
FIXTURE = pathlib.Path("data/zelda/states/octorok_shot_in_flight.state")


def _slots(ram: bytes) -> list[tuple[int, int]]:
    return [(ram[X_BASE + i], ram[Y_BASE + i]) for i in range(N_SLOTS)]


def _toward(prev, cur, link) -> bool:
    """True if the object moved a projectile-like step (>=2px) roughly toward Link."""
    (px, py), (cx, cy), (lx, ly) = prev, cur, link
    vx, vy = cx - px, cy - py
    if abs(vx) + abs(vy) < 2 or abs(vx) + abs(vy) > 12:   # too slow (enemy walk) or teleport
        return False
    # closing distance to Link?
    return (abs(cx - lx) + abs(cy - ly)) < (abs(px - lx) + abs(py - ly))


def probe(frames: int, from_checkpoint: bool) -> None:
    game = ZeldaGame()
    game.cli_name = "zelda"

    # Drive with the real director so Billy actually reaches shooting octoroks and plays naturally.
    # stable-retro allows one emulator per process, so use the Director's own session.
    from billy.director import Director
    from billy.knowledge import KnowledgeBase, SkillLibrary
    director = Director(game, KnowledgeBase(), use_llm=False, skills=SkillLibrary())
    sess = director.session
    sess.reset()
    sess.wait_until_live()
    director.boot()
    if from_checkpoint:
        try:
            director.resume_from_checkpoint()
        except Exception:
            pass

    prev_slots = None
    prev_hp = None
    hits: dict[int, int] = {}     # slot -> times it showed projectile-like motion
    samples: list[str] = []
    saved = False

    orig_observe = director._observe

    def hooked():
        nonlocal prev_slots, prev_hp, saved
        obs = orig_observe()
        ram = sess.read_state().ram
        s = obs.raw
        link = (s.link_x, s.link_y)
        slots = _slots(ram)
        if prev_slots is not None:
            for i in range(7, N_SLOTS):          # skip Link(0) + the 6 known enemy slots
                if slots[i] == (0, 0) or prev_slots[i] == (0, 0):
                    continue
                if _toward(prev_slots[i], slots[i], link):
                    hits[i] = hits.get(i, 0) + 1
                    if len(samples) < 20:
                        samples.append(
                            f"  slot {i:2d}: {prev_slots[i]} -> {slots[i]}  (Link {link})")
                    if not saved:                # capture the first shot-in-flight as a fixture
                        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
                        FIXTURE.write_bytes(sess.clone_state())
                        saved = True
                        samples.append(f"  >> saved shot-in-flight fixture: {FIXTURE}")
            # damage moment: an enemy shot connected — log the object field
            if prev_hp is not None and s.health < prev_hp:
                active = [(i, slots[i]) for i in range(7, N_SLOTS) if slots[i] != (0, 0)]
                samples.append(f"  [HIT] hp {prev_hp}->{s.health} slots7-15 active={active}")
        prev_slots, prev_hp = slots, s.health
        return obs

    director._observe = hooked
    import os
    os.environ.setdefault("BILLY_MAX_FRAMES", str(frames))
    try:
        director.run_attempt(1)
    finally:
        director._observe = orig_observe
        if getattr(director, "pool", None) is not None:
            director.pool.close()
        sess.close()

    print("\n=== projectile-slot candidates (object table 0x70/0x84, slots 7-15) ===")
    if hits:
        for slot, n in sorted(hits.items(), key=lambda kv: -kv[1]):
            print(f"  slot {slot:2d}: {n} projectile-like steps  "
                  f"(X @0x{X_BASE+slot:02X}, Y @0x{Y_BASE+slot:02X})")
    else:
        print("  none detected — enemies may not have fired this run (try more --frames, or a "
              "red-octorok screen). Object-table slots 7-15 stayed enemy/static.")
    print("\n=== sample motions / hits ===")
    for line in samples:
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6000)
    ap.add_argument("--from-checkpoint", action="store_true",
                    help="resume from data/checkpoints/zelda (reach octoroks faster)")
    args = ap.parse_args()
    probe(args.frames, args.from_checkpoint)


if __name__ == "__main__":
    main()
