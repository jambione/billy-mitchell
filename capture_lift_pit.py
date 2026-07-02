#!/usr/bin/env python3
"""Capture a settled SMB 1-3 pit-edge savestate for lift search / RL.

Tries, in order:
  1. Director play-through (--play) until on-ground in [x-min, x-max] on 1-3
  2. Scan frames while coasting after a tree-top section rollout (--roll-section)

Near the pit lip, snapshots are taken immediately (settle would slide Mario off).
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("BILLY_HEADLESS", "1")

STATE_OUT = "data/rl/states/smb_1_3_lift.state"


def _from_state_capture(from_state: str, x_min: int, x_max: int, out: str,
                        max_frames: int, rl_sections: bool) -> bool:
    from billy.director import Director
    from billy.games.smb import SmbGame
    from billy.knowledge import KnowledgeBase, SkillLibrary

    sections = None
    if rl_sections:
        from billy.rl.section_policy import SectionController, default_smb_sections
        sections = SectionController(default_smb_sections())

    game = SmbGame()
    director = Director(game, KnowledgeBase(), use_llm=False, skills=SkillLibrary(), sections=sections)
    ok = director.capture_savestate_from(from_state, "1-3", x_min, x_max, out,
                                         max_frames=max_frames)
    director.session.close()
    return ok


def _play_capture(x_min: int, x_max: int, out: str, max_frames: int, rl_sections: bool) -> bool:
    from billy.director import Director
    from billy.games.smb import SmbGame
    from billy.knowledge import KnowledgeBase, SkillLibrary

    sections = None
    if rl_sections:
        from billy.rl.section_policy import SectionController, default_smb_sections
        sections = SectionController(default_smb_sections())

    game = SmbGame()
    director = Director(game, KnowledgeBase(), use_llm=False, skills=SkillLibrary(), sections=sections)
    director.boot()
    ok = director.capture_savestate("1-3", x_min, x_max, out, max_frames=max_frames)
    director.session.close()
    return ok


def _roll_capture(from_state: str, model: str, goal_x: int, x_min: int, x_max: int,
                  out: str, coast_frames: int) -> bool:
    from stable_baselines3 import PPO

    from billy.abstractions import Step
    from billy.games.smb import SmbGame
    from billy.games.smb.capture_util import save_snapshot
    from billy.rl import features
    from billy.rl.section_env import SECTION_ACTIONS
    from billy.systems.nes import controller as C

    game = SmbGame()
    session = game.system.connect()
    with open(from_state, "rb") as f:
        session.reset()
        session.env.em.set_state(f.read())
        session._refresh_ram()

    def observe():
        st = session.read_state()
        return game.observe(st.frame, st.ram)

    ppo = PPO.load(model, device="cpu")
    obs = observe()
    print(f"[capture] roll start {obs.level_label} x={obs.progress} ground={obs.raw.on_ground}")
    for _ in range(96):
        if obs.dead or obs.progress >= goal_x:
            break
        if save_snapshot(session, observe, out, x_min=x_min, x_max=x_max, level_label="1-3"):
            session.close()
            return True
        action, _ = ppo.predict(features.featurize(obs.raw), deterministic=True)
        names, hold = SECTION_ACTIONS[int(action)]
        session.send_plan([Step(hold, C.mask_from_names(list(names)))])
        obs = observe()

    for i in range(coast_frames):
        if obs.dead:
            break
        if save_snapshot(session, observe, out, x_min=x_min, x_max=x_max, level_label="1-3"):
            session.close()
            return True
        session.send_plan([Step(1, C.NEUTRAL)])
        obs = observe()
        if i % 30 == 0:
            print(f"[capture] coast f={i} x={obs.progress} ground={obs.raw.on_ground} "
                  f"vx={obs.raw.x_speed} gap={obs.raw.gap_info()}")

    print(f"[capture] roll miss — last {obs.level_label} x={obs.progress} "
          f"ground={obs.raw.on_ground} vx={obs.raw.x_speed} dead={obs.dead}")
    session.close()
    return False


def _save_approach_reference(out: str, x_min: int, x_max: int) -> bool:
    """Keep the best-known sliding pit-lip state when automated capture cannot settle vx."""
    from billy.games.smb import SmbGame
    from billy.games.smb.capture_util import near_pit, save_snapshot

    candidates = [
        "data/rl/states/smb_1_3_lift.state",
        "data/rl/states/smb_1_3_lift_636.state",
    ]
    game = SmbGame()
    session = game.system.connect()
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            session.reset()
            session.env.em.set_state(f.read())
            session._refresh_ram()
        if save_snapshot(session, lambda: game.observe(session.read_state().frame,
                                                       session.read_state().ram),
                         out, x_min=x_min, x_max=x_max, level_label="1-3"):
            session.close()
            return True
        obs = game.observe(session.read_state().frame, session.read_state().ram)
        if (obs.level_label == "1-3" and obs.raw.on_ground
                and x_min <= obs.progress <= x_max and near_pit(obs)):
            with open(out, "wb") as wf:
                wf.write(session.clone_state())
            print(f"[capture] approach reference {obs.level_label} x={obs.progress} "
                  f"vx={obs.raw.x_speed} (sliding lip — section RL crosses from here)")
            session.close()
            return True
    session.close()
    return False


def _probe_search(state: str, depth: int, beam: int) -> int:
    from billy.games.smb import SmbGame
    from billy.games.smb.lift_search import frame_lift_search, persist_lift_plan

    game = SmbGame()
    session = game.system.connect()
    with open(state, "rb") as f:
        session.reset()
        session.env.em.set_state(f.read())
        session._refresh_ram()

    def observe():
        st = session.read_state()
        return game.observe(st.frame, st.ram)

    start = observe()
    print(f"[probe] {start.level_label} x={start.progress} vx={start.raw.x_speed} "
          f"ground={start.raw.on_ground} gap={start.raw.gap_info()}")
    plan, reach, crossed = frame_lift_search(session, observe, depth=depth, beam=beam)
    print(f"[probe] crossed={crossed} reach={reach} steps={len(plan) if plan else 0}")
    if crossed and plan:
        persist_lift_plan(plan, reach)
    session.close()
    return 0 if crossed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Capture 1-3 pit-edge savestate + optional lift search.")
    p.add_argument("--out", default=STATE_OUT)
    p.add_argument("--x-min", type=int, default=620)
    p.add_argument("--x-max", type=int, default=700)
    p.add_argument("--play", action="store_true", help="drive Billy from 1-1 (slow)")
    p.add_argument("--roll-section", default="", metavar="MODEL",
                   help="roll this PPO model from --from-state and scan frames")
    p.add_argument("--from-state", default="data/rl/states/smb_1_3_at_486.state")
    p.add_argument("--goal-x", type=int, default=700)
    p.add_argument("--coast-frames", type=int, default=240)
    p.add_argument("--max-frames", type=int, default=120_000)
    p.add_argument("--rl-sections", action="store_true", default=True)
    p.add_argument("--no-rl-sections", action="store_false", dest="rl_sections")
    p.add_argument("--probe", action="store_true", help="run frame_lift_search after capture")
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--beam", type=int, default=28)
    args = p.parse_args(argv)

    ok = False
    if args.play:
        ok = _play_capture(args.x_min, args.x_max, args.out, args.max_frames, args.rl_sections)
    elif args.roll_section:
        ok = _roll_capture(args.from_state, args.roll_section, args.goal_x,
                           args.x_min, args.x_max, args.out, args.coast_frames)
    else:
        # default: Director from tree-top handoff, then roll, then full play-through
        print(f"[capture] trying Director from {args.from_state} …")
        ok = _from_state_capture(args.from_state, args.x_min, args.x_max, args.out,
                                 args.max_frames, args.rl_sections)
        if not ok:
            ok = _roll_capture(args.from_state, "data/rl/section_1_3", args.goal_x,
                               args.x_min, args.x_max, args.out, args.coast_frames)
        if not ok:
            print("[capture] roll failed — trying Director play-through …")
            ok = _play_capture(args.x_min, args.x_max, args.out, args.max_frames, args.rl_sections)

    if not ok:
        print("[capture] automated capture failed — saving best-known approach reference …")
        ok = _save_approach_reference(args.out, args.x_min, args.x_max)
    if not ok:
        return 1
    if args.probe:
        return _probe_search(args.out, args.depth, args.beam)
    return 0


if __name__ == "__main__":
    sys.exit(main())