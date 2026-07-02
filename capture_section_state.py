#!/usr/bin/env python3
"""Capture a hazard savestate for section sub-policy training.

Drives Billy (optionally with --rl-sections) until Mario is on-ground in the target
level within [x-min, x-max], then writes session.clone_state() to --out.

Examples:
    # From tree-top entrance, roll the trained section model and save at the landing:
    BILLY_HEADLESS=1 .venv/bin/python capture_section_state.py \\
        --from-state data/rl/states/smb_1_3_section.state \\
        --roll-section data/rl/section_1_3 --goal-x 700 \\
        --wait-ground --out data/rl/states/smb_1_3_lift.state

    # From a live Billy run (uses Director + existing cache):
    BILLY_HEADLESS=1 .venv/bin/python capture_section_state.py \\
        --play --rl-sections --level 1-3 --x-min 700 --x-max 760 \\
        --out data/rl/states/smb_1_3_lift.state
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("BILLY_HEADLESS", "1")


def _roll_from_state(state: str, model: str, goal_x: int, wait_ground: bool,
                     landing_waits: int, out: str) -> int:
    from stable_baselines3 import PPO

    from billy.abstractions import Step
    from billy.games.smb import SmbGame
    from billy.rl import features
    from billy.rl.section_env import SECTION_ACTIONS
    from billy.systems.nes import controller as C

    game = SmbGame()
    session = game.system.connect()
    with open(state, "rb") as f:
        session.reset()
        session.env.em.set_state(f.read())
        session._refresh_ram()

    def observe():
        st = session.read_state()
        return game.observe(st.frame, st.ram)

    ppo = PPO.load(model, device="cpu")
    obs = observe()
    print(f"[capture] start {obs.level_label} x={obs.progress} ground={obs.raw.on_ground}")
    for _ in range(96):
        if obs.dead or obs.progress >= goal_x:
            break
        action, _ = ppo.predict(features.featurize(obs.raw), deterministic=True)
        names, hold = SECTION_ACTIONS[int(action)]
        session.send_plan([Step(hold, C.mask_from_names(list(names)))])
        obs = observe()

    for _ in range(landing_waits):
        if obs.dead:
            break
        session.send_plan([Step(4, C.mask_from_names([]))])
        obs = observe()
    if wait_ground and not obs.dead:
        for _ in range(120):
            session.send_plan([Step(4, C.mask_from_names([]))])
            obs = observe()
            if obs.raw.on_ground:
                break
            if obs.dead:
                break

    print(f"[capture] end   {obs.level_label} x={obs.progress} ground={obs.raw.on_ground} "
          f"vx={obs.raw.x_speed} dead={obs.dead}")
    if obs.dead:
        print("[capture] Mario died before capture — try a different --from-state or --goal-x.")
        session.close()
        return 1
    from billy.games.smb.capture_util import settle_mario
    ok, _ = settle_mario(session, observe)
    obs = observe()
    print(f"[capture] settled x={obs.progress} ground={obs.raw.on_ground} vx={obs.raw.x_speed}")
    if not ok:
        print("[capture] could not settle Mario before save.")
        session.close()
        return 1
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        f.write(session.clone_state())
    print(f"[capture] saved -> {out}")
    session.close()
    return 0


def _play_capture(level: str, x_min: int, x_max: int, rl_sections: bool, max_frames: int, out: str) -> int:
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
    ok = director.capture_savestate(level, x_min, x_max, out, max_frames=max_frames)
    director.session.close()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Capture a section training savestate.")
    p.add_argument("--out", default="data/rl/states/captured.state")
    p.add_argument("--from-state", default="", help="restore this .state before rolling a section model")
    p.add_argument("--roll-section", default="", metavar="MODEL", help="PPO model to roll out from --from-state")
    p.add_argument("--goal-x", type=int, default=700)
    p.add_argument("--wait-ground", action="store_true", help="coast until Mario lands before saving")
    p.add_argument("--landing-waits", type=int, default=0,
                   help="noop steps after --goal-x before --wait-ground (tree-top handoff)")
    p.add_argument("--play", action="store_true", help="drive Billy from level start instead of --from-state")
    p.add_argument("--rl-sections", action="store_true")
    p.add_argument("--level", default="1-3")
    p.add_argument("--x-min", type=int, default=700)
    p.add_argument("--x-max", type=int, default=760)
    p.add_argument("--max-frames", type=int, default=30_000)
    args = p.parse_args(argv)

    if args.play:
        return _play_capture(args.level, args.x_min, args.x_max, args.rl_sections, args.max_frames, args.out)
    if not args.from_state or not args.roll_section:
        p.error("use --play, or both --from-state and --roll-section")
    return _roll_from_state(args.from_state, args.roll_section, args.goal_x,
                            args.wait_ground, args.landing_waits, args.out)


if __name__ == "__main__":
    sys.exit(main())