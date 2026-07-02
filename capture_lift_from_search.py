#!/usr/bin/env python3
"""Capture a settled 1-3 lift savestate via expanded search (bypasses 1-1/1-2)."""
from __future__ import annotations

import json
import os

os.environ.setdefault("BILLY_HEADLESS", "1")

from billy import config
from billy.abstractions import Step, plan_frames
from billy.games.smb import SmbGame
from billy.games.smb.capture_util import settle_mario, zero_x_speed
from billy.games.smb.reflexes import SmbReflex
from billy.systems.nes import controller as C

STATE_IN = "data/rl/states/smb_1_3_at_486.state"
STATE_OUT = "data/rl/states/smb_1_3_lift.state"
PLAN_OUT = "data/rl/lift_cached.plan.json"


def _micro_search(session, game, reflex, candidates, start_x):
    snap = session.clone_state()
    best_plan, best_score, best_reach = candidates[0], -10**9, start_x
    with session.search_mode():
        for plan in candidates:
            session.restore(snap)
            session.send_plan(plan)
            obs = game.observe(session.read_state().frame, session.read_state().ram)
            reached = obs.progress
            coasted = 0
            while coasted < config.SEARCH_HORIZON_FRAMES and not obs.dead:
                session.send_plan(reflex.advance_plan(obs))
                obs = game.observe(session.read_state().frame, session.read_state().ram)
                reached = max(reached, obs.progress)
                coasted += max(1, plan_frames(reflex.advance_plan(obs)))
                if coasted >= config.SEARCH_HORIZON_FRAMES:
                    break
            score = reached if not obs.dead else reached - 100000
            if score > best_score:
                best_score, best_plan, best_reach = score, plan, reached
    session.restore(snap)
    return best_plan, best_reach


def main() -> int:
    game = SmbGame()
    session = game.system.connect()
    with open(STATE_IN, "rb") as f:
        session.reset()
        session.env.em.set_state(f.read())
        session._refresh_ram()

    def observe():
        st = session.read_state()
        return game.observe(st.frame, st.ram)

    reflex = SmbReflex()
    start = observe()
    plan, _ = _micro_search(session, game, reflex, reflex.expanded_candidates(start), start.progress)
    session.send_plan(plan)
    landed = observe()
    for _ in range(30):
        session.send_plan([Step(4, 0)])
        landed = observe()
        if landed.raw.on_ground and landed.progress >= 600:
            break

    zero_x_speed(session)
    for _ in range(8):
        session.send_plan([Step(4, 0)])
        landed = observe()
        if landed.raw.on_ground and abs(landed.raw.x_speed) <= 1:
            break

    ok, _ = settle_mario(session, observe, allow_left=False)
    settled = observe()
    print(f"[capture] landed x={landed.progress} settled x={settled.progress} "
          f"vx={settled.raw.x_speed} ground={settled.raw.on_ground} gap={settled.raw.gap_info()}")
    if not ok or not settled.raw.on_ground:
        session.close()
        return 1

    os.makedirs(os.path.dirname(STATE_OUT) or ".", exist_ok=True)
    with open(STATE_OUT, "wb") as f:
        f.write(session.clone_state())
    print(f"[capture] saved -> {STATE_OUT}")

    # probe best idle-only crossing from this state
    best_alive = (0, 0)
    for wait in range(1, 120):
        session.restore(session.clone_state())
        session.send_plan([Step(wait, 0)])
        o = observe()
        if not o.dead and o.progress > best_alive[0]:
            best_alive = (o.progress, wait)
        if not o.dead and o.progress >= 900:
            json.dump({"plan": [[wait, 0]], "reach_after": o.progress, "kind": "idle"},
                      open(PLAN_OUT, "w"))
            print(f"[capture] idle CROSS wait={wait} x={o.progress}")
            session.close()
            return 0

    print(f"[capture] best idle alive x={best_alive[0]} @ wait={best_alive[1]}f")
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())