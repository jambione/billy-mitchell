"""Seed a game-scoped SolutionCache from verified teleop/remix demos on disk."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..abstractions import Game, Step
from ..config import DATA_DIR


def seed_demos(game_id: str, cache, game: Game, *, session=None) -> int:
    """Bank demos under data/rl/demos/{game_id}/ that verify but aren't cached yet."""
    if not game_id:
        return 0
    demo_dir = DATA_DIR / "rl" / "demos" / game_id
    if not demo_dir.is_dir():
        return 0
    from ..teleop import verify_demo

    os.environ.setdefault("BILLY_HEADLESS", "1")
    seeded = 0
    own_session = session is None
    if own_session:
        session = game.system.connect()
        session.reset()
        session.wait_until_live()

    def observe():
        st = session.read_state()
        return game.observe(st.frame, st.ram, getattr(st, "rgb", None))

    try:
        for demo_path in sorted(demo_dir.glob("*.demo.json")):
            state_path = demo_path.with_name(demo_path.name.replace(".demo.json", ".state"))
            if not state_path.is_file():
                continue
            plan = [Step(f, b) for f, b in json.loads(demo_path.read_text())["steps"]]
            start_state = state_path.read_bytes()
            result = verify_demo(session, game, start_state, plan,
                                 min_progress=game.remix_min_progress())
            if not result.bankable:
                continue
            with session.search_mode():
                session.restore(start_state)
                start_obs = observe()
            if cache.get(start_obs.level_key, start_obs.progress, start_obs.elevation):
                continue
            cache.put(start_obs.level_key, start_obs.progress, plan, result.end_progress,
                      y=start_obs.elevation, force=True)
            seeded += 1
    finally:
        if own_session:
            close = getattr(session, "close", None)
            if close:
                close()
    return seeded