"""Teleop demo-capture core: RLE recorder, verify gate, banking, and the stale-cache exemption."""
from __future__ import annotations

import contextlib

from billy.abstractions import Observation, Step
from billy.knowledge.cache import SolutionCache, bucket_of
from billy.systems.nes import controller as c
from billy.teleop import TeleopRecorder, bank_demo, verify_demo


def test_recorder_run_length_encodes():
    rec = TeleopRecorder()
    for _ in range(90):
        rec.record(c.RIGHT, 1)
    for _ in range(16):
        rec.record(c.mask(c.RIGHT, c.B), 1)
    plan = rec.plan()
    assert plan == [Step(90, c.RIGHT), Step(16, c.mask(c.RIGHT, c.B))]
    assert rec.frame_count() == 106


def test_recorder_skips_zero_and_merges_neutral():
    rec = TeleopRecorder()
    rec.record(c.NEUTRAL, 0)        # ignored
    rec.record(c.NEUTRAL, 5)
    rec.record(c.NEUTRAL, 3)        # merges
    rec.record(c.UP, 2)
    assert rec.plan() == [Step(8, c.NEUTRAL), Step(2, c.UP)]


# --- fakes for verify_demo (no ROM) -----------------------------------------------------
class _FakeSession:
    """Progress advances by total plan frames; dies if a plan holds the 'death' button (SELECT)."""
    def __init__(self):
        self.progress = 0
        self.dead = False

    @contextlib.contextmanager
    def search_mode(self):
        yield

    def restore(self, state):
        self.progress, self.dead = int(state), False

    def send_plan(self, plan):
        for s in plan:
            if s.buttons & c.SELECT:
                self.dead = True
            self.progress += s.frames

    def read_state(self):
        return type("St", (), {"frame": self.progress, "ram": b"", "rgb": None})()


class _FakeGame:
    def observe(self, frame, ram, rgb=None):
        # frame doubles as progress here; death flagged out-of-band via session in the test wrapper
        return Observation(frame=frame, progress=frame, score=0, level_label="overworld #121",
                           level_key=("overworld", 121), dead=False, summary="", ascii_map="",
                           raw=None, elevation=125)


def _verify(session, plan):
    # observe() in verify_demo reads session.read_state(); wire death through a game shim
    game = _FakeGame()
    base_observe = game.observe
    game.observe = lambda f, r, rgb=None: Observation(
        **{**base_observe(f, r).__dict__, "dead": session.dead})
    return verify_demo(session, game, b"100", plan, min_progress=8)


def test_verify_demo_bankable_when_survives_and_advances():
    res = _verify(_FakeSession(), [Step(60, c.RIGHT)])
    assert res.survived and res.advanced and res.bankable
    assert res.start_progress == 100 and res.end_progress == 160


def test_verify_demo_not_bankable_on_death():
    res = _verify(_FakeSession(), [Step(10, c.SELECT)])  # SELECT => death in fake
    assert not res.survived and not res.bankable


def test_verify_demo_not_bankable_without_progress():
    res = _verify(_FakeSession(), [Step(4, c.RIGHT)])    # +4 < min_progress(8)
    assert res.survived and not res.advanced and not res.bankable


def test_bank_demo_keys_match_director_lookup():
    cache = SolutionCache(path="/tmp/_teleop_test_solutions.jsonl")
    cache.entries.clear()
    obs = Observation(frame=0, progress=2769, score=0, level_label="overworld #121",
                      level_key=("overworld", 121), dead=False, summary="", ascii_map="",
                      raw=None, elevation=125)
    plan = [Step(40, c.RIGHT), Step(12, c.UP), Step(40, c.RIGHT)]
    key = bank_demo(cache, obs, plan, reach=3100)
    assert key == bucket_of(("overworld", 121), 2769, 125)
    # Director looks up exactly this way:
    got = cache.get(obs.level_key, obs.progress, obs.elevation)
    assert got is not None and got.reach_after == 3100 and list(got.plan) == plan


# --- stale_cache exemption: a far-advancing survivor (e.g. a dodging demo) must replay ----
def _scene_121(**kw):
    base = dict(map_location=121, sword_level=1, max_hearts=3, health=2, link_x=40, link_y=125,
                in_cave=False, in_dungeon=False, scrolling=False, at_left_edge=False,
                at_right_edge=False, at_top_edge=False, at_bottom_edge=False,
                visited_screens=[119, 120, 121], enemies=[])
    base.update(kw)
    s = type("Scene", (), base)()
    s.enemy_count = lambda: len(base["enemies"])
    s.item_count = lambda: 0
    s.nearest_enemy = lambda within=0: None
    return s


def _obs_121(progress=2769):
    return Observation(frame=0, progress=progress, score=0, level_label="overworld #121",
                       level_key=("overworld", 121), dead=False, summary="", ascii_map="",
                       raw=_scene_121(), elevation=125)


def test_stale_cache_trusts_far_advancing_demo_with_verticals():
    from billy.games.zelda.hazard_hooks import ZeldaHazardHooks
    hooks = ZeldaHazardHooks()
    obs = _obs_121(progress=2769)
    # A dodging demo: contains UP/DOWN, reaches far past the node (crosses the screen).
    demo = type("E", (), {"plan": [Step(40, c.RIGHT), Step(12, c.UP), Step(60, c.RIGHT)],
                          "reach_after": 2769 + 300})()
    assert hooks.stale_cache(obs, demo) is False  # trusted, not staled


def test_stale_cache_still_stales_barely_moving_vertical_wander():
    from billy.games.zelda.hazard_hooks import ZeldaHazardHooks
    hooks = ZeldaHazardHooks()
    obs = _obs_121(progress=2769)
    wander = type("E", (), {"plan": [Step(12, c.UP), Step(12, c.DOWN)],
                            "reach_after": 2769 + 10})()  # barely advances
    assert hooks.stale_cache(obs, wander) is True
