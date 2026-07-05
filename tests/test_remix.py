"""Remix gauntlet: medal timing, goal detection, and manifest integrity."""
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import remix
from remix import CHALLENGES, Challenge, _BY_ID


def _obs(level_key=(0, 1, 3), progress=0, in_play=True, place=(9, 9, 1)):
    raw = types.SimpleNamespace(in_play=in_play, place=place)
    return types.SimpleNamespace(level_key=level_key, progress=progress, raw=raw)


# --- medals: seconds-to-finish, faster is shinier -------------------------------------------

def test_medal_thresholds():
    ch = Challenge("t", "T", "smb", "", "advance", time_s=40, medals=(12, 22))
    assert ch.medal(8.0) == "gold"
    assert ch.medal(18.0) == "silver"
    assert ch.medal(30.0) == "bronze"          # finished, just slow
    assert ch.medal(None) == "none"            # never finished


# --- goal detection --------------------------------------------------------------------------

def test_clear_goal_fires_on_level_change():
    ch = Challenge("c", "C", "smb", "", "clear", time_s=60, medals=(45, 65))
    start = _obs(level_key=(0, 1, 1))
    assert not ch.reached(start, _obs(level_key=(0, 1, 2)))   # same world/stage → not yet
    assert ch.reached(start, _obs(level_key=(0, 2, 3)))       # stage advanced → done


def test_advance_goal_needs_the_pixel_gain():
    ch = Challenge("a", "A", "smb", "", "advance", goal_px=180, time_s=40, medals=(12, 22))
    start = _obs(progress=1000)
    assert not ch.reached(start, _obs(progress=1120))
    assert ch.reached(start, _obs(progress=1200))


def test_psii_exit_needs_a_new_outdoor_place():
    ch = Challenge("p", "P", "psii", "", "psii_exit", time_s=60, medals=(25, 45))
    start = _obs(place=(9, 9, 1))
    assert not ch.reached(start, _obs(place=(9, 9, 1)))       # same place
    assert not ch.reached(start, _obs(place=(6, 6, 0)))       # a building (indoor)
    assert ch.reached(start, _obs(place=(11, 9, 1)))          # new OUTDOOR place = out of town


def test_a_dead_or_not_in_play_frame_is_never_a_win():
    ch = Challenge("a", "A", "smb", "", "advance", goal_px=10, time_s=40, medals=(12, 22))
    start = _obs(progress=1000)
    assert not ch.reached(start, _obs(progress=2000, in_play=False))


# --- manifest integrity ----------------------------------------------------------------------

def test_ids_unique_and_indexed():
    ids = [c.id for c in CHALLENGES]
    assert len(ids) == len(set(ids))
    assert set(_BY_ID) == set(ids)


def test_every_challenge_targets_a_real_game():
    from run import GAMES
    for c in CHALLENGES:
        assert c.game in GAMES, f"{c.id} → unknown game {c.game}"


def test_declared_start_states_exist():
    root = Path(remix.__file__).resolve().parent
    for c in CHALLENGES:
        if c.from_state:
            assert (root / c.from_state).is_file(), f"{c.id} start state missing: {c.from_state}"
