"""Remix gauntlet: medal scoring, best-folding, and manifest integrity."""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import remix
from remix import CHALLENGES, Challenge, _BY_ID


def _result(**kw):
    base = dict(outcome="timeout", max_x=0, level_frontier=0, fastest_clear_frames=0)
    base.update(kw)
    return types.SimpleNamespace(**base)


# --- clear challenges: fastest time wins ----------------------------------------------------

def test_clear_medals_are_time_gated():
    ch = Challenge("t", "T", "smb", "", kind="clear", medals=(38, 48))
    assert ch.medal(35.0) == "gold"
    assert ch.medal(40.0) == "silver"
    assert ch.medal(60.0) == "bronze"        # cleared at all, just slow
    assert ch.medal(None) == "none"          # never cleared


def test_clear_attempt_value_only_counts_a_clear():
    ch = Challenge("t", "T", "smb", "", kind="clear", medals=(38, 48))
    assert ch.attempt_value(_result(outcome="clear", fastest_clear_frames=2394)) == 39.9
    assert ch.attempt_value(_result(outcome="game_over", fastest_clear_frames=0)) is None


def test_clear_folds_to_the_faster_time():
    ch = Challenge("t", "T", "smb", "", kind="clear", medals=(38, 48))
    assert ch.better(42.0, 39.0) == 39.0
    assert ch.better(None, 39.0) == 39.0
    assert ch.better(None, None) is None


# --- reach challenges: farther / longer wins ------------------------------------------------

def test_reach_medals_are_progress_gated():
    ch = Challenge("r", "R", "shmup", "", kind="reach", medals=(320, 230, 150))
    assert ch.medal(400) == "gold"
    assert ch.medal(252) == "silver"
    assert ch.medal(160) == "bronze"
    assert ch.medal(100) == "none"           # below the bronze floor


def test_reach_attempt_value_needs_the_bronze_floor():
    ch = Challenge("r", "R", "shmup", "", kind="reach", medals=(320, 230, 150))
    assert ch.attempt_value(_result(max_x=252)) == 252.0
    assert ch.attempt_value(_result(max_x=120)) is None        # under the floor = not a pass
    # level_frontier is taken when it's the larger progress signal
    assert ch.attempt_value(_result(max_x=100, level_frontier=300)) == 300.0


def test_reach_folds_to_the_farther_value():
    ch = Challenge("r", "R", "shmup", "", kind="reach", medals=(320, 230, 150))
    assert ch.better(200.0, 260.0) == 260.0
    assert ch.better(260.0, None) == 260.0


# --- manifest integrity ---------------------------------------------------------------------

def test_challenge_ids_are_unique():
    ids = [c.id for c in CHALLENGES]
    assert len(ids) == len(set(ids))
    assert set(_BY_ID) == set(ids)


def test_every_challenge_targets_a_real_game():
    from run import GAMES
    for c in CHALLENGES:
        assert c.game in GAMES, f"{c.id} points at unknown game {c.game}"


def test_clear_medals_have_two_thresholds_reach_have_three():
    for c in CHALLENGES:
        assert len(c.medals) == (2 if c.kind == "clear" else 3), c.id
