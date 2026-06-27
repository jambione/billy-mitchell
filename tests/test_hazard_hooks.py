"""Tests for game-agnostic hazard hooks and SMB pit/lift wiring."""
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.abstractions import Observation
from billy.games.smb.hazard_hooks import (
    SmbHazardHooks,
    is_pit_death,
    pit_approach_zone,
    pit_cacheable,
    pit_goal_x,
)
from billy.hazard_hooks import NullHazardHooks


@dataclass
class _FakeScene:
    mario_x: int = 1000

    def gap_info(self, max_tiles: int = 8):
        return (20, 3)

    def pipe_entry_spot(self, max_tiles: int = 6):
        return None


def _obs(x: int, label: str = "2-2", lk=(1, 1, 2)) -> Observation:
    return Observation(
        frame=1, progress=x, score=0, level_label=label, level_key=lk,
        dead=False, summary="", ascii_map="", raw=_FakeScene(mario_x=x),
        elevation=0,
    )


def test_pit_death_and_cacheable():
    assert is_pit_death("2-2", 1124)
    assert is_pit_death("2-2", 1155)
    assert not is_pit_death("2-2", 900)
    assert pit_goal_x(1155) == 1171
    assert pit_cacheable(1171, 1155)
    assert not pit_cacheable(1146, 1155)
    assert not pit_cacheable(1120, 1124)


def test_pit_approach_zone():
    assert pit_approach_zone(_obs(1030))
    assert not pit_approach_zone(_obs(500, label="1-1", lk=(0, 0, 0)))


def test_smb_hooks_exempt_pit_from_stall_break():
    hooks = SmbHazardHooks()
    assert hooks.stall_break_exempt(_obs(1030))
    assert hooks.in_special_zone(_obs(1030))
    assert hooks.learn_cacheable("2-2", 1155, 1171)
    assert not hooks.learn_cacheable("2-2", 1155, 1146)


def test_null_hooks_defaults():
    hooks = NullHazardHooks()
    obs = _obs(100)
    assert hooks.commit_chunk_size(obs, 6) == 6
    assert not hooks.in_special_zone(obs)
    assert hooks.learn_cacheable("1-1", 100, 200)