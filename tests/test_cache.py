"""Unit tests for the position-keyed SolutionCache (the compounding-learning core)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.abstractions import Step  # noqa: E402
from billy.knowledge.cache import SolutionCache, bucket_of  # noqa: E402
from billy.systems.nes import controller  # noqa: E402


def _plan():
    return [Step(4, controller.mask(controller.RIGHT, controller.B)),
            Step(28, controller.mask(controller.RIGHT, controller.B, controller.A))]


def test_bucket_quantizes_by_tile():
    assert bucket_of(0, 0, 0) == (0, 0, 0)
    assert bucket_of(0, 0, 15) == (0, 0, 0)
    assert bucket_of(0, 0, 16) == (0, 0, 1)
    assert bucket_of(1, 2, 700) == (1, 2, 43)


def test_put_get_and_keep_better(tmp_path):
    c = SolutionCache(path=tmp_path / "solutions.jsonl")
    c.put(0, 0, 700, _plan(), reach_after=780)
    e = c.get(0, 0, 700)
    assert e is not None and e.reach_after == 780
    # same bucket (700-703 all map to bucket 43), worse reach -> not replaced
    c.put(0, 0, 701, [Step(8, controller.RIGHT)], reach_after=720)
    assert c.get(0, 0, 700).reach_after == 780
    # same bucket, better reach -> replaced
    c.put(0, 0, 703, [Step(8, controller.RIGHT)], reach_after=900)
    assert c.get(0, 0, 700).reach_after == 900


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "solutions.jsonl"
    c = SolutionCache(path=p)
    c.put(0, 0, 700, _plan(), reach_after=780)
    c.put(4, 1, 1200, [Step(16, controller.A)], reach_after=1260)
    # reload from disk
    c2 = SolutionCache(path=p)
    assert len(c2) == 2
    e = c2.get(0, 0, 700)
    assert e.reach_after == 780
    assert [(s.frames, s.buttons) for s in e.plan] == [(s.frames, s.buttons) for s in _plan()]


def test_fail_drops_entry_for_research(tmp_path):
    c = SolutionCache(path=tmp_path / "solutions.jsonl")
    c.put(0, 0, 700, _plan(), reach_after=780)
    assert c.get(0, 0, 700) is not None
    c.record_fail(0, 0, 700)
    assert c.get(0, 0, 700) is None  # dropped so search refreshes it


def test_solved_frontier(tmp_path):
    c = SolutionCache(path=tmp_path / "solutions.jsonl")
    c.put(0, 0, 300, _plan(), reach_after=360)
    c.put(0, 0, 900, _plan(), reach_after=980)
    c.put(0, 0, 600, _plan(), reach_after=660)
    # frontier = highest solved bucket in px (900 // 16 * 16 = 896)
    assert c.solved_frontier(0, 0) == (900 // 16) * 16
    assert c.solved_frontier(1, 1) == 0
