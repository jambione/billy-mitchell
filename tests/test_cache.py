"""Unit tests for the position-keyed SolutionCache (the compounding-learning core).

The cache is game-agnostic: it keys on (level_key, progress_bucket) built from the engine's
generic Observation fields, so any future game reuses it unchanged.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.abstractions import Step  # noqa: E402
from billy.knowledge.cache import SolutionCache, bucket_of  # noqa: E402
from billy.systems.nes import controller  # noqa: E402

LK = (0, 0)  # a level_key (game-agnostic tuple)


def _plan():
    return [Step(4, controller.mask(controller.RIGHT, controller.B)),
            Step(28, controller.mask(controller.RIGHT, controller.B, controller.A))]


def test_bucket_quantizes_by_tile():
    # key is (level, x_band, y_band) — y defaults to 0 (one band) so the x quantization is unchanged
    assert bucket_of(LK, 0) == ((0, 0), 0, 0)
    assert bucket_of(LK, 15) == ((0, 0), 0, 0)
    assert bucket_of(LK, 16) == ((0, 0), 1, 0)
    assert bucket_of((1, 2), 700) == ((1, 2), 43, 0)
    # the y band disambiguates a high road from a low road at the SAME x
    assert bucket_of(LK, 16, 0) != bucket_of(LK, 16, 128)


def test_put_get_and_keep_better(tmp_path):
    c = SolutionCache(path=tmp_path / "solutions.jsonl")
    c.put(LK, 700, _plan(), reach_after=780)
    e = c.get(LK, 700)
    assert e is not None and e.reach_after == 780
    # same bucket (700-703 all map to bucket 43), worse reach -> not replaced
    c.put(LK, 701, [Step(8, controller.RIGHT)], reach_after=720)
    assert c.get(LK, 700).reach_after == 780
    # same bucket, better reach -> replaced
    c.put(LK, 703, [Step(8, controller.RIGHT)], reach_after=900)
    assert c.get(LK, 700).reach_after == 900


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "solutions.jsonl"
    c = SolutionCache(path=p)
    c.put(LK, 700, _plan(), reach_after=780)
    c.put((4, 1), 1200, [Step(16, controller.A)], reach_after=1260)
    # reload from disk
    c2 = SolutionCache(path=p)
    assert len(c2) == 2
    e = c2.get(LK, 700)
    assert e.reach_after == 780
    assert [(s.frames, s.buttons) for s in e.plan] == [(s.frames, s.buttons) for s in _plan()]
    # other-level entry survived too
    assert c2.get((4, 1), 1200).reach_after == 1260


def test_fail_drops_entry_for_research(tmp_path):
    c = SolutionCache(path=tmp_path / "solutions.jsonl")
    c.put(LK, 700, _plan(), reach_after=780)
    assert c.get(LK, 700) is not None
    c.record_fail(LK, 700)
    assert c.get(LK, 700) is None  # dropped so search refreshes it


def test_solved_frontier_is_per_level(tmp_path):
    c = SolutionCache(path=tmp_path / "solutions.jsonl")
    c.put(LK, 300, _plan(), reach_after=360)
    c.put(LK, 900, _plan(), reach_after=980)
    c.put(LK, 600, _plan(), reach_after=660)
    assert c.solved_frontier(LK) == (900 // 16) * 16  # highest solved bucket in px
    assert c.solved_frontier((1, 1)) == 0             # different level: nothing solved


def test_game_scoped_entries_do_not_collide(tmp_path):
    """smb and smb_lost share level_key (0,0,0) at 1-1 — cache must not cross-pollute."""
    p = tmp_path / "solutions.jsonl"
    smb = SolutionCache(path=p, game_id="smb")
    lost = SolutionCache(path=p, game_id="smb_lost")
    lk = (0, 0, 0)
    smb.put(lk, 700, _plan(), reach_after=780)
    lost.put(lk, 1039, _plan(), reach_after=1082)
    assert smb.get(lk, 700).reach_after == 780
    assert smb.get(lk, 1039) is None
    assert lost.get(lk, 1039).reach_after == 1082
    assert lost.get(lk, 700) is None
    # both persist in one file
    smb3 = SolutionCache(path=p, game_id="smb")
    lost3 = SolutionCache(path=p, game_id="smb_lost")
    assert smb3.get(lk, 700).reach_after == 780
    assert lost3.get(lk, 1039).reach_after == 1082


def test_nearby_reaching_finds_high_reach_entry_behind(tmp_path):
    """The demo-rot case: a 2510px demo at bucket 47 must be findable from bucket 51."""
    from billy.knowledge.cache import SolutionCache
    from billy.abstractions import Step
    cache = SolutionCache(path=tmp_path / "s.jsonl")
    lk = (0, 1, 2)
    cache.put(lk, 752, [Step(2724, 0x80)], reach_after=3266, y=200)   # the demo (yband 8)
    cache.put(lk, 816, [Step(6, 0x80)], reach_after=874, y=100)       # a short local hop
    # From x=816 (bucket 51), any elevation: the demo 4 buckets back qualifies…
    e = cache.nearby_reaching(lk, 816, min_gain=200)
    assert e is not None and e.reach_after == 3266
    # …but not from too far past it, and not when only low-gain entries are near.
    assert cache.nearby_reaching(lk, 950, min_gain=200) is None
    assert cache.nearby_reaching(lk, 816, min_gain=3000) is None
