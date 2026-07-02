"""Route strategist: plans over the recorded graph, preferring warps toward completion.

The graph mechanics are game-agnostic; goal-directed ranking works for ordinal (SMB
world/stage) keys out of the box and degrades to frontier exploration otherwise."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.knowledge.routes import RouteGraph
from billy.strategist import RouteStrategist


def _linear(tmp_path) -> RouteGraph:
    g = RouteGraph(path=tmp_path / "routes.jsonl")
    g.record((0, 0, 0), (0, 1, 1), "clear", at=3266, dst_label="1-2")
    g.record((0, 1, 1), (0, 1, 2), "screen", at=90, dst_label="1-2b")
    g.record((0, 1, 2), (0, 2, 3), "clear", at=3266, dst_label="1-3")
    g.record((0, 2, 3), (0, 3, 4), "clear", at=2514, dst_label="1-4")
    return g


def test_linear_plan_follows_the_march(tmp_path):
    s = RouteStrategist(_linear(tmp_path))
    assert s.next_hop((0, 0, 0)) == (0, 1, 1)
    obj = s.objective((0, 0, 0))
    assert obj.kind == "advance" and not obj.via_warp and obj.target == (0, 1, 1)
    # The planned path reaches the furthest known level.
    assert s.best_path((0, 0, 0))[-1] == (0, 3, 4)


def test_warp_is_preferred_over_the_grind(tmp_path):
    g = _linear(tmp_path)
    # Discover the 1-2 warp zone: 1-2b ⤳ World 4 (skips ahead).
    g.record((0, 1, 2), (3, 0, 0), "clear", at=3350, dst_label="4-1")
    s = RouteStrategist(g)
    # From 1-2b the strategist should route to the warp (reaches higher rank), not 1-3.
    obj = s.objective((0, 1, 2))
    assert obj.via_warp and obj.target == (3, 0, 0), "strategist ignored the discovered warp"
    # The whole plan from 1-1 should go through the warp and reach World 4.
    assert s.best_path((0, 0, 0))[-1] == (3, 0, 0)
    assert any("⤳" in lbl for lbl in s.plan_labels((0, 0, 0)))


def test_prompt_section_lists_warps(tmp_path):
    g = _linear(tmp_path)
    g.record((0, 1, 2), (3, 0, 0), "clear", at=3350, dst_label="4-1")
    txt = RouteStrategist(g).prompt_section((0, 0, 0))
    assert "Route plan" in txt and "known warps" in txt


def test_unknown_location_advises_forward(tmp_path):
    s = RouteStrategist(RouteGraph(path=tmp_path / "routes.jsonl"))
    obj = s.objective((5, 5, 5))
    assert obj.target is None and obj.kind == "advance"
    assert s.prompt_section((5, 5, 5)) == ""


def test_non_ordinal_keys_degrade_to_frontier(tmp_path):
    # Zelda-style keys: default rank can't order them, so the planner still returns a path
    # (frontier exploration) without crashing, and prefers a warp if one exists.
    g = RouteGraph(path=tmp_path / "routes.jsonl")
    g.record(("overworld", 119), ("overworld", 120), "screen", at=2375, dst_label="#120")
    g.record(("overworld", 120), ("dungeon-1", 55), "clear", at=4184, dst_label="L1")
    s = RouteStrategist(g)
    path = s.best_path(("overworld", 119))
    assert path[0] == ("overworld", 119) and len(path) >= 2   # plans, doesn't crash
