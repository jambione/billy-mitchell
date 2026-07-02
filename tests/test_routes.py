"""Route memory: discovered level topology persists, aggregates, and spots warps."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.knowledge.routes import RouteEdge, RouteGraph


def test_record_and_roundtrip(tmp_path):
    path = tmp_path / "routes.jsonl"
    g = RouteGraph(path=path)
    g.record((0, 0, 0), (0, 1, 1), "clear", at=3266, dst_label="1-2")
    g.record((0, 1, 1), (0, 1, 2), "screen", at=90, dst_label="1-2")
    g2 = RouteGraph(path=path)
    assert len(g2) == 2
    edges = g2.edges_from((0, 0, 0))
    assert len(edges) == 1 and edges[0].dst == (0, 1, 1) and edges[0].kind == "clear"


def test_repeat_observations_bump_hits_not_edges(tmp_path):
    g = RouteGraph(path=tmp_path / "routes.jsonl")
    for _ in range(3):
        g.record((0, 0, 0), (0, 1, 1), "clear", at=3266)
    assert len(g) == 1
    assert g.edges_from((0, 0, 0))[0].hits == 3


def test_warp_detection(tmp_path):
    g = RouteGraph(path=tmp_path / "routes.jsonl")
    # Sequential clear: 1-2 -> 1-3 is NOT a warp.
    g.record((0, 1, 2), (0, 2, 3), "clear", at=3266, dst_label="1-3")
    # Skip: 1-2 -> 4-1 IS a warp (the 1-2 warp zone).
    g.record((0, 1, 2), (3, 0, 0), "clear", at=3350, dst_label="4-1")
    warps = g.warps()
    assert len(warps) == 1 and warps[0].dst == (3, 0, 0)
    assert "WARP" in g.describe()


def test_non_ordinal_keys_never_warp(tmp_path):
    g = RouteGraph(path=tmp_path / "routes.jsonl")
    g.record(("overworld", 119), ("overworld", 62), "screen", at=2900)
    assert g.warps() == []


def test_self_and_empty_edges_ignored(tmp_path):
    g = RouteGraph(path=tmp_path / "routes.jsonl")
    g.record((0, 0, 0), (0, 0, 0), "screen", at=10)
    g.record((), (0, 1, 1), "clear", at=10)
    assert len(g) == 0


def test_corrupt_lines_skipped(tmp_path):
    path = tmp_path / "routes.jsonl"
    path.write_text('{"src": [0,0,0], "dst": [0,1,1], "kind": "clear", "at": 3266}\n'
                    'not json at all\n'
                    '{"missing": "fields"}\n')
    g = RouteGraph(path=path)
    assert len(g) == 1
