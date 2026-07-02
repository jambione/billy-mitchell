"""Parallel micro-search plumbing (no emulator: chunking + order preservation)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.search_pool import MIN_CANDIDATES, split_chunks


def test_split_chunks_preserves_order_and_content():
    items = list(range(11))
    for n in (1, 2, 3, 4, 11, 20):
        chunks = split_chunks(items, n)
        flat = [x for c in chunks for x in c]
        assert flat == items, f"n={n} broke ordering"
        assert all(c for c in chunks), "no empty chunks"


def test_split_chunks_balanced():
    chunks = split_chunks(list(range(10)), 4)
    sizes = [len(c) for c in chunks]
    assert max(sizes) - min(sizes) <= 1


def test_min_candidates_threshold_is_sane():
    # Documented behavior: tiny candidate sets stay serial (IPC would beat the win).
    assert 2 <= MIN_CANDIDATES <= 16
