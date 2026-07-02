"""Tests for the learning ledger (compounding visibility)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.knowledge.cache import SolutionCache
from billy.learning import LearningLedger, format_learning_line


def test_ledger_counts_banks_and_drops(tmp_path):
    cache = SolutionCache(path=tmp_path / "solutions.jsonl")
    ledger = LearningLedger(path=tmp_path / "learning.jsonl")
    ledger.set_attempt_num(1)
    ledger.begin_attempt(cache)
    lk = (0, 0, 0)
    ledger.bank(lk, 100, 200, "search")
    ledger.drop(lk, 100, "replay_fail")
    ledger.replay()
    learn = ledger.finish_attempt(cache, lk)
    assert learn.banks == 1
    assert learn.drops == 1
    assert learn.replay_hits == 1
    assert "banked" in format_learning_line(learn, "1-1")