"""Rolling LLM memory tier tests (no LLM calls)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.agents.rolling_memory import RollingGameMemory  # noqa: E402


def test_memory_accumulates_and_prompts():
    mem = RollingGameMemory(rollup_every=100)
    mem.note("entered cave")
    mem.note("got sword")
    text = mem.prompt_section()
    assert "entered cave" in text
    assert "got sword" in text


def test_memory_reset_clears():
    mem = RollingGameMemory()
    mem.note("a")
    mem.long_summary = "old run"
    mem.reset()
    assert not mem.short_term
    assert mem.long_summary == ""
    assert mem.turn_count == 0