"""Tests for whole-trajectory tape recording/replay."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.abstractions import Step
from billy.knowledge.tape import TapeLibrary, append_plan
from billy.systems.nes import controller


def test_tape_persistence(tmp_path):
    path = tmp_path / "tapes.jsonl"
    lib = TapeLibrary(path=path)
    lk = (0, 0, 0)
    plan = [Step(4, controller.RIGHT), Step(8, controller.mask(controller.RIGHT, controller.A))]
    lib.put(lk, plan, frontier=900, clears_level=True)
    lib2 = TapeLibrary(path=path)
    e = lib2.get(lk)
    assert e is not None
    assert e.frontier == 900
    assert e.clears_level
    assert len(e.plan) == 2


def test_append_plan_extends_recording():
    record: list[Step] = []
    append_plan(record, [Step(2, 0), Step(4, controller.RIGHT)])
    append_plan(record, [Step(6, controller.A)])
    assert len(record) == 3
    assert record[1].frames == 4