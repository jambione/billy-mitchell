"""Skill distillation: banked maneuvers become transferable sequence skills (search-only)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.abstractions import Step
from billy.knowledge.distill import MIN_GAIN_PX, distill_solution, plan_signature
from billy.knowledge.skills import SkillLibrary
from billy.systems.nes import controller as c

PLAN = [Step(20, c.mask(c.RIGHT, c.B)), Step(26, c.mask(c.RIGHT, c.A, c.B)), Step(10, c.RIGHT)]


def _lib(tmp_path):
    return SkillLibrary(path=tmp_path / "skills.jsonl")


def test_distills_significant_maneuver(tmp_path):
    lib = _lib(tmp_path)
    ok = distill_solution(lib, summary="pit ahead, platforms above", level_label="1-3",
                          plan=PLAN, start_x=630, reach=760, source="learn_section")
    assert ok and len(lib) == 1
    s = lib.skills[0]
    assert s.kind == "sequence"
    assert s.payload["console"] == "nes"
    assert s.payload["gained"] == 130


def test_gates_small_gains_and_trivial_plans(tmp_path):
    lib = _lib(tmp_path)
    assert not distill_solution(lib, summary="s", level_label="1-1",
                                plan=PLAN, start_x=100, reach=100 + MIN_GAIN_PX - 1,
                                source="search")
    assert not distill_solution(lib, summary="s", level_label="1-1",
                                plan=[Step(4, c.RIGHT)], start_x=100, reach=400,
                                source="search")   # <8 frames: trivial
    assert not distill_solution(lib, summary="", level_label="1-1",
                                plan=PLAN, start_x=100, reach=400, source="search")
    assert len(lib) == 0


def test_dedupes_identical_plans(tmp_path):
    lib = _lib(tmp_path)
    assert distill_solution(lib, summary="a", level_label="1-3", plan=PLAN,
                            start_x=630, reach=760, source="x")
    assert not distill_solution(lib, summary="b", level_label="2-1", plan=PLAN,
                                start_x=100, reach=300, source="y")
    assert len(lib) == 1


def test_sequence_skill_instantiates_exact_plan(tmp_path):
    lib = _lib(tmp_path)
    distill_solution(lib, summary="pit ahead", level_label="1-3", plan=PLAN,
                     start_x=630, reach=760, source="demo")
    plans = lib.skills[0].instantiate(view=None, profile=None)
    assert plans == [PLAN]


def test_console_gating_in_candidates(tmp_path):
    lib = _lib(tmp_path)
    distill_solution(lib, summary="pit ahead", level_label="1-3", plan=PLAN,
                     start_x=630, reach=760, source="demo", console="nes")
    # Same console: the sequence plan is offered. (profile=None is fine — sequence skills
    # don't consult the physics profile, and no other kinds are in this library.)
    assert lib.candidates(None, None, "pit ahead", console="nes") == [PLAN]
    # Different console: gated out (an NES mask means nothing on a SNES pad).
    assert lib.candidates(None, None, "pit ahead", console="snes") == []


def test_persistence_roundtrip_with_payload(tmp_path):
    lib = _lib(tmp_path)
    distill_solution(lib, summary="pit ahead", level_label="1-3", plan=PLAN,
                     start_x=630, reach=760, source="demo")
    lib2 = SkillLibrary(path=tmp_path / "skills.jsonl")
    assert len(lib2) == 1
    assert lib2.skills[0].payload["sig"] == plan_signature(PLAN)
    assert lib2.skills[0].instantiate(None, None) == [PLAN]


def test_starter_skills_unaffected(tmp_path):
    lib = _lib(tmp_path).seed_starter()
    n = len(lib)
    assert n >= 3
    assert all(not s.payload for s in lib.skills)
