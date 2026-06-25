"""Tests for the cross-game Skill layer (embedding-retrieved transferable tactics).

Runs with the embedder offline (no LM Studio in CI), so it exercises the graceful flat-fallback
path: skills still instantiate into search candidates.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.games.common.platformer import PhysicsProfile  # noqa: E402
from billy.games.smb.perception import build_scene  # noqa: E402
from billy.knowledge.skills import STARTER_SKILLS, SkillLibrary  # noqa: E402


def test_seed_starter_is_idempotent(tmp_path):
    lib = SkillLibrary(path=tmp_path / "skills.jsonl").seed_starter()
    assert len(lib) == len(STARTER_SKILLS)
    lib.seed_starter()  # again
    assert len(lib) == len(STARTER_SKILLS)  # no duplicates


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "skills.jsonl"
    SkillLibrary(path=p).seed_starter()
    lib2 = SkillLibrary(path=p)
    assert len(lib2) == len(STARTER_SKILLS)
    assert {s.kind for s in lib2.skills} == {"gap_jump", "stomp", "wall_jump"}


def test_candidates_instantiate_into_plans(tmp_path):
    lib = SkillLibrary(path=tmp_path / "skills.jsonl").seed_starter()
    scene = build_scene(bytes(0x800), 0)
    cands = lib.candidates(scene, PhysicsProfile(), summary="enemy ahead, pit ahead", k=3)
    assert cands and all(isinstance(c, list) and c for c in cands)


def test_empty_library_yields_no_candidates(tmp_path):
    lib = SkillLibrary(path=tmp_path / "skills.jsonl")  # not seeded
    scene = build_scene(bytes(0x800), 0)
    assert lib.candidates(scene, PhysicsProfile(), summary="anything") == []
