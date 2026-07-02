"""KnowledgeBase lesson scoping: lessons are advice, and advice must not cross games.

The observed bug: an SMB pit tactic ("sprint then hold right+A") surfaced in a Zelda LLM
prompt because lessons had no game tag. These pin the fix (embeddings stubbed so the test is
deterministic and needs no LM Studio — the game filter runs before the similarity rank)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import billy.knowledge.store as store
from billy.knowledge.store import KnowledgeBase


def _no_embed(monkeypatch):
    monkeypatch.setattr(store, "_safe_embed", lambda text: [])


def test_retrieve_scopes_to_game(tmp_path, monkeypatch):
    _no_embed(monkeypatch)
    kb = KnowledgeBase(path=tmp_path / "lessons.jsonl")
    kb.add("pit after the first pipes in 1-1", "sprint then hold right+A", "cleared",
           "1-1", game="smb")
    kb.add("dungeon room with 3 stalfos", "kill all, grab the key", "cleared",
           "dungeon-1 #114", game="zelda")

    smb = kb.retrieve("anything", game="smb")
    assert len(smb) == 1 and smb[0].game == "smb"
    zelda = kb.retrieve("anything", game="zelda")
    assert len(zelda) == 1 and zelda[0].game == "zelda"
    assert all("pipes" not in l.situation for l in zelda), "SMB tactic leaked into Zelda"


def test_unscoped_retrieve_sees_all(tmp_path, monkeypatch):
    _no_embed(monkeypatch)
    kb = KnowledgeBase(path=tmp_path / "lessons.jsonl")
    kb.add("s1", "t1", "o", game="smb")
    kb.add("s2", "t2", "o", game="zelda")
    assert len(kb.retrieve("anything")) == 2   # no game arg = unfiltered (back-compat)


def test_legacy_untagged_lesson_is_dormant_and_never_leaks(tmp_path, monkeypatch):
    # A pre-scoping lesson persisted with game="".
    path = tmp_path / "lessons.jsonl"
    path.write_text('{"situation": "old smb spot", "tactic": "jump", "outcome": "ok", '
                    '"world_stage": "1-2", "game": "", "uses": 0, "impact_score": 0.0, '
                    '"embedding": []}\n')
    _no_embed(monkeypatch)
    kb = KnowledgeBase(path=path)
    assert kb.retrieve("x", game="smb") == []          # dormant under strict scope
    assert kb.retrieve("x", game="zelda") == []         # and never leaks to another game
    # Playing SMB re-learns the spot tagged, restoring SMB advice without touching Zelda.
    kb.add("old smb spot", "jump", "ok", "1-2", game="smb")
    assert len(kb.retrieve("x", game="smb")) == 1
    assert kb.retrieve("x", game="zelda") == []


def test_persisted_game_tag_roundtrips(tmp_path, monkeypatch):
    _no_embed(monkeypatch)
    path = tmp_path / "lessons.jsonl"
    KnowledgeBase(path=path).add("s", "t", "o", "1-1", game="smb")
    reloaded = KnowledgeBase(path=path)
    assert reloaded.lessons[0].game == "smb"
