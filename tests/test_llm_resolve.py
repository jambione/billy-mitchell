"""Chat-model auto-resolution against LM Studio's /models listing (no network)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import billy.llm as llm


class _Resp:
    def __init__(self, data):
        self._data = data
    def raise_for_status(self):
        pass
    def json(self):
        return {"data": [{"id": i} for i in self._data]}


class _FakeSession:
    def __init__(self, ids=None, fail=False):
        self.ids = ids or []
        self.fail = fail
    def get(self, url, timeout=None):
        if self.fail:
            import requests
            raise requests.ConnectionError("down")
        return _Resp(self.ids)


def _with(monkeypatch, session):
    monkeypatch.setattr(llm, "_session", session)
    monkeypatch.setattr(llm, "_resolved_chat_model", None)


def test_configured_model_used_when_listed(monkeypatch):
    _with(monkeypatch, _FakeSession([llm.config.CHAT_MODEL, "other-model"]))
    assert llm.resolve_chat_model() == llm.config.CHAT_MODEL


def test_falls_back_to_first_loaded_chat_model(monkeypatch):
    _with(monkeypatch, _FakeSession(["text-embedding-nomic-embed-text-v1.5",
                                     "qwen2.5-coder-7b-instruct"]))
    assert llm.resolve_chat_model() == "qwen2.5-coder-7b-instruct"   # embedding model skipped


def test_unreachable_server_keeps_configured_id_uncached(monkeypatch):
    _with(monkeypatch, _FakeSession(fail=True))
    assert llm.resolve_chat_model() == llm.config.CHAT_MODEL
    assert llm._resolved_chat_model is None   # not cached — retries next call


def test_resolution_cached_after_success(monkeypatch):
    _with(monkeypatch, _FakeSession(["some-model"]))
    first = llm.resolve_chat_model()
    monkeypatch.setattr(llm, "_session", _FakeSession(fail=True))
    assert llm.resolve_chat_model() == first
