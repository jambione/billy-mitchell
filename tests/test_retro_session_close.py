"""RetroSession teardown must close the pyglet viewer so Remix doesn't leave zombie windows."""
from __future__ import annotations

from billy.systems.nes.retro_session import RetroSession


class _FakeViewer:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_retro_session_close_shuts_viewer():
    viewer = _FakeViewer()
    s = RetroSession.__new__(RetroSession)
    s._viewer = viewer
    s.env = type("E", (), {"close": lambda self: None})()
    RetroSession.close(s)
    assert viewer.closed
    assert s._viewer is None