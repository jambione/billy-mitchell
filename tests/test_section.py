"""Unit tests for the hazard-scoped RL section controller — no emulator or model file needed.

These cover the wiring contract: graceful degradation when no model loads (so the reflex-only build
is untouched), section matching by (level_label, x-range, on-ground), and the action vocabulary.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.rl.section_env import SECTION_ACTIONS, N_SECTION_ACTIONS  # noqa: E402
from billy.rl.section_policy import (  # noqa: E402
    Section, SectionController, default_smb_sections,
)


def test_section_actions_wellformed():
    """Every action is (button-names tuple, positive hold-frames); jumps hold A for an arc."""
    assert N_SECTION_ACTIONS == len(SECTION_ACTIONS) >= 4
    for names, hold in SECTION_ACTIONS:
        assert isinstance(names, tuple)
        assert isinstance(hold, int) and hold > 0
    # at least one sustained-jump action (holds A for a real arc, not a 4-frame tap)
    assert any("A" in names and hold >= 12 for names, hold in SECTION_ACTIONS)


def test_default_sections_cover_1_3():
    secs = default_smb_sections()
    assert any(s.label == "1-3" and s.x_lo < s.goal_x for s in secs)


def test_controller_degrades_without_model():
    """A missing/unloadable model -> empty controller that is a pure no-op (reflex-only build safe)."""
    ctrl = SectionController([Section(label="1-3", x_lo=100, x_hi=560, goal_x=700,
                                      model_path="/nonexistent/model.zip")])
    assert len(ctrl) == 0

    class _Raw:
        on_ground = True
        is_dying = False

    class _Obs:
        level_label = "1-3"
        progress = 200
        raw = _Raw()

    # No registered model -> never matches, never touches the session.
    assert ctrl._match(_Obs()) is None
    assert ctrl.cross(_Obs(), session=None, observe=None) is None


def test_match_respects_label_range_and_ground():
    """_match only fires for the right level, within the x-band, and on the ground."""
    sec = Section(label="1-3", x_lo=100, x_hi=560, goal_x=700, model_path="x")
    ctrl = SectionController.__new__(SectionController)   # skip model loading
    ctrl.sections = [(sec, object())]                     # a stand-in "model"

    def obs(label, x, ground):
        raw = type("R", (), {"on_ground": ground, "is_dying": False})()
        return type("O", (), {"level_label": label, "progress": x, "raw": raw})()

    assert ctrl._match(obs("1-3", 200, True)) is not None
    assert ctrl._match(obs("1-3", 700, True)) is None      # past x_hi
    assert ctrl._match(obs("1-2", 200, True)) is None      # wrong level
    assert ctrl._match(obs("1-3", 200, False)) is None     # airborne
