"""B1 fixture catalog tests (no ROM)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.games.zelda.fixtures import (  # noqa: E402
    EAST_MARCH_MILESTONES,
    MANIFEST,
    load_manifest,
    state_path,
)


def test_east_march_milestone_names():
    assert "east_row8_screen_123" in EAST_MARCH_MILESTONES
    assert "east_row8_screen_127" in EAST_MARCH_MILESTONES


def test_manifest_path_under_data():
    assert MANIFEST.name == "manifest.json"
    assert "zelda" in str(MANIFEST)


def test_state_path_helper():
    p = state_path("east_row8_sword")
    assert p.suffix == ".state"
    assert p.name == "east_row8_sword.state"


def test_load_manifest_is_list():
    assert isinstance(load_manifest(), list)