"""Named Zelda savestates for ROM probes, tests, and hazard-scoped RL (B1)."""
from __future__ import annotations

import json
from pathlib import Path

from ...abstractions import Observation

STATES_DIR = Path(__file__).resolve().parents[3] / "data" / "zelda" / "states"
MANIFEST = STATES_DIR / "manifest.json"

# Milestone names the east-march driver captures after a successful Director run.
EAST_MARCH_MILESTONES = (
    "east_row8_sword",
    "east_row8_screen_123",
    "east_row8_screen_124",
    "east_row8_screen_127",
    "level1_entrance_overworld",
)


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    return json.loads(MANIFEST.read_text())


def state_path(name: str) -> Path:
    return STATES_DIR / f"{name}.state"


def restore_named_state(session, name: str) -> None:
    """Load a named .state into an open session."""
    path = state_path(name)
    if not path.exists():
        raise FileNotFoundError(f"named state not found: {path}")
    with path.open("rb") as f:
        session.reset()
        session.env.em.set_state(f.read())
        session._refresh_ram()


def list_named_states() -> list[dict]:
    return load_manifest()


def manifest_row(name: str) -> dict | None:
    for row in load_manifest():
        if row.get("name") == name:
            return row
    return None