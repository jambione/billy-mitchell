"""The Genesis system: the SAME in-process retro transport, Genesis controller + RAM size.

Two console-specific bits beyond the SNES precedent:
- Custom integrations: titles stable-retro doesn't ship (e.g. Phantasy Star II) live in
  emulator/integrations/<Game>-Genesis-v0/ (rom.sha + metadata + Start.state anchor; the ROM
  itself is copied in as gitignored rom.md). This module registers that path once.
- Actions.ALL: the Genesis FILTERED action space silently strips START (verified against the
  core: `data.filter_action(START_bit) == 0`), which would make title screens and PSII's
  START-opened menu unreachable. Sessions are created unfiltered.
"""
from __future__ import annotations

from pathlib import Path

from stable_retro.data import Integrations

from ...abstractions import Session, System
from ..nes.retro_session import RetroSession
from . import controller as genesis_controller
from .controller import GenesisController

RAM_SIZE = 0x10000   # Genesis 68k work RAM: 64KB at $FF0000-$FFFFFF

_INTEGRATIONS_DIR = Path(__file__).resolve().parents[3] / "emulator" / "integrations"
_registered = False


def register_custom_integrations() -> None:
    """Idempotently add the repo's custom integration folder to stable-retro's search path."""
    global _registered
    if not _registered and _INTEGRATIONS_DIR.is_dir():
        Integrations.add_custom_path(str(_INTEGRATIONS_DIR))
        _registered = True


class GenesisSystem(System):
    name = "genesis"
    ram_size = RAM_SIZE

    def __init__(self, retro_game: str | None = None, retro_inttype=None) -> None:
        self.controller = GenesisController()
        self.retro_game = retro_game
        # ALL searches stock stable/experimental/contrib AND the repo's custom integrations.
        self.retro_inttype = Integrations.ALL if retro_inttype is None else retro_inttype
        register_custom_integrations()

    def connect(self) -> Session:
        import stable_retro as retro
        return RetroSession(game=self.retro_game, inttype=self.retro_inttype,
                            controller_mod=genesis_controller, ram_size=RAM_SIZE,
                            restricted_actions=retro.Actions.ALL)

    def launch_command(self, rom: str) -> str:
        return "python emulator/make_psii_state.py   # one-time custom-integration bring-up"
