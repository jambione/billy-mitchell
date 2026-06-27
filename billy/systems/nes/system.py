"""The NES system: in-process stable-retro transport + NES controller."""
from __future__ import annotations

from ...abstractions import Session, System
from .controller import NesController
from .retro_session import RetroSession

RAM_SIZE = 0x800


class NesSystem(System):
    name = "nes"
    ram_size = RAM_SIZE

    def __init__(self, retro_game: str | None = None, retro_inttype=None) -> None:
        self.controller = NesController()
        self.retro_game = retro_game   # stable-retro integration id (None -> RetroSession default)
        self.retro_inttype = retro_inttype

    def connect(self) -> Session:
        """An in-process stable-retro session (no external emulator, no file IPC)."""
        return RetroSession(game=self.retro_game, inttype=self.retro_inttype)

    def launch_command(self, rom: str) -> str:
        # Kept for the System ABC; stable-retro runs in-process, nothing to launch.
        return "python -m retro.import roms/   # one-time ROM import"
