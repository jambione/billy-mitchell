"""The NES system: file-IPC transport + NES controller + FCEUX launch."""
from __future__ import annotations

from pathlib import Path

from ... import config, ipc
from ...abstractions import Session, System
from .controller import NesController

BRIDGE_LUA = Path(__file__).resolve().parent / "bridge.lua"
RAM_SIZE = 0x800


class NesSystem(System):
    name = "nes"
    ram_size = RAM_SIZE

    def __init__(self) -> None:
        self.controller = NesController()

    def connect(self) -> Session:
        """A lock-step file-IPC session to the FCEUX bridge."""
        return ipc.Bridge()

    def launch_command(self, rom: str) -> str:
        return f"fceux --loadlua {BRIDGE_LUA} {rom}"
