"""The SNES system: the SAME in-process retro transport, SNES controller + WRAM size.

Proof of the architecture seam: `RetroSession` (systems/nes/retro_session.py) is
console-agnostic — button vocabulary and RAM size are constructor parameters. This module
supplies the SNES values; everything above the Session contract (Director, cache, tapes,
teleop, search) is untouched.
"""
from __future__ import annotations

from ...abstractions import Session, System
from ..nes.retro_session import RetroSession
from . import controller as snes_controller
from .controller import SnesController

RAM_SIZE = 0x20000   # SNES work RAM: 128KB at $7E0000-$7FFFFF (retro get_ram exposes it first)


class SnesSystem(System):
    name = "snes"
    ram_size = RAM_SIZE

    def __init__(self, retro_game: str | None = None, retro_inttype=None) -> None:
        self.controller = SnesController()
        self.retro_game = retro_game
        self.retro_inttype = retro_inttype

    def connect(self) -> Session:
        return RetroSession(game=self.retro_game, inttype=self.retro_inttype,
                            controller_mod=snes_controller, ram_size=RAM_SIZE)

    def launch_command(self, rom: str) -> str:
        return "python -m retro.import roms/   # one-time ROM import"
