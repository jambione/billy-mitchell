"""SMB2-Japan game adapter — same SMB1 engine, so it inherits SMB's perception + reflex wholesale
and only points at the SMB2-Japan stable-retro integration."""
from __future__ import annotations

from ..smb.game import SmbGame


class SmbLostGame(SmbGame):
    name = "Super Mario Bros 2 (Japan)"
    RETRO_GAME = "SuperMarioBros2Japan-Nes-v0"
