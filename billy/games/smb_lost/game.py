"""SMB2-Japan game adapter — same SMB1 engine, so it inherits SMB's perception + reflex wholesale
and only points at the SMB2-Japan stable-retro integration."""
from __future__ import annotations

from ..smb.game import SmbGame


class SmbLostGame(SmbGame):
    name = "Super Mario Bros 2 (Japan)"
    RETRO_GAME = "SuperMarioBros2Japan-Nes-v0"

    def remix_director_sections(self):
        from ...rl.section_policy import SectionController, default_smb_lost_sections
        return SectionController(default_smb_lost_sections(), game_id="smb_lost")

    def remix_anchor_ok(self, source: str) -> bool:
        """Approach captures at the pit lip are valid tape anchors (moving-hazard parity)."""
        return source.startswith("start of") or source.startswith("approach")
