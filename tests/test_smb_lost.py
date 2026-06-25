"""SMB2-Japan (Lost Levels) adapter — boots and is controllable via the SAME SMB perception/reflex.

Requires the SMB2-Japan ROM to be imported into stable-retro (user-supplied, copyright). Skips
cleanly when the integration/ROM isn't present, so CI without the ROM stays green.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.games.smb_lost import SmbLostGame  # noqa: E402


def _has_rom() -> bool:
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import stable_retro as retro
        env = retro.make("SuperMarioBros2Japan-Nes-v0", render_mode="rgb_array")
        env.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_rom(), reason="SMB2-Japan ROM not imported into stable-retro")
def test_smb_lost_boots_and_advances():
    game = SmbLostGame()
    assert game.RETRO_GAME == "SuperMarioBros2Japan-Nes-v0"
    session = game.system.connect()
    session.wait_until_live()
    obs = game.boot(session)
    assert obs.level_label == "1-1" and obs.progress > 0 and not obs.dead
    # reuses SMB's reflex unchanged (same engine)
    from billy.games.common.platformer import PlatformerReflex
    assert isinstance(game.make_reflex(), PlatformerReflex)
    session.close()
