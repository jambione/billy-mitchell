"""SMB Tier-1 reflex policy.

The whole platformer policy now lives in the game-neutral `games/common/platformer.py`
(`PlatformerReflex`); SMB just supplies its physics profile. SMB's `Scene` already implements the
`PlatformerView` probe surface, so no adapter glue is needed. Behaviour is identical to the previous
hand-written SMB reflex — the 1-1 clear benchmark is the regression guard.
"""
from __future__ import annotations

from ..common.platformer import PlatformerReflex
from .tuning import PROFILE


class SmbReflex(PlatformerReflex):
    def __init__(self) -> None:
        super().__init__(PROFILE)
