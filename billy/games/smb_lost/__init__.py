"""Super Mario Bros 2 (Japan) / "The Lost Levels" — NES game plugin.

Runs the SMB1 engine, so it reuses SMB's perception (same RAM map) and the shared platformer
reflex unchanged; only the stable-retro integration id and label differ. This is the cross-game
transfer testbed: an empty SolutionCache but a SkillLibrary seeded from SMB tactics.
"""
from .game import SmbLostGame

__all__ = ["SmbLostGame"]
