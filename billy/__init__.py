"""Billy Mitchell — an agentic NES game-player that learns.

A two-tier loop drives Super Mario Bros in FCEUX: a fast Python/Lua layer handles the
virtual controller + RAM perception + reflex execution every frame, while the LLM ("Billy")
is consulted only at decision points and a "Coach" distills lessons into a knowledge base.
"""
__all__ = ["config"]
