"""Zelda-specific tuning constants."""

from __future__ import annotations

# Screen-edge thresholds (pixels, same scale as link_x RAM).
SCREEN_EDGE_LO = 16
SCREEN_EDGE_HI = 220

# Overworld screen ids increment +1 east, +16 south on the 16x8 grid.
SCREEN_EAST = 1
SCREEN_SOUTH = 16
SCREEN_NORTH = -16
SCREEN_WEST = -1

# Known dungeon entrance overworld screens (Level 1–9). Billy biases toward these once adjacent.
DUNGEON_ENTRANCE_SCREENS: frozenset[int] = frozenset({
    6, 33, 40, 44, 45, 54, 58, 66, 67,   # common entrance tiles (approx overworld ids)
})

# Link's first overworld screen (has the north cave mouth).
START_SCREEN = 119

# Level-1 Eagle entrance — north forest cave stairs (screen 0x06).
LEVEL_1_ENTRANCE = 6

# FAQ map (8,4) entrance marker and east-coast march target.
LEVEL_1_OVERWORLD = 55
SEA_EAST_SCREEN = 127

# Screen ids that cannot be reached via a cardinal step (cliffs, water, etc.).
BLOCKED_NEIGHBORS: dict[int, frozenset[int]] = {
    119: frozenset({119 + SCREEN_SOUTH}),   # start screen — south is a cliff
}

# Combat
ATTACK_RANGE = 48
RETREAT_HEALTH = 1          # back off at 1 heart
STUCK_FRAMES = 120
REFLEX_FRAMES = 8

# Anti-oscillation: don't immediately walk back onto these screens.
BACKTRACK_MEMORY = 4