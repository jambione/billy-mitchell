"""PRG0 RAM addresses for stable-retro LegendOfZeldaPRG0-Nes.

Documented here for the Integration UI workflow (B1): map variables in the UI,
export to data.json, and keep this module as the source of truth for perception.py.
Addresses are byte offsets into NES work RAM (0x0000-0x07FF).
"""
from __future__ import annotations

# Link / world
LINK_X = 0x70          # 112
LINK_Y = 0x84          # 132
LINK_DIRECTION = 0x98  # 152
GAME_MODE = 0x12       # 18 — 5/7 overworld, 11 cave interior, etc.
CURRENT_LEVEL = 0x10   # 16 — 0 overworld, 1-8 dungeon
MAP_LOCATION = 0xEB    # 235 — overworld screen or dungeon room id
NEXT_LOCATION = 0xEC   # 236
SCROLLING = 0xE8       # 232 — 255 when not scrolling

# Inventory / progress
HEARTS_BYTE = 0x667    # 1647 — low nibble health, high max-1
PARTIAL_HEART = 0x668    # 1648
TRIFORCE_PIECES = 0x669  # 1649
SWORD_LEVEL = 0x657      # 1623
BOMBS = 0x658            # 1624
RUPEES = 0x66D           # 1645
KEYS = 0x66E             # 1646

# Dungeon exploration map (compass / room revealed bits)
DUNGEON_MAP_START = 0x640  # 1600 — 64 bytes, one bit per dungeon room
COMPASS_FLAGS = 0x667      # overlap with hearts in some docs — use room flags below

# Room state (dungeon doors / blocks — probe with Integration UI for Level 1)
ROOM_STATE = 0xF9          # 249
ROOM_STATE2 = 0xFA         # 250

# Enemies @ 113/133 + types @ 848; ground items @ 173-178
ENEMY_X_BASE = 0x71
ENEMY_Y_BASE = 0x85
ENEMY_TYPE_BASE = 0x350

# Visited overworld screens history
VISITED_SCREENS_BASE = 0x621  # 1569

# stable-retro data.json export shape (info variables)
RETRO_DATA_JSON: dict = {
    "info": {
        "link_x": {"address": LINK_X, "type": "|u1"},
        "link_y": {"address": LINK_Y, "type": "|u1"},
        "game_mode": {"address": GAME_MODE, "type": "|u1"},
        "current_level": {"address": CURRENT_LEVEL, "type": "|u1"},
        "map_location": {"address": MAP_LOCATION, "type": "|u1"},
        "sword_level": {"address": SWORD_LEVEL, "type": "|u1"},
        "keys": {"address": KEYS, "type": "|u1"},
        "bombs": {"address": BOMBS, "type": "|u1"},
        "rupees": {"address": RUPEES, "type": "|u1"},
        "room_state": {"address": ROOM_STATE, "type": "|u1"},
    }
}