"""Dungeon perception helpers (no ROM)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.games.zelda.dungeon import read_dungeon_state, dungeon_summary  # noqa: E402
from billy.games.zelda.ram_map import LINK_X, MAP_LOCATION, RETRO_DATA_JSON  # noqa: E402


def test_dungeon_state_from_ram():
    ram = bytearray(0x800)
    ram[MAP_LOCATION] = 12
    ram[16] = 1
    ram[1646] = 2
    ram[249] = 0x03
    st = read_dungeon_state(ram, room_id=12, keys_held=2)
    assert st.room_id == 12
    assert st.keys_held == 2
    assert st.locked_doors
    assert "dungeon-room" in dungeon_summary(st)


def test_ram_map_matches_data_json():
    assert RETRO_DATA_JSON["info"]["link_x"]["address"] == LINK_X
    assert RETRO_DATA_JSON["info"]["map_location"]["address"] == MAP_LOCATION