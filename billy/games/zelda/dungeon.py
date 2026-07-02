"""Dungeon state decoding for Level 1+ adapter work (B1).

Feeds perception.py; addresses live in ram_map.py for Integration UI sync.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ram_map import DUNGEON_MAP_START, ROOM_STATE, ROOM_STATE2


def _u8(ram: bytes, addr: int) -> int:
    return ram[addr]


@dataclass(frozen=True)
class DungeonState:
    """Per-room dungeon signals Billy needs for keys/locks routing."""
    room_id: int
    keys_held: int
    room_state: int
    room_state2: int
    map_bits_set: int      # count of explored dungeon rooms (compass proxy)
    locked_doors: bool     # heuristic from room_state bits


def read_dungeon_state(ram: bytes, *, room_id: int, keys_held: int) -> DungeonState:
    room_state = _u8(ram, ROOM_STATE)
    room_state2 = _u8(ram, ROOM_STATE2)
    bits = sum(bin(_u8(ram, DUNGEON_MAP_START + i)).count("1") for i in range(64))
    # Low bits of room_state often track door/open state — treat nonzero as "something to solve"
    locked = (room_state & 0x0F) != 0 or (room_state2 & 0x03) != 0
    return DungeonState(
        room_id=room_id,
        keys_held=keys_held,
        room_state=room_state,
        room_state2=room_state2,
        map_bits_set=bits,
        locked_doors=locked,
    )


def dungeon_summary(state: DungeonState | None) -> str:
    if state is None:
        return ""
    lock = "locked" if state.locked_doors else "open"
    return (f"dungeon-room=#{state.room_id} keys={state.keys_held} "
            f"room_state=0x{state.room_state:02x} doors={lock} "
            f"map_bits={state.map_bits_set}")