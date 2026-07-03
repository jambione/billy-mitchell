"""Phantasy Star II (Genesis) work-RAM addresses — every entry probed against the live ROM
by probe_psii_ram.py (see each comment for the evidence). Genesis work RAM is 64KB; these are
offsets into `session.read_state().ram` (the $FF0000 bank).

Verified so far (explore-the-town milestone). Battle/party addresses (HP, EXP, meseta,
in-battle flag) are DEFERRED until Billy's own exploration reaches the overworld — the
follow-up probe will run from that checkpoint.
"""

# Player (lead character) map position in pixels, u16 big-endian. Probe: y 40->104 over 64
# DOWN frames; x 216->184 over 32 LEFT frames (1 px/frame walking, 16px tiles).
PLAYER_X_HI = 0xE409     # u16 BE at 0xE409-0xE40A
PLAYER_Y_HI = 0xE40D     # u16 BE at 0xE40D-0xE40E

# Place identity: two bytes that are stable within a place, distinct across places, and
# reversible (probe tour: Paseo=9/9, Rolf's house=6/6 — re-entering Paseo reads 9/9 again).
# Not semantically decoded (may be map-metadata indices); used only as a comparable key.
PLACE_A = 0xE9E4
PLACE_B = 0xE9EA

# 1 outdoors (town/overworld), 0 inside a building. Probe: flipped 1->0->1 across the
# doorstep -> house -> doorstep round trip, stable within each.
OUTDOOR = 0xD07B

# 1 while the field command menu is open, 0 otherwise. Probe: 0->1 on logical-A (physical C)
# tap, 1->0 on logical-B cancel, twice in a row (0xDE54/0xDE58/0xE000/0xE004 mirror it).
MENU_OPEN = 0xDE04


def u16(ram: bytes, addr: int) -> int:
    return (ram[addr] << 8) | ram[addr + 1]
