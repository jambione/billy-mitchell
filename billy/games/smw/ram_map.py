"""Super Mario World (SNES) RAM map — WRAM-relative offsets ($7E0000 base).

Source: the community SMW RAM map (SMWCentral), addresses quoted as $7E:xxxx and used here as
offsets into the session's `ram` (stable-retro's Snes9x `get_ram()` exposes the 128KB WRAM
first, so $7Exxxx == ram[0xxxxx]).

STATUS: UNVERIFIED against a live ROM (roms/smw.sfc not present). Every offset below is
well-established in the SMW community, but the first live boot must run `probe_smw_ram.py`
to confirm the WRAM base holds — see games/smw/STATUS.md for the checklist.
"""

# --- player -------------------------------------------------------------------------------
PLAYER_X = 0x0094          # 16-bit little-endian: x position within the level
PLAYER_Y = 0x0096          # 16-bit little-endian: y position (larger = lower)
PLAYER_STATE = 0x0071      # animation/lock state: 0 = normal control, 9 = dying
PLAYER_IN_AIR = 0x0072     # nonzero = airborne
POWERUP = 0x0019           # 0 small, 1 big, 2 cape, 3 fire
LIVES = 0x0DBE
COINS = 0x0DBF
SCORE = 0x0F34             # 3 bytes little-endian, value = score / 10

# --- game state ---------------------------------------------------------------------------
GAME_MODE = 0x0100         # 0x14 = playing a level
TRANSLEVEL = 0x13BF        # current translevel number (level identity)
END_LEVEL_TIMER = 0x1493   # nonzero during the end-of-level walk (goal tape crossed)
EVENTS_TRIGGERED = 0x1F2E  # count of overworld events triggered — MONOTONIC level-clear counter
ON_GROUND = 0x13EF         # nonzero = standing on ground (incl. sprites/slopes)

# --- sprites (12 slots) -------------------------------------------------------------------
SPRITE_COUNT = 12
SPRITE_ID = 0x009E         # sprite number per slot (0 = none only with status empty)
SPRITE_STATUS = 0x14C8     # 0 = empty; 8+ = alive/active
SPRITE_X_LO = 0x00E4
SPRITE_X_HI = 0x14E0
SPRITE_Y_LO = 0x00D8
SPRITE_Y_HI = 0x14D4

GAME_MODE_LEVEL = 0x14
