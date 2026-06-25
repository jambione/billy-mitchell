"""Super Mario Bros reflex tuning — physics-derived constants for the SMB policy.

Kept with the game (not the engine) so each title can tune its own feel.
"""
# Reflex cadence / stall detection
REFLEX_STEP_FRAMES = 4       # frames advanced per routine reflex exchange
BUMP_FRAMES = 14             # brief on-ground stall => hop (catch unseen blocks)
STUCK_FRAMES = 80            # prolonged no-progress => escalate to Billy

# Gap-aware jump (launch near the pit edge; A-hold scales with pit width)
JUMP_TRIGGER_PX = 28
JUMP_BASE_FRAMES = 18
JUMP_PER_TILE_FRAMES = 4
JUMP_MIN_FRAMES = 16
JUMP_MAX_FRAMES = 34

# Airborne control
AIRBORNE_STEP_FRAMES = 6     # carry momentum right when no landing target
AIR_STEER_FRAMES = 3         # short steps while steering toward a landing spot

# Wall / pipe / stair jumping
OBSTACLE_TRIGGER_PX = 24
OBSTACLE_BASE_FRAMES = 24
OBSTACLE_PER_HEIGHT_FRAMES = 5

# Stomping enemies
STOMP_RANGE = 60             # commit early so Mario is above the enemy on contact
STOMP_HOLD_FRAMES = 16

# Bonking ? blocks / bricks for coins & power-ups
BONK_TRIGGER_PX = 14
BONK_HOLD_FRAMES = 22
