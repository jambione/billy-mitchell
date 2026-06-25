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

# SMB's physics profile for the shared platformer reflex (games/common/platformer.py).
from ..common.platformer import PhysicsProfile  # noqa: E402

PROFILE = PhysicsProfile(
    reflex_step_frames=REFLEX_STEP_FRAMES, bump_frames=BUMP_FRAMES, stuck_frames=STUCK_FRAMES,
    jump_trigger_px=JUMP_TRIGGER_PX, jump_base_frames=JUMP_BASE_FRAMES,
    jump_per_tile_frames=JUMP_PER_TILE_FRAMES, jump_min_frames=JUMP_MIN_FRAMES,
    jump_max_frames=JUMP_MAX_FRAMES, airborne_step_frames=AIRBORNE_STEP_FRAMES,
    air_steer_frames=AIR_STEER_FRAMES, obstacle_trigger_px=OBSTACLE_TRIGGER_PX,
    obstacle_base_frames=OBSTACLE_BASE_FRAMES, obstacle_per_height_frames=OBSTACLE_PER_HEIGHT_FRAMES,
    stomp_range=STOMP_RANGE, stomp_hold_frames=STOMP_HOLD_FRAMES,
    bonk_trigger_px=BONK_TRIGGER_PX, bonk_hold_frames=BONK_HOLD_FRAMES,
)
