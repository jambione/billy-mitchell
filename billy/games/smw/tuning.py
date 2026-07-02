"""SMW physics profile — starting values, to be tuned on first live runs.

SMW's Mario accelerates more gradually than SMB1's and jumps float longer (plus the spin
jump / cape verbs live in the SNES controller). These numbers start from SMB1's profile with
the community-known differences applied; treat them as priors, not truth, until the 1-1
(Yoshi's Island 1/2) benchmark exists.
"""
from __future__ import annotations

from ...games.common.platformer import PhysicsProfile

SMW_PROFILE = PhysicsProfile(
    reflex_step_frames=4,
    bump_frames=14,
    stuck_frames=90,          # slower accel: give runs a beat longer before "stuck"
    jump_trigger_px=30,
    jump_base_frames=20,      # floatier arc than SMB1
    jump_per_tile_frames=4,
    jump_min_frames=16,
    jump_max_frames=38,
    airborne_step_frames=6,
    air_steer_frames=3,
    obstacle_trigger_px=24,
    obstacle_base_frames=26,
    obstacle_per_height_frames=5,
    stomp_range=60,
    stomp_hold_frames=16,
    bonk_trigger_px=14,
    bonk_hold_frames=22,
    enemy_react_px=72,
)
