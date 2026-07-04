"""ShmupTracker on synthetic fixed-camera frames (no emulator).

The synthetic world: a black space background with a bright ship low on screen and a couple of
enemy sprites up top. These pin the contracts the shmup adapter needs — appearance-based
sprite isolation (motion-free), player tracking of an IDLE ship, monotonic survival progress,
and the area-collapse death signal.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.vision.shmup import ShmupTracker, bright_blobs

H, W = 224, 320


def _space(ship=(150, 190), enemies=((80, 60), (220, 70))):
    """Black field, a bright ship, and enemy sprites."""
    f = np.zeros((H, W, 3), dtype=np.uint8)
    if ship is not None:
        x, y = ship
        f[y:y + 12, x:x + 16] = (80, 160, 250)      # blue-white ship
    for (ex, ey) in enemies:
        f[ey:ey + 14, ex:ex + 16] = (230, 60, 60)   # red enemy
    return f


def test_bright_blobs_finds_sprites_without_motion():
    blobs = bright_blobs(_space(), hud_rows=24)
    assert len(blobs) >= 3, "ship + two enemies should each be a blob"


def test_tracks_an_idle_ship():
    """A ship that never moves must stay tracked — appearance, not motion."""
    t = ShmupTracker()
    last = None
    for i in range(20):
        last = t.update(_space(ship=(150, 190)), frame_no=i * 4)
    assert last.player is not None, "idle ship lost — appearance tracking failed"
    px, py = last.player[0], last.player[1]
    assert abs(px - 150) < 24 and abs(py - 190) < 24


def test_survival_progress_is_monotonic():
    t = ShmupTracker()
    prog = [t.update(_space(), frame_no=i).progress for i in range(15)]
    assert prog[-1] > prog[0]
    assert all(b >= a for a, b in zip(prog, prog[1:]))


def test_area_collapse_reads_as_death():
    t = ShmupTracker()
    for i in range(12):                          # populated field builds a baseline
        v = t.update(_space(), frame_no=i)
        assert not v.dead
    # the ship and enemies vanish (a death clears the field): area collapses
    v = t.update(_space(ship=None, enemies=()), frame_no=12)
    v = t.update(_space(ship=None, enemies=()), frame_no=13)
    assert v.dead, "a sprite-area collapse must read as death"


def test_rebase_clears_death_and_progress():
    t = ShmupTracker()
    for i in range(12):
        t.update(_space(), frame_no=i)
    t.update(_space(ship=None, enemies=()), frame_no=12)
    t.rebase()
    v = t.update(_space(), frame_no=0)
    assert not v.dead and v.progress == 1
