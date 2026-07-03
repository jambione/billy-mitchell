"""Pixel perception primitives on synthetic frames (no emulator, no RAM map).

The synthetic world: a tiled background that scrolls, a sprite-sized player blob with its
own motion. These pin the core contracts — scroll estimation (with tile-period ambiguity),
sprite isolation, scene fingerprints, and the tracker's progress/on_ground/death logic.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.vision import PixelTracker, estimate_scroll, find_blobs, to_gray
from billy.vision.core import (background_color, color_histogram,
                               fingerprint_distance, ground_distance, motion_mask)

H, W = 224, 240
SKY = (92, 148, 252)      # SMB-ish sky
GROUND_Y = 192


def _world(width=2000, seed=7):
    """A deterministic tiled world strip: sky + textured ground + pillars."""
    rng = np.random.default_rng(seed)
    world = np.zeros((H, width, 3), dtype=np.uint8)
    world[:] = SKY
    # ground with per-tile texture
    for tx in range(0, width, 16):
        shade = 120 + int(rng.integers(0, 80))
        world[GROUND_Y:, tx:tx + 16] = (shade, shade // 2, 20)
    # pillars every ~150px with unique heights (breaks the 16px periodicity)
    for i, px in enumerate(range(80, width - 40, 150)):
        top = 120 + (i * 13) % 60
        world[top:GROUND_Y, px:px + 16] = (30 + i * 5 % 100, 200, 60)
    return world


def _frame(world, cam_x, player=None):
    """Crop the camera window; optionally stamp a player blob at screen coords."""
    f = world[:, cam_x:cam_x + W].copy()
    if player is not None:
        x, y = player
        f[y:y + 16, x:x + 12] = (252, 60, 60)
    return f


def test_scroll_estimation_exact_and_tile_ambiguity():
    world = _world()
    for dx in (0, 1, 3, 7, 12, 24, 40):
        a, b = _frame(world, 300), _frame(world, 300 + dx)
        assert estimate_scroll(a, b, expect=dx) == dx
        # even with a wrong-but-close expectation, unambiguous texture wins
        assert estimate_scroll(a, b, expect=max(0, dx - 3)) == dx


def test_motion_mask_isolates_the_sprite():
    world = _world()
    a = _frame(world, 300, player=(60, 176))
    b = _frame(world, 308, player=(64, 176))   # camera +8, player moves differently
    mask = motion_mask(a, b, 8)
    blobs = find_blobs(mask)
    assert blobs, "sprite motion not detected"
    x, y, w, h = blobs[0]
    assert abs(x - 58) < 20 and abs(y - 172) < 20   # around the player, roughly


def test_background_color_finds_sky():
    bg = background_color(_frame(_world(), 300))
    assert tuple(bg) == SKY


def test_fingerprint_stable_across_scroll_but_not_across_scenes():
    world = _world()
    h1 = color_histogram(_frame(world, 200))
    h2 = color_histogram(_frame(world, 800))
    assert fingerprint_distance(h1, h2) < 0.5     # same level, scrolled
    cave = np.zeros((H, W, 3), dtype=np.uint8)    # different scene: dark cave
    cave[:, :] = (12, 12, 24)
    cave[GROUND_Y:, :] = (80, 80, 96)
    h3 = color_histogram(cave)
    assert fingerprint_distance(h1, h3) > 0.75


def test_tracker_progress_follows_the_march():
    world = _world()
    t = PixelTracker()
    cam, px = 100, 80
    last = None
    for i in range(30):
        cam += 4                      # camera scrolls right
        f = _frame(world, cam, player=(px, 176))
        last = t.update(f, frame_no=i * 4)
    assert last.in_play
    assert last.player is not None, "player never acquired"
    # progress ≈ camera travel (120px) + on-screen x; require the right ballpark + monotone
    assert last.progress > 100
    assert last.camera_x == 30 * 4 - 4  # first update has no prev frame -> no scroll


def test_ground_distance_finds_the_ground_line():
    f = _frame(_world(), 300)
    bg = background_color(f)
    assert ground_distance(f, bg, 40, 60, GROUND_Y - 10) == 10   # ground 10 rows below
    assert ground_distance(f, bg, 40, 60, GROUND_Y + 4) == 0     # already inside ground
    assert ground_distance(f, bg, 40, 60, 100, max_scan=20) is None   # open sky: no support


def test_tracker_on_ground_from_ground_line():
    world = _world()
    # Standing: feet on the ground line while the camera scrolls.
    t = PixelTracker()
    cam, last = 100, None
    for i in range(20):
        cam += 4
        last = t.update(_frame(world, cam, player=(80, GROUND_Y - 16)), frame_no=i * 4)
    assert last.player is not None
    assert last.on_ground, "feet on the ground line must read grounded"
    # Stationary: no motion = no fresh blob, but ground support is static evidence.
    for i in range(20, 23):
        last = t.update(_frame(world, cam, player=(80, GROUND_Y - 16)), frame_no=i * 4)
    assert last.on_ground, "a player who stopped moving did not stop standing"
    # Airborne: same march with the player high in the sky.
    t2 = PixelTracker()
    cam = 100
    for i in range(20):
        cam += 4
        last = t2.update(_frame(world, cam, player=(80, 120)), frame_no=i * 4)
    assert last.player is not None
    assert not last.on_ground, "a player in open sky must not read grounded"


def test_tracker_death_on_pit_fall():
    world = _world()
    t = PixelTracker()
    y = 150
    for i in range(24):
        f = _frame(world, 300, player=(120, min(H - 17, y)))
        v = t.update(f, frame_no=i)
        y += 6                        # falling…
    # now the player is gone below the screen; a few more updates without them
    for i in range(24, 30):
        v = t.update(_frame(world, 300), frame_no=i)
    assert v.dead, "pit fall (fell below viewport, then lost) must read as dead"


def test_tracker_blackout_is_not_in_play():
    t = PixelTracker()
    world = _world()
    t.update(_frame(world, 300, player=(80, 176)), frame_no=0)
    black = np.zeros((H, W, 3), dtype=np.uint8)
    v = t.update(black, frame_no=1)
    assert not v.in_play


def test_tracker_area_change_bumps_identity_and_rebases():
    world = _world()
    t = PixelTracker()
    for i in range(6):
        t.update(_frame(world, 300 + i * 4, player=(80, 176)), frame_no=i)
    assert t.area_seq == 0
    cave = np.zeros((H, W, 3), dtype=np.uint8)
    cave[:, :] = (12, 12, 24)
    cave[GROUND_Y:, :] = (80, 80, 96)
    v = None
    for i in range(6, 12):
        v = t.update(cave, frame_no=i)
    assert v.area_seq == 1, "sustained scene change must bump the area identity"
    assert v.camera_x == 0, "progress re-bases in the new area"


def test_tracked_session_time_travels_the_tracker():
    """clone/restore must snapshot+restore TRACKER state alongside emulator state —
    otherwise invisible search rollouts desync pixel perception from the machine."""
    from billy.games.pixel.game import _TrackedSession

    class _FakeSession:
        def __init__(self):
            self.n = 0
        def clone_state(self):
            return bytes([self.n])
        def restore(self, snap):
            self.n = snap[0]
        def save_state(self, slot=0):
            pass
        def load_state(self, slot=0):
            pass

    t = PixelTracker()
    s = _TrackedSession(_FakeSession(), t)
    t.camera_x = 500
    snap = s.clone_state()
    t.camera_x = 900          # play/search moved on…
    s.restore(snap)           # …then the engine rewound
    assert t.camera_x == 500, "tracker did not time-travel with the emulator"
    s2 = _TrackedSession(_FakeSession(), t)
    s2.save_state(3)
    t.camera_x = 1234
    s2.load_state(3)
    assert t.camera_x == 500
