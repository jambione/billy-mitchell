"""Pixel perception — playing from the screen instead of a RAM map.

`core` holds pure frame primitives (scroll estimation, blob detection, fingerprints);
`tracker` holds the stateful PixelTracker that turns a frame stream into the signals the
engine needs (progress, player position, on_ground, death, level identity). The goal: a
game adapter built ONLY on these lets Billy learn any side-scroller without RAM offsets.
"""
from .core import estimate_scroll, find_blobs, frame_fingerprint, to_gray
from .tracker import PixelTracker, PixelView

__all__ = ["estimate_scroll", "find_blobs", "frame_fingerprint", "to_gray",
           "PixelTracker", "PixelView"]
