"""Tests for the auto-stuck trainer (death clustering + remedy wiring)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.config import CACHE_BUCKET_PX
from billy.games.smb.hazard_hooks import SmbHazardHooks
from billy.stuck_trainer import StuckTracker, auto_state_path, write_trail_snapshot


def test_stuck_tracker_clusters_deaths(tmp_path):
    path = tmp_path / "stuck.json"
    tracker = StuckTracker(path=path)
    tracker.note_death("1-3", 769, frontier=576)
    tracker.note_death("1-3", 771, frontier=580)
    rec = tracker.stuck_at("1-3", 769, threshold=2)
    assert rec is not None
    assert rec.deaths == 2
    assert rec.last_death_x == 771


def test_frontier_advance_resets_death_count(tmp_path):
    tracker = StuckTracker(path=tmp_path / "stuck.json")
    tracker.note_death("1-3", 769, frontier=576)
    tracker.note_death("1-3", 769, frontier=576)
    tracker.note_death("1-3", 769, frontier=900)
    rec = tracker.stuck_at("1-3", 769, threshold=2)
    assert rec is None


def test_remediation_marks_done(tmp_path):
    tracker = StuckTracker(path=tmp_path / "stuck.json")
    tracker.note_death("1-3", 769, frontier=576)
    tracker.note_death("1-3", 769, frontier=576)
    tracker.mark_remediated("1-3", 769)
    assert tracker.stuck_at("1-3", 769, threshold=2) is None


def test_auto_state_path_deterministic():
    p = auto_state_path("1-3", 769, 724)
    assert "1_3" in p
    assert "d48" in p  # 769 // 16
    assert "x724" in p


def test_smb_lift_stuck_remedy():
    hooks = SmbHazardHooks()
    remedy = hooks.stuck_remedy("1-3", 769)
    assert remedy is not None
    assert remedy.kind == "frame_search"
    assert remedy.goal_x >= 880
    assert len(remedy.savestate_paths) >= 2


def test_smb_approach_capture_band():
    hooks = SmbHazardHooks()
    band = hooks.approach_capture_band("1-3", 769)
    assert band is not None
    lo, hi = band
    assert lo <= 508 < hi
    assert lo < 769 < hi + 64


def test_write_trail_snapshot(tmp_path):
    snap = b"\x00\x01fake-state"
    out = tmp_path / "auto" / "1_3_d48_x720.state"
    assert write_trail_snapshot(snap, str(out), level_label="1-3", approach_x=720)
    assert out.read_bytes() == snap


def test_tracker_persists(tmp_path):
    path = tmp_path / "stuck.json"
    t1 = StuckTracker(path=path)
    t1.note_death("1-3", 769, frontier=576)
    t2 = StuckTracker(path=path)
    assert ("1-3", 769 // CACHE_BUCKET_PX) in t2.records