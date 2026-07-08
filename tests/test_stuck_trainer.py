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
    tracker.note_death("smb", "1-3", 769, frontier=576)
    tracker.note_death("smb", "1-3", 771, frontier=580)
    rec = tracker.stuck_at("smb", "1-3", 769, threshold=2)
    assert rec is not None
    assert rec.deaths == 2
    assert rec.last_death_x == 771


def test_stuck_tracker_scopes_by_game(tmp_path):
    tracker = StuckTracker(path=tmp_path / "stuck.json")
    tracker.note_death("smb", "1-1", 100, frontier=80)
    tracker.note_death("smb_lost", "1-1", 1057, frontier=3152)
    assert tracker.stuck_at("smb", "1-1", 100, threshold=1).deaths == 1
    assert tracker.stuck_at("smb_lost", "1-1", 1057, threshold=1).deaths == 1
    assert tracker.stuck_at("smb_lost", "1-1", 100, threshold=1) is None


def test_frontier_advance_resets_death_count(tmp_path):
    tracker = StuckTracker(path=tmp_path / "stuck.json")
    tracker.note_death("smb", "1-3", 769, frontier=576)
    tracker.note_death("smb", "1-3", 769, frontier=576)
    tracker.note_death("smb", "1-3", 769, frontier=900)
    rec = tracker.stuck_at("smb", "1-3", 769, threshold=2)
    assert rec is None


def test_remediation_marks_done(tmp_path):
    tracker = StuckTracker(path=tmp_path / "stuck.json")
    tracker.note_death("smb", "1-3", 769, frontier=576)
    tracker.note_death("smb", "1-3", 769, frontier=576)
    tracker.mark_remediated("smb", "1-3", 769)
    assert tracker.stuck_at("smb", "1-3", 769, threshold=2) is None


def test_mark_level_remediated_clears_all_buckets(tmp_path):
    tracker = StuckTracker(path=tmp_path / "stuck.json")
    tracker.note_death("smb", "3-4", 525, frontier=288)
    tracker.note_death("smb", "3-4", 525, frontier=288)
    tracker.note_death("smb", "3-4", 800, frontier=288)  # different bucket
    n = tracker.mark_level_remediated("smb", "3-4")
    assert n == 2
    assert tracker.stuck_at("smb", "3-4", 525, threshold=1) is None
    assert tracker.stuck_at("smb", "3-4", 800, threshold=1) is None


def test_new_death_after_remediate_starts_fresh_streak(tmp_path):
    tracker = StuckTracker(path=tmp_path / "stuck.json")
    tracker.note_death("smb", "3-4", 525, frontier=288)
    tracker.note_death("smb", "3-4", 525, frontier=288)
    tracker.mark_level_remediated("smb", "3-4")
    assert tracker.stuck_at("smb", "3-4", 525, threshold=2) is None
    tracker.note_death("smb", "3-4", 525, frontier=288)  # first death after teach
    assert tracker.stuck_at("smb", "3-4", 525, threshold=2) is None
    tracker.note_death("smb", "3-4", 525, frontier=288)  # second — stuck again
    assert tracker.stuck_at("smb", "3-4", 525, threshold=2) is not None


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


def test_smb_generic_platformer_stuck_remedy():
    hooks = SmbHazardHooks()
    remedy = hooks.stuck_remedy("1-1", 1057)
    assert remedy is not None
    assert remedy.goal_x == 1057 + 24
    assert remedy.bank_x_lo <= 1057 <= remedy.bank_x_hi + 64
    band = hooks.approach_capture_band("1-1", 1057)
    assert band is not None
    assert band[0] <= 1057 - 50 <= band[1]


def test_smb_approach_capture_band():
    hooks = SmbHazardHooks()
    band = hooks.approach_capture_band("1-3", 769)
    assert band is not None
    lo, hi = band
    assert lo <= 508 < hi
    assert lo < 769 < hi + 64


def test_smb_34_approach_capture_includes_jump_takeoff():
    """3-4 pit@525: Billy's last on-ground frame before the jump is ~359, not in a 100px band."""
    hooks = SmbHazardHooks()
    lo, hi = hooks.approach_capture_band("3-4", 525)
    assert lo <= 359 <= hi


def test_write_trail_snapshot(tmp_path):
    snap = b"\x00\x01fake-state"
    out = tmp_path / "auto" / "1_3_d48_x720.state"
    assert write_trail_snapshot(snap, str(out), level_label="1-3", approach_x=720)
    assert out.read_bytes() == snap


def test_tracker_persists(tmp_path):
    path = tmp_path / "stuck.json"
    t1 = StuckTracker(path=path)
    t1.note_death("smb", "1-3", 769, frontier=576)
    t2 = StuckTracker(path=path)
    assert ("smb", "1-3", 769 // CACHE_BUCKET_PX) in t2.records
    assert t2.records[("smb", "1-3", 769 // CACHE_BUCKET_PX)].game == "smb"