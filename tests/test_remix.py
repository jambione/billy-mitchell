"""Remix: dynamic multi-game wall queue — request parsing, resolve, frontier, roster."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import remix


def test_campaign_games_are_real():
    from run import GAMES
    for g in remix.CAMPAIGN:
        assert g in GAMES, f"campaign roster has unknown game {g}"
    for g in remix._ENDING:
        assert g in remix.CAMPAIGN


def test_read_requests_parses_and_skips_junk(tmp_path, monkeypatch):
    f = tmp_path / "demo_requests.jsonl"
    f.write_text('{"game":"smb","level_label":"1-3","death_bucket":48,"death_x":771}\n'
                 'not json\n'
                 '{"game":"zelda","level_label":"dungeon-1","death_bucket":10,"death_x":300}\n')
    monkeypatch.setattr(remix, "REQUESTS_FILE", f)
    reqs = remix._read_requests()
    assert [r["game"] for r in reqs] == ["smb", "zelda"]       # junk line dropped


def test_resolve_request_removes_only_the_taught_wall(tmp_path, monkeypatch):
    f = tmp_path / "demo_requests.jsonl"
    cleared = tmp_path / "cleared.jsonl"
    a = {"game": "smb", "level_label": "1-3", "death_bucket": 48, "death_x": 771}
    b = {"game": "smb", "level_label": "2-2", "death_bucket": 72, "death_x": 1158}
    f.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    monkeypatch.setattr(remix, "REQUESTS_FILE", f)
    monkeypatch.setattr(remix, "CLEARED_FILE", cleared)
    remix._resolve_request(a)
    left = remix._read_requests()
    assert len(left) == 1 and left[0]["level_label"] == "2-2"   # only 1-3 removed
    assert json.loads(cleared.read_text().strip())["level_label"] == "1-3"


def test_resolve_matches_on_game_level_and_bucket(tmp_path, monkeypatch):
    # same level label in two games must not cross-resolve
    f = tmp_path / "demo_requests.jsonl"
    monkeypatch.setattr(remix, "REQUESTS_FILE", f)
    monkeypatch.setattr(remix, "CLEARED_FILE", tmp_path / "c.jsonl")
    smb = {"game": "smb", "level_label": "1-1", "death_bucket": 5, "death_x": 100}
    lost = {"game": "smb_lost", "level_label": "1-1", "death_bucket": 5, "death_x": 100}
    f.write_text(json.dumps(smb) + "\n" + json.dumps(lost) + "\n")
    remix._resolve_request(smb)
    assert [r["game"] for r in remix._read_requests()] == ["smb_lost"]


def test_furthest_reads_checkpoint_meta(tmp_path, monkeypatch):
    ck = tmp_path / "checkpoints"
    (ck / "smb").mkdir(parents=True)
    (ck / "smb" / "furthest.json").write_text('{"label":"2-2","progress":160}')
    monkeypatch.setattr(remix.config, "CHECKPOINTS_DIR", ck)
    assert remix._furthest("smb") == ("2-2", 160)
    assert remix._furthest("zelda") == ("start", 0)           # missing = start


def test_past_margin_is_a_real_crossing():
    # the teach goal is death_x + margin: a toe over the edge is not a pass
    assert remix.PAST_MARGIN >= 16


def test_dropin_prefers_the_level_checkpoint(tmp_path, monkeypatch):
    # A tight approach state can coast over the wall; the level-start checkpoint gives real
    # runway, so we must prefer it when present.
    ck = tmp_path / "checkpoints" / "smb"
    ck.mkdir(parents=True)
    (ck / "1_3.state").write_bytes(b"LEVELSTART")
    approach = tmp_path / "approach.state"
    approach.write_bytes(b"APPROACH")
    monkeypatch.setattr(remix.config, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    req = {"game": "smb", "level_label": "1-3", "state": str(approach), "death_x": 771}
    data, source = remix._dropin_state(req)
    assert data == b"LEVELSTART" and "start of 1-3" in source


def test_dropin_falls_back_to_approach_without_a_checkpoint(tmp_path, monkeypatch):
    approach = tmp_path / "approach.state"
    approach.write_bytes(b"APPROACH")
    monkeypatch.setattr(remix.config, "CHECKPOINTS_DIR", tmp_path / "none")
    req = {"game": "zelda", "level_label": "dungeon-1", "state": str(approach), "death_x": 300}
    data, source = remix._dropin_state(req)
    assert data == b"APPROACH" and source == "approach"


# --- the take-control gate: nothing wins until YOU play (the reported bug) -------------------

class _FakeSession:
    """Enough of the session surface for _play_wall's pre-arm gate (no emulator)."""
    def __init__(self, polls):
        self._polls, self._i, self.steps = polls, 0, 0

    def teleop_reset(self):
        pass

    def set_overlay(self, lines):
        pass

    def teleop_poll(self):
        v = self._polls[self._i] if self._i < len(self._polls) else (0, False, False)
        self._i += 1
        return v

    def teleop_step(self, mask):
        self.steps += 1


def test_no_input_never_auto_wins():
    # The exact regression: the drop-in's momentum must NOT complete the wall for you.
    fake = _FakeSession([(0, False, False)] * 6000)     # you never touch the controls
    outcome = remix._play_wall(fake, object(), target=795, death_x=771)[0]
    assert outcome == "afk"                              # not "win"


def test_esc_before_taking_control_skips():
    fake = _FakeSession([(0, False, True)])             # ESC right away
    assert remix._play_wall(fake, object(), 795, 771)[0] == "skip"
