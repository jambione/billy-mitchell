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


def test_dropin_prefers_remix_approach_over_level_checkpoint(tmp_path, monkeypatch):
    ck = tmp_path / "checkpoints" / "smb"
    ck.mkdir(parents=True)
    (ck / "2_4.state").write_bytes(b"LEVELSTART")
    approach = tmp_path / "auto" / "2_4_d114_remix.state"
    approach.parent.mkdir(parents=True)
    approach.write_bytes(b"AT_PIT")
    monkeypatch.setattr(remix.config, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    req = {"game": "smb", "level_label": "2-4", "state": str(approach), "death_x": 1832}
    data, source = remix._dropin_state(req)
    assert data == b"AT_PIT" and "approach at 2-4" in source


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


def _smb_req(death_x=771):
    return {"game": "smb", "level_label": "1-3", "death_x": death_x}


def test_no_input_never_auto_wins():
    # The exact regression: the drop-in's momentum must NOT complete the wall for you.
    fake = _FakeSession([(0, False, False)] * 6000)     # you never touch the controls
    outcome = remix._play_wall(fake, object(), "smb", _smb_req(), death_x=771)[0]
    assert outcome == "afk"                              # not "win"


def test_esc_before_taking_control_skips():
    fake = _FakeSession([(0, False, True)])             # ESC right away
    assert remix._play_wall(fake, object(), "smb", _smb_req(), 771)[0] == "skip"


def test_enter_before_move_does_not_arm():
    # ENTER alone during the wait loop must not start recording (was instant-win bug).
    fake = _FakeSession([(0, True, False)] * 10)
    assert remix._play_wall(fake, object(), "smb", _smb_req(), 771)[0] == "afk"


def test_resolve_clears_whole_level_not_one_bucket(tmp_path, monkeypatch):
    f = tmp_path / "demo_requests.jsonl"
    cleared = tmp_path / "cleared.jsonl"
    a = {"game": "zelda", "level_label": "overworld #121", "death_bucket": 178, "death_x": 2857}
    b = {"game": "zelda", "level_label": "overworld #121", "death_bucket": 177, "death_x": 2841}
    f.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    monkeypatch.setattr(remix, "REQUESTS_FILE", f)
    monkeypatch.setattr(remix, "CLEARED_FILE", cleared)
    remix._resolve_request(a)
    assert remix._read_requests() == []


def test_discover_walls_from_stuck_json(tmp_path, monkeypatch):
    """demo_requests empty + stuck.json has deaths → remix still has walls to teach."""
    data = tmp_path / "data"
    data.mkdir()
    ck = data / "checkpoints" / "smb"
    ck.mkdir(parents=True)
    (ck / "furthest.json").write_text('{"label":"2-2","progress":160}')
    (ck / "2_2.state").write_bytes(b"CK")
    (data / "stuck.json").write_text(json.dumps([
        {   # behind frontier — must not surface
            "level_label": "1-3",
            "death_bucket": 17,
            "deaths": 87,
            "last_death_x": 283,
            "remediated": False,
            "captured_states": [],
        },
        {
            "level_label": "2-3",
            "death_bucket": 67,
            "deaths": 43,
            "last_death_x": 1085,
            "remediated": False,
            "captured_states": [],
        },
    ]))
    monkeypatch.setattr(remix.config, "DATA_DIR", data)
    monkeypatch.setattr(remix.config, "CHECKPOINTS_DIR", data / "checkpoints")
    monkeypatch.setattr(remix, "REQUESTS_FILE", data / "demo_requests.jsonl")
    monkeypatch.setattr(remix, "CLEARED_FILE", data / "remix_cleared.jsonl")

    walls = remix._discover_walls(["smb"])
    assert len(walls) == 1
    assert walls[0]["level_label"] == "2-3"
    assert walls[0]["death_x"] == 1085
    assert walls[0]["state"].endswith("2_2.state") or "2_3_" in walls[0]["state"]


def test_discover_skips_when_banked_despite_stuck_history(tmp_path, monkeypatch):
    """Cleared + solution in cache → don't re-queue even if stuck.json still has old deaths."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "solutions.jsonl").write_text(
        json.dumps({"level": ["overworld", 121], "bucket": 173, "plan": [[1, 1]], "reach_after": 2974}) + "\n")
    cleared = data / "remix_cleared.jsonl"
    cleared.write_text(json.dumps({
        "game": "zelda", "level_label": "overworld #121", "death_bucket": 178, "death_x": 2857,
    }) + "\n")
    (data / "stuck.json").write_text(json.dumps([
        {"level_label": "overworld #121", "death_bucket": 178, "deaths": 57,
         "last_death_x": 2857, "remediated": False, "captured_states": []},
    ]))
    monkeypatch.setattr(remix.config, "DATA_DIR", data)
    monkeypatch.setattr(remix.config, "SOLUTIONS_FILE", data / "solutions.jsonl")
    monkeypatch.setattr(remix, "REQUESTS_FILE", data / "demo_requests.jsonl")
    monkeypatch.setattr(remix, "CLEARED_FILE", cleared)
    assert remix._discover_walls(["zelda"]) == []


def test_discover_requeues_when_still_stuck(tmp_path, monkeypatch):
    """A level in remix_cleared but still dying in stuck.json must come back."""
    data = tmp_path / "data"
    zelda_states = data / "zelda" / "states"
    zelda_states.mkdir(parents=True)
    (zelda_states / "teleop_121.state").write_bytes(b"Z")
    cleared = data / "remix_cleared.jsonl"
    cleared.write_text(json.dumps({
        "game": "zelda", "level_label": "overworld #121", "death_bucket": 178, "death_x": 2857,
    }) + "\n")
    (data / "stuck.json").write_text(json.dumps([
        {"level_label": "overworld #121", "death_bucket": 178, "deaths": 57,
         "last_death_x": 2857, "remediated": False, "captured_states": []},
    ]))
    (data / "solutions.jsonl").write_text("")   # cleared but teach never banked
    monkeypatch.setattr(remix.config, "DATA_DIR", data)
    monkeypatch.setattr(remix.config, "SOLUTIONS_FILE", data / "solutions.jsonl")
    monkeypatch.setattr(remix.config, "CHECKPOINTS_DIR", data / "checkpoints")
    monkeypatch.setattr(remix, "REQUESTS_FILE", data / "demo_requests.jsonl")
    monkeypatch.setattr(remix, "CLEARED_FILE", cleared)
    walls = remix._discover_walls(["zelda"])
    assert len(walls) == 1 and walls[0]["level_label"] == "overworld #121"


def test_discover_skips_already_cleared_levels(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    cleared = data / "remix_cleared.jsonl"
    cleared.write_text(json.dumps({
        "game": "zelda", "level_label": "overworld #121", "death_bucket": 178, "death_x": 2857,
    }) + "\n")
    (data / "stuck.json").write_text(json.dumps([
        {"level_label": "overworld #121", "death_bucket": 182, "deaths": 2,
         "last_death_x": 2921, "remediated": False, "captured_states": []},
    ]))
    monkeypatch.setattr(remix.config, "DATA_DIR", data)
    monkeypatch.setattr(remix.config, "CHECKPOINTS_DIR", data / "checkpoints")
    monkeypatch.setattr(remix, "REQUESTS_FILE", data / "demo_requests.jsonl")
    monkeypatch.setattr(remix, "CLEARED_FILE", cleared)
    assert remix._discover_walls(["zelda"]) == []


def test_discover_skips_already_cleared_walls(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    cleared = data / "remix_cleared.jsonl"
    cleared.write_text(json.dumps({
        "game": "smb", "level_label": "2-3", "death_bucket": 67, "death_x": 1085,
    }) + "\n")
    (data / "stuck.json").write_text(json.dumps([{
        "level_label": "2-3", "death_bucket": 67, "deaths": 43,
        "last_death_x": 1085, "remediated": False, "captured_states": [],
    }]))
    monkeypatch.setattr(remix.config, "DATA_DIR", data)
    monkeypatch.setattr(remix.config, "CHECKPOINTS_DIR", data / "checkpoints")
    monkeypatch.setattr(remix, "REQUESTS_FILE", data / "demo_requests.jsonl")
    monkeypatch.setattr(remix, "CLEARED_FILE", cleared)
    assert remix._discover_walls(["smb"]) == []


def test_discover_zelda_uses_teleop_state(tmp_path, monkeypatch):
    data = tmp_path / "data"
    zelda_states = data / "zelda" / "states"
    zelda_states.mkdir(parents=True)
    (zelda_states / "teleop_121.state").write_bytes(b"Z")
    (data / "stuck.json").write_text(json.dumps([{
        "level_label": "overworld #121",
        "death_bucket": 182,
        "deaths": 27,
        "last_death_x": 2921,
        "remediated": False,
        "captured_states": [],
    }]))
    monkeypatch.setattr(remix.config, "DATA_DIR", data)
    monkeypatch.setattr(remix.config, "CHECKPOINTS_DIR", data / "checkpoints")
    monkeypatch.setattr(remix, "REQUESTS_FILE", data / "demo_requests.jsonl")
    monkeypatch.setattr(remix, "CLEARED_FILE", data / "remix_cleared.jsonl")

    walls = remix._discover_walls(["zelda"])
    assert walls[0]["game"] == "zelda"
    assert "teleop_121.state" in walls[0]["state"]


# --- tape-bank + on-screen replay: the demo now sticks (moving hazards) AND you see it land ---

def test_anchor_is_level_entry_gates_by_game_and_source():
    # Only a real level/screen entry is a valid tape anchor (restored AT level-begin).
    assert remix._anchor_is_level_entry("smb", "start of 1-3") is True
    assert remix._anchor_is_level_entry("smb", "approach at 1-3") is False   # mid-level → cache only
    assert remix._anchor_is_level_entry("zelda", "approach at overworld#121") is True   # screen-keyed


def test_bank_tape_skips_mid_level_smb_anchor(tmp_path, monkeypatch):
    # A mid-level SMB approach must NOT become a per-level tape — it would replay from mid-level
    # on every entry, skipping the level's first half. The cache entry carries it instead.
    from types import SimpleNamespace

    from billy.abstractions import Step
    from billy.knowledge.tape import TapeLibrary
    monkeypatch.setattr(remix.config, "TAPES_FILE", tmp_path / "tapes.jsonl")
    obs = SimpleNamespace(level_key=(0, 1, 3))
    result = SimpleNamespace(end_level_key=(0, 1, 3), end_progress=800)
    banked = remix._bank_tape("smb", obs, b"MIDLEVEL", [Step(5, 1)], result, "approach at 1-3")
    assert banked is False
    assert len(TapeLibrary(path=tmp_path / "tapes.jsonl")) == 0


def test_bank_tape_writes_entry_anchored_tape_from_level_start(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from billy.abstractions import Step
    from billy.knowledge.tape import TapeLibrary
    monkeypatch.setattr(remix.config, "TAPES_FILE", tmp_path / "tapes.jsonl")
    obs = SimpleNamespace(level_key=(0, 1, 3))
    result = SimpleNamespace(end_level_key=(0, 1, 3), end_progress=800)   # crossed, level not cleared
    assert remix._bank_tape("smb", obs, b"LEVELSTART", [Step(5, 1)], result, "start of 1-3") is True
    lib = TapeLibrary(path=tmp_path / "tapes.jsonl")
    entry = lib.get((0, 1, 3))
    assert entry is not None
    assert entry.entry_state == b"LEVELSTART"      # the anchor that makes it reproduce
    assert entry.clears_level is False             # a mid-level crossing is a partial tape


def test_bank_tape_zelda_screen_crossing_clears_the_unit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from billy.abstractions import Step
    from billy.knowledge.tape import TapeLibrary
    monkeypatch.setattr(remix.config, "TAPES_FILE", tmp_path / "tapes.jsonl")
    obs = SimpleNamespace(level_key=("overworld", 121))
    result = SimpleNamespace(end_level_key=("overworld", 122), end_progress=60)
    assert remix._bank_tape("zelda", obs, b"SCREEN", [Step(4, 2)], result, "approach at overworld#121")
    entry = TapeLibrary(path=tmp_path / "tapes.jsonl").get(("overworld", 121))
    assert entry.clears_level is True              # crossing east advanced the screen unit


class _ReplaySession:
    """Captures the visible-replay calls without an emulator."""
    def __init__(self):
        self.restored = None
        self.masks: list[int] = []
        self.overlays: list[list] = []

    def restore(self, snap):
        self.restored = snap

    def set_overlay(self, lines):
        self.overlays.append(lines)

    def teleop_step(self, mask):
        self.masks.append(mask)


def test_replay_taught_line_restores_and_replays_every_frame():
    from billy.abstractions import Step
    sess = _ReplaySession()
    remix._replay_taught_line(sess, b"ENTRY", [Step(3, 1), Step(2, 4)], "cross the pit")
    assert sess.restored == b"ENTRY"               # replays from the exact taught state
    assert sess.masks == [1, 1, 1, 4, 4]           # every frame, in order (run-length expanded)
    assert sess.overlays and any("WATCH BILLY" in l for l in sess.overlays[0])


def test_replay_taught_line_is_non_fatal_on_window_error():
    # A windowing hiccup mid-replay must never blow up a successful bank.
    class _Boom(_ReplaySession):
        def teleop_step(self, mask):
            raise RuntimeError("viewer died")
    from billy.abstractions import Step
    remix._replay_taught_line(_Boom(), b"E", [Step(1, 1)], "goal")   # no exception escapes
