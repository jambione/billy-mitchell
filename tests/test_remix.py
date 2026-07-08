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


def _smb_game():
    from billy.games.smb.game import SmbGame
    return SmbGame()


def test_no_input_never_auto_wins():
    # The exact regression: the drop-in's momentum must NOT complete the wall for you.
    fake = _FakeSession([(0, False, False)] * 6000)     # you never touch the controls
    outcome = remix._play_wall(fake, _smb_game(), _smb_req())[0]
    assert outcome == "afk"                              # not "win"


def test_esc_before_taking_control_skips():
    fake = _FakeSession([(0, False, True)])             # ESC right away
    assert remix._play_wall(fake, _smb_game(), _smb_req())[0] == "skip"


def test_enter_before_move_does_not_arm():
    # ENTER alone during the wait loop must not start recording (was instant-win bug).
    fake = _FakeSession([(0, True, False)] * 10)
    assert remix._play_wall(fake, _smb_game(), _smb_req())[0] == "afk"


def test_resolve_clears_whole_level_not_one_bucket(tmp_path, monkeypatch):
    f = tmp_path / "demo_requests.jsonl"
    cleared = tmp_path / "cleared.jsonl"
    data = tmp_path / "data"
    data.mkdir()
    (data / "stuck.json").write_text("[]")
    a = {"game": "zelda", "level_label": "overworld #121", "death_bucket": 178, "death_x": 2857}
    b = {"game": "zelda", "level_label": "overworld #121", "death_bucket": 177, "death_x": 2841}
    f.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    monkeypatch.setattr(remix, "REQUESTS_FILE", f)
    monkeypatch.setattr(remix, "CLEARED_FILE", cleared)
    monkeypatch.setattr(remix.config, "DATA_DIR", data)
    remix._resolve_request(a)
    assert remix._read_requests() == []


def test_resolve_marks_stuck_remediated_so_wall_does_not_requeue(tmp_path, monkeypatch):
    """After a successful teach, historical death counts must not re-offer the same wall."""
    data = tmp_path / "data"
    data.mkdir()
    auto = data / "rl" / "states" / "auto"
    auto.mkdir(parents=True)
    (auto / "3_4_d32_x359.state").write_bytes(b"APPROACH")
    ck = data / "checkpoints" / "smb"
    ck.mkdir(parents=True)
    (ck / "furthest.json").write_text('{"label":"3-4","progress":105}')
    stuck = data / "stuck.json"
    stuck.write_text(json.dumps([
        {"game": "smb", "level_label": "3-4", "death_bucket": 32, "deaths": 60,
         "last_death_x": 525, "frontier_at_first": 288, "remediated": False,
         "captured_states": []},
    ]))
    reqs = data / "demo_requests.jsonl"
    req = {"game": "smb", "level_label": "3-4", "death_bucket": 32, "death_x": 525,
           "deaths": 60, "state": str(auto / "3_4_d32_x359.state")}
    reqs.write_text(json.dumps(req) + "\n")
    cleared = data / "remix_cleared.jsonl"
    monkeypatch.setattr(remix.config, "DATA_DIR", data)
    monkeypatch.setattr(remix.config, "CHECKPOINTS_DIR", data / "checkpoints")
    monkeypatch.setattr(remix, "REQUESTS_FILE", reqs)
    monkeypatch.setattr(remix, "CLEARED_FILE", cleared)

    remix._resolve_request(req)

    assert remix._read_requests() == []
    rows = json.loads(stuck.read_text())
    assert rows[0]["remediated"] is True
    assert rows[0]["deaths"] == 0
    assert remix._still_stuck_on_level("smb", "3-4") is False
    assert remix._discover_walls(["smb"]) == []
    # open_walls must not re-offer from leftover history either
    assert all(w["level_label"] != "3-4" for w in remix._open_walls(["smb"], persist=False))


def test_open_walls_drops_stale_taught_requests(tmp_path, monkeypatch):
    """demo_requests leftovers for a taught, non-stuck wall are purged on open."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "stuck.json").write_text(json.dumps([
        {"game": "smb", "level_label": "3-4", "death_bucket": 32, "deaths": 0,
         "last_death_x": 525, "remediated": True, "captured_states": []},
    ]))
    cleared = data / "remix_cleared.jsonl"
    cleared.write_text(json.dumps({
        "game": "smb", "level_label": "3-4", "death_bucket": 32, "death_x": 525,
    }) + "\n")
    reqs = data / "demo_requests.jsonl"
    reqs.write_text(json.dumps({
        "game": "smb", "level_label": "3-4", "death_bucket": 32, "death_x": 525, "deaths": 60,
    }) + "\n")
    monkeypatch.setattr(remix.config, "DATA_DIR", data)
    monkeypatch.setattr(remix, "REQUESTS_FILE", reqs)
    monkeypatch.setattr(remix, "CLEARED_FILE", cleared)

    walls = remix._open_walls(["smb"], persist=True)
    assert walls == []
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


def test_open_walls_discovers_games_without_explicit_requests(tmp_path, monkeypatch):
    """smb_lost on demo_requests must not hide smb walls discovered from stuck.json."""
    data = tmp_path / "data"
    data.mkdir()
    auto = data / "rl" / "states" / "auto"
    auto.mkdir(parents=True)
    (auto / "3_4_d32_x359.state").write_bytes(b"APPROACH")
    ck = data / "checkpoints" / "smb"
    ck.mkdir(parents=True)
    (ck / "furthest.json").write_text('{"label":"3-4","progress":105}')
    (data / "demo_requests.jsonl").write_text(json.dumps({
        "game": "smb_lost", "level_label": "1-2", "death_bucket": 91,
        "death_x": 1460, "deaths": 14, "state": str(auto / "1_2_d91_x1427.state"),
    }) + "\n")
    (data / "stuck.json").write_text(json.dumps([
        {"game": "smb", "level_label": "3-4", "death_bucket": 32, "deaths": 44,
         "last_death_x": 525, "remediated": False, "captured_states": []},
    ]))
    (data / "remix_cleared.jsonl").write_text(json.dumps({
        "game": "smb", "level_label": "3-4", "death_bucket": 32, "death_x": 525,
    }) + "\n")
    (data / "solutions.jsonl").write_text(
        json.dumps({"level": [0, 3, 4], "bucket": 22, "plan": [[1, 1]], "reach_after": 546}) + "\n")
    monkeypatch.setattr(remix.config, "DATA_DIR", data)
    monkeypatch.setattr(remix.config, "CHECKPOINTS_DIR", data / "checkpoints")
    monkeypatch.setattr(remix.config, "SOLUTIONS_FILE", data / "solutions.jsonl")
    monkeypatch.setattr(remix, "REQUESTS_FILE", data / "demo_requests.jsonl")
    monkeypatch.setattr(remix, "CLEARED_FILE", data / "remix_cleared.jsonl")

    walls = remix._open_walls(["smb", "smb_lost"], persist=False)
    games = {w["game"] for w in walls}
    assert games == {"smb", "smb_lost"}
    smb = next(w for w in walls if w["game"] == "smb")
    assert smb["level_label"] == "3-4"
    assert smb["death_x"] == 525


def test_discover_skips_when_banked_despite_stuck_history(tmp_path, monkeypatch):
    """Cleared + banked solution + deaths below threshold → don't re-queue stale history."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "solutions.jsonl").write_text(
        json.dumps({"level": ["overworld", 121], "bucket": 173, "plan": [[1, 1]], "reach_after": 2974}) + "\n")
    cleared = data / "remix_cleared.jsonl"
    cleared.write_text(json.dumps({
        "game": "zelda", "level_label": "overworld #121", "death_bucket": 178, "death_x": 2857,
    }) + "\n")
    (data / "stuck.json").write_text(json.dumps([
        {"level_label": "overworld #121", "death_bucket": 178, "deaths": 2,
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
    (data / "solutions.jsonl").write_text(
        json.dumps({"level": [0, 2, 3], "bucket": 67, "plan": [[1, 1]], "reach_after": 1200}) + "\n")
    (data / "stuck.json").write_text(json.dumps([{
        "level_label": "2-3", "death_bucket": 67, "deaths": 2,
        "last_death_x": 1085, "remediated": False, "captured_states": [],
    }]))
    monkeypatch.setattr(remix.config, "DATA_DIR", data)
    monkeypatch.setattr(remix.config, "CHECKPOINTS_DIR", data / "checkpoints")
    monkeypatch.setattr(remix.config, "SOLUTIONS_FILE", data / "solutions.jsonl")
    monkeypatch.setattr(remix, "REQUESTS_FILE", data / "demo_requests.jsonl")
    monkeypatch.setattr(remix, "CLEARED_FILE", cleared)
    assert remix._discover_walls(["smb"]) == []


def test_discover_smb_lost_wall_with_game_field(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    auto = data / "rl" / "states" / "auto"
    auto.mkdir(parents=True)
    (auto / "1_1_d66_x1000.state").write_bytes(b"APPROACH")
    (data / "stuck.json").write_text(json.dumps([{
        "game": "smb_lost",
        "level_label": "1-1",
        "death_bucket": 66,
        "deaths": 12,
        "last_death_x": 1057,
        "frontier_at_first": 3152,
        "remediated": False,
        "captured_states": [],
    }]))
    monkeypatch.setattr(remix.config, "DATA_DIR", data)
    monkeypatch.setattr(remix.config, "CHECKPOINTS_DIR", data / "checkpoints")
    monkeypatch.setattr(remix, "REQUESTS_FILE", data / "demo_requests.jsonl")
    monkeypatch.setattr(remix, "CLEARED_FILE", data / "remix_cleared.jsonl")

    walls = remix._discover_walls(["smb_lost"])
    assert len(walls) == 1
    assert walls[0]["game"] == "smb_lost"
    assert walls[0]["death_x"] == 1057
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


def test_prepare_approach_reuses_existing_auto_state(tmp_path, monkeypatch):
    auto = tmp_path / "rl" / "states" / "auto"
    auto.mkdir(parents=True)
    (auto / "3_4_d21_x500.state").write_bytes(b"APPROACH")
    monkeypatch.setattr(remix.config, "DATA_DIR", tmp_path)
    from billy.games.smb.game import SmbGame
    req = {"level_label": "3-4", "death_x": 525}
    assert remix._prepare_approach(req, SmbGame()) == str(auto / "3_4_d21_x500.state")


def test_fake_game_joins_remix_via_game_hooks_only():
    """Phase 1 acceptance: a new Game plugs into teach flow without remix.py edits."""
    from types import SimpleNamespace

    from billy.abstractions import Game, Observation

    class _FakeRemixGame(Game):
        name = "Fake Quest"
        system = SimpleNamespace(name="fake")

        def observe(self, frame, ram, rgb=None):
            raise NotImplementedError

        def boot(self, session):
            raise NotImplementedError

        def make_reflex(self):
            raise NotImplementedError

        def remix_goal(self, req):
            return f"beat boss at {req['level_label']}"

        def remix_min_progress(self):
            return 99

        def remix_win(self, obs, req, start_obs):
            return obs.progress >= int(req["death_x"]) + 50

        def remix_dropin_ok(self, obs, req):
            return obs.progress < int(req["death_x"])

        def remix_anchor_ok(self, source):
            return "entry" in source

        def remix_wall_at(self, req):
            return f"boss room {req['level_label']}"

    game = _FakeRemixGame()
    req = {"level_label": "dungeon-9", "death_x": 400}
    assert game.remix_goal(req) == "beat boss at dungeon-9"
    assert game.remix_min_progress() == 99
    assert game.remix_wall_at(req) == "boss room dungeon-9"
    assert game.remix_anchor_ok("entry checkpoint") is True
    assert game.remix_anchor_ok("mid-fight") is False
    start = Observation(frame=0, progress=380, score=0, level_label="dungeon-9",
                        level_key=("dungeon", 9), dead=False, summary="", ascii_map="",
                        raw=SimpleNamespace(in_play=True))
    assert game.remix_dropin_ok(start, req) is True
    win = Observation(**{**start.__dict__, "progress": 460})
    assert game.remix_win(win, req, start) is True


def test_bank_tape_skips_mid_level_smb_anchor(tmp_path, monkeypatch):
    # A mid-level SMB approach must NOT become a per-level tape — it would replay from mid-level
    # on every entry, skipping the level's first half. The cache entry carries it instead.
    from types import SimpleNamespace

    from billy.abstractions import Step
    from billy.games.smb.game import SmbGame
    from billy.knowledge.tape import TapeLibrary
    monkeypatch.setattr(remix.config, "TAPES_FILE", tmp_path / "tapes.jsonl")
    obs = SimpleNamespace(level_key=(0, 1, 3))
    result = SimpleNamespace(end_level_key=(0, 1, 3), end_progress=800)
    banked = remix._bank_tape(SmbGame(), obs, b"MIDLEVEL", [Step(5, 1)], result, "approach at 1-3")
    assert banked is False
    assert len(TapeLibrary(path=tmp_path / "tapes.jsonl")) == 0


def test_bank_tape_writes_entry_anchored_tape_from_level_start(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from billy.abstractions import Step
    from billy.games.smb.game import SmbGame
    from billy.knowledge.tape import TapeLibrary
    monkeypatch.setattr(remix.config, "TAPES_FILE", tmp_path / "tapes.jsonl")
    obs = SimpleNamespace(level_key=(0, 1, 3))
    result = SimpleNamespace(end_level_key=(0, 1, 3), end_progress=800)   # crossed, level not cleared
    assert remix._bank_tape(SmbGame(), obs, b"LEVELSTART", [Step(5, 1)], result, "start of 1-3") is True
    lib = TapeLibrary(path=tmp_path / "tapes.jsonl")
    entry = lib.get((0, 1, 3))
    assert entry is not None
    assert entry.entry_state == b"LEVELSTART"      # the anchor that makes it reproduce
    assert entry.clears_level is False             # a mid-level crossing is a partial tape


def test_bank_tape_zelda_screen_crossing_clears_the_unit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from billy.abstractions import Step
    from billy.games.zelda.game import ZeldaGame
    from billy.knowledge.tape import TapeLibrary
    monkeypatch.setattr(remix.config, "TAPES_FILE", tmp_path / "tapes.jsonl")
    obs = SimpleNamespace(level_key=("overworld", 121))
    result = SimpleNamespace(end_level_key=("overworld", 122), end_progress=60)
    assert remix._bank_tape(ZeldaGame(), obs, b"SCREEN", [Step(4, 2)], result,
                            "approach at overworld#121")
    entry = TapeLibrary(path=tmp_path / "tapes.jsonl").get(("overworld", 121))
    assert entry.clears_level is True              # crossing east advanced the screen unit


class _ReplaySession:
    """Captures the visible-replay calls without an emulator."""
    def __init__(self):
        self.restored = None
        self.masks: list[int] = []
        self.overlays: list[list] = []
        self.viewer_refreshed = False

    def restore(self, snap):
        self.restored = snap

    def ensure_viewer(self):
        self.viewer_refreshed = True
        return True

    def set_overlay(self, lines):
        self.overlays.append(lines)

    def teleop_step(self, mask):
        self.masks.append(mask)


def test_replay_taught_line_restores_and_replays_every_frame():
    from billy.abstractions import Step
    sess = _ReplaySession()
    remix._replay_taught_line(sess, b"ENTRY", [Step(3, 1), Step(2, 4)], "cross the pit")
    assert sess.restored == b"ENTRY"               # replays from the exact taught state
    assert sess.viewer_refreshed                     # same window, not a second spawn
    assert sess.masks == [1, 1, 1, 4, 4]           # every frame, in order (run-length expanded)
    assert sess.overlays and any("WATCH BILLY" in l for l in sess.overlays[0])


def test_replay_taught_line_is_non_fatal_on_window_error():
    # A windowing hiccup mid-replay must never blow up a successful bank.
    class _Boom(_ReplaySession):
        def teleop_step(self, mask):
            raise RuntimeError("viewer died")
    from billy.abstractions import Step
    remix._replay_taught_line(_Boom(), b"E", [Step(1, 1)], "goal")   # no exception escapes


# --- BC warm-start: the 4th carrier — persist the seed train_section.py --demo consumes --------

def test_bc_seed_writes_state_and_demo_for_smb(tmp_path, monkeypatch):
    import json as _json
    from types import SimpleNamespace

    from billy.abstractions import Step
    monkeypatch.setattr(remix.config, "DATA_DIR", tmp_path)
    obs = SimpleNamespace(level_label="1-3", progress=630)
    result = SimpleNamespace(end_progress=780)
    path = remix._write_bc_seed("smb", obs, b"STATE", [Step(6, 1), Step(3, 3)], result)
    assert path is not None
    demo = tmp_path / "rl" / "demos" / "smb" / "1_3_x630.demo.json"
    state = tmp_path / "rl" / "demos" / "smb" / "1_3_x630.state"
    assert state.read_bytes() == b"STATE"
    assert _json.loads(demo.read_text())["steps"] == [[6, 1], [3, 3]]   # the exact input stream


def test_bc_seed_skipped_for_non_section_games(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from billy.abstractions import Step
    monkeypatch.setattr(remix.config, "DATA_DIR", tmp_path)
    obs = SimpleNamespace(level_label="overworld #121", progress=200)
    result = SimpleNamespace(end_progress=260)
    # SectionEnv is SMB-shaped — no BC carrier for Zelda.
    assert remix._write_bc_seed("zelda", obs, b"S", [Step(1, 1)], result) is None
    assert not (tmp_path / "rl" / "demos" / "zelda").exists()


# --- the compounding march: teach → re-scout → next wall, in one sitting ----------------------

def test_teach_game_marches_until_no_walls_left():
    # Each taught line clears that wall; re-scout surfaces the next. Loop ends when walls run out.
    remaining = [
        [{"game": "smb", "level_label": "3-4", "death_x": 525}],
        [{"game": "smb", "level_label": "4-1", "death_x": 300}],
        [],                                          # after two teaches, Billy is unstuck
    ]
    scouted = []
    resolved = []
    taught = remix._teach_game(
        "smb", scout_attempts=1, rescout=True,
        teach=lambda req: True,                      # you clear every wall
        scout=lambda g, n: scouted.append(g) or "next",
        open_walls=lambda games: remaining.pop(0) if remaining else [],
        resolve=lambda req: resolved.append(req["level_label"]))
    assert taught == 2
    assert resolved == ["3-4", "4-1"]
    assert scouted == ["smb", "smb"]                 # re-scouted after each taught line


def test_teach_game_stops_when_a_wall_cannot_be_taught():
    # A skipped/failed wall ends the march (it stays open for next time) — no re-scout after it.
    scouted = []
    taught = remix._teach_game(
        "smb", scout_attempts=1, rescout=True,
        teach=lambda req: False,                     # you skip the wall
        scout=lambda g, n: scouted.append(g) or "x",
        open_walls=lambda games: [{"game": "smb", "level_label": "3-4", "death_x": 525}],
        resolve=lambda req: None)
    assert taught == 0 and scouted == []             # never advanced, never re-scouted


def test_teach_game_no_rescout_teaches_open_walls_without_hunting():
    # --no-scout: teach every wall already on file; never re-scout for new ones.
    open_list = [
        {"game": "smb", "level_label": "3-4", "death_x": 525},
        {"game": "smb", "level_label": "4-1", "death_x": 300},
    ]

    def _open(games):
        return list(open_list[:1]) if open_list else []

    def resolve(req):
        if open_list and open_list[0]["level_label"] == req["level_label"]:
            open_list.pop(0)

    scouted = []
    taught = remix._teach_game(
        "smb", scout_attempts=1, rescout=False,
        teach=lambda req: True, scout=lambda g, n: scouted.append(g),
        open_walls=_open, resolve=resolve)
    assert taught == 2 and scouted == [] and open_list == []


def test_teach_game_respects_the_per_game_cap():
    # An always-stuck game (bank never actually unblocks him) must not loop forever.
    taught = remix._teach_game(
        "smb", scout_attempts=1, rescout=True,
        teach=lambda req: True,
        scout=lambda g, n: "same",
        open_walls=lambda games: [{"game": "smb", "level_label": "3-4", "death_x": 525}],
        resolve=lambda req: None)
    assert taught == remix.MAX_WALLS_PER_GAME


# --- one emulator per process: teach window vs re-scout / approach -----------------------

class _FakeTeachSession:
    def __init__(self, events: list, name: str):
        self.events = events
        self.name = name
        self.closed = False
        events.append(f"open:{name}")

    def wait_until_live(self):
        pass

    def close(self):
        self.closed = True
        self.events.append(f"close:{self.name}")


def test_teach_game_closes_window_before_rescout(monkeypatch):
    """stable-retro allows one emulator per process — re-scout must not open while teach lives."""
    events: list[str] = []
    n = {"i": 0}

    def open_sess(game_key):
        n["i"] += 1
        return _FakeTeachSession(events, f"s{n['i']}")

    monkeypatch.setattr(remix, "_open_teach_session", open_sess)
    monkeypatch.setattr(remix, "_attach_approach_state", lambda req: dict(req))
    monkeypatch.setattr(remix, "_approach_needs_live_drive", lambda req: False)

    remaining = [
        [{"game": "smb", "level_label": "3-4", "death_x": 525}],
        [{"game": "smb", "level_label": "4-1", "death_x": 300}],
        [],
    ]

    def teach(req, session=None):
        assert session is not None and not session.closed
        events.append(f"teach:{req['level_label']}")
        return True

    def scout(g, n_attempts):
        # Teach session must already be closed before scout opens its own emulator.
        assert events[-1].startswith("close:"), events
        events.append("scout")
        return "next"

    taught = remix._teach_game(
        "smb", scout_attempts=1, rescout=True, reuse_window=True,
        teach=teach, scout=scout,
        open_walls=lambda games: remaining.pop(0) if remaining else [],
        resolve=lambda req: None)
    assert taught == 2
    # teach → close → scout → open next → teach → close → scout
    assert events == [
        "open:s1", "teach:3-4", "close:s1", "scout",
        "open:s2", "teach:4-1", "close:s2", "scout",
    ]


def test_teach_game_no_rescout_teaches_all_open_walls_one_window(monkeypatch):
    """--no-scout marches through every open wall in one window (no second spawn)."""
    events: list[str] = []
    n = {"i": 0}

    def open_sess(game_key):
        n["i"] += 1
        return _FakeTeachSession(events, f"s{n['i']}")

    monkeypatch.setattr(remix, "_open_teach_session", open_sess)
    monkeypatch.setattr(remix, "_attach_approach_state", lambda req: dict(req))
    monkeypatch.setattr(remix, "_approach_needs_live_drive", lambda req: False)

    open_list = [
        {"game": "smb", "level_label": "3-4", "death_x": 525},
        {"game": "smb", "level_label": "4-1", "death_x": 300},
    ]

    def open_walls(games):
        return list(open_list[:1]) if open_list else []

    def resolve(req):
        if open_list and open_list[0]["level_label"] == req["level_label"]:
            open_list.pop(0)

    taught_sessions = []

    def teach(req, session=None):
        taught_sessions.append(session)
        events.append(f"teach:{req['level_label']}")
        return True

    taught = remix._teach_game(
        "smb", scout_attempts=1, rescout=False, reuse_window=True,
        teach=teach, scout=lambda g, n: events.append("scout") or "x",
        open_walls=open_walls, resolve=resolve)
    assert taught == 2
    assert events.count("scout") == 0
    assert n["i"] == 1                                 # one window only
    assert taught_sessions[0] is taught_sessions[1]     # same session object
    assert events == ["open:s1", "teach:3-4", "teach:4-1", "close:s1"]


def test_teach_game_releases_window_before_live_approach(monkeypatch):
    """Live approach capture needs the emulator — teach window must be closed first."""
    events: list[str] = []
    n = {"i": 0}

    def open_sess(game_key):
        n["i"] += 1
        return _FakeTeachSession(events, f"s{n['i']}")

    monkeypatch.setattr(remix, "_open_teach_session", open_sess)

    # First wall: no live drive. Second wall: needs live approach → must close s1 first.
    drive = {"3-4": False, "4-1": True}

    def needs_drive(req):
        return drive[req["level_label"]]

    def attach(req):
        events.append(f"approach:{req['level_label']}")
        return dict(req)

    monkeypatch.setattr(remix, "_approach_needs_live_drive", needs_drive)
    monkeypatch.setattr(remix, "_attach_approach_state", attach)

    open_list = [
        {"game": "smb", "level_label": "3-4", "death_x": 525},
        {"game": "smb", "level_label": "4-1", "death_x": 300},
    ]

    def open_walls(games):
        return list(open_list[:1]) if open_list else []

    def resolve(req):
        if open_list and open_list[0]["level_label"] == req["level_label"]:
            open_list.pop(0)

    def teach(req, session=None):
        events.append(f"teach:{req['level_label']}")
        return True

    taught = remix._teach_game(
        "smb", scout_attempts=1, rescout=False, reuse_window=True,
        teach=teach, scout=lambda g, n: "x",
        open_walls=open_walls, resolve=resolve)
    assert taught == 2
    assert events == [
        "approach:3-4", "open:s1", "teach:3-4",
        "close:s1", "approach:4-1", "open:s2", "teach:4-1", "close:s2",
    ]
