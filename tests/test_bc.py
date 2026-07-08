"""Behavior-cloning demo pipeline: demo→action mapping (pure) and the demo-request remedy.

No torch/SB3 needed — bc_pretrain/collect_bc_pairs are exercised by train_section.py runs;
these tests cover the deterministic mapping and request plumbing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.abstractions import Step
from billy.rl.bc import demo_to_actions, load_demo, trim_demo_start
from billy.rl.section_env import SECTION_ACTIONS
from billy.systems.nes import controller as C


def _plan_of(action_idxs):
    """Build the exact input stream those section actions would produce."""
    plan = []
    for k in action_idxs:
        names, hold = SECTION_ACTIONS[k]
        plan.append(Step(hold, C.mask_from_names(list(names))))
    return plan


def test_roundtrip_exact_vocabulary_stream():
    """A demo composed exactly of vocabulary actions maps back to an equivalent stream."""
    src = [1, 0, 6, 2, 4]     # run-jump, cruise, wait, short-jump, walk
    idxs = demo_to_actions(_plan_of(src))
    # The mapping is greedy over windows, so indices may differ where masks coincide
    # (e.g. cruise x2 vs one longer window) — but the RECONSTRUCTED stream must match.
    def flatten(plan):
        out = []
        for s in plan:
            out.extend([s.buttons] * s.frames)
        return out
    assert flatten(_plan_of(idxs)) == flatten(_plan_of(src))


def test_human_long_hold_maps_to_repeated_cruise():
    """A 90-frame human RIGHT+B hold becomes repeated cruise actions, not garbage."""
    plan = [Step(90, C.mask_from_names(["right", "B"]))]
    idxs = demo_to_actions(plan)
    assert idxs, "mapping must not be empty"
    cruise = [k for k, (names, hold) in enumerate(SECTION_ACTIONS)
              if set(names) == {"right", "B"}]
    assert all(k in cruise for k in idxs)


def test_jump_hold_prefers_full_arc_action():
    """A held right+A+B jump maps to the sustained run-jump, not chopped 4-frame moves."""
    plan = [Step(26, C.mask_from_names(["right", "A", "B"]))]
    idxs = demo_to_actions(plan)
    full_arc = [k for k, (names, hold) in enumerate(SECTION_ACTIONS)
                if set(names) == {"right", "A", "B"} and hold == 26]
    assert idxs[0] in full_arc


def test_load_demo_reads_teleop_format(tmp_path):
    p = tmp_path / "x.demo.json"
    p.write_text('{"steps": [[90, 130], [16, 3]]}')
    plan = load_demo(p)
    assert plan == [Step(90, 130), Step(16, 3)]


def test_request_demo_writes_once_and_dedups(tmp_path, monkeypatch):
    from billy import config
    from billy.stuck_trainer import StuckRecord, StuckRemedy, request_demo

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    remedy = StuckRemedy(kind="frame_search", level_label="1-3", death_x=760, goal_x=900)
    record = StuckRecord(level_label="1-3", death_bucket=47, deaths=6, last_death_x=760)
    states = [str(tmp_path / "1_3_d47_x700.state"), str(tmp_path / "1_3_d47_x741.state")]
    for s in states:
        open(s, "wb").close()
    req = tmp_path / "demo_requests.jsonl"

    got = request_demo("smb", remedy, record, states, requests_file=req)
    assert got == states[1], "should pick the FURTHEST approach state"
    lines = [l for l in req.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    inbox = tmp_path / "remix_inbox.txt"
    assert inbox.is_file()
    assert "smb 1-3" in inbox.read_text()

    # Second call: deduped, no new line.
    request_demo("smb", remedy, record, states, requests_file=req)
    lines = [l for l in req.read_text().splitlines() if l.strip()]
    assert len(lines) == 1


def test_trim_demo_start_drops_take_control_wiggles():
    plan = [Step(2, C.mask_from_names(["left", "B"])),
            Step(10, C.mask_from_names(["right", "B"]))]
    trimmed = trim_demo_start(plan)
    assert trimmed == [plan[1]]


def test_trim_demo_start_skips_bare_right_runup_before_jump():
    """A long right-only cruise before right+B jump is dropped at pit-lip savestates."""
    plan = [Step(19, C.mask_from_names(["right"])),
            Step(15, C.mask_from_names(["right", "A", "B"]))]
    trimmed = trim_demo_start(plan)
    assert trimmed == [plan[1]]


def test_collect_bc_pairs_needs_matching_game_rom():
    """BC collection dies immediately when the savestate ROM ≠ SectionEnv's game adapter."""
    from billy.rl.bc import collect_bc_pairs
    from billy.rl.section_env import SectionEnv
    from billy.games.smb_lost import SmbLostGame

    demo = "data/rl/demos/smb_lost/1_1_x1040.demo.json"
    state = "data/rl/demos/smb_lost/1_1_x1040.state"
    if not os.path.isfile(demo) or not os.path.isfile(state):
        return
    from billy.rl.bc import collect_bc_pairs_from_plan
    plan = load_demo(demo)
    wrong = SectionEnv(state, level_label="1-1", goal_x=1081, start_x=1030, back_x=950,
                       randomize_frames=0, landing_waits=2)
    assert len(collect_bc_pairs_from_plan(wrong, plan)) == 0
    wrong.close()
    right = SectionEnv(state, level_label="1-1", goal_x=1081, start_x=1030, back_x=950,
                       randomize_frames=0, landing_waits=2, game=SmbLostGame)
    pairs = collect_bc_pairs_from_plan(right, plan)
    right.close()
    assert len(pairs) > 0, "smb_lost state must replay on SmbLostGame"


def test_request_demo_no_states_is_noop(tmp_path):
    from billy.stuck_trainer import StuckRecord, StuckRemedy, request_demo

    remedy = StuckRemedy(kind="frame_search", level_label="1-3", death_x=760, goal_x=900)
    record = StuckRecord(level_label="1-3", death_bucket=47, deaths=6)
    assert request_demo("smb", remedy, record, [],
                        requests_file=tmp_path / "r.jsonl") is None
