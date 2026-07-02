"""Tests for whole-trajectory tape recording/replay."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.abstractions import Step
from billy.knowledge.tape import TapeLibrary, append_plan
from billy.systems.nes import controller


def test_tape_persistence(tmp_path):
    path = tmp_path / "tapes.jsonl"
    lib = TapeLibrary(path=path)
    lk = (0, 0, 0)
    plan = [Step(4, controller.RIGHT), Step(8, controller.mask(controller.RIGHT, controller.A))]
    lib.put(lk, plan, frontier=900, clears_level=True)
    lib2 = TapeLibrary(path=path)
    e = lib2.get(lk)
    assert e is not None
    assert e.frontier == 900
    assert e.clears_level
    assert len(e.plan) == 2


def test_append_plan_extends_recording():
    record: list[Step] = []
    append_plan(record, [Step(2, 0), Step(4, controller.RIGHT)])
    append_plan(record, [Step(6, controller.A)])
    assert len(record) == 3
    assert record[1].frames == 4


def test_partial_tape_never_displaces_a_clear(tmp_path):
    lib = TapeLibrary(path=tmp_path / "tapes.jsonl")
    lk = (0, 0, 0)
    clear_plan = [Step(100, controller.RIGHT)]
    lib.put(lk, clear_plan, frontier=3000, clears_level=True)
    # A later frontier-march partial reaches further in raw progress terms but must not win.
    lib.put(lk, [Step(10, controller.RIGHT)], frontier=3200, clears_level=False)
    e = lib.get(lk)
    assert e.clears_level and len(e.plan) == 1 and e.plan[0].frames == 100


def test_extended_tape_with_higher_frontier_replaces(tmp_path):
    lib = TapeLibrary(path=tmp_path / "tapes.jsonl")
    lk = (0, 0, 0)
    prefix = [Step(50, controller.RIGHT)]
    lib.put(lk, prefix, frontier=800, clears_level=False)
    extended = prefix + [Step(30, controller.mask(controller.RIGHT, controller.A))]
    lib.put(lk, extended, frontier=1100, clears_level=False)
    e = lib.get(lk)
    assert e.frontier == 1100 and len(e.plan) == 2


def test_shorter_partial_does_not_regress_partial(tmp_path):
    lib = TapeLibrary(path=tmp_path / "tapes.jsonl")
    lk = (0, 0, 0)
    lib.put(lk, [Step(50, controller.RIGHT)], frontier=1100, clears_level=False)
    lib.put(lk, [Step(10, controller.RIGHT)], frontier=700, clears_level=False)
    assert lib.get(lk).frontier == 1100


def test_record_fail_drops_after_limit_and_allows_replacement(tmp_path):
    lib = TapeLibrary(path=tmp_path / "tapes.jsonl")
    lk = (0, 1, 2)
    # A corrupt tape with an inflated frontier would block honest replacements forever...
    lib.put(lk, [Step(10, controller.RIGHT)], frontier=3267, clears_level=True)
    lib.put(lk, [Step(900, controller.RIGHT)], frontier=3266, clears_level=True)
    assert lib.get(lk).frontier == 3267          # blocked (the bug this guards against)
    # ...until repeated verify failures drop it:
    lib.record_fail(lk)
    assert lib.get(lk) is not None               # one strike — still there
    lib.record_fail(lk)
    assert lib.get(lk) is None                   # dropped at FAIL_LIMIT
    lib.put(lk, [Step(900, controller.RIGHT)], frontier=3266, clears_level=True)
    assert lib.get(lk).frontier == 3266          # honest tape takes the slot


def test_record_hit_resets_fail_streak(tmp_path):
    lib = TapeLibrary(path=tmp_path / "tapes.jsonl")
    lk = (0, 0, 0)
    lib.put(lk, [Step(100, controller.RIGHT)], frontier=300, clears_level=False)
    lib.record_fail(lk)
    lib.record_hit(lk)                           # a success clears the strike
    lib.record_fail(lk)
    assert lib.get(lk) is not None               # 1 fail since last hit — survives


def test_tape_consume_preserves_input_stream():
    """The Director's chunked consume must replay the exact stored input stream, and the
    consumed chunks (re-recorded via _commit) must RLE back to the same stream — the property
    that lets an exhausted tape extend instead of self-corrupting into a suffix."""
    from billy.director import Director
    from billy.teleop import TeleopRecorder

    d = Director.__new__(Director)   # no session/game — only the tape-consume fields
    plan = [Step(90, controller.RIGHT), Step(16, controller.mask(controller.RIGHT, controller.A)),
            Step(7, 0), Step(33, controller.RIGHT)]
    d._tape_replay = [Step(s.frames, s.buttons) for s in plan]
    d._tape_mode = True

    rec = TeleopRecorder()           # RLE re-encoder, same as recording the committed chunks
    while True:
        chunk = d._tape_consume()
        if not chunk:
            break
        for s in chunk:
            rec.record(s.buttons, s.frames)
    assert rec.plan() == plan

def test_entry_state_anchor_roundtrips(tmp_path):
    """A clearing tape can carry its entry savestate (sidecar file) so replay reproduces a
    moving hazard (1-3's lift) deterministically. It must survive a save/reload."""
    lib = TapeLibrary(path=tmp_path / "tapes.jsonl")
    lk = (0, 2, 3)
    plan = [Step(50, controller.RIGHT)]
    state = b"\x00\x01\x02SAVESTATE\xff" * 4
    lib.put(lk, plan, frontier=2514, clears_level=True, entry_state=state)
    # sidecar written, JSONL references it
    assert (tmp_path / "tape_states" / "0_2_3.state").exists()
    reloaded = TapeLibrary(path=tmp_path / "tapes.jsonl")
    e = reloaded.get(lk)
    assert e is not None and e.entry_state == state


def test_extension_keeps_the_anchor(tmp_path):
    """A self-recorded extension that doesn't re-supply an anchor must keep the demo's."""
    lib = TapeLibrary(path=tmp_path / "tapes.jsonl")
    lk = (0, 2, 3)
    state = b"ANCHOR" * 8
    lib.put(lk, [Step(50, controller.RIGHT)], frontier=800, clears_level=True, entry_state=state)
    # a later, further-reaching clear with no explicit anchor keeps the original
    lib.put(lk, [Step(60, controller.RIGHT)], frontier=2514, clears_level=True)
    assert lib.get(lk).entry_state == state


def test_unanchored_partial_tape_has_no_state(tmp_path):
    lib = TapeLibrary(path=tmp_path / "tapes.jsonl")
    lk = (0, 0, 0)
    lib.put(lk, [Step(50, controller.RIGHT)], frontier=1200, clears_level=False)
    assert lib.get(lk).entry_state is None
    assert not (tmp_path / "tape_states").exists()
