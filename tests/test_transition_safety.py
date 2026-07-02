"""Transition safety: a plan is only verified UP TO an area transition, so neither search
rollouts nor live commits may run its tail blind into the new area — and a mid-plan death
must be SEEN, not masked by the dying flags decaying before the plan ends.

(The regression these guard: 1-2's exit-pipe plans banked with a transition bonus carried
unverified tail frames that walked Mario straight into 1-3's first pit, inside a single
commit — so the main loop never saw the clear, the checkpoint never advanced, and
learn-from-death had no same-level snapshots to search. 78 identical deaths, zero learning.)
"""
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.abstractions import Observation, Step
from billy.director import _TRANSITION_BONUS, Director, rollout_candidate
from billy.systems.nes import controller


class _Raw:
    on_ground = True


def _obs(frame, x, key, dead=False):
    return Observation(frame=frame, progress=x, score=0, level_label="L", level_key=key,
                       dead=dead, summary="", ascii_map="", raw=_Raw(), elevation=0)


class _Session:
    """Counts emulated frames; the test's observe() derives world state from the count."""

    def __init__(self):
        self.frames = 0

    def send_plan(self, plan):
        self.frames += sum(s.frames for s in plan)

    def clone_state(self):
        return ("snap", self.frames)


class _Game:
    def search_area_advance(self, start_key, end_key):
        return end_key > start_key

    def level_cleared(self, prev_key, new_key):
        return new_key[:2] > prev_key[:2]

    def screen_changed(self, prev_key, new_key):
        return prev_key != new_key and not self.level_cleared(prev_key, new_key)


class _Reflex:
    def advance_plan(self, obs):
        return [Step(4, 0)]


def test_rollout_sees_mid_plan_death_despite_flag_decay():
    # Death at frame 30; by frame 90 the dying flags have decayed (level reload) — the old
    # full-plan-then-observe rollout called this a survivor.
    sess = _Session()

    def observe():
        dead = 30 <= sess.frames < 90
        return _obs(sess.frames, min(sess.frames * 2, 300), (0, 1, 2), dead=dead)

    survived, reached, *_ = rollout_candidate(sess, observe, _Reflex(), _Game(),
                                              [Step(120, controller.RIGHT)], settle=50)
    assert not survived, "mid-plan death was masked by flag decay at plan end"


def test_rollout_stops_at_mid_plan_transition():
    # Key advances at frame 20 — the candidate must score AT the crossing (transition bonus)
    # and must NOT execute its remaining frames in the new area.
    sess = _Session()

    def observe():
        key = (0, 2, 3) if sess.frames >= 20 else (0, 1, 2)
        return _obs(sess.frames, 100 + sess.frames, key)

    survived, reached, *_ = rollout_candidate(sess, observe, _Reflex(), _Game(),
                                              [Step(200, controller.RIGHT)], settle=50)
    assert survived
    assert reached >= _TRANSITION_BONUS
    assert sess.frames < 200, "plan tail was executed past the transition"


def test_commit_stops_at_level_key_change_and_records_prefix():
    sess = _Session()
    d = Director.__new__(Director)
    d.session = sess
    d._tape_record = []

    class _Hooks:
        def commit_chunk_size(self, obs, default):
            return default

        def approach_snapshot_band(self, obs):
            return None

    d.hooks = _Hooks()
    d._approach_trail = deque(maxlen=4)

    def observe():
        key = (0, 2, 3) if sess.frames >= 12 else (0, 1, 2)
        return _obs(sess.frames, 40 if sess.frames >= 12 else 3200 + sess.frames, key)

    d._observe = observe
    obs, _ = d._commit([Step(60, controller.RIGHT)], deque(maxlen=8), last_snap_x=5000)
    assert obs.level_key == (0, 2, 3), "commit should surface the transition observation"
    assert sess.frames < 60, "commit replayed the unverified tail into the new area"
    # The executed prefix (and only it) extends the old level's tape, RLE-merged.
    assert d._tape_record, "executed prefix must be recorded for tape continuity"
    assert sum(s.frames for s in d._tape_record) == sess.frames
    assert all(s.buttons == controller.RIGHT for s in d._tape_record)


def test_commit_unchanged_when_no_transition():
    sess = _Session()
    d = Director.__new__(Director)
    d.session = sess
    d._tape_record = []

    class _Hooks:
        def commit_chunk_size(self, obs, default):
            return default

        def approach_snapshot_band(self, obs):
            return None

    d.hooks = _Hooks()
    d._approach_trail = deque(maxlen=4)
    d._observe = lambda: _obs(sess.frames, 100 + sess.frames, (0, 0, 0))
    plan = [Step(30, controller.RIGHT), Step(10, controller.mask(controller.RIGHT, controller.A))]
    obs, _ = d._commit(plan, deque(maxlen=48), last_snap_x=0)
    assert sess.frames == 40
    # Chunked execution must RLE back to the exact original stream (tape-extend property).
    assert d._tape_record == plan
