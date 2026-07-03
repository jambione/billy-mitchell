"""Generic pixel platformer adapter — the Game contract implemented from pixels alone.

No RAM map: perception is billy/vision's PixelTracker on the rgb frame stream. Point it at
any imported side-scroller ROM:

    BILLY_RETRO_GAME=SuperMarioBros-Nes-v0 .venv/bin/python run.py --game pixel --no-llm

The engine's whole learning stack (tapes, cache, search, demos) works unchanged on top.

THE subtlety: the tracker is stateful (camera position, player track), but the Director
time-travels the emulator constantly (clone/restore for invisible search, savestate slots for
respawns). A tracker that lives in wall-clock time desyncs from an emulator that rewinds — so
`_TrackedSession` wraps the transport and snapshots/restores the tracker state ALONGSIDE every
emulator savestate. Perception time-travels with the machine.
"""
from __future__ import annotations

import os

from ...abstractions import Decision, Game, Observation, Plan, ReflexPolicy, Session, Step
from ...systems.nes import controller as C
from ...systems.nes.system import NesSystem
from ...vision import PixelTracker


class _TrackedSession:
    """Delegating Session wrapper that keeps PixelTracker state in lockstep with emulator
    time-travel (clone/restore + save/load slots)."""

    def __init__(self, inner: Session, tracker: PixelTracker):
        self._inner = inner
        self._tracker = tracker
        self._stash: dict[int, dict] = {}       # hash(snapshot) -> tracker state
        self._slot_stash: dict[int, dict] = {}  # save_state slot -> tracker state

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _snap_tracker(self) -> dict:
        t = self._tracker
        return {k: (v.copy() if hasattr(v, "copy") and not isinstance(v, tuple) else v)
                for k, v in t.__dict__.items()}

    def _apply(self, state: dict) -> None:
        self._tracker.__dict__.update(
            {k: (v.copy() if hasattr(v, "copy") and not isinstance(v, tuple) else v)
             for k, v in state.items()})

    def clone_state(self):
        snap = self._inner.clone_state()
        self._stash[hash(snap)] = self._snap_tracker()
        if len(self._stash) > 512:
            for k in list(self._stash)[:256]:
                del self._stash[k]
        return snap

    def restore(self, snapshot) -> None:
        self._inner.restore(snapshot)
        state = self._stash.get(hash(snapshot))
        if state is not None:
            self._apply(state)
        else:
            self._tracker.respawned()   # unknown past — re-base rather than desync

    def save_state(self, slot: int = 0) -> None:
        self._inner.save_state(slot)
        self._slot_stash[slot] = self._snap_tracker()

    def load_state(self, slot: int = 0) -> None:
        self._inner.load_state(slot)
        state = self._slot_stash.get(slot)
        if state is not None:
            self._apply(state)
        else:
            self._tracker.respawned()


class PixelReflex(ReflexPolicy):
    """Minimal forward-motion reflex: run right, periodic sustained hops. Deliberately dumb —
    the cache/search/tape stack owns the hazards; this just supplies motion + candidates."""

    def __init__(self):
        self._tick = 0

    def reset(self, obs: Observation) -> None:
        self._tick = 0

    def note_level_advance(self, obs: Observation) -> None:
        self._tick = 0

    def step(self, obs: Observation) -> Decision:
        self._tick += 1
        run = C.mask(C.RIGHT, C.B)
        jump = C.mask(C.RIGHT, C.B, C.A)
        if self._tick % 4 == 0:
            return Decision(plan=[Step(20, jump), Step(4, run)], note="hop")
        return Decision(plan=[Step(24, run)], note="run")

    def advance_plan(self, obs: Observation) -> Plan:
        return [Step(4, C.mask(C.RIGHT, C.B))]

    def danger_candidates(self, obs: Observation) -> list[Plan]:
        run = C.mask(C.RIGHT, C.B)
        jump = C.mask(C.RIGHT, C.B, C.A)
        out: list[Plan] = [[Step(24, run)], [Step(8, 0), Step(24, run)]]
        for hold in (8, 14, 20, 26, 32):
            out.append([Step(hold, jump), Step(8, run)])
            out.append([Step(6, run), Step(hold, jump)])
        out.append([Step(12, C.LEFT), Step(20, jump)])
        return out

    def expanded_candidates(self, obs: Observation) -> list[Plan]:
        return self.danger_candidates(obs)


class _TrackedSystem(NesSystem):
    """NES system whose sessions carry the tracker through emulator time-travel."""

    def __init__(self, retro_id: str, tracker: PixelTracker):
        super().__init__(retro_id)
        self._tracker = tracker

    def connect(self) -> Session:
        return _TrackedSession(super().connect(), self._tracker)


class PixelPlatformerGame(Game):
    """Any side-scroller, perceived from the screen. `BILLY_RETRO_GAME` picks the ROM."""
    name = "Pixel Platformer"

    def __init__(self) -> None:
        retro_id = os.environ.get("BILLY_RETRO_GAME", "SuperMarioBros-Nes-v0")
        self.tracker = PixelTracker()
        self.system = _TrackedSystem(retro_id, self.tracker)
        # observe() is (frame, ram, rgb); keep the last good view for garbage frames.
        self._last_progress = 0

    def observe(self, frame: int, ram: bytes, rgb=None) -> Observation:
        if rgb is None:
            return Observation(frame=frame, progress=self._last_progress, score=0,
                               level_label="pixel #0", level_key=("pixel", 0), dead=False,
                               summary="no frame", ascii_map="", raw=None)
        view = self.tracker.update(rgb, frame)
        if view.in_play:
            self._last_progress = view.progress
        label = f"pixel #{view.area_seq}"
        return Observation(
            frame=frame,
            progress=view.progress if view.in_play else self._last_progress,
            score=0,
            level_label=label,
            level_key=("pixel", view.area_seq),
            dead=view.dead,
            summary=view.summary(),
            ascii_map="",
            raw=view,
            # Don't key memory on a signal we can't reproduce: blob bottoms jitter ±8px
            # (cell quantization), so raw elevation splits identical spots into different
            # y-bands and banked solutions never replay. One band until pixel ground
            # estimation is solid; x-buckets carry the position identity.
            elevation=0,
        )

    def make_reflex(self) -> ReflexPolicy:
        return PixelReflex()

    def level_cleared(self, prev_key: tuple, new_key: tuple) -> bool:
        return False   # generically unknowable from pixels (v1) — treat scenes as screens

    def screen_changed(self, prev_key: tuple, new_key: tuple) -> bool:
        return prev_key != new_key

    def search_area_advance(self, start_key: tuple, end_key: tuple) -> bool:
        return end_key > start_key

    def boot(self, session: Session) -> Observation:
        """Press START past a title screen if needed, then wiggle to seed player tracking."""
        session.reset()

        def obs() -> Observation:
            st = session.read_state()
            return self.observe(st.frame, st.ram, getattr(st, "rgb", None))

        for _ in range(4):                       # title screens want START
            o = obs()
            if o.raw is not None and o.raw.in_play:
                break
            session.send_plan([Step(8, C.START), Step(30, 0)])
        # Wiggle: motion is what makes the player visible to pixel tracking.
        for _ in range(6):
            session.send_plan([Step(8, C.mask(C.RIGHT, C.B)), Step(4, 0)])
            o = obs()
            if o.raw is not None and o.raw.player is not None:
                break
        return obs()
