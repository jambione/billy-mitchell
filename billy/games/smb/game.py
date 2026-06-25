"""Super Mario Bros on the NES — the Game adapter binding perception + reflexes + lifecycle."""
from __future__ import annotations

from ...abstractions import BootError, Game, Observation, ReflexPolicy, Session
from ...systems.nes import controller
from ...systems.nes.system import NesSystem
from .perception import build_scene
from .reflexes import SmbReflex


class SmbGame(Game):
    name = "Super Mario Bros"

    def __init__(self) -> None:
        self.system = NesSystem()
        # Last in-play (progress, level_label, level_key) — reused on death/transition frames
        # so the engine never sees the 0xFFFF overflow (x=65535 / "256-256" garbage).
        self._last_good: tuple[int, str, tuple] = (0, "1-1", (0, 0))

    def observe(self, frame: int, ram: bytes) -> Observation:
        s = build_scene(ram, frame)
        if s.in_play:
            progress, level_label, level_key = s.mario_x, s.world_stage, (s.world, s.stage)
            self._last_good = (progress, level_label, level_key)
        else:
            progress, level_label, level_key = self._last_good  # don't trust garbage RAM
        return Observation(
            frame=frame,
            progress=progress,
            score=s.score,
            level_label=level_label,
            level_key=level_key,
            dead=s.is_dying,
            summary=s.summary(),
            ascii_map=s.ascii_view(),
            raw=s,
        )

    def make_reflex(self) -> ReflexPolicy:
        return SmbReflex()

    def boot(self, session: Session) -> Observation:
        """Bring the game to a controllable in-play state and return the first observation.

        With the in-process stable-retro integration the env resets directly into 1-1, so we
        just confirm control with a tiny nudge. (The old FCEUX path needed a title-screen +
        Start-press dance; that's handled by the integration's start state now.)"""
        def obs() -> Observation:
            st = session.read_state()
            return self.observe(st.frame, st.ram)

        session.reset()
        before = obs()
        session.send_plan(controller.run_right(6, sprint=False))  # control test: nudge right
        after = obs()
        if not (after.raw.time > 0 and after.raw.mario_x >= before.raw.mario_x):
            raise BootError("could not gain control after reset — is the SMB integration loaded?")
        return after
