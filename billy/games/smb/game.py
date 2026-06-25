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

    def observe(self, frame: int, ram: bytes) -> Observation:
        s = build_scene(ram, frame)
        return Observation(
            frame=frame,
            progress=s.mario_x,
            score=s.score,
            level_label=s.world_stage,
            level_key=(s.world, s.stage),
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
