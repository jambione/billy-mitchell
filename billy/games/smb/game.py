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
        """Reset to the title screen, press Start until Mario is controllable, return the
        first in-play observation. Robust regardless of prior cartridge state."""
        def obs() -> Observation:
            st = session.read_state()
            return self.observe(st.frame, st.ram)

        obs()                              # first frame -> establishes a pending action
        session.soft_reset()
        obs()
        for _ in range(10):                # let the title screen settle
            session.send_plan(controller.idle(20))
            obs()
        for _ in range(80):
            session.send_plan(controller.press_start())
            before = obs()
            session.send_plan(controller.run_right(6, sprint=False))  # control test: nudge right
            after = obs()
            a, b = after.raw, before.raw
            if a.world == 0 and a.time > 0 and a.mario_x > b.mario_x:
                break
        else:
            raise BootError("could not gain control after pressing Start — is SMB loaded?")
        session.send_plan(controller.idle(2))
        return obs()
