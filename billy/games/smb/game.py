"""Super Mario Bros on the NES — the Game adapter binding perception + reflexes + lifecycle."""
from __future__ import annotations

from ...abstractions import BootError, Game, Observation, ReflexPolicy, Session
from ...systems.nes import controller
from ...systems.nes.system import NesSystem
from .perception import build_scene
from .reflexes import SmbReflex


class SmbGame(Game):
    name = "Super Mario Bros"
    RETRO_GAME = "SuperMarioBros-Nes-v0"   # stable-retro integration id (subclasses override)

    def __init__(self) -> None:
        self.system = NesSystem(self.RETRO_GAME)
        # Last in-play (progress, level_label, level_key) — reused on death/transition frames
        # so the engine never sees the 0xFFFF overflow (x=65535 / "256-256" garbage).
        self._last_good: tuple[int, str, tuple] = (0, "1-1", (0, 0, 0))

    def observe(self, frame: int, ram: bytes) -> Observation:
        s = build_scene(ram, frame)
        if s.in_play:
            # level_key includes the AREA (0x0760): a level like 1-2 has multiple areas joined by
            # pipes, and entering a pipe warps Mario to a new area where x RESETS. Keying on
            # (world, stage, area) keeps post-pipe solutions in their own cache region (no collision
            # with start-of-level buckets) and lets the engine SEE the pipe warp as a forward
            # transition (area advances), which is how Billy gets past 1-2's mandatory exit pipe.
            progress, level_label, level_key = s.mario_x, s.world_stage, (s.world, s.stage, s.area)
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
        # Liveness via in_play (plausible level + sane world-x), not the timer: SMB2-Japan reads
        # time=0 in its start state, so a time>0 check would wrongly reject it.
        if not (after.raw.in_play and after.raw.mario_x >= before.raw.mario_x):
            raise BootError("could not gain control after reset — is the integration loaded?")
        return after
