"""Super Mario World on the SNES — the cross-console Game adapter.

The engine seam holds: this file supplies observe/boot/reflex; the Director, cache, tapes,
search, teleop, and stuck trainer are untouched. The shared PlatformerReflex drives play with
the LOGICAL button vocabulary (A=jump, B=run) that systems/snes/controller.py translates to
the SNES pad — zero new reflex code, the same carry-forward that made SMB2-Japan free.

Level identity: SMW is non-linear (an overworld map between levels), so the linear
"world-stage increases" clear rule doesn't apply. level_key = (events_triggered, mode_group,
translevel): EVENTS_TRIGGERED is a monotonic counter that bumps when a level is beaten, so the
engine's `new_key[:2] > prev_key[:2]` clear detection works unchanged; translevel changes on
the map are mere screen_changed events (their own tape/cache regions).
"""
from __future__ import annotations

from ...abstractions import BootError, Game, Observation, ReflexPolicy, Session
from ...systems.snes import controller
from ...systems.snes.system import SnesSystem
from ..common.platformer import PlatformerReflex
from .perception import build_scene
from .tuning import SMW_PROFILE


class SmwGame(Game):
    name = "Super Mario World"
    RETRO_GAME = "SuperMarioWorld-Snes"   # stable-retro's stock SNES integration

    def __init__(self) -> None:
        self.system = SnesSystem(self.RETRO_GAME)
        self._last_good: tuple[int, str, tuple] = (0, "smw", (0, 0, 0))

    def observe(self, frame: int, ram: bytes, rgb=None) -> Observation:
        s = build_scene(ram, frame, rgb=rgb)
        if s.in_play:
            progress = s.mario_x
            level_label = f"smw-{s.translevel}"
            level_key = (s.events_triggered, 0, s.translevel)
            self._last_good = (progress, level_label, level_key)
        else:
            progress, level_label, level_key = self._last_good
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
            elevation=s.mario_y,
        )

    def make_reflex(self) -> ReflexPolicy:
        return PlatformerReflex(SMW_PROFILE)

    def boot(self, session: Session) -> Observation:
        """stable-retro's SMW integration boots into a level via its default savestate; confirm
        control with a nudge, like SMB. If the integration lands on a menu/map instead, this is
        the place for the Start-press dance — see STATUS.md."""
        def obs() -> Observation:
            st = session.read_state()
            return self.observe(st.frame, st.ram)

        session.reset()
        o = obs()
        for _ in range(120):
            if o.raw.in_play:
                return o
            session.send_plan(controller.press_start())
            o = obs()
        raise BootError(
            "SMW did not reach a playable state — check the integration's default state "
            "(see games/smw/STATUS.md verification checklist)")
