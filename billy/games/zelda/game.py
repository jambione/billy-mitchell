"""The Legend of Zelda (NES) — Game adapter binding perception + top-down reflexes."""
from __future__ import annotations

from stable_retro.data import Integrations

from ...abstractions import BootError, Game, Observation, ReflexPolicy, Session
from ...systems.nes import controller
from ...systems.nes.system import NesSystem
from .perception import build_scene
from .reflex import ZeldaReflex


class ZeldaGame(Game):
    name = "The Legend of Zelda"
    RETRO_GAME = "LegendOfZeldaPRG0-Nes"

    def __init__(self) -> None:
        self.system = NesSystem(self.RETRO_GAME, retro_inttype=Integrations.EXPERIMENTAL)
        self._last_good: tuple[int, str, tuple] = (0, "overworld #119", ("overworld", 119))

    def observe(self, frame: int, ram: bytes, rgb=None) -> Observation:
        s = build_scene(ram, frame, rgb=rgb)
        if s.in_play:
            level_label = s.room_label
            level_key = (s.realm, s.map_location)
            progress = s.objective_score()
            self._last_good = (progress, level_label, level_key)
        else:
            progress, level_label, level_key = self._last_good

        return Observation(
            frame=frame,
            progress=progress,
            score=s.rupees,
            level_label=level_label,
            level_key=level_key,
            dead=s.is_dying,
            summary=s.summary(),
            ascii_map=s.ascii_view(),
            raw=s,
            elevation=s.link_y,
        )

    def make_reflex(self) -> ReflexPolicy:
        return ZeldaReflex()

    def hazard_hooks(self):
        from .hazard_hooks import ZeldaHazardHooks
        return ZeldaHazardHooks()

    def level_cleared(self, prev_key: tuple, new_key: tuple) -> bool:
        """Realm change only (overworld ↔ dungeon) — not every screen hop."""
        return prev_key and new_key and prev_key[0] != new_key[0]

    def screen_changed(self, prev_key: tuple, new_key: tuple) -> bool:
        return (prev_key != new_key
                and prev_key and new_key
                and prev_key[0] == new_key[0])

    def search_area_advance(self, start_key: tuple, end_key: tuple) -> bool:
        return False   # room id jumps are not SMB-style pipe warps

    def boot(self, session: Session) -> Observation:
        def obs() -> Observation:
            st = session.read_state()
            return self.observe(st.frame, st.ram)

        session.reset()
        before = obs()
        session.send_plan(controller.run_right(8, sprint=False))
        after = obs()
        if not (after.raw.in_play and after.raw.link_x >= before.raw.link_x):
            raise BootError("could not gain control after reset — is the Zelda ROM imported?")
        return after