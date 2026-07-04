"""Phantasy Star II (Genesis) — Game adapter: RAM Scene + exploration-credit progress.

Zelda's shape, RPG-tuned: level identity is the probed place key, progress within a place is
monotone EXPLORATION credit (distinct 16px tiles seen this visit) — so search/tapes value
"cover new ground", the RPG's equivalent of the platformer's x-frontier. Battles/EXP/death
are not yet perceived (their RAM addresses are deferred until Billy's own exploration
reaches the overworld; town has no encounters, so dead=False is honest here, and an attempt
ends on the frame budget rather than game over).
"""
from __future__ import annotations

from ...abstractions import BootError, Game, Observation, ReflexPolicy, Session, Step
from ...systems.genesis import controller as C
from ...systems.genesis.system import GenesisSystem
from .perception import build_scene
from .reflex import PsiiReflex


class PsiiGame(Game):
    name = "Phantasy Star II"
    RETRO_GAME = "PhantasyStarII-Genesis-v0"

    TILE_CREDIT = 16     # progress units per newly-seen 16px tile (this visit)

    def __init__(self) -> None:
        self.system = GenesisSystem(self.RETRO_GAME)
        self._last_good: tuple[int, str, tuple] = (0, "psii", ("psii", 0, 0, 0))
        # Per-place visited tiles, kept for the whole session: re-entering a map RESUMES its
        # exploration count (progress stays monotone within a visit, and covering new ground
        # always beats re-treading — the search signal an explorer needs).
        self._place_tiles: dict[tuple, set[tuple]] = {}

    def observe(self, frame: int, ram: bytes, rgb=None) -> Observation:
        s = build_scene(ram, frame)
        if s.in_play:
            level_key = ("psii",) + s.place
            tiles = self._place_tiles.setdefault(level_key, set())
            tiles.add(s.tile)
            progress = len(tiles) * self.TILE_CREDIT   # monotone: the set only grows
            label = f"psii {s.place[0]}.{s.place[1]}{'+' if s.outdoor else '-'}"
            self._last_good = (progress, label, level_key)
        else:
            progress, label, level_key = self._last_good
        return Observation(
            frame=frame,
            progress=progress,
            score=0,
            level_label=label,
            level_key=level_key,
            dead=False,      # no battle perception yet — town has no encounters
            summary=s.summary(),
            ascii_map="",
            raw=s,
            elevation=s.y // 16,
        )

    def make_reflex(self) -> ReflexPolicy:
        return PsiiReflex()

    # NOTE deliberately no tape_moves() override: non-empty opts into the whole-trajectory
    # EVOLVE loop (right for reactive shmups), which would REPLACE the cache/search/reflex
    # path. An RPG wants position-keyed learning + screen banking — the Zelda mode.

    def guide_query(self, obs: Observation) -> str:
        s = obs.raw
        if s is None:
            return obs.summary
        where = "outside in Mota" if s.outdoor else "inside a building in Paseo"
        return f"Phantasy Star II, early game near Paseo, {where}; what should Rolf do next?"

    def level_cleared(self, prev_key: tuple, new_key: tuple) -> bool:
        return False                       # story flags come with the battle-probe follow-up

    def screen_changed(self, prev_key: tuple, new_key: tuple) -> bool:
        return prev_key != new_key         # every place hop banks a route transition

    def search_area_advance(self, start_key: tuple, end_key: tuple) -> bool:
        return False                       # place ids aren't ordered — no warp semantics

    def boot(self, session: Session) -> Observation:
        """The integration's Start.state anchors IN PLAY (Rolf's doorstep, Paseo) — reset and
        settle a moment; no dance needed (that one-time cost lives in emulator/make_psii_state)."""
        session.reset()
        session.send_plan([Step(30, 0)])
        st = session.read_state()
        obs = self.observe(st.frame, st.ram, getattr(st, "rgb", None))
        if not (obs.raw and obs.raw.in_play):
            raise BootError(f"PSII boot not in play: {obs.summary}")
        return obs
