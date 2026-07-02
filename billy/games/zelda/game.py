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

    # Combat progress: killing an enemy is worth this many progress units, and clearing a
    # screen that HAD enemies is worth a room-clear bonus. Without this, progress is
    # position-only, so a fight — where Link stands still and swings — reads as ZERO gain:
    # human fight demos couldn't bank, search couldn't value a kill, and learn-from-death
    # had nothing to optimise on combat-walled screens (the overworld #121 wall).
    KILL_CREDIT = 64
    ROOM_CLEAR_BONUS = 256

    def __init__(self) -> None:
        self.system = NesSystem(self.RETRO_GAME, retro_inttype=Integrations.EXPERIMENTAL)
        self._last_good: tuple[int, str, tuple] = (0, "overworld #119", ("overworld", 119))
        self._monotone_level_key: tuple | None = None
        self._monotone_frontier: int = 0
        self._screen_enemy_hi: int = 0    # most enemies seen at once on this screen
        self._kills_credit: int = 0       # monotonic kills-observed credit for this screen

    def _combat_credit(self, s, level_key: tuple) -> int:
        """Monotonic per-screen combat progress (kills + room-clear), tracked adapter-side
        exactly like the monotone frontier: it only ratchets up within a screen visit."""
        if level_key != self._monotone_level_key:     # new screen — reset combat tracking
            self._screen_enemy_hi = s.enemy_count()
            self._kills_credit = 0
        else:
            self._screen_enemy_hi = max(self._screen_enemy_hi, s.enemy_count())  # late spawns
            self._kills_credit = max(self._kills_credit,
                                     self._screen_enemy_hi - s.enemy_count())
        credit = self._kills_credit * self.KILL_CREDIT
        if self._screen_enemy_hi > 0 and s.enemy_count() == 0:
            credit += self.ROOM_CLEAR_BONUS           # fought a populated screen down to zero
        return credit

    def observe(self, frame: int, ram: bytes, rgb=None) -> Observation:
        s = build_scene(ram, frame, rgb=rgb)
        if s.in_play:
            level_label = s.room_label
            level_key = (s.realm, s.map_location)
            raw_progress = s.objective_score() + self._combat_credit(s, level_key)
            if level_key != self._monotone_level_key:
                self._monotone_level_key = level_key
                self._monotone_frontier = raw_progress
            else:
                self._monotone_frontier = max(self._monotone_frontier, raw_progress)
            progress = self._monotone_frontier
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

    _LEVEL_WORDS = ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")

    def guide_query(self, obs: Observation) -> str:
        """Walkthrough-vocabulary retrieval query. The raw summary is telemetry
        ("dungeon-1 #115 link=(120,141)") with no words in common with FAQ prose, so cosine
        retrieval surfaced junk inside dungeons — Billy stood in Level 1 'not knowing' a
        walkthrough he had fully ingested. Phrase the situation the way the FAQ writes it."""
        s = obs.raw
        if s is None or not getattr(s, "in_dungeon", False):
            return obs.summary
        lvl = max(1, int(getattr(s, "current_level", 1) or 1))
        word = self._LEVEL_WORDS[lvl - 1] if lvl <= len(self._LEVEL_WORDS) else str(lvl)
        bits = [f"Level {word} Level {lvl} dungeon walkthrough room"]
        if s.enemy_count() > 0:
            bits.append(f"kill all the enemies ({s.enemy_count()} in this room), "
                        f"one will drop a key")
        else:
            bits.append("room cleared, which door next")
        if getattr(s, "keys", 0) > 0:
            bits.append(f"{s.keys} keys held, open the locked door")
        d = getattr(s, "dungeon", None)
        if d is not None and getattr(d, "locked_doors", None):
            bits.append("locked door blocks the way")
        bits.append(obs.summary)
        return " ".join(bits)

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