"""Generic shoot-'em-up adapter — the Game contract for a fixed-camera shooter.

Proves pixel generality beyond the platformer: no RAM map, no scroll, no ground. Perception is
billy/vision's ShmupTracker on the rgb stream (player ship + survival, all from pixels). The
one thing pixels can't cleanly give on this ROM is a robust TERMINAL — Airstriker's death has
no clean universal pixel signature (it's confounded by the player's own bullet area) — so the
terminal and score come from the emulator integration's `info` (lives/score), which every
stable-retro game ships. That is NOT a per-game RAM map we authored (the thing the bet is
about); it's the same standard channel as `reward`/`done`. The SUBSTANTIVE perception — finding
and tracking the ship the reflex must fly — is pixel-derived and general.

    BILLY_RETRO_GAME=Airstriker-Genesis-v0 .venv/bin/python run.py --game shmup --no-llm

The engine's whole learning stack rides on top; a shmup from a deterministic boot is a
TAPE-shaped problem (a fixed input trajectory reproduces), so search + tapes that survive
longer are what compounds.
"""
from __future__ import annotations

import os

from ...abstractions import Decision, Game, Observation, Plan, ReflexPolicy, Session, Step
from ...systems.nes import controller as C
from ...vision import ShmupTracker
from ..pixel.game import _TrackedSystem   # reuse: snapshots tracker state through time-travel


class ShmupReflex(ReflexPolicy):
    """Always FIRE (free and always-on in a shmup — it also keeps the field populated so the
    tracker's signals stay stable); movement is a dodge spread the search picks from. The ship
    lives at the bottom, threats fall from the top, so 'dodge' = strafe sideways / tuck up."""

    def __init__(self) -> None:
        self._tick = 0

    def reset(self, obs: Observation) -> None:
        self._tick = 0

    def note_level_advance(self, obs: Observation) -> None:
        self._tick = 0

    def _dodge_dir(self, obs: Observation) -> int:
        """Strafe AWAY from the nearest thing above the ship (pixel enemies), else drift."""
        view = obs.raw
        if view is None or view.player is None:
            return C.RIGHT
        px = view.player[0] + view.player[2] / 2
        above = [e for e in view.enemies if e[1] + e[3] < view.player[1]]
        if not above:
            return C.LEFT if (self._tick // 6) % 2 else C.RIGHT
        nearest = min(above, key=lambda e: abs(e[0] + e[2] / 2 - px) + (view.player[1] - e[1]))
        ex = nearest[0] + nearest[2] / 2
        return C.LEFT if ex > px else C.RIGHT     # step opposite the incoming threat

    def step(self, obs: Observation) -> Decision:
        self._tick += 1
        d = self._dodge_dir(obs)
        return Decision(plan=[Step(6, C.mask(C.B, d))], note="fire+dodge")

    def advance_plan(self, obs: Observation) -> Plan:
        return [Step(4, C.mask(C.B, self._dodge_dir(obs)))]

    def danger_candidates(self, obs: Observation) -> list[Plan]:
        fire = C.B
        out: list[Plan] = [[Step(18, C.mask(fire, C.LEFT))],
                           [Step(18, C.mask(fire, C.RIGHT))],
                           [Step(18, C.mask(fire, C.UP))],
                           [Step(12, C.mask(fire))],                       # hold position, fire
                           [Step(10, C.mask(fire, C.LEFT, C.UP))],
                           [Step(10, C.mask(fire, C.RIGHT, C.UP))]]
        for hold in (8, 16, 24):
            out.append([Step(hold, C.mask(fire, C.LEFT)), Step(hold, C.mask(fire, C.RIGHT))])
            out.append([Step(hold, C.mask(fire, C.RIGHT)), Step(hold, C.mask(fire, C.LEFT))])
        return out

    def expanded_candidates(self, obs: Observation) -> list[Plan]:
        return self.danger_candidates(obs)


class ShmupGame(Game):
    """Any fixed-camera shooter, perceived from the screen. `BILLY_RETRO_GAME` picks the ROM."""
    name = "Shmup"

    _SCORE_W = 1          # score folded into progress alongside survival (both monotonic)

    def __init__(self) -> None:
        retro_id = os.environ.get("BILLY_RETRO_GAME", "Airstriker-Genesis-v0")
        self.tracker = ShmupTracker()
        self.system = _TrackedSystem(retro_id, self.tracker)
        self._session: Session | None = None
        self._last_progress = 0

    # --- info (terminal/score) plumbed from the session's integration info ------------------
    def _info(self) -> dict:
        if self._session is None:
            return {}
        try:
            return self._session.read_state().info or {}
        except Exception:
            return {}

    def observe(self, frame: int, ram: bytes, rgb=None) -> Observation:
        info = self._info()
        lives = info.get("lives")
        score = int(info.get("score", 0) or 0)
        if rgb is None:
            return Observation(frame=frame, progress=self._last_progress, score=score,
                               level_label="shmup", level_key=("shmup", 0), dead=False,
                               summary="no frame", ascii_map="", raw=None)
        view = self.tracker.update(rgb, frame)
        # DEATH from the integration's lives (pixels can't give a robust terminal here); the
        # ship + enemies driving the reflex are all pixel-perceived. Terminal = ALL lives gone
        # (game over), not the first hit — that gives tape evolution a rich target (survive
        # every wave through all lives) with room to keep compounding, instead of maxing out
        # the first life in one generation.
        dead = lives is not None and lives <= 0
        # PROGRESS = survival (pixel, monotonic, reproducible) + score (kills). Both reward the
        # search for a longer/better trajectory from the deterministic boot.
        progress = view.progress + self._SCORE_W * score
        self._last_progress = progress
        return Observation(
            frame=frame, progress=progress, score=score, level_label="shmup",
            level_key=("shmup", 0), dead=dead, summary=f"{view.summary()} lives={lives}",
            ascii_map="", raw=view, elevation=0)

    def make_reflex(self) -> ReflexPolicy:
        return ShmupReflex()

    def tape_moves(self) -> list[int]:
        """Movement vocabulary for tape evolution — always firing, every dodge direction (the
        ship lives at the bottom, so left/right/up + diagonals cover the escapes)."""
        f = C.B
        return [C.mask(f), C.mask(f, C.LEFT), C.mask(f, C.RIGHT), C.mask(f, C.UP),
                C.mask(f, C.LEFT, C.UP), C.mask(f, C.RIGHT, C.UP), C.mask(f, C.DOWN)]

    def level_cleared(self, prev_key: tuple, new_key: tuple) -> bool:
        return False

    def screen_changed(self, prev_key: tuple, new_key: tuple) -> bool:
        return False      # one continuous fixed-camera screen

    def search_area_advance(self, start_key: tuple, end_key: tuple) -> bool:
        return False

    def boot(self, session: Session) -> Observation:
        """Press START, then settle just long enough to be live — and NO longer. The anchor
        (this state, checkpointed to slot 0) is where tape evolution begins, so it must land
        EARLY, while the whole trajectory is still malleable and all lives are intact: idling
        here loses a life around frame ~190, which would anchor evolution at a doomed state
        with no headroom. (The integration's `gameover` is a counter and `terminated` a fixed
        timer, so we can't gate on them — a short fixed dance is the reliable boot.)"""
        self._session = session
        session.reset()
        self.tracker.rebase()
        session.send_plan([Step(8, C.START), Step(32, 0)])
        st = session.read_state()
        return self.observe(st.frame, st.ram, getattr(st, "rgb", None))
