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

    def observe(self, frame: int, ram: bytes, rgb=None) -> Observation:
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
            elevation=s.mario_y,   # 2nd route coordinate (larger = lower on screen)
        )

    def make_reflex(self) -> ReflexPolicy:
        return SmbReflex()

    def hazard_hooks(self):
        from .hazard_hooks import SmbHazardHooks
        return SmbHazardHooks()

    def remix_demo_end_ok(self, result, req: dict) -> bool:
        want = req.get("level_label", "")
        return not want or result.end_label == want

    def remix_dropin_is_safe(self, obs, req: dict) -> bool:
        """Not on the pit lip, not sprinting, not past the wall — teachable runway."""
        if not super().remix_dropin_is_safe(obs, req):
            return False
        from .capture_util import near_pit
        # Standing on the lip is certain death once the human takes control.
        if near_pit(obs):
            gap = obs.raw.gap_info() if hasattr(obs.raw, "gap_info") else None
            if gap is not None and gap[0] < 40:
                return False
        return True

    def remix_stabilize_dropin(self, session, observe, req: dict):
        """Bleed x-speed and back off the pit lip so the human isn't launched into a cliff."""
        from .capture_util import near_pit, settle_mario
        from ...abstractions import Step
        from ...systems.nes import controller as C

        death_x = int(req.get("death_x", 0))
        obs = observe()
        if obs.dead:
            return False, obs
        # Bleed sprint / jump momentum.
        ok, _ = settle_mario(session, observe, allow_left=True, max_frames=180)
        obs = observe()
        if obs.dead:
            return False, obs
        # Back away from the lip if we're on it (or too close to the wall).
        for _ in range(40):
            if obs.dead:
                return False, obs
            too_close_wall = death_x and obs.progress >= death_x - 24
            lip = near_pit(obs)
            gap = obs.raw.gap_info() if hasattr(obs.raw, "gap_info") else None
            on_lip = lip and gap is not None and gap[0] < 40
            if self.remix_on_ground(obs) and not too_close_wall and not on_lip:
                if abs(int(obs.raw.x_speed)) <= 2:
                    break
            session.send_plan([Step(4, C.LEFT if (too_close_wall or on_lip) else C.NEUTRAL)])
            obs = observe()
        settle_mario(session, observe, allow_left=False, max_frames=60)
        obs = observe()
        return self.remix_dropin_is_safe(obs, req), obs

    def remix_capture_ready(self, session, observe, req: dict):
        """Snapshot only settled solid ground with runway — never the pit lip."""
        from .capture_util import capture_ready, near_pit, settle_mario
        label = req.get("level_label", "")
        x_min, x_max = self.remix_approach_progress_window(req)
        obs = observe()
        if obs.dead or obs.level_label != label:
            return False, obs
        if not (self.remix_on_ground(obs) and x_min <= obs.progress <= x_max):
            return False, obs
        # Refuse lip captures for Remix teach (human needs runway, not certain death).
        if near_pit(obs):
            gap = obs.raw.gap_info() if hasattr(obs.raw, "gap_info") else None
            if gap is not None and gap[0] < 40:
                return False, obs
        ok, _ = settle_mario(session, observe, allow_left=False)
        obs2 = observe()
        if ok and capture_ready(obs2, x_min=x_min, x_max=x_max, level_label=label):
            if self.remix_dropin_is_safe(obs2, req):
                return True, obs2
        return False, obs2

    def remix_director_sections(self):
        from ...rl.section_policy import SectionController, default_smb_sections
        return SectionController(default_smb_sections())

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
