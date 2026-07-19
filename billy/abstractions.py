"""The contracts that decouple the engine from any specific system / controller / game.

Layering:  Game ─uses→ System ─uses→ Controller.  The engine (director, agents, knowledge,
metrics) depends ONLY on the interfaces here, so a new console is a new `systems/<x>/` and a
new title is a new `games/<y>/` — the engine never changes.

Also defines the generic input primitives (Step / Plan) and the data the engine passes
around (Observation, Decision).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

# --- input primitives -------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    """Hold `buttons` (a controller bitmask) for `frames` frames."""
    frames: int
    buttons: int

    def __post_init__(self) -> None:
        if not (1 <= self.frames <= 0xFFFF):
            raise ValueError(f"step frames out of range: {self.frames}")
        # 16-bit mask: bits 0-7 are the shared NES-layout roles, bits 8+ are console extras
        # (e.g. SNES SPIN/X/L/R). NES plans never exceed 0xFF, so persisted data is unchanged.
        if not (0 <= self.buttons <= 0xFFFF):
            raise ValueError(f"step buttons out of range: {self.buttons}")


Plan = Sequence[Step]


class BootError(RuntimeError):
    """Raised by Game.boot when it can't reach a playable state."""


def encode_plan(plan: Plan) -> bytes:
    """Encode steps as nsteps(u8) then per step dur(u16 LE) + buttonmask(u16 LE). Legacy of
    the external-bridge era (stable-retro runs in-process); kept for tooling round-trips."""
    steps = list(plan)
    if len(steps) > 0xFF:
        raise ValueError(f"too many steps in one plan: {len(steps)} (max 255)")
    out = bytearray([len(steps)])
    for s in steps:
        out += s.frames.to_bytes(2, "little")
        out += s.buttons.to_bytes(2, "little")
    return bytes(out)


def plan_frames(plan: Plan) -> int:
    return sum(s.frames for s in plan)


# --- engine data ------------------------------------------------------------------------
@dataclass
class Observation:
    """What a Game exposes to the (game-agnostic) engine each frame."""
    frame: int
    progress: int          # monotonic within-level progress (stuck/danger/metrics use this)
    score: int
    level_label: str       # human label, e.g. "1-1"
    level_key: tuple       # ordinal for "advanced to the next level" comparisons
    dead: bool
    summary: str           # compact text for the LLM
    ascii_map: str         # small visual for the LLM
    raw: Any = None        # game-specific scene, for the game's own reflex policy
    elevation: int = 0     # generic 2nd coordinate: the route key is (level, progress, elevation),
                           # so a high road and a low road at the same `progress` are distinct nodes.
                           # (SMB sets this to mario_y, where a LARGER value is lower on screen.)


@dataclass
class Decision:
    """A reflex tier's choice for one exchange."""
    plan: Plan
    needs_billy: bool = False
    note: str = ""
    search_candidates: "list[Plan] | None" = None   # variants for danger-zone micro-search


# --- the three layers -------------------------------------------------------------------
class Controller(ABC):
    """An input device: the button set + name<->bitmask encoding."""
    name: str
    neutral: int = 0

    @abstractmethod
    def mask_from_names(self, names: object) -> int: ...

    @abstractmethod
    def names_from_mask(self, mask: int) -> list[str]: ...


@runtime_checkable
class Session(Protocol):
    """Lock-step transport to a game running on a system."""
    def read_state(self): ...
    def send_plan(self, plan: Plan) -> None: ...
    def save_state(self, slot: int = 0) -> None: ...
    def load_state(self, slot: int = 0) -> None: ...
    def soft_reset(self) -> None: ...
    def wait_until_live(self, timeout_s: float = ...) -> None: ...
    def reset(self) -> None: ...


class System(ABC):
    """An emulated platform: transport + controller + RAM size + how to launch it."""
    name: str
    ram_size: int
    controller: Controller

    @abstractmethod
    def connect(self) -> Session: ...

    @abstractmethod
    def launch_command(self, rom: str) -> str: ...


class ReflexPolicy(ABC):
    """A game's fast tier: an action every frame, or `needs_billy` to escalate to the LLM."""
    @abstractmethod
    def reset(self, obs: Observation) -> None: ...

    @abstractmethod
    def note_level_advance(self, obs: Observation) -> None: ...

    @abstractmethod
    def step(self, obs: Observation) -> Decision: ...

    def advance_plan(self, obs: Observation) -> Plan:
        """A short 'keep making forward progress' input the engine uses to COAST during a
        micro-search rollout — so a candidate is evaluated while the agent keeps moving through
        the hazard, not standing still. Default is a no-op; games that have a clear 'forward'
        (e.g. run right) should override so learn-from-death can actually traverse a death zone."""
        return [Step(2, 0)]


class Game(ABC):
    """A specific title on a System, using that System's Controller."""
    name: str
    system: System

    @abstractmethod
    def observe(self, frame: int, ram: bytes, rgb=None) -> Observation: ...

    @abstractmethod
    def boot(self, session: Session) -> Observation:
        """Drive the emulator into a playable state (e.g. press Start) and return the first
        in-play observation. The engine snapshots the checkpoint afterward."""

    @abstractmethod
    def make_reflex(self) -> ReflexPolicy: ...

    def hazard_hooks(self) -> "HazardHooks":
        """Optional per-game hazard callbacks (lift zones, pit approaches, etc.)."""
        from .hazard_hooks import null_hooks
        return null_hooks()

    def guide_query(self, obs: "Observation") -> str:
        """Text used to retrieve walkthrough steps for the current situation.

        Defaults to the LLM summary, but summaries are telemetry ("#115 link=(120,141)
        hearts=2/3") while walkthroughs are prose ("kill all the Stalfos, one drops a key") —
        cosine retrieval needs shared VOCABULARY. Games override this to speak walkthrough
        language (level names, enemy words, objectives) so the right steps surface."""
        return obs.summary

    def level_cleared(self, prev_key: tuple, new_key: tuple) -> bool:
        """True when the player finished a major stage (SMB world-stage, Zelda dungeon, etc.).

        Screen/area hops within a stage are NOT clears — see `screen_changed`."""
        return new_key[:2] > prev_key[:2]

    def screen_changed(self, prev_key: tuple, new_key: tuple) -> bool:
        """True on an in-run screen/area change that should not count as a level clear."""
        return prev_key != new_key and not self.level_cleared(prev_key, new_key)

    def checkpoint_ready(self, obs: "Observation") -> bool:
        """Whether `obs` is a safe spot to bank the attempt-start / cross-session checkpoint.

        Default is SMB's model: on-ground and near the level start in x-pixels (so the respawn
        state and any entry-anchored tape verify from a start-like position). Games whose
        `progress` isn't an x-pixel (e.g. Zelda's objective score) override this."""
        return getattr(obs.raw, "on_ground", True) and 16 < obs.progress < 240

    def route_rank(self, obs: "Observation") -> int | None:
        """A monotonic 'how far along the intended route' scalar, for games whose `level_key`
        is NOT ordinal (Zelda's screen grid, etc.). Drives the cross-session checkpoint frontier:
        the furthest is banked by MAX rank instead of by `level_key > prev_key`.

        Default `None` = fall back to the ordinal-`level_key` ratchet (SMB world/stage/area).
        Returning non-None also opts the game into checkpointing on `screen_changed`, not just
        `level_cleared` — screen-progressing games advance without ever clearing a 'level'."""
        return None

    def search_area_advance(self, start_key: tuple, end_key: tuple) -> bool:
        """Whether a micro-search rollout 'warped' to a new area (pipe, new Zelda screen, etc.).

        Zelda returns False so exploration progress is measured by `progress` only, not room id."""
        return end_key > start_key

    def tape_moves(self) -> list[int]:
        """Mutation vocabulary for TAPE EVOLUTION — the button masks a movement window may take
        while search hill-climbs a whole input trajectory. Non-empty opts the game into the
        evolve loop (for reactive games whose deaths are positioning problems local search can't
        fix: a shmup, a moving-enemy gauntlet). Default empty = position-keyed learning only."""
        return []

    @property
    def evolves_tapes(self) -> bool:
        return bool(self.tape_moves())

    # --- Remix teach contract (Phase 1) -------------------------------------------------------
    # Optional hooks so a new title joins the gauntlet without editing remix.py. Defaults suit
    # progress-keyed platformers (SMB family); Zelda/PSII override for screen/battle wins.

    def remix_past_margin(self) -> int:
        """How far past Billy's death spot a crossing must end (anti-trivial-win for platformers)."""
        return 24

    def remix_goal(self, req: dict) -> str:
        """Human-readable challenge line shown during teach."""
        death_x = int(req.get("death_x", 0))
        return f"get past x≈{death_x} and keep going"

    def remix_min_progress(self) -> int:
        """Minimum progress gain for verify_demo to bank a crossing."""
        return 8

    def remix_win(self, obs: "Observation", req: dict, start_obs: "Observation") -> bool:
        """True when the human crossed the wall (survived and advanced past Billy's death spot)."""
        death_x = int(req.get("death_x", 0))
        return (getattr(obs.raw, "in_play", True)
                and obs.progress >= death_x + self.remix_past_margin())

    def remix_dropin_ok(self, obs: "Observation", req: dict) -> bool:
        """False if the drop-in already coasted to/past the wall before the human takes control."""
        death_x = int(req.get("death_x", 0))
        return obs.progress < death_x

    def remix_anchor_ok(self, source: str) -> bool:
        """True when the drop-in is a legitimate level/screen entry for entry-anchored tapes."""
        return source.startswith("start of")

    def remix_overlay_hint(self, obs: "Observation", req: dict) -> str:
        """Progress readout shown during teach (overlay line 2)."""
        death_x = int(req.get("death_x", 0))
        return f"x={obs.progress}  →  need {death_x + self.remix_past_margin()}"

    def remix_wall_at(self, req: dict) -> str:
        """Where Billy is stuck — shown in the teach banner."""
        return f"x≈{int(req.get('death_x', 0))}"

    def remix_demo_end_ok(self, result, req: dict) -> bool:
        """Extra verify gate before banking (e.g. demo must end on the taught level)."""
        return True

    def stuck_death_threshold(self) -> int:
        """Deaths at one hazard before Billy files a demo request / Remix surfaces the wall."""
        from . import config
        return config.STUCK_DEATH_THRESHOLD

    def remix_needs_approach_capture(self, req: dict) -> bool:
        """Drive Billy headless to just-before-the-wall when no drop-in state exists yet."""
        return True

    def remix_approach_progress_window(self, req: dict) -> tuple[int, int]:
        """Progress band [lo, hi] for a teachable drop-in just before the wall.

        Leave runway before the death (not on the cliff lip) so the human is not dropped
        into certain death or full-sprint into a pit."""
        death_x = int(req.get("death_x", 0))
        x_min = max(32, death_x - 120)
        # Stop ~2 tiles short of the death — lip snapshots coast/fall into the hazard.
        x_max = max(x_min + 16, death_x - 32)
        return x_min, x_max

    def remix_on_ground(self, obs: "Observation") -> bool:
        """Whether obs is safe to snapshot (on-ground equivalent — airborne won't reproduce)."""
        return getattr(obs.raw, "on_ground", True)

    def remix_dropin_level_ok(self, obs: "Observation", req: dict) -> bool:
        """Drop-in must be on the taught level/screen."""
        return obs.level_label == req.get("level_label", "")

    def remix_dropin_is_safe(self, obs: "Observation", req: dict) -> bool:
        """Human teach drop-in: solid ground, controlled speed, not already past/at the wall."""
        if obs.dead or not self.remix_on_ground(obs):
            return False
        if not self.remix_dropin_level_ok(obs, req):
            return False
        death_x = int(req.get("death_x", 0))
        if death_x and obs.progress >= death_x - 16:
            return False  # too close — certain death / auto-coast past the teach
        vx = abs(int(getattr(obs.raw, "x_speed", 0) or 0))
        if vx > 6:
            return False  # sprinting into the hazard
        return True

    def remix_stabilize_dropin(self, session: "Session", observe, req: dict
                               ) -> tuple[bool, "Observation"]:
        """After restore: bleed momentum so the human is not launched into a cliff.

        Default: require already-safe on-ground. Platformers override with settle + back off."""
        obs = observe()
        return self.remix_dropin_is_safe(obs, req), obs

    def remix_capture_ready(self, session: "Session", observe, req: dict
                            ) -> tuple[bool, "Observation"]:
        """Settle motion and confirm obs is reproducible before approach snapshot."""
        obs = observe()
        if obs.dead or not self.remix_on_ground(obs):
            return False, obs
        if not self.remix_dropin_level_ok(obs, req):
            return False, obs
        x_min, x_max = self.remix_approach_progress_window(req)
        if not (x_min <= obs.progress <= x_max):
            return False, obs
        if not self.remix_dropin_is_safe(obs, req):
            return False, obs
        return True, obs

    def remix_director_sections(self):
        """Optional SectionController for headless approach capture (SMB family)."""
        return None
