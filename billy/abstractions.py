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
        if not (0 <= self.buttons <= 0xFF):
            raise ValueError(f"step buttons out of range: {self.buttons}")


Plan = Sequence[Step]


class BootError(RuntimeError):
    """Raised by Game.boot when it can't reach a playable state."""


def encode_plan(plan: Plan) -> bytes:
    """Encode steps as nsteps(u8) then per step dur(u16 LE) + buttonmask(u8). Matches the
    decoder in each system's bridge. (8-bit masks today; widen per-controller when needed.)"""
    steps = list(plan)
    if len(steps) > 0xFF:
        raise ValueError(f"too many steps in one plan: {len(steps)} (max 255)")
    out = bytearray([len(steps)])
    for s in steps:
        out += s.frames.to_bytes(2, "little")
        out.append(s.buttons)
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
    def observe(self, frame: int, ram: bytes) -> Observation: ...

    @abstractmethod
    def boot(self, session: Session) -> Observation:
        """Drive the emulator into a playable state (e.g. press Start) and return the first
        in-play observation. The engine snapshots the checkpoint afterward."""

    @abstractmethod
    def make_reflex(self) -> ReflexPolicy: ...
