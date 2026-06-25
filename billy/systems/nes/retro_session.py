"""In-process NES transport backed by stable-retro (libretro).

Replaces the old FCEUX + file-IPC bridge. Because the emulator lives in this process we get:
  • no file handshake (the chronic `action.bin.tmp` race is gone),
  • frame-perfect, deterministic stepping,
  • silent state cloning (`clone_state`/`restore`) so micro-search can evaluate candidates on a
    COPY of the game and only the winner ever touches the live, on-screen run — no visible rewind.

It implements the same 7-method `Session` Protocol the Director already depends on, so nothing
upstream changes. RAM perception is unchanged: the NES work-RAM (0x0000-0x07FF) is the first
2 KB of `env.get_ram()`, and our address map reads it as before.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

import retro

from . import controller

RAM_SIZE = 0x800
_GAME = os.environ.get("BILLY_RETRO_GAME", "SuperMarioBros-Nes-v0")


@dataclass(frozen=True)
class State:
    """What read_state() hands back — mirrors the old ipc.State shape the engine expects."""
    frame: int
    ram: bytes
    done: bool = False


class RetroSession:
    """A stable-retro env presented through the engine's lock-step Session contract."""

    def __init__(self, render: bool | None = None) -> None:
        # Watchable by default; set BILLY_HEADLESS=1 for fast benchmarks (no window).
        if render is None:
            render = os.environ.get("BILLY_HEADLESS", "0") != "1"
        self._render = render
        self.env = retro.make(_GAME, render_mode="human" if render else None)
        # Map our controller button names -> stable-retro action-vector indices.
        # env.buttons looks like ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A'].
        self._btn_index = {name.upper(): i for i, name in enumerate(self.env.buttons) if name}
        self._n_buttons = len(self.env.buttons)
        self._frame = 0
        self._ram = bytes(RAM_SIZE)
        self._slots: dict[int, bytes] = {}
        self._done = False
        self._started = False

    # --- engine Session contract --------------------------------------------------------
    def reset(self) -> None:
        out = self.env.reset()
        self._started = True
        self._done = False
        self._frame = 0
        self._refresh_ram()

    def wait_until_live(self, timeout_s: float = 180.0) -> None:
        """In-process: the env is live as soon as it's reset. Ensure that has happened."""
        if not self._started:
            self.reset()

    def read_state(self, timeout_s: float | None = None) -> State:
        return State(frame=self._frame, ram=self._ram, done=self._done)

    def send_plan(self, plan) -> None:
        """Execute every frame of the plan on the live env, then publish the resulting RAM."""
        for step in plan:
            action = self._action_from_mask(step.buttons)
            for _ in range(step.frames):
                self._step_once(action)
        self._refresh_ram()

    def save_state(self, slot: int = 0) -> None:
        self._slots[slot] = self.env.em.get_state()

    def load_state(self, slot: int = 0) -> None:
        snap = self._slots.get(slot)
        if snap is not None:
            self.env.em.set_state(snap)
            self._refresh_ram()

    def soft_reset(self) -> None:
        """No title-screen dance needed — the integration boots in-play. Reset to start state."""
        self.reset()

    # --- micro-search support: clone the whole machine, evaluate, restore silently --------
    def clone_state(self) -> bytes:
        """Snapshot the full emulator state (for invisible candidate evaluation)."""
        return self.env.em.get_state()

    def restore(self, snapshot: bytes) -> None:
        self.env.em.set_state(snapshot)
        self._refresh_ram()

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass  # pyglet/Cocoa teardown is cosmetic on macOS

    # --- internals ----------------------------------------------------------------------
    def _action_from_mask(self, mask: int) -> np.ndarray:
        act = np.zeros(self._n_buttons, dtype=np.int8)
        for name in controller.names_from_mask(mask):
            idx = self._btn_index.get(name.upper())
            if idx is not None:
                act[idx] = 1
        return act

    def _step_once(self, action: np.ndarray) -> None:
        result = self.env.step(action)
        # gymnasium 5-tuple (obs, reward, terminated, truncated, info) or legacy 4-tuple.
        if len(result) == 5:
            _, _, terminated, truncated, _ = result
            self._done = bool(terminated or truncated)
        else:
            _, _, self._done, _ = result
        self._frame += 1

    def _refresh_ram(self) -> None:
        ram = self.env.get_ram()
        self._ram = bytes(np.asarray(ram, dtype=np.uint8)[:RAM_SIZE])
