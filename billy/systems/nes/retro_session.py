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

import contextlib
import os
import time
from dataclasses import dataclass

import numpy as np

import stable_retro as retro
from stable_retro.data import Integrations

from . import controller

RAM_SIZE = 0x800
_GAME = os.environ.get("BILLY_RETRO_GAME", "SuperMarioBros-Nes-v0")
_INTTYPE = os.environ.get("BILLY_RETRO_INTTYPE", "").lower()


class _Viewer:
    """A tiny, best-effort pyglet window to watch Billy. Any failure degrades to headless.

    Doubles as the teleop keyboard source: while the window has focus it tracks held keys and
    maps them to an NES button mask (arrows + Z/X = A/B, Tab/RShift = Start/Select). Enter ends
    a teleop demo; Esc aborts it. `teleop_poll()` is a no-op until a window exists."""

    def __init__(self, scale: int = 3) -> None:
        import pyglet
        self._pyglet = pyglet
        self.window = None
        self.scale = scale
        self._keys: set[int] = set()
        self._finish = False
        self._abort = False

    def _bind_keys(self) -> None:
        key = self._pyglet.window.key
        self._KEYMAP = {
            key.UP: controller.UP, key.DOWN: controller.DOWN,
            key.LEFT: controller.LEFT, key.RIGHT: controller.RIGHT,
            key.Z: controller.A, key.X: controller.B,
            key.TAB: controller.START, key.RSHIFT: controller.SELECT,
            key.LSHIFT: controller.SELECT,
        }
        ENTER, ESC = key.ENTER, key.ESCAPE

        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == ENTER:
                self._finish = True
            elif symbol == ESC:
                self._abort = True
            elif symbol in self._KEYMAP:
                self._keys.add(symbol)

        @self.window.event
        def on_key_release(symbol, modifiers):
            self._keys.discard(symbol)

    def current_mask(self) -> int:
        m = 0
        for sym in self._keys:
            m |= self._KEYMAP.get(sym, 0)
        return m

    def teleop_poll(self) -> tuple[int, bool, bool]:
        """(held-button mask, finish_requested, abort_requested) — pumps window events first."""
        if self.window is None:
            return 0, False, False
        self.window.switch_to()
        self.window.dispatch_events()
        return self.current_mask(), self._finish, self._abort

    def reset_teleop(self) -> None:
        self._keys.clear()
        self._finish = False
        self._abort = False

    def show(self, frame: np.ndarray) -> None:
        h, w, _ = frame.shape
        if self.window is None:
            self.window = self._pyglet.window.Window(
                width=w * self.scale, height=h * self.scale, caption="Billy Mitchell", vsync=False)
            self._bind_keys()
        img = self._pyglet.image.ImageData(w, h, "RGB", frame.tobytes(), pitch=-w * 3)
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.clear()
        tex = img.get_texture()
        tex.width, tex.height = w * self.scale, h * self.scale
        tex.blit(0, 0)
        self.window.flip()

    def close(self) -> None:
        if self.window is not None:
            with contextlib.suppress(Exception):
                self.window.close()


@dataclass(frozen=True)
class State:
    """What read_state() hands back — mirrors the old ipc.State shape the engine expects."""
    frame: int
    ram: bytes
    done: bool = False
    rgb: object = None   # latest rgb_array frame (optional; Zelda vision uses this)


def _resolve_inttype(inttype) -> Integrations:
    """Map None / env override / string aliases to a stable-retro Integrations value."""
    if inttype is not None:
        return inttype
    if _INTTYPE in ("experimental", "exp"):
        return Integrations.EXPERIMENTAL
    if _INTTYPE in ("all",):
        return Integrations.ALL
    if _INTTYPE in ("stable",):
        return Integrations.STABLE
    return Integrations.STABLE


class RetroSession:
    """A stable-retro env presented through the engine's lock-step Session contract."""

    def __init__(self, render: bool | None = None, game: str | None = None,
                 inttype=None) -> None:
        # Watchable by default; set BILLY_HEADLESS=1 for fast benchmarks (no window).
        if render is None:
            render = os.environ.get("BILLY_HEADLESS", "0") != "1"
        game = game or _GAME   # integration id: arg > BILLY_RETRO_GAME env > SMB default
        inttype = _resolve_inttype(inttype)
        # Always render to an offscreen array; WE decide which frames reach the screen, so
        # micro-search frames stay hidden (the live run never visibly rewinds).
        try:
            self.env = retro.make(game, render_mode="rgb_array", inttype=inttype)
        except FileNotFoundError:
            if inttype == Integrations.STABLE:
                self.env = retro.make(game, render_mode="rgb_array",
                                      inttype=Integrations.EXPERIMENTAL)
            else:
                raise
        self._viewer = _Viewer() if render else None
        self._show = render          # True only while executing committed (live) play
        self._realtime = render and os.environ.get("BILLY_TURBO", "0") != "1"
        # Map our controller button names -> stable-retro action-vector indices.
        # env.buttons looks like ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A'].
        self._btn_index = {name.upper(): i for i, name in enumerate(self.env.buttons) if name}
        self._n_buttons = len(self.env.buttons)
        self._frame = 0
        self._ram = bytes(RAM_SIZE)
        self._rgb: np.ndarray | None = None
        self._slots: dict[int, bytes] = {}
        self._done = False
        self._started = False

    # --- teleop (human-in-the-loop demo capture) ----------------------------------------
    def ensure_viewer(self) -> bool:
        """Make sure the watch window exists (so it can take keyboard focus). Returns success."""
        if self._viewer is None:
            return False
        if self._rgb is None:
            try:
                self._rgb = np.asarray(self.env.render())
            except Exception:
                return False
        prev = self._show
        self._show = True
        self._display()
        self._show = prev
        return self._viewer is not None and self._viewer.window is not None

    def teleop_poll(self) -> tuple[int, bool, bool]:
        """(held NES mask, finish?, abort?) from the watch window's keyboard."""
        if self._viewer is None:
            return 0, False, False
        return self._viewer.teleop_poll()

    def teleop_reset(self) -> None:
        if self._viewer is not None:
            self._viewer.reset_teleop()

    def teleop_step(self, mask: int) -> None:
        """Advance ONE frame with a human-held button mask, displayed and paced in real time."""
        prev_show, prev_rt = self._show, self._realtime
        self._show, self._realtime = True, True   # pace at 60fps even if BILLY_TURBO is set
        try:
            self._step_once(self._action_from_mask(mask))
        finally:
            self._show, self._realtime = prev_show, prev_rt
        self._refresh_ram()

    @contextlib.contextmanager
    def search_mode(self):
        """Within this block, stepped frames are NOT displayed — micro-search stays invisible."""
        prev = self._show
        self._show = False
        try:
            yield
        finally:
            self._show = prev

    # --- engine Session contract --------------------------------------------------------
    def reset(self) -> None:
        out = self.env.reset()
        self._started = True
        self._done = False
        self._frame = 0
        self._refresh_ram()
        try:
            self._rgb = np.asarray(self.env.render())
        except Exception:
            self._rgb = None

    def wait_until_live(self, timeout_s: float = 180.0) -> None:
        """In-process: the env is live as soon as it's reset. Ensure that has happened."""
        if not self._started:
            self.reset()

    def read_state(self, timeout_s: float | None = None) -> State:
        return State(frame=self._frame, ram=self._ram, done=self._done, rgb=self._rgb)

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
        try:
            self._rgb = np.asarray(self.env.render())
        except Exception:
            self._rgb = None
        if self._show and self._viewer is not None:
            self._display()

    def _display(self) -> None:
        try:
            frame = self._rgb if self._rgb is not None else np.asarray(self.env.render())
            self._viewer.show(frame)
            if self._realtime:
                time.sleep(1 / 60)
        except Exception:
            self._viewer = None  # disable display on any windowing failure; keep playing

    def _refresh_ram(self) -> None:
        ram = self.env.get_ram()
        self._ram = bytes(np.asarray(ram, dtype=np.uint8)[:RAM_SIZE])
