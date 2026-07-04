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


def _load_pad_map() -> dict:
    """Gamepad mapping: defaults → saved calibration (data/pad_map.json, written by
    `teleop.py calibrate`) → BILLY_PAD_* env overrides. See systems/nes/pad_map.py."""
    from .pad_map import load_pad_map
    return load_pad_map()


class _Viewer:
    """A tiny, best-effort pyglet window to watch Billy. Any failure degrades to headless.

    Doubles as the teleop keyboard source: while the window has focus it tracks held keys and
    maps them to an NES button mask (arrows + Z/X = A/B, Tab/RShift = Start/Select). Enter ends
    a teleop demo; Esc aborts it. `teleop_poll()` is a no-op until a window exists."""

    def __init__(self, scale: int = 3, controller_mod=None) -> None:
        import pyglet
        self._pyglet = pyglet
        self._c = controller_mod or controller   # button vocabulary (defaults to NES)
        self.window = None
        self.scale = scale
        self._keys: set[int] = set()
        self._finish = False
        self._abort = False
        self._takeover = False    # T pressed: the human wants the controller (live demo)
        self.joystick = None
        self._joy_map = _load_pad_map()
        # On-screen overlay (calibration prompts etc.): first line renders big/highlighted.
        # Labels are rebuilt only when the lines change (per-frame Label construction is slow).
        self._overlay: tuple[str, ...] = ()
        self._overlay_key: tuple[str, ...] | None = None
        self._overlay_widgets: list = []

    def _open_joystick(self) -> None:
        """Best-effort: open the first gamepad (e.g. 8Bitdo SN30 Pro) for teleop input.

        On macOS, pyglet HID (gamepad) events are NOT delivered by window.dispatch_events();
        they come through the platform event loop, which we must start and step each frame."""
        try:
            sticks = self._pyglet.input.get_joysticks()
            if sticks:
                self.joystick = sticks[0]
                self.joystick.open(window=self.window)
                self._pyglet.app.platform_event_loop.start()
        except Exception:
            self.joystick = None

    def _dir_active(self, j, spec: dict, dz: float, btns) -> bool:
        """Evaluate one calibrated direction spec against the live pad (see pad_map.py)."""
        src = str(spec.get("src"))
        if src == "button":
            idx = spec.get("idx", -1)
            return isinstance(idx, int) and 0 <= idx < len(btns) and bool(btns[idx])
        if src == "hat":
            # Exact-tuple match: some pads' hats come through the OS HID layer rotated or
            # scrambled, so per-axis signs lie. Whatever (hat_x, hat_y) the pad emitted while
            # the user HELD this direction during calibration is this direction, verbatim.
            want = spec.get("value")
            return (want is not None
                    and (getattr(j, "hat_x", 0), getattr(j, "hat_y", 0)) == tuple(want))
        if src == "hat_x" or src == "hat_y":
            val = getattr(j, src, 0)
        elif src == "stick_x":                       # legacy alias for axis x
            val = getattr(j, "x", 0.0) or 0.0
        elif src == "stick_y":                       # legacy alias for axis y
            val = getattr(j, "y", 0.0) or 0.0
        elif src.startswith("axis_"):                # generic HID axis (x/y/z/rx/ry/rz)
            val = getattr(j, src[5:], 0.0) or 0.0
        else:
            return False
        threshold = 0.5 if src.startswith("hat") else dz
        return (val - spec.get("rest", 0.0)) * spec.get("sign", 1) > threshold

    def _joy_mask(self) -> int:
        j = self.joystick
        if j is None:
            return 0
        m = 0
        dz = float(self._joy_map.get("deadzone", 0.4))
        btns = getattr(j, "buttons", []) or []
        dirs = self._joy_map.get("dirs") or {}
        if dirs:
            # Calibrated per-direction specs own movement: each direction reads whatever the
            # wizard saw change when the user held it (d-pad button / hat axis / stick axis).
            for name, spec in dirs.items():
                bit = getattr(self._c, str(name).upper(), 0)
                if bit and isinstance(spec, dict) and self._dir_active(j, spec, dz, btns):
                    m |= bit
        else:
            # Legacy heuristics (defaults + pre-directional maps): LEFT ANALOG STICK with
            # optional per-axis inversion; d-pad hat only via `use_hat`.
            x = getattr(j, "x", 0.0) or 0.0
            y = getattr(j, "y", 0.0) or 0.0
            if self._joy_map.get("invert_x"):
                x = -x
            if self._joy_map.get("invert_y"):
                y = -y
            if x < -dz:
                m |= self._c.LEFT
            if x > dz:
                m |= self._c.RIGHT
            if y < -dz:
                m |= self._c.UP
            if y > dz:
                m |= self._c.DOWN
            if self._joy_map.get("use_hat"):
                try:
                    if j.hat_x < -0.5:
                        m |= self._c.LEFT
                    if j.hat_x > 0.5:
                        m |= self._c.RIGHT
                except Exception:
                    pass
        def pressed(idx):
            return isinstance(idx, int) and 0 <= idx < len(btns) and btns[idx]
        # Roles are LOGICAL and resolved against the active controller module, so one saved
        # map serves every console (e.g. "SPIN" only binds when the SNES controller is active).
        from .pad_map import ROLE_KEYS
        for role in ROLE_KEYS:
            if role == "FINISH":
                continue
            bit = getattr(self._c, role, 0)
            if bit and pressed(self._joy_map.get(role, -1)):
                m |= bit
        if pressed(self._joy_map.get("FINISH", -1)):
            self._finish = True
        return m

    def _bind_keys(self) -> None:
        key = self._pyglet.window.key
        self._KEYMAP = {
            key.UP: self._c.UP, key.DOWN: self._c.DOWN,
            key.LEFT: self._c.LEFT, key.RIGHT: self._c.RIGHT,
            key.Z: self._c.A, key.X: self._c.B,
            key.TAB: self._c.START, key.RSHIFT: self._c.SELECT,
            key.LSHIFT: self._c.SELECT,
        }
        # Consoles with extra buttons contribute additional teleop keys, e.g. SNES
        # {"C": SPIN, "S": X, "Q": L, "W": R} — controller modules opt in via VIEWER_KEYS.
        for key_name, bit in getattr(self._c, "VIEWER_KEYS", {}).items():
            sym = getattr(key, key_name, None)
            if sym is not None:
                self._KEYMAP[sym] = bit
        ENTER, ESC = key.ENTER, key.ESCAPE

        TAKEOVER = key.T

        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == ENTER:
                self._finish = True
            elif symbol == ESC:
                self._abort = True
            elif symbol == TAKEOVER:
                self._takeover = True
            elif symbol in self._KEYMAP:
                self._keys.add(symbol)

        @self.window.event
        def on_key_release(symbol, modifiers):
            self._keys.discard(symbol)

    def current_mask(self) -> int:
        m = 0
        for sym in self._keys:
            m |= self._KEYMAP.get(sym, 0)
        return m | self._joy_mask()

    def teleop_poll(self) -> tuple[int, bool, bool]:
        """(held-button mask, finish_requested, abort_requested) — pumps window events first."""
        if self.window is None:
            return 0, False, False
        self.window.switch_to()
        self.window.dispatch_events()
        if self.joystick is not None:
            # Pump HID/gamepad events (not delivered via window.dispatch_events on macOS).
            try:
                self._pyglet.app.platform_event_loop.step(0)
            except Exception:
                pass
        return self.current_mask(), self._finish, self._abort

    def reset_teleop(self) -> None:
        self._keys.clear()
        self._finish = False
        self._abort = False   # a latched ESC must not leak into the next prompt/stage
        self._takeover = False
        self._abort = False

    def show(self, frame: np.ndarray) -> None:
        h, w, _ = frame.shape
        if self.window is None:
            self.window = self._pyglet.window.Window(
                width=w * self.scale, height=h * self.scale, caption="Billy Mitchell", vsync=False)
            self._bind_keys()
            self._open_joystick()
        img = self._pyglet.image.ImageData(w, h, "RGB", frame.tobytes(), pitch=-w * 3)
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.clear()
        tex = img.get_texture()
        tex.width, tex.height = w * self.scale, h * self.scale
        tex.blit(0, 0)
        if self._overlay:
            self._draw_overlay(w * self.scale, h * self.scale)
        self.window.flip()

    def set_overlay(self, lines) -> None:
        """On-screen text banner (e.g. calibration prompts). None/[] clears it."""
        self._overlay = tuple(str(x) for x in lines) if lines else ()

    def _draw_overlay(self, width: int, height: int) -> None:
        try:
            if self._overlay != self._overlay_key:
                shapes, text = self._pyglet.shapes, self._pyglet.text
                title, rest = self._overlay[0], self._overlay[1:]
                band_h = 46 + 22 * len(rest)
                widgets = [shapes.Rectangle(0, height - band_h, width, band_h,
                                            color=(0, 0, 0))]
                widgets[0].opacity = 185
                widgets.append(text.Label(
                    title, x=width // 2, y=height - 24, anchor_x="center", anchor_y="center",
                    font_size=18, bold=True, color=(255, 220, 60, 255)))
                for i, line in enumerate(rest):
                    widgets.append(text.Label(
                        line, x=width // 2, y=height - 48 - 22 * i,
                        anchor_x="center", anchor_y="center",
                        font_size=12, color=(235, 235, 235, 255)))
                self._overlay_widgets = widgets
                self._overlay_key = self._overlay
            # pyglet 1.5: tex.blit leaves GL_TEXTURE_2D enabled, which corrupts untextured
            # shape drawing (the backdrop would sample the game frame). Disable it first;
            # harmless no-op wrapped for pyglet 2.x core profiles.
            with contextlib.suppress(Exception):
                from pyglet import gl
                gl.glDisable(gl.GL_TEXTURE_2D)
            for wdg in self._overlay_widgets:
                wdg.draw()
        except Exception as e:
            # Never break the frame loop — but report once and stop retrying.
            if self._overlay_key is not None or self._overlay_widgets:
                print(f"[viewer] overlay draw failed ({type(e).__name__}: {e}) — "
                      f"banner disabled (terminal prompts still apply)")
            self._overlay_widgets, self._overlay_key = [], None
            self._overlay = ()

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
    info: dict = None    # integration info (lives/score/…) — optional, RAM-map-free games use it


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


class _AudioSink:
    """Best-effort speaker output for LIVE, real-time play — the game's music + SFX.

    stable-retro already generates audio every frame (`em.get_audio()` → int16 stereo at
    `em.get_audio_rate()`); we just stream it. A PortAudio output stream pulls from a bounded
    ring buffer that each live emulator step feeds. Search rollouts and turbo/headless runs
    never feed it (the caller gates on show+realtime), so fast paths stay silent and the audio
    only ever hears the frames you're actually watching. Any failure (no device, no PortAudio,
    CI) degrades to silence — never breaks play. Latency is bounded by dropping the oldest
    audio when the buffer runs long (a real-time run that briefly stutters must not build a
    growing delay); underflow pads with silence."""

    _MAX_BUFFER_SEC = 0.20      # cap queued audio → cap lip-sync lag (drop oldest beyond this)

    def __init__(self, rate: int, channels: int = 2) -> None:
        import threading
        self.rate = int(rate)
        self.channels = channels
        self._buf = np.zeros((0, channels), dtype=np.int16)
        self._lock = threading.Lock()
        self._cap = int(self.rate * self._MAX_BUFFER_SEC)
        self._stream = None

    def start(self) -> bool:
        try:
            import sounddevice as sd
            self._stream = sd.OutputStream(
                samplerate=self.rate, channels=self.channels, dtype="int16",
                blocksize=0, callback=self._callback)
            self._stream.start()
            return True
        except Exception as e:      # no device / no PortAudio / unsupported rate → stay silent
            print(f"[audio] sound disabled ({type(e).__name__}: {e}) — play continues silent")
            self._stream = None
            return False

    def _callback(self, outdata, frames, time_info, status) -> None:
        with self._lock:
            have = min(frames, len(self._buf))
            if have:
                outdata[:have] = self._buf[:have]
                self._buf = self._buf[have:]
            if have < frames:
                outdata[have:] = 0      # underflow → silence, no crash

    def feed(self, samples) -> None:
        if self._stream is None or samples is None:
            return
        a = np.asarray(samples, dtype=np.int16)
        if a.ndim == 1:
            a = np.stack([a, a], axis=1)
        if a.shape[1] != self.channels:
            a = a[:, : self.channels]
        with self._lock:
            self._buf = np.concatenate([self._buf, a]) if len(self._buf) else a
            if len(self._buf) > self._cap:      # running behind → drop oldest, keep it live
                self._buf = self._buf[-self._cap:]

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class RetroSession:
    """A stable-retro env presented through the engine's lock-step Session contract."""

    def __init__(self, render: bool | None = None, game: str | None = None,
                 inttype=None, controller_mod=None, ram_size: int | None = None,
                 restricted_actions=None) -> None:
        # Watchable by default; set BILLY_HEADLESS=1 for fast benchmarks (no window).
        if render is None:
            render = os.environ.get("BILLY_HEADLESS", "0") != "1"
        game = game or _GAME   # integration id: arg > BILLY_RETRO_GAME env > SMB default
        inttype = _resolve_inttype(inttype)
        # Console parameterization: this transport is console-agnostic — the button vocabulary
        # and work-RAM size are the only per-console bits, supplied by the system's controller
        # module (defaults: NES). SNES passes systems/snes/controller + its WRAM size.
        self.controller = controller_mod or controller
        self.ram_size = ram_size or RAM_SIZE
        # Logical->physical button-name translation (e.g. our logical A/jump is SNES's "B").
        self._retro_names = getattr(self.controller, "RETRO_NAMES", {})
        # Always render to an offscreen array; WE decide which frames reach the screen, so
        # micro-search frames stay hidden (the live run never visibly rewinds).
        # `restricted_actions` (e.g. Actions.ALL) is a per-console need: the Genesis FILTERED
        # space silently strips START, so its system opts out of filtering.
        make_kwargs = {"render_mode": "rgb_array"}
        if restricted_actions is not None:
            make_kwargs["use_restricted_actions"] = restricted_actions
        try:
            self.env = retro.make(game, inttype=inttype, **make_kwargs)
        except FileNotFoundError:
            if inttype == Integrations.STABLE:
                self.env = retro.make(game, inttype=Integrations.EXPERIMENTAL, **make_kwargs)
            else:
                raise
        self._viewer = _Viewer(controller_mod=self.controller) if render else None
        self._show = render          # True only while executing committed (live) play
        self._realtime = render and os.environ.get("BILLY_TURBO", "0") != "1"
        # Sound: on for watchable real-time play (the music is half the fun), off for headless/
        # turbo/search. BILLY_SOUND=0 mutes; =1 forces even if something else would skip it.
        self._audio: _AudioSink | None = None
        _sound_env = os.environ.get("BILLY_SOUND")
        if _sound_env != "0" and (self._realtime or _sound_env == "1"):
            try:
                rate = int(self.env.em.get_audio_rate())
                sink = _AudioSink(rate)
                if sink.start():
                    self._audio = sink
            except Exception as e:
                print(f"[audio] unavailable ({type(e).__name__}: {e}) — play continues silent")
        # Map our controller button names -> stable-retro action-vector indices.
        # env.buttons looks like ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A'].
        self._btn_index = {name.upper(): i for i, name in enumerate(self.env.buttons) if name}
        self._n_buttons = len(self.env.buttons)
        self._frame = 0
        self._ram = bytes(self.ram_size)
        self._rgb: np.ndarray | None = None
        self._info: dict = {}
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

    def takeover_requested(self) -> bool:
        """True once if the human pressed T in the watch window (live-demo takeover).
        Reading clears the latch. Always False headless."""
        v = self._viewer
        if v is None or not getattr(v, "_takeover", False):
            return False
        v._takeover = False
        return True

    def teleop_reset(self) -> None:
        if self._viewer is not None:
            self._viewer.reset_teleop()

    def set_pad_map(self, mapping: dict) -> None:
        """Apply a (freshly calibrated) pad mapping to the live viewer without reconnecting."""
        if self._viewer is not None:
            self._viewer._joy_map = dict(mapping)

    def reopen_joystick(self) -> bool:
        """Rescan for a gamepad (e.g. it was asleep when the window opened). True if present."""
        v = self._viewer
        if v is None:
            return False
        if v.joystick is None:
            v._open_joystick()
        return v.joystick is not None

    def set_overlay(self, lines) -> None:
        """Draw a text banner over the game frame (calibration prompts). None clears it."""
        if self._viewer is not None:
            self._viewer.set_overlay(lines)

    def pad_state(self):
        """Raw gamepad state for calibration: pressed button indices, hat, sticks (or None)."""
        v = self._viewer
        if v is None or v.joystick is None:
            return None
        j = v.joystick

        def ax(name):
            return round(getattr(j, name, 0.0) or 0.0, 3)

        btns = [i for i, b in enumerate(getattr(j, "buttons", []) or []) if b]
        return {
            "buttons": btns,
            "hat": (getattr(j, "hat_x", 0), getattr(j, "hat_y", 0)),
            "stick": (ax("x"), ax("y")),
            # ALL HID axes: pads route d-pads/sticks to surprising axes per mode/OS, and some
            # axes float — the calibration wizard decides per direction which one is real.
            "axes": {n: ax(n) for n in ("x", "y", "z", "rx", "ry", "rz")},
            "name": getattr(j.device, "name", "?"),
        }

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
        self._info = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
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
        return State(frame=self._frame, ram=self._ram, done=self._done, rgb=self._rgb,
                     info=self._info)

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
        if self._audio is not None:
            self._audio.close()
            self._audio = None
        try:
            self.env.close()
        except Exception:
            pass  # pyglet/Cocoa teardown is cosmetic on macOS

    # --- internals ----------------------------------------------------------------------
    def _action_from_mask(self, mask: int) -> np.ndarray:
        act = np.zeros(self._n_buttons, dtype=np.int8)
        for name in self.controller.names_from_mask(mask):
            name = name.upper()
            idx = self._btn_index.get(self._retro_names.get(name, name))
            if idx is not None:
                act[idx] = 1
        return act

    def _step_once(self, action: np.ndarray) -> None:
        result = self.env.step(action)
        # gymnasium 5-tuple (obs, reward, terminated, truncated, info) or legacy 4-tuple.
        if len(result) == 5:
            _, _, terminated, truncated, self._info = result
            self._done = bool(terminated or truncated)
        else:
            _, _, self._done, self._info = result
        self._frame += 1
        try:
            self._rgb = np.asarray(self.env.render())
        except Exception:
            self._rgb = None
        if self._show and self._realtime and self._audio is not None:
            # Only the frames you're actually watching reach the speakers — search runs under
            # search_mode() (show=False) stay silent, so no fast-forward chipmunk audio.
            try:
                self._audio.feed(self.env.em.get_audio())
            except Exception:
                pass
        if self._show and self._viewer is not None:
            self._display()

    def _display(self) -> None:
        try:
            frame = self._rgb if self._rgb is not None else np.asarray(self.env.render())
            self._viewer.show(frame)
            if self._realtime:
                time.sleep(1 / 60)
        except Exception as e:
            # Disable display on a windowing failure but SAY SO — a silent kill here freezes
            # the window and eats all pad/keyboard input, which reads as "nothing works".
            import traceback
            print(f"[viewer] display failed ({type(e).__name__}: {e}) — window disabled, "
                  f"play continues headless")
            traceback.print_exc()
            self._viewer = None

    def _refresh_ram(self) -> None:
        ram = self.env.get_ram()
        self._ram = bytes(np.asarray(ram, dtype=np.uint8)[:self.ram_size])
        # Recompute integration info (lives/score/…) from the CURRENT RAM. After a state
        # restore no env.step runs, so without this `info` would stay stale from before the
        # rewind — and an info-terminal game (shmup) would read a phantom death on respawn.
        try:
            self._info = self.env.data.lookup_all()
        except Exception:
            pass
