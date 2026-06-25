"""Lock-step file IPC between the Python brain and the FCEUX Lua bridge.

Each exchange: the bridge publishes a state (req_id, frame, 2KB RAM) and blocks; Python
reads it, decides, and writes an action echoing that req_id; the bridge executes and
publishes the next state. The req_id handshake guarantees we never act on a stale frame and
never drop one. Writes are atomic (temp file + os.replace) so the reader always sees a whole
message.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from .abstractions import Plan, encode_plan


@dataclass(frozen=True)
class State:
    """One observed frame from the emulator."""
    req_id: int
    frame: int
    done: bool
    ram: bytes  # 2048 bytes of NES work RAM (0x0000-0x07FF)


class Bridge:
    """Owns the two lock-step files and the request handshake."""

    def __init__(self, state_file: Path = config.STATE_FILE, action_file: Path = config.ACTION_FILE):
        self.state_file = Path(state_file)
        self.action_file = Path(action_file)
        self._last_req = 0          # highest req_id we have consumed
        self._pending_req: int | None = None  # awaiting our action

    def reset(self) -> None:
        """Clear any stale files from a previous run so handshakes start clean."""
        config.ensure_dirs()
        for f in (self.state_file, self.action_file):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        self._last_req = 0
        self._pending_req = None

    def wait_until_live(self, timeout_s: float = 180.0) -> None:
        """Block until the bridge has published a full state file, without consuming it.

        Used as a friendly 'waiting for FCEUX' preflight; leaves the handshake untouched so
        the first real read_state() still sees frame #1.
        """
        expected = config.STATE_HEADER + config.RAM_SIZE
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                if self.state_file.stat().st_size == expected:
                    return
            except (FileNotFoundError, OSError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("FCEUX bridge never came online — run ./emulator/run_fceux.sh")
            time.sleep(0.05)

    # --- read side ----------------------------------------------------------------------
    def read_state(self, timeout_s: float | None = None) -> State:
        """Block until the bridge publishes a frame newer than the last consumed one."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        expected = config.STATE_HEADER + config.RAM_SIZE
        while True:
            try:
                data = self.state_file.read_bytes()
            except (FileNotFoundError, OSError):
                data = b""
            if len(data) == expected:
                req = int.from_bytes(data[0:4], "little")
                if req != self._last_req:
                    frame = int.from_bytes(data[4:8], "little")
                    done = data[8] != 0
                    ram = data[config.STATE_HEADER:]
                    self._last_req = req
                    self._pending_req = req
                    return State(req_id=req, frame=frame, done=done, ram=ram)
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("no new state from FCEUX bridge — is run_fceux.sh running?")
            time.sleep(config.POLL_INTERVAL_S)

    # --- write side ---------------------------------------------------------------------
    def _send(self, command: int, payload: bytes = b"") -> None:
        if self._pending_req is None:
            raise RuntimeError("send() called before read_state(); nothing to acknowledge")
        req = self._pending_req
        body = req.to_bytes(4, "little") + bytes([command]) + payload
        tmp = self.action_file.with_suffix(self.action_file.suffix + ".tmp")

        # Write the temp file with retries (in case the filesystem is under contention)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                tmp.write_bytes(body)
                time.sleep(0.001)  # small pause to ensure disk flush
                os.replace(tmp, self.action_file)  # atomic publish
                self._pending_req = None
                return
            except (FileNotFoundError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.01 * (2 ** attempt))  # exponential backoff: 10ms, 20ms, 40ms, 80ms
                else:
                    raise RuntimeError(f"IPC send failed after {max_retries} retries: {e}") from e

    def send_plan(self, plan: Plan) -> None:
        """Acknowledge the current frame with a button plan for the bridge to execute."""
        self._send(config.CMD_RUN_PLAN, encode_plan(plan))

    def save_state(self, slot: int = 0) -> None:
        """Snapshot the emulator into a slot. Slot 0 marks the level start; 1+ are for
        micro-search checkpoints."""
        self._send(config.CMD_SAVESTATE, bytes([slot]))

    def load_state(self, slot: int = 0) -> None:
        """Restore a slot's snapshot — a one-frame rewind."""
        self._send(config.CMD_LOADSTATE, bytes([slot]))

    def soft_reset(self) -> None:
        self._send(config.CMD_SOFT_RESET)
