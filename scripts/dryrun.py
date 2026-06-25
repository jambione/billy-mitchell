#!/usr/bin/env python3
"""End-to-end dry run of the Billy brain against a FAKE emulator.

The FakeEmu thread speaks the exact same binary file protocol as emulator/billy_bridge.lua
(state.bin / action.bin, req_id handshake, command + button-plan decoding), so this exercises
the real ipc.Bridge, Director, Executor, Commentator, and metrics — proving the loop and the
wire format without needing FCEUX. The toy world has an enemy, a wall (forces a STUCK ->
decision), and a finish line (level clear).

    python scripts/dryrun.py            # pure reflex (no LLM)
    python scripts/dryrun.py --llm      # also drive Billy/Coach if LM Studio is up
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

# Point the whole system at a throwaway runtime dir BEFORE importing billy.config.
_RT = tempfile.mkdtemp(prefix="billy-dry-")
os.environ["BILLY_RUNTIME"] = _RT
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy import config  # noqa: E402
from billy.director import Director  # noqa: E402
from billy.games.smb import SmbGame  # noqa: E402
from billy.knowledge import KnowledgeBase  # noqa: E402
from billy.systems.nes import controller as nes  # noqa: E402

PIT_T0, PIT_T1 = 14, 15   # floor tiles (page 0) carved into a pit -> world x 224..255
PIT_X0, PIT_X1 = 224, 255
ENEMY_X = 120     # an enemy before the pit
WALL_X = 400      # a wall after the pit: blocks unless Mario jumps
CLEAR_X = 600     # crossing this finishes the level


class FakeEmu(threading.Thread):
    """A minimal SMB stand-in that talks the bridge protocol over the real files."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.state_file = config.STATE_FILE
        self.action_file = config.ACTION_FILE
        self._stop = threading.Event()
        self.x, self.stage, self.frame, self.air, self.dead = 40, 0, 0, 0, False
        self._snaps: dict[int, tuple] = {}

    def stop(self) -> None:
        self._stop.set()

    # --- world model ---------------------------------------------------------------------
    def _build_ram(self) -> bytes:
        ram = bytearray(0x800)
        ram[0x6D], ram[0x86] = self.x // 256, self.x % 256
        ram[0x03B8] = 100                      # mario_y = 116
        ram[0x075C] = self.stage
        ram[0x075A] = 2                        # lives
        ram[0x07F8], ram[0x07F9], ram[0x07FA] = 3, 0, 0  # time 300
        ram[0x001D] = 0 if self.air == 0 else 1          # float state (0 = on ground)
        ram[0x00B5] = 2 if self.dead else 1              # y-viewport (>1 => fell/dead)
        # Solid floor across both tile pages, then carve a pit at tiles 14-15 (page 0).
        for page in (0, 1):
            for sub_x in range(16):
                ram[0x500 + page * 208 + 6 * 16 + sub_x] = 1
        ram[0x500 + 6 * 16 + PIT_T0] = 0
        ram[0x500 + 6 * 16 + PIT_T1] = 0
        if self.stage == 0 and abs(self.x - ENEMY_X) < 80:
            ram[0x0F] = 1
            ram[0x6E], ram[0x87] = ENEMY_X // 256, ENEMY_X % 256
            ram[0xCF] = 100
        return bytes(ram)

    def _apply_plan(self, data: bytes) -> None:
        nsteps, pos = data[5], 6
        for _ in range(nsteps):
            dur = data[pos] + data[pos + 1] * 256
            mask = data[pos + 2]
            pos += 3
            right = mask & nes.RIGHT
            sprint = mask & nes.B
            jump = mask & nes.A
            if jump and self.air == 0 and not self.dead:
                self.air = dur + 14            # leave the ground; arc lasts ~A-hold + tail
            for _ in range(dur):
                if self.dead:
                    break
                self.frame += 1
                if self.air > 0:
                    self.air -= 1
                if right:
                    nx = self.x + (2 if sprint else 1)
                    if self.x <= WALL_X < nx and self.air == 0 and self.stage == 0:
                        nx = WALL_X            # wall blocks unless airborne
                    self.x = nx
                # Pit: standing in it with no air left => you fall.
                if self.air == 0 and PIT_X0 <= self.x <= PIT_X1 and self.stage == 0:
                    self.dead = True
            if self.x >= CLEAR_X:
                self.stage += 1                # finished a level -> loop into the "next" one
                self.x, self.air, self.dead = 40, 0, False

    # --- protocol (mirrors billy_bridge.lua) --------------------------------------------
    def _write_state(self, req: int) -> None:
        body = (req.to_bytes(4, "little") + self.frame.to_bytes(4, "little") +
                b"\x00" + self._build_ram())
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_bytes(body)
        os.replace(tmp, self.state_file)

    def _wait_action(self, req: int) -> bytes | None:
        while not self._stop.is_set():
            try:
                data = self.action_file.read_bytes()
            except (FileNotFoundError, OSError):
                data = b""
            if len(data) >= 5 and int.from_bytes(data[0:4], "little") == req:
                return data
            self._write_state(req)   # republish (mirrors bridge.lua) so a late reset can't deadlock
            time.sleep(0.0005)
        return None

    def run(self) -> None:
        try:
            self._run()
        except Exception:  # surface thread crashes instead of silently hanging the Director
            import traceback
            print("[FakeEmu] CRASHED:")
            traceback.print_exc()
            self._stop.set()

    def _run(self) -> None:
        config.ensure_dirs()
        req = 0
        while not self._stop.is_set():
            req += 1
            self._write_state(req)
            data = self._wait_action(req)
            if data is None:
                return
            cmd = data[4]
            slot = data[5] if len(data) > 5 else 0
            if cmd == config.CMD_SAVESTATE:
                self._snaps[slot] = (self.x, self.stage, self.frame, self.air, self.dead)
            elif cmd == config.CMD_LOADSTATE and slot in self._snaps:
                self.x, self.stage, self.frame, self.air, self.dead = self._snaps[slot]
            elif cmd == config.CMD_SOFT_RESET:
                self.x, self.stage, self.frame, self.air, self.dead = 40, 0, 0, 0, False
            else:
                self._apply_plan(data)


def main() -> int:
    use_llm = "--llm" in sys.argv

    # Watchdog: if the loop deadlocks, bail loudly instead of hanging forever.
    def _watchdog() -> None:
        time.sleep(float(os.environ.get("DRYRUN_WATCHDOG", "30")))
        print("[dryrun] WATCHDOG timeout — the loop is stuck; aborting.")
        os._exit(2)
    threading.Thread(target=_watchdog, daemon=True).start()

    # The FakeEmu republishes state each poll, so run_session's reset() can't deadlock it.
    emu = FakeEmu()
    emu.start()
    director = Director(SmbGame(), KnowledgeBase(config.DATA_DIR / "_dry_lessons.jsonl"),
                        use_llm=use_llm)
    results = director.run_session(attempts=1)
    emu.stop()

    total = sum(r.levels_cleared for r in results)
    ok = total >= 2
    print(f"\n[dryrun] {'PASS' if ok else 'FAIL'} — cleared {total} toy level(s) "
          f"continuously (gap jump + checkpoint + level advance).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
