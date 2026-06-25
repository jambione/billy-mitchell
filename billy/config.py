"""Engine configuration for Billy Mitchell (game-agnostic).

Endpoints, model ids, filesystem paths, the file-IPC protocol, and the loop tunables shared
by every game. Game-specific physics/reflex constants live with each game (e.g.
games/smb/tuning.py); controller button bits live with each system (systems/nes/controller.py).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Repo / runtime paths ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# Shared lock-step IPC directory. The Lua bridge reads the same path from $BILLY_RUNTIME
# (see emulator/run_fceux.sh). Keep it on local disk for fast page-cache reads/writes.
RUNTIME_DIR = Path(os.environ.get("BILLY_RUNTIME", str(REPO_ROOT / ".runtime")))
STATE_FILE = RUNTIME_DIR / "state.bin"     # Lua -> Python  (req_id, frame, 2KB RAM)
ACTION_FILE = RUNTIME_DIR / "action.bin"   # Python -> Lua  (req_id echo, command, plan)
ROMS_DIR = REPO_ROOT / "roms"
DATA_DIR = REPO_ROOT / "data"              # lessons.jsonl, metrics, savestates metadata
LESSONS_FILE = DATA_DIR / "lessons.jsonl"
METRICS_FILE = DATA_DIR / "metrics.jsonl"

# --- LM Studio (OpenAI-compatible) ------------------------------------------------------
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
# Strategy/analysis model. Qwen2.5-Coder-7B (MLX, 4-bit) is fast on Apple Silicon; swap to a
# general instruct model via BILLY_CHAT_MODEL for richer persona/strategy if you load one.
CHAT_MODEL = os.environ.get("BILLY_CHAT_MODEL", "qwen2.5-coder-7b-instruct-mlx")
EMBED_MODEL = os.environ.get("BILLY_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
LLM_TIMEOUT_S = float(os.environ.get("BILLY_LLM_TIMEOUT", "120"))

# --- IPC protocol -----------------------------------------------------------------------
RAM_SIZE = 0x800            # NES work RAM 0x0000-0x07FF dumped wholesale by the bridge
STATE_HEADER = 9            # bytes: req_id(4 LE) + frame(4 LE) + done_flag(1)
POLL_INTERVAL_S = 0.0005    # tight poll on the lock-step files

# Action commands (action.bin byte[4]).
CMD_RUN_PLAN = 0            # execute the button plan that follows
CMD_SAVESTATE = 1           # snapshot current state into a slot (byte after cmd = slot)
CMD_LOADSTATE = 2           # restore a slot's snapshot (instant rewind/reset)
CMD_SOFT_RESET = 3          # NES soft reset

# Savestate micro-search: at a risky pit, snapshot, try a few jump variants, and commit the
# one that survives and gets furthest — rewinding the rest. The instant-loadstate
# architecture makes this nearly free, and it's how SMB bots "never miss" a pit.
MICRO_SEARCH = True
SEARCH_SLOT = 1             # savestate slot reserved for search checkpoints (0 = level start)
SEARCH_HORIZON_FRAMES = 54  # frames to simulate each candidate (until landed or dead)

# --- Engine loop tunables (game-agnostic) -----------------------------------------------
# Game-specific reflex/physics constants live with each game (e.g. games/smb/tuning.py).
BILLY_PLAN_MAX_FRAMES = 90  # cap on a single Billy LLM macro before re-observing
DANGER_RADIUS = 80          # consult Billy this many progress-units before a death spot
DANGER_BUCKET = 24          # quantize death positions into zones of this width
KB_TOP_K = 4                # lessons retrieved into Billy's prompt
MAX_ATTEMPT_FRAMES = 60 * 60 * 10  # safety cap (~10 game-minutes) per attempt

# Continuous playthrough: keep going past each level clear, checkpointing the new level.
MAX_LEVELS_PER_ATTEMPT = 8  # stop an attempt after clearing this many levels
RESPAWNS_PER_ATTEMPT = 3    # on death, retry from the current level's checkpoint this many times


def ensure_dirs() -> None:
    """Create runtime/data dirs if missing. Cheap and idempotent."""
    for d in (RUNTIME_DIR, DATA_DIR, ROMS_DIR):
        d.mkdir(parents=True, exist_ok=True)
