"""Engine configuration for Billy Mitchell (game-agnostic).

Endpoints, model ids, filesystem paths, and the loop tunables shared by every game. The
emulator now runs in-process (stable-retro), so there is no file-IPC protocol here anymore.
Game-specific physics/reflex constants live with each game (e.g. games/smb/tuning.py).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Repo / data paths ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
ROMS_DIR = REPO_ROOT / "roms"
DATA_DIR = REPO_ROOT / "data"              # lessons.jsonl, solutions.jsonl, metrics
LESSONS_FILE = DATA_DIR / "lessons.jsonl"
SOLUTIONS_FILE = DATA_DIR / "solutions.jsonl"   # position-keyed solution cache (the policy)
SKILLS_FILE = DATA_DIR / "skills.jsonl"         # transferable abstract tactics (cross-game)
METRICS_FILE = DATA_DIR / "metrics.jsonl"

RAM_SIZE = 0x800            # NES work RAM 0x0000-0x07FF (first 2KB of env.get_ram())

# --- LM Studio (OpenAI-compatible) ------------------------------------------------------
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
# Strategy/analysis model. Qwen2.5-Coder-7B (MLX, 4-bit) is fast on Apple Silicon; swap to a
# general instruct model via BILLY_CHAT_MODEL for richer persona/strategy if you load one.
CHAT_MODEL = os.environ.get("BILLY_CHAT_MODEL", "qwen2.5-coder-7b-instruct-mlx")
EMBED_MODEL = os.environ.get("BILLY_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
LLM_TIMEOUT_S = float(os.environ.get("BILLY_LLM_TIMEOUT", "120"))

# --- Engine loop tunables (game-agnostic) -----------------------------------------------
# Game-specific reflex/physics constants live with each game (e.g. games/smb/tuning.py).
BILLY_PLAN_MAX_FRAMES = 90  # cap on a single Billy LLM macro before re-observing
KB_TOP_K = 4                # lessons retrieved into Billy's prompt
# Safety cap (~10 game-minutes) per attempt; override with BILLY_MAX_FRAMES for quick benchmarks.
MAX_ATTEMPT_FRAMES = int(os.environ.get("BILLY_MAX_FRAMES", 60 * 60 * 10))

# Continuous playthrough: keep going past each level clear, checkpointing the new level.
# Env-overridable so a "full clear" run can let one attempt blaze through many levels.
MAX_LEVELS_PER_ATTEMPT = int(os.environ.get("BILLY_MAX_LEVELS", 8))   # stop an attempt after this many clears
RESPAWNS_PER_ATTEMPT = int(os.environ.get("BILLY_RESPAWNS", 3))       # retries from checkpoint per attempt

# --- Micro-search + solution cache (the compounding-learning core) ----------------------
MICRO_SEARCH = True
SEARCH_SLOT = 1             # savestate slot reserved for search checkpoints (0 = level start)
SEARCH_HORIZON_FRAMES = 50  # frames to simulate each candidate at a live danger (quick sim)
LEARN_HORIZON_FRAMES = 150  # longer rollout for learn-from-death (must traverse the death zone)
MIN_RUNWAY_PX = 24          # learn-from-death needs at least this much room before a hazard
CACHE_BUCKET_PX = 16        # solution-cache key granularity (one NES tile)
MAX_BUCKET_VISITS = int(os.environ.get("BILLY_MAX_BUCKET_VISITS", 8))  # same hazard N times w/o passing -> give up
# Trust-replay (default): replay cached plans verbatim; a deterministic emulator reproduces them.
# Set BILLY_VERIFY_REPLAY=1 to instead clone-check each replay (re-searches on mismatch — can drift).
VERIFY_REPLAY = os.environ.get("BILLY_VERIFY_REPLAY", "0") == "1"
# Dense brute-force grid at hard walls (thorough but ~40 candidates -> slow). Off by default.
EXPANDED_SEARCH = os.environ.get("BILLY_EXPANDED_SEARCH", "0") == "1"


def ensure_dirs() -> None:
    """Create the data dir if missing. Cheap and idempotent."""
    for d in (DATA_DIR, ROMS_DIR):
        d.mkdir(parents=True, exist_ok=True)
