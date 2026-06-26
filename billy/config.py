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
# Route-awareness (Phase 1): a cache/route node is keyed on (level, x_band, y_band) — the y band
# disambiguates a high road from a low road at the SAME x (the high ledge vs the main path), which a
# 1-D x-only key conflated. Coarse so one flat platform stays a single node.
CACHE_Y_BAND_PX = 24
# When micro-search scores candidates, penalise one that lands on a node already proven to dead-end
# (stall-breaker fired there), and add a small GRAVITY tiebreak preferring the lower/grounded path
# when x-progress is comparable — so Billy stops greedily climbing into dead-ends / past low exits.
# (SMB elevation = mario_y, where LARGER = lower on screen, so a positive weight prefers staying low.)
ROUTE_DEAD_PENALTY = 50_000
# Phase 2a — backward dead-end propagation: when a node dead-ends, also mark+drop the last N nodes of
# the APPROACH that led there, so next pass the search avoids that branch (e.g. the high road) and the
# re-search routes the other way. The 2-D key means marking the high-road nodes dead leaves the low
# road (same x, different y band) open. Keep small — marking too much of the approach dead nukes
# legit path nodes and regresses the route.
DEADEND_BACKTRACK = 4
# Gravity tiebreak (prefer the lower road on comparable x): a blanket version cost 1-1 score (lower
# paths = fewer coins) without cracking 1-2's exit, so it's OFF by default. The real route-finding is
# dead-end memory + Phase-2 backtracking, not a blanket bias. Kept as a tunable knob.
ELEVATION_TIEBREAK = 0.0
MAX_BUCKET_VISITS = int(os.environ.get("BILLY_MAX_BUCKET_VISITS", 8))  # same hazard N times w/o passing -> give up
# Trust-replay (default): replay cached plans verbatim; a deterministic emulator reproduces them.
# Set BILLY_VERIFY_REPLAY=1 to instead clone-check each replay (re-searches on mismatch — can drift).
VERIFY_REPLAY = os.environ.get("BILLY_VERIFY_REPLAY", "0") == "1"
# Dense brute-force grid at hard walls the focused spread can't crack (e.g. 1-2's low-ceiling +
# enemy ledge at x=908). Fires ONLY as a fallback when the focused micro-search fails to progress,
# and stops at the first surviving+advancing candidate (early-exit), so its ~40-candidate cost is
# paid only at genuine walls and only until one works — then it's cached forever. On by default;
# set BILLY_EXPANDED_FALLBACK=0 to disable. (BILLY_EXPANDED_SEARCH kept as a back-compat alias.)
EXPANDED_FALLBACK = (os.environ.get("BILLY_EXPANDED_FALLBACK",
                                    os.environ.get("BILLY_EXPANDED_SEARCH", "1")) == "1")
EXPANDED_SEARCH = EXPANDED_FALLBACK   # back-compat alias


def ensure_dirs() -> None:
    """Create the data dir if missing. Cheap and idempotent."""
    for d in (DATA_DIR, ROMS_DIR):
        d.mkdir(parents=True, exist_ok=True)
