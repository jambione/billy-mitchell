# CLAUDE.md — working notes for this repo

Billy Mitchell is an agentic NES game-player that **learns an exact, position-keyed policy** and
carries abstract tactics across games. Read this before changing the learning loop.

## Run / test (always use the venv)
```bash
BILLY_HEADLESS=1 .venv/bin/python -m pytest -q tests/        # tests
# Prove the compounding loop (clears 1-1 every attempt; prints search↓/replay↑ curve):
BILLY_HEADLESS=1 BILLY_REPEAT_LEVEL=1 BILLY_MAX_FRAMES=8000 .venv/bin/python -u run.py --attempts 10 --no-llm
```
- `.venv` has Python 3.14 + `stable-retro` (cp314 wheel) + `requests`/`numpy`. The ROMs and `data/`
  are gitignored. One-time setup: `./emulator/setup_retro.sh`.
- Env knobs: `BILLY_HEADLESS=1` (no window), `BILLY_TURBO=1` (no realtime pacing), `BILLY_MAX_FRAMES`
  (cap an attempt), `BILLY_REPEAT_LEVEL=1` (end each attempt at first clear → repeat the same level),
  `BILLY_RETRO_GAME` (integration id override).

## Architecture (the seam that matters)
Game-agnostic engine talks only to contracts in `billy/abstractions.py`:
`Session` (7 methods) · `Game` (observe/boot/make_reflex) · `Observation` (generic `progress`,
`level_key`) · `ReflexPolicy`. A new console = `systems/<x>/`; a new title = `games/<y>/`.

Decision flow per hazard (in `director.py` `run_attempt`):
**cache-hit → verify on clone → replay** ⚡; else **micro-search on a clone (invisible) → bank** 🔍;
else **LLM**. On death → **learn-from-death** (search a survivor past the death, bank it) 🧠.

## Tiers
1. **Reflex** — `games/common/platformer.py` `PlatformerReflex(PhysicsProfile)`; routine play, no LLM.
2. **SolutionCache** — `knowledge/cache.py`; exact verified sequences keyed `(level_key, x_bucket)`.
   The compounding memory. Persisted tiny (button steps only) to `data/solutions.jsonl`.
3. **Micro-search** — `director.py` `_micro_search`/`_rollout`; runs on `session.clone_state()` under
   `search_mode()` so frames never display (no visible rewind). Candidates = reflex spread + Skills.
4. **LLM (Billy/Coach)** — only when search finds nothing / persona. Off the hot loop.

## Invariants — do not break
- **Exact-replay only.** The cache replays the *exact* button sequence from the same state. Embeddings
  (Skills/KB) may **seed search candidates**, but must **never** drive a blind replay (fuzzy match →
  wrong action → death).
- **Don't cache non-progress.** A "solution" must survive *and* advance (`reach > start + MIN`); a
  plan that merely avoids death is a stall. See `_MIN_PROGRESS_PX`.
- **Rollout `settle` is a POST-candidate budget**, not a total. (The Phase-0 bug: a ~50-frame jump
  consumed a 50-frame total budget, so a death just past the landing was never simulated.) Keep the
  coast-forward loop running after every candidate.
- **Snapshot/learn/replay only ON-GROUND** spots — airborne states don't reproduce across passes.
- **Behaviour-preserving refactors** to the reflex must keep the 1-1 clear + identical compounding
  curve (the regression guard).

## Known limits (honest)
- Within-level compounding is **partial**: static geometry caches+replays; moving enemies re-search
  each pass (position-bucketed replay is timing-sensitive → replay-verify falls back to live-search).
  Billy still clears 1-1 every attempt. Search→0 would need whole-trajectory determinism.
- Cross-game transfer: the **shared reflex** is the primary carry-forward (SMB2-Japan plays with zero
  new reflex code). The **Skill layer** adds candidate diversity that can unblock a hazard (seeded run
  reached further on Lost Levels 1-1), at the cost of more rollouts.

## Deferred (not built yet): VLM/pixel perception, hierarchical strategist, self-improvement/offline
distillation, parallel sessions, dashboards. In-memory cosine is fine — no vector DB needed at this scale.
```
End PR/commit messages with: Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```
