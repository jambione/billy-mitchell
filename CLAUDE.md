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
The retro transport (`systems/nes/retro_session.py`) is console-parameterized (controller module +
RAM size); `systems/snes/` + `games/smw/` ride it with LOGICAL buttons (A=jump→SNES B, B=run→SNES Y
via `RETRO_NAMES`) so the shared reflex carries across consoles. SMW is scaffold-only until a ROM
lands — bring-up checklist in `billy/games/smw/STATUS.md`.

Decision flow per hazard (in `director.py` `run_attempt`):
**cache-hit → verify on clone → replay** ⚡; else **micro-search on a clone (invisible) → bank** 🔍;
else **LLM**. On death → **learn-from-death** (search a survivor past the death, bank it) 🧠.

## Tiers
1. **Reflex** — `games/common/platformer.py` `PlatformerReflex(PhysicsProfile)`; routine play, no LLM.
2. **Tapes** — `knowledge/tape.py`; whole-trajectory input streams per level/screen, tape-first replay
   after a clone-verify (`tape%` column). Tapes EXTEND when exhausted (replayed chunks re-seed the
   recording — never replace a tape with a suffix), chain across screens, and persist partials to the
   frontier on a timed-out-alive attempt.
3. **SolutionCache** — `knowledge/cache.py`; exact verified sequences keyed `(level_key, x_bucket)`.
   The compounding memory. Persisted tiny (button steps only) to `data/solutions.jsonl`.
   **Reachback**: on a miss (or weak hit), a HIGH-reach entry a few buckets behind is clone-verified
   from the live state before replay — demos bind to their exact state, so verify decides honestly.
4. **Micro-search** — `director.py` `_micro_search`/`rollout_candidate`; runs on
   `session.clone_state()` under `search_mode()` so frames never display (no visible rewind).
   Candidates = reflex spread + Skills, incl. distilled `sequence` skills (`knowledge/distill.py` —
   significant banks auto-become transferable, console-gated, search-seeded only).
   `BILLY_PARALLEL_SEARCH=N` fans candidates out to N emulator workers (`search_pool.py`).
5. **Human demos** — PULL: when search + stuck-training all miss, `stuck_trainer.request_demo`
   files a teleop command (`data/demo_requests.jsonl`). PUSH: **T in the watch window** = live
   takeover (`director._human_takeover`) — the segment banks from Billy's own live state (the
   durable carrier; mid-level savestate demos often fail verify from shifted approaches).
   One demo = cache entry + tape (`teleop.py --tape`) + distilled skill + BC warm-start
   (`train_section.py --demo`, `billy/rl/bc.py`).
6. **Walkthrough guide** — `knowledge/guide.py`; a text FAQ at `walkthrough/<SYSTEM>/<game>` is
   ingested once (LLM read merged with heuristic parse → `data/guides/`). Seeds search candidates
   + LLM prompt context. Advice only, never authority.
7. **LLM (Billy/Coach)** — only when search finds nothing / persona. Off the hot loop.

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
