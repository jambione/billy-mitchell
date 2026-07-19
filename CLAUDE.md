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
   frontier on a timed-out-alive attempt. A clearing tape may carry an **entry-state anchor**
   (`entry_state` sidecar) restored at level begin — this is what makes a MOVING hazard (1-3's lift,
   phase-set at level load) reproduce: the input stream only replays from that exact state. Anchored
   tapes are never dropped on a verify miss (a miss means the live approach differed, not that the
   tape is wrong). This cleared 1-3 → Billy now marches into World 2.
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

Session-level: `knowledge/routes.py` records every observed transition into a persisted map
(`data/routes.jsonl`; multi-level-skip clears = WORLD WARPs, e.g. 1-2 → 4-1). `strategist.py`
`RouteStrategist` DECIDES with it — plans the warp-preferring path to the furthest-known level,
names the next objective (logged on entry, fed to the LLM). Warp preference comes from a
per-game progress `rank` (SMB world/stage ordinal is the default; other games can supply
`Game.route_rank` — else it degrades to frontier exploration). The furthest level-start
checkpoint is saved to `data/checkpoints/<game>/` — `run.py --resume` continues the march there
next session. Zelda's `progress` includes monotonic per-screen COMBAT credit (kills + room-clear)
so fight demos/search bank on combat-walled screens.

## Invariants — do not break
- **Exact-replay only.** The cache replays the *exact* button sequence from the same state. Embeddings
  (Skills/KB) may **seed search candidates**, but must **never** drive a blind replay (fuzzy match →
  wrong action → death).
- **Don't cache non-progress.** A "solution" must survive *and* advance (`reach > start + MIN`); a
  plan that merely avoids death is a stall. See `_MIN_PROGRESS_PX`.
- **Rollout `settle` is a POST-candidate budget**, not a total. (The Phase-0 bug: a ~50-frame jump
  consumed a 50-frame total budget, so a death just past the landing was never simulated.) Keep the
  coast-forward loop running after every candidate.
- **A plan is verified only UP TO a transition.** Rollouts chunk-step and stop scoring at a
  level/area advance (mid-plan death must be SEEN, not masked by flag decay through the reload);
  `_commit` stops replaying at a level_key change for the same reason. (The 1-3 x=283 loop: a
  transition-bonused exit-pipe plan replayed its unverified tail straight into the next level's
  first pit, inside one commit — no 🏁, checkpoint stuck, learn-from-death starved.)
- **Snapshot/learn/replay only ON-GROUND** spots — airborne states don't reproduce across passes.
- **Behaviour-preserving refactors** to the reflex must keep the 1-1 clear + identical compounding
  curve (the regression guard).

## Known limits (honest)
- Within-level compounding is **partial** for POSITION-KEYED cache entries: static geometry
  caches+replays; moving enemies re-search each pass (position-bucketed replay is timing-sensitive →
  replay-verify falls back to live-search). The **whole-level tape** (esp. entry-state-anchored) is the
  answer where it exists — it reproduces the whole trajectory incl. moving lifts deterministically
  (this is how 1-3 clears). Search→0 within a level needs that whole-trajectory determinism.
- Cross-game transfer: the **shared reflex** is the primary carry-forward (SMB2-Japan plays with zero
  new reflex code). The **Skill layer** adds candidate diversity that can unblock a hazard (seeded run
  reached further on Lost Levels 1-1), at the cost of more rollouts.

## Deferred (not built yet): VLM/pixel perception, hierarchical strategist, self-improvement/offline
distillation, parallel sessions, dashboards. In-memory cosine is fine — no vector DB needed at this scale.
```
End PR/commit messages with: Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

## Agent skills

### Issue tracker

GitHub Issues on `jambione/billy-mitchell` via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default role labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: root `CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.
