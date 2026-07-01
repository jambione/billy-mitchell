# Billy Mitchell — Strategic Roadmap

Where Billy is going, and how. This is the **strategic** view (the big build-up); for the immediate
tactical next steps see [NEXT_STEPS.md](NEXT_STEPS.md), and for the architecture see
[CLAUDE.md](CLAUDE.md).

## Where we are

- Clears SMB **1-1 and 1-2** every attempt; **crosses 1-3's tree-top section** via a hazard-scoped RL
  sub-policy; reuses the shared platformer reflex on **SMB2-Japan** with no new code.
- The engine is already game-agnostic ([billy/abstractions.py](billy/abstractions.py):
  `Game`/`System`/`Observation`/`ReflexPolicy`), driven by a cache-first loop: exact-replay → reflex →
  invisible micro-search on clones → learn-from-death → LLM.

## Two priorities driving this roadmap

1. **Exponential velocity** — be *blown away* by how fast Billy improves each attempt, and have him
   play the level **safely and at his best** (top score, "leveling up the character" = power-ups /
   survivability so the level gets *easier*).
2. **Play new games — The Legend of Zelda.**

Made "infinitely better" along the way by generalizing the engine where a real second genre forces
it, and by making each solved thing make the *next* thing faster.

## Guiding principles (every phase)

- **Exact-replay invariant stays:** embeddings / RL / skills only **seed search**, never blind-replay.
- **Regression-guarded:** every phase keeps the 1-1/1-2 clear + identical compounding curve, tests green.
- **Generalize only when forced:** widen a seam when Zelda actually demands it — measured, not speculative.

## Phase 1 — Exponential velocity (priority 1; fastest, lowest-risk, SMB-contained)

**1A. Deterministic whole-trajectory tape — the headline. ✅ SHIPPED (July 2026).**
`billy/knowledge/tape.py` + tape-first replay in the Director. Since July 2026 tapes also
**extend instead of self-corrupting** (replayed chunks re-seed the recording, so an exhausted tape
grows a suffix rather than being replaced by one), **screen-segment tapes chain** (finished on every
screen change, verified by area-advance), and **timed-out-alive attempts persist a partial tape to
the frontier** — the next attempt fast-forwards there with zero search and spends its whole budget
on new ground. Measured: 1-1 re-clears at **tape%=100, search=0, 0.7s wall-clock**.
- Honest caveat: mid-level re-entry after a death still uses the per-`(level,x)` cache.

**1B. Objective-aware play — best score + level up the character.**
Today `_micro_search` optimizes reach (progress) only; score and power state are tracked but never
*pursued*. Add a game-defined **objective value** the search optimizes *after* survival+progress: SMB =
score + power (`size`: small/big/fire) + coins. Make power-up acquisition deliberate (route toward
mushrooms/flowers — perception + grab reflex already exist) so Billy becomes Fire Mario.
- Serves velocity too: Fire/Big Mario survives a hit and one-shots enemies → enemy hazards stop
  forcing re-search. "Play best" and "learn faster" are the same lever.
- `Observation` gains a generic `objective`; `_micro_search` scoring adds an objective tiebreak (after
  survived/progress, before elevation); the cache keeps the higher-objective plan on ties.
  [billy/games/smb/perception.py](billy/games/smb/perception.py) already exposes `size`/`coins`/`powerups`.

**1C. Visible learning curve — "be blown away." ✅ SHIPPED (July 2026).**
The compounding-curve table now shows search / replay / **tape%** / bank / learn / frontier /
reached / **time(s)** per attempt plus sparkline trends ([billy/metrics.py](billy/metrics.py)).

**1D. Demo pipeline ×3 + pull-based teaching. ✅ SHIPPED (July 2026).**
One human demo now produces three artifacts: an exact cache entry, a whole-trajectory tape
(`teleop.py play --tape`), and a BC warm-start for section sub-policies
(`train_section.py --demo x.demo.json` — PPO fine-tunes a policy that already knows the crossing).
And teaching is **pull-based**: when search + pit/frame search + section training ALL miss a hazard,
`stuck_trainer.request_demo` asks for one demo with a ready-to-run teleop command
(`data/demo_requests.jsonl`).

**1E. Parallel micro-search. ✅ SHIPPED (July 2026, opt-in).**
`BILLY_PARALLEL_SEARCH=<n>` spins up n emulator workers ([billy/search_pool.py](billy/search_pool.py))
that evaluate candidates concurrently with the same rollout code; serial stays the default and the
regression baseline.

## Phase 2 — Generalize the engine (the refactor Zelda forces; small, regression-guarded)

**2A. Generic progress / frontier-novelty.** `Observation.progress` is platformer-shaped (Mario x).
Generalize it to a game-defined **frontier** signal: SMB keeps x; Zelda = exploration novelty
(distinct screens/rooms entered) + objective milestones. The engine's stuck/danger/metrics/
learn-from-death key off `progress`+`level_key` unchanged — only the *game* computes them.
`cache.bucket_of` already keys `(level_key, progress, elevation)`, so Zelda supplies room-id as
`level_key` and Link x/y via `progress`/`elevation`. SMB behavior identical (regression guard).

**2B. Skill distillation / transfer. ✅ SHIPPED (July 2026).**
[billy/knowledge/distill.py](billy/knowledge/distill.py): every *significant* banked maneuver
(≥60px real gain) auto-distills into a `sequence` Skill — the proven exact plan + an embedding of
the situation it solved — which seeds micro-search at similar hazards within and across games
(console-gated; always clone-verified before commit, so the exact-replay invariant holds). Demos
distill too (the demo's third artifact). `BILLY_DISTILL=0` to disable.

## Phase 3 — The Legend of Zelda (first target: boot + explore + survive)

**Status (June 2026):** Adapter in [billy/games/zelda/](billy/games/zelda/); details in
[billy/games/zelda/STATUS.md](billy/games/zelda/STATUS.md). Boots, 17 tests green, FAQ walkthrough
wired, NW cave entry works, learning compounded on overworld combat (#72) in earlier runs. **Blocked**
on start-cave wooden sword pickup (Link stalls at cave y≈141; `current_sword` never flips).

**3A. Custom integration.** ✅ Experimental stable-retro `LegendOfZeldaPRG0-Nes`; ROM gitignored;
[emulator/setup_retro.sh](emulator/setup_retro.sh) extended.

**3B. `games/zelda/` adapter.** ✅ perception, reflex, vision, items, walkthrough, curiosity, explore,
hazard_hooks. Remaining: sword pickup macro, dungeon rooms, cave text.

**3C. Prove transfer.** 🔄 Director + cache work without engine changes. Next: FAQ step 1 (sword) →
east to sea (#127) → Level 1 entrance.

## Phase 4 — SNES / Super Mario World (scaffold ✅ July 2026; live pending ROM)

The cross-console proof. `RetroSession` is now console-parameterized (controller module + RAM
size); [billy/systems/snes/](billy/systems/snes/) supplies a 12-button controller whose LOGICAL
names keep the NES bit layout (A=jump→SNES B, B=run→SNES Y, new SPIN=SNES A), so the shared
platformer reflex and every engine tier (cache/tapes/search/teleop/stuck-trainer) drive SMW with
zero new engine code. [billy/games/smw/](billy/games/smw/) has WRAM perception + non-linear level
identity (monotonic events-triggered clear counter). **Blocked on the ROM**: drop it in `roms/`,
run `probe_smw_ram.py`, follow [billy/games/smw/STATUS.md](billy/games/smw/STATUS.md).

## Sequencing & honest scope

Phase 1 first (fastest "wow", low risk, SMB-contained). Phase 2 is the small connective generalization.
Phase 3 is the big lift (a new genre) and benefits from 1+2. **1A (tape)** and **1B (objective play)**
are the headline velocity wins; **Phase 3** is the headline new capability. Each phase is independently
valuable and independently regression-guarded.
