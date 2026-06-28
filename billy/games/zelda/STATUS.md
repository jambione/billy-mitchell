# Billy Mitchell — Zelda Status & Demo Guide

Last updated: **June 28, 2026**. FAQ route: `walkthrough/NES/zelda` (Dan Simpson v1.9).

This document is the handoff snapshot: **what Billy can do today**, **what compounding looks like
in practice**, and **what remains** before FAQ screen #127 and Level 1.

---

## What Billy is (30-second pitch)

Billy is not a frame-by-frame LLM bot. He plays NES games through a **four-tier stack**:

| Tier | Role | Zelda today |
|------|------|-------------|
| **Reflex** | Routine movement/combat every frame | Top-down Zelda reflex + ROM-verified cave macro |
| **SolutionCache** | Exact, position-keyed verified sequences — **discover once, replay forever** | 6 banked overworld solutions (#120–#121) |
| **Micro-search** | Invisible rollouts on cloned emulator state | East-march combat, lip walks, cave drift recovery |
| **Learn-from-death** | After a death, search backward for a survivor *past* the death spot and bank it | Active on #121; blocked at #124 until knockback fix is live-tested |
| **LLM** (optional) | Persona + stuck improvisation | Rolling memory wired; off hot loop with `--no-llm` |

The compounding curve is the product demo: **search↓ replay↑ frontier↑** across attempts.

**Architecture seam (do not break):** game logic stays in `billy/games/zelda/`; never modify
`billy/director.py` or `billy/knowledge/cache.py`.

---

## What Billy can do right now (proven)

### ✅ Start cave → wooden sword (P0 — done)

- ROM-verified macro in `start_cave.py`: approach → enter → text → climb → pickup → exit.
- Reflex commits the **full contiguous interior plan** in one Decision (not one step per tick).
- Live Director (`--attempts 5`, cache persisted): **search 0 / replay 328** per attempt,
  `sword_level=1`, reaches overworld **#120**, FAQ phase `east_to_sea`.

**Demo command:**
```bash
BILLY_HEADLESS=1 BILLY_MAX_FRAMES=8000 .venv/bin/python run.py --game zelda --attempts 3 --no-llm
# Expect: cave clears, sword=1, screen → #120, heavy replay counts
```

### ✅ East march compounding (#119 → #121)

Billy marches FAQ row 8 east with ROM-tuned screen hops (`east_march.py`):

- **70f hop** when 192 ≤ x < 200; **48f edge** at true lip (x ≥ 220).
- Alternating RIGHT / RIGHT+B cross macros clear combat screens.
- Lip-walk plans on deep screens (#123+) to avoid multi-screen overshoot.
- `hazard_hooks.try_frame_search()` rolls combat candidates on clones before banking.

**Observed compounding (June 28, 2026 — post replay-fix):**

| Attempt | search | replay | reached_x | furthest screen | notes |
|---------|--------|--------|-----------|-----------------|-------|
| 1 (5-attempt cave run) | 0 | 328 | 2440 | #120 | Cave + early east cache |
| 8 (`--fresh`, pre-fix) | 140 | **0** | 2857 | #121 | All cache treated stale — no replays |
| 3 (post-fix, cache kept) | 56 | **20** | **2921** | #121 | Replays working; frontier +64px |

Solution cache entry `overworld #121` bucket 177 has **60 replay hits** — evidence the cache is
actually driving play, not re-searching every step.

### ✅ Unit test coverage

**56 tests** across `test_zelda.py`, `test_zelda_march_dungeon.py`, `test_zelda_fixtures.py`
(plus `test_start_cave.py`, `test_rolling_memory.py` in broader suite).

Covers: cave macro, east march stale rules, monotonic progress, dungeon combat scaffold, fixtures.

### ✅ B1 fixtures scaffold

| Asset | Path | Purpose |
|-------|------|---------|
| RAM map | `ram_map.py` | PRG0 address source of truth |
| Retro data | `data/zelda/retro_data.json` | Integration UI sync target |
| State capture | `capture_zelda_state.py` | Named savestates + manifest |
| Path probe | `probe_zelda_path.py` | JSONL recorder + plan verifier |
| Named states | `data/zelda/states/manifest.json` | Checkpoint catalog |

Captured milestones: `cave_after_enter`, `east_row8_screen_124`, `east_row8_sword`.

### ✅ B3 LLM rolling memory

`billy/agents/rolling_memory.py` wired into Director → Billy/Coach prompts. Advisory only;
never touches SolutionCache.

### 🔶 Dungeon adapter (P2 — scaffolded, not live-proven)

- `dungeon.py`, `dungeon_nav.py` — room decode, greedy explore, key-door reflex, combat hook.
- Wired in `reflex.py` / `hazard_hooks.py` but **not exercised** in a live Director run yet.

---

## Milestone map (FAQ east-to-sea)

Row 8 march: screen **119** (start) → **127** (sea).

| Screen | First life | Retry lives | Status |
|--------|------------|-------------|--------|
| #119 cave | Reliable (macro) | Reliable (cache replay) | ✅ Done |
| #120 | Reliable | Reliable (cross macro + cache) | ✅ Done |
| #121 | Reaches, dies mid-screen | Reaches, dies @2921 | 🔶 **Current wall** |
| #122 | — | — | ❌ Not reached (post-fix runs) |
| #123 | Dies ~3819 (historical) | Reached via replay (historical) | 🔶 Regressing — need to re-march |
| #124 | — | Reached (historical `progress≈3812`) | 🔶 Blocked on #121; learn-from-death fix ready |
| #127 `SEA_EAST_SCREEN` | — | — | ❌ Target not reached |

**Best historical retry chain:** `#119 → #120 → #121 → #122 → #123 → #124`, then died at
`#124@3783` every time (knockback shrank progress → zero learn-from-death runway).

---

## Recent fixes (June 28, 2026)

### 1. Monotonic progress (`game.py`)

Combat knockback drops `link_x`, which shrank `objective_score()` and made death progress **less
than** entry snapshots → negative learn-from-death runway on #124.

**Fix:** on the same `level_key`, `observe()` keeps `progress = max(raw, frontier)`; resets on
screen/realm change. Does not touch Director or cache.

### 2. Replay fix (`hazard_hooks.py`)

`stale_cache()` staled **all** plans < 56 frames on every east-march screen — intended to stop
#119↔#120 lip ping-pong, but it also killed short learn-from-death combat survivors on #121.

**Fix:** only stale short **transition** hops (`is_east_march_plan`); allow short **combat** plans
on #121+; keep ping-pong guard on #119–#120 only.

**Before → after:** `replay=0` → `replay=20`, `search=140` → `search=56`, `reached_x=2857` → `2921`.

---

## What's left (priority order)

### P1 — Break through #121 → #127 (immediate)

1. **#121 combat death @2921** — banked replays work, but survivor does not clear the death zone
   or scroll to #122. Tune learn-from-death horizon / combat candidates on row 8; ensure learned
   pass at ~2745 replays and advances past 2921.
2. **Re-march to #123–#124** — prior session reached #124 on retry lives; current cache only
   compounds through #121. Need longer run **without `--fresh`** to rebuild deep-screen cache.
3. **Validate #124 learn-from-death** — monotonic progress should unlock
   `learned to pass overworld #124@...` in Director log; then lip-walk chains #125–#127.
4. **Recapture healthy savestates** — `east_row8_screen_124.state` was snapped at 0 hearts
   (`in_play=False`); re-capture at full health for fixture/bootstrap use. Snap #123 too.
5. **#119↔#120 ping-pong guard** — still possible if short edge cache poisons; narrowed stale
   rules help but watch for regressions.

**Success metrics:**
- Director log: `learned to pass overworld #124@...`
- Screen log: `#125`, `#126`, `#127`
- `visited` includes `SEA_EAST_SCREEN` (127) → phase switches to `pre_level1_hearts`

### P2 — FAQ after sea (#127)

- Bomb shop (#111), heart containers, Level 1 approach (screen #55).
- **Dungeon adapter live exercise** — keys, doors, boss; hazard-scoped RL optional.
- Cave text OCR or RAM text buffer for old-man hints.

### B1 — Integration UI (ongoing)

- Refine `data.json` + RAM map via stable-retro Integration UI.
- Named states per dungeon room for replay verification and RL fixtures.

### B2 / B3 / B4 (plan Part B — not started on Zelda)

- External benchmark harness (LMGame Bench) — optional scoreboard.
- Game Boy console adapter — optional cross-console demo.
- VLM perception assist — only when RAM blocks dungeon progress.

---

## Plan vs delivery (handoff checklist)

| Item | Plan ref | Status |
|------|----------|--------|
| **Part A** — cave macro + compounding | P0 | ✅ Done |
| **B3** — LLM rolling memory | B3 | ✅ Done |
| **P1** — East march to #127 | P1 | 🔶 Partial — #121 wall, #124 not re-validated |
| **B1** — Integration UI / fixtures | B1 | 🔶 Partial — scaffold + 3 named states |
| **P2** — Dungeon adapter | P2 | 🔶 Scaffolded, not live-proven |

---

## How to demo Billy (show the learning curve)

### Quick visual (windowed, 3 attempts)
```bash
cd repo/billy-mitchell
.venv/bin/python run.py --game zelda --attempts 3 --no-llm
# Watch: cave macro → east march → cache replays on repeat passes
```

### Benchmark (headless, compounding metrics)
```bash
BILLY_HEADLESS=1 BILLY_MAX_FRAMES=400000 .venv/bin/python run.py --game zelda --attempts 8 --no-llm
# Do NOT pass --fresh if you want cache to compound across attempts
# Read the "Compounding curve" table at the end: search↓ replay↑ frontier↑
```

### Capture east-march milestones
```bash
BILLY_HEADLESS=1 .venv/bin/python capture_zelda_state.py drive-east --attempts 3
.venv/bin/python capture_zelda_state.py list
```

### Unit tests (no ROM for most)
```bash
.venv/bin/python -m pytest tests/test_zelda.py tests/test_zelda_march_dungeon.py \
    tests/test_zelda_fixtures.py tests/test_start_cave.py -q
```

### What to point at in the log

| Log line | Meaning |
|----------|---------|
| `🔍 solved (reach N) — remembered` | Micro-search found a new survivor; banked in cache |
| `replay …` (in attempt summary `replay=N`) | Cache replay fired N times — **compounding working** |
| `🧠 learned to pass overworld #X@Y` | Learn-from-death banked a past-death survivor |
| `🗺️ screen → overworld #N` | Screen transition — frontier advancing |
| `search=S replay=R` in attempt footer | S should drop, R should rise across attempts |

---

## File map

| File | Role |
|------|------|
| `game.py` | Game adapter; **monotonic progress** in `observe()` |
| `perception.py` | RAM → Scene; `objective_score()` uses `max_hearts` not current health |
| `walkthrough.py` | FAQ route phases & screen grid |
| `reflex.py` | Live policy + cave macro + east march + dungeon branch |
| `start_cave.py` | ROM-verified wooden sword macro |
| `east_march.py` | FAQ row-8 hops, lip-walk, entry guard, cross reps |
| `hazard_hooks.py` | Combat zones, stale_cache, learn-from-death hooks |
| `dungeon_nav.py` | Dungeon explore + key doors + combat |
| `fixtures.py` | Named savestate loader for tests / capture |
| `capture_zelda_state.py` | CLI for milestone snapshots |
| `data/zelda/states/` | Named `.state` files + `manifest.json` |
| `data/solutions.jsonl` | Persisted SolutionCache (the policy) |

---

## SMB (for context — Billy's other proven title)

Billy's original demo game. Hazard-scoped RL (1-3 lift sub-policy), full compounding through
world 1-3, stuck trainer, section policies. Zelda reuses the **same Director loop** unchanged.

---

*Next engineering session: push #121 past 2921 → #122, rebuild cache to #124, confirm monotonic
progress unlocks #124 learn-from-death, capture healthy #123/#124 savestates.*