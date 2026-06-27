# Zelda adapter — status & path forward

Last updated: June 2026. FAQ source: `walkthrough/NES/zelda` (Dan Simpson v1.9).

## What we accomplished

### Engine integration (Phase 3C — in progress)

- **`ZeldaGame`** registered in `run.py`; boots via stable-retro experimental integration
  `LegendOfZeldaPRG0-Nes`.
- **Same Director loop** as SMB: cache-first replay → reflex → invisible micro-search →
  learn-from-death. No engine forks required.
- **17 unit tests** in `tests/test_zelda.py` (ROM-gated boot test skips without ROM).

### Perception (`perception.py`, `items.py`, `vision.py`)

- RAM decoder for Link position, hearts, rupees, screen id, game mode, enemies.
- **Phantom enemy filter** — slots with `enemy_type == 0` ignored (fixes false combat lock).
- **Ground items** — drop slots @173–178 decoded separately from enemies; ASCII map uses `I`.
- **NW cave vision** — scans left half of frame for black-square mouths (not center/top).
- **Cave interiors** — game mode 11 treated as in-play so progress/learning continue indoors.
- **`enemy_ahead()`** shim for shared commentary (fixes Director crash on Zelda runs).

### Walkthrough-driven routing (`walkthrough.py`, `curiosity.py`, `explore.py`)

FAQ first-quest phases encoded as screen IDs (origin grid 8,8 = screen 119):

| Phase | FAQ step | Target |
|-------|----------|--------|
| `wooden_sword` | Get wooden sword in NW cave | Screen 119, stay on map |
| `east_to_sea` | Head right 8 screens | Screen 127 |
| `level_1_approach` | Right, up 4, left | Screen 55 (Eagle entrance) |

- Start cave targets **NW mouth** ~(60, 76), not the north screen edge (#103).
- After sword: march **east** along row 8 per FAQ (not north through #103 first).

### Reflex (`reflex.py`, `hazard_hooks.py`)

- Top-down movement, sword combat, screen-edge transitions.
- **Start-cave state machine** — approach NW → enter → dismiss text (`A`) → walk toward drops.
- Item pickup priority over exploration when no enemies nearby.
- Cave zones exempt from stall-breaker; expanded micro-search candidates for cave (`A`, `UP`, diagonals).

### Learning (observed behavior)

- **Earlier overworld wandering** (before strict FAQ gating): cache compounded — attempt 2 hit
  **0 searches / 5 replays**, frontier advanced to screen **#72**.
- **Current bottleneck run**: Director timeouts on screen **#119** looping in cave interior;
  `sword_level` never increments → FAQ `east_to_sea` phase never starts.
- Cache entries exist for screen-119 cave wander (~118 solutions) but do not solve sword pickup.

## Current blocker (ROM-validated)

Billy reliably:

1. Approaches NW cave on screen 119
2. Enters cave interior (mode 11)
3. Dismisses old-man text (drop type 1 → 2 after `A`)
4. Climbs from (112, 213) to **(112, 141)**

He **cannot**:

- Move above **y ≈ 141** inside the cave (hard ceiling row)
- Reach displayed drops at **y = 128** (120, 72, 168)
- Increment `current_sword` @ RAM 1623 in brute-force probing

Until sword acquisition works (or an alternate success signal is confirmed), Billy cannot
complete FAQ step 1 or march east to the sea.

## Path forward (priority order)

### P0 — Solve start-cave sword (unblocks everything)

1. **ROM probe** — confirm stable-retro cave interior matches retail (old-man room vs wrong template).
   Check `start.state` and whether `current_sword` is the correct field.
2. **Scripted macro** — bank a verified frame-perfect sequence in `start_cave.py` once found:
   enter → text dismiss → climb path above y=141 (if one exists).
3. **Micro-search in cave** — Director already treats cave as special zone; ensure search banks
   any plan that increases `objective_score` or flips sword RAM.
4. **Alternate success signal** — if sword RAM never moves, detect pickup via `screen_item`,
   `dungeon_item`, or post-cave `B` button slash; relax `requires_start_cave_inspection` gate.
5. **Pragmatic timeout** — after N frames in cave without sword, log + allow FAQ `east_to_sea`
   (degraded path; Billy fights without sword until revisit).

### P1 — Item loop & learning compounding

- Finish pickup collision (walk RIGHT along y=141 row, then UP if path opens).
- Bank cave macro in `data/solutions.jsonl` → attempt N+1 replays for free.
- Re-enable overworld exploration learning (combat on #72+ was working).

### P2 — FAQ milestones after sword

- East to sea (#127) → bomb shop (#111) → hearts/Level 1 prep per walkthrough §1.
- Level 1 dungeon adapter (room keys, keys/bombs, dungeon modes).
- Cave text OCR or RAM text buffer for old-man hints.

### P3 — Prove cross-genre transfer

- Run without `--fresh` for compounding: `BILLY_HEADLESS=1 .venv/bin/python run.py --game zelda --attempts 10 --no-llm`
- Target metric: search↓ replay↑, reach screen 127, then Level 1 entrance (#55 / stairs #6).

## Run commands

```bash
# Unit tests (no ROM required except boot test)
.venv/bin/python -m pytest tests/test_zelda.py -q

# Play (persist cache — omit --fresh)
BILLY_HEADLESS=1 .venv/bin/python run.py --game zelda --attempts 3 --no-llm

# Quick benchmark (caps frames)
BILLY_HEADLESS=1 BILLY_MAX_FRAMES=5000 .venv/bin/python run.py --game zelda --attempts 3 --no-llm
```

## File map

| File | Role |
|------|------|
| `game.py` | Game adapter, Observation binding |
| `perception.py` | RAM → Scene |
| `items.py` | Ground drop decoding |
| `vision.py` | NW cave mouth RGB detection |
| `walkthrough.py` | FAQ route phases & screen grid |
| `curiosity.py` | Cave approach & inspection gates |
| `explore.py` | Direction scoring with FAQ priority |
| `reflex.py` | Live policy + cave interior FSM |
| `hazard_hooks.py` | Combat/cave zones for Director |
| `tuning.py` | Constants (START_SCREEN=119, etc.) |