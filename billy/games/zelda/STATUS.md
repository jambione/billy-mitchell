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

### Reflex (`reflex.py`, `hazard_hooks.py`, `start_cave.py`)

- Top-down movement, sword combat, screen-edge transitions.
- **ROM-verified start-cave macro** (`start_cave.py`) — phase playback: text → climb →
  pickup (RIGHT → DOWN → DOWN → UP+LEFT → LEFT → A) → exit (long DOWN).
- Reflex emits one macro step per tick; cave interior is not a special zone during macro
  (Director uses reflex plan directly). Macro drift falls back to micro-search via
  `macro_candidates()` in `hazard_hooks.extra_candidates`.
- **Cave timeout** — after 900 frames without sword, FAQ `east_to_sea` fallback (degraded).
- Item pickup priority over exploration when no enemies nearby.
- Cave zones exempt from stall-breaker; expanded micro-search candidates for cave.

### Learning (observed behavior)

- **Earlier overworld wandering** (before strict FAQ gating): cache compounded — attempt 2 hit
  **0 searches / 5 replays**, frontier advanced to screen **#72**.
- **Current bottleneck run**: Director timeouts on screen **#119** looping in cave interior;
  `sword_level` never increments → FAQ `east_to_sea` phase never starts.
- Cache entries exist for screen-119 cave wander (~118 solutions) but do not solve sword pickup.

## P0 solved — start-cave wooden sword (June 2026)

Brute-force ROM probing found a reproducible sequence (banked in `start_cave.py`):

1. **Approach:** LEFT 35, UP 25, LEFT 15, UP 15
2. **Enter:** UP+LEFT 50, LEFT 40, UP+LEFT 40
3. **Text:** A × 35 (12 frames each)
4. **Climb:** UP × 18 (4 frames each) → link ≈ (112, 141)
5. **Sword pickup:** RIGHT 4 → DOWN 8 → DOWN 16 → UP+LEFT 12 → LEFT 12 → A 8 → `sword_level = 1`
6. **Exit:** DOWN 120 → overworld mode 5

The old item-walk reflex could not reach drops at y=128 (ceiling at y≈141). The pickup trick
ducks under the lip with RIGHT/DOWN then UP+LEFT into the old man.

**Wired:** reflex phase macro, `macro_candidates()` for Director search, 900-frame cave timeout
→ FAQ `east_to_sea` fallback. Director treats `stale_cache` as a cache miss (no search storms).
Pre-sword / in-cave cache on screen 119 is stale so reflex macro owns the route.

**Validated (June 2026):**

- ROM one-shot: `FULL_FROM_APPROACH` → `sword_level=1`, exit → overworld #119.
- Unit tests: 24 passing (`test_zelda.py` + `test_start_cave.py`).
- Director 3×25k frames: search 8→1→1, replay 1683→1690, frontier 2208px; still times out on
  #119 before screen #120 in the frame cap (live phase playback slower than one-shot macro).

## Path forward (priority order)

### P1 — East march & learning compounding

- Tune live cave exit (continue DOWN until `in_cave` clears); raise frame cap or bank full macro.
- Confirm overworld #120+ after sword in multi-attempt runs without `--fresh`.
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
| `reflex.py` | Live policy + cave macro playback |
| `start_cave.py` | ROM-verified wooden sword macro |
| `hazard_hooks.py` | Combat/cave zones for Director |
| `tuning.py` | Constants (START_SCREEN=119, etc.) |