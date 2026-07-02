# Super Mario World (SNES) — Scaffold Status

**As of July 1, 2026: SCAFFOLD, not yet live.** `roms/` has no SMW ROM, so nothing below has
run against the real game. Everything is unit-tested against synthetic WRAM fixtures.

## What exists

| Piece | File | State |
|---|---|---|
| SNES system + 12-button controller | `billy/systems/snes/` | ✅ built; RETRO_NAMES maps logical A(jump)/B(run) → SNES B/Y so the shared reflex carries over unchanged |
| Console-agnostic session | `billy/systems/nes/retro_session.py` | ✅ parameterized (controller module + RAM size); NES behavior unchanged |
| WRAM map | `ram_map.py` | ⚠️ community-sourced offsets, **unverified live** |
| Perception → PlatformerView | `perception.py` | ✅ player/sprites/score/mode; ⚠️ tile queries (gap/obstacle) return None until map16 reads verified |
| Game adapter | `game.py` | ✅ observe/boot/level-identity; clear detection via monotonic EVENTS_TRIGGERED counter |
| Physics profile | `tuning.py` | ⚠️ priors (floatier jumps than SMB1), untuned |
| RAM probe | `probe_smw_ram.py` (repo root) | ✅ verifies offsets live in one command |

## Bring-up checklist (once you have the ROM)

1. Drop the ROM in `roms/` (e.g. `smw.sfc`) and import: `.venv/bin/python -m stable_retro.import roms/`
   — the stock `SuperMarioWorld-Snes` integration should claim it.
2. `BILLY_HEADLESS=1 .venv/bin/python probe_smw_ram.py` — holds RIGHT for 60 frames and
   checks: PLAYER_X increases, GAME_MODE==0x14, ON_GROUND flips sensibly, LIVES==5.
   Any miss = the WRAM base assumption is wrong; the probe prints a scan to relocate it.
3. `BILLY_HEADLESS=1 BILLY_MAX_FRAMES=6000 .venv/bin/python run.py --game smw --attempts 2 --no-llm`
   — expect cruise + search/learn-from-death play (tile queries are off, so pits are learned,
   not sighted). Cache/tapes/teleop all work day one: they're engine-side.
4. Tune `tuning.py` jump frames against the first pit that search can't crack.
5. Wire Layer-1 map16 tile reads (gap_info/obstacle_ahead) — the reflex then *sees* pits like
   it does in SMB, and search load drops.
6. Teleop works immediately: `.venv/bin/python teleop.py play --game smw --from-state <s> --bank`
   (keyboard adds C=spin jump, S=X, Q/W=L/R).

## Design notes

- **Level identity**: SMW is non-linear (overworld map), so `level_key = (events_triggered,
  0, translevel)` — the events counter is monotonic and bumps per beaten level, making the
  engine's `[:2] >` clear rule hold without Director changes. Map walking = screen changes.
- **Spin jump** is a new logical button (`SPIN`, SNES A). `spin_jump_right()` should join the
  search candidate spread once live — it bounces off hazards normal jumps can't.
- **Distilled NES skills won't fire here** (console-gated by design); SMW builds its own
  sequence library, but the abstract starter skills (gap_jump/stomp) instantiate fine since
  they use logical buttons.
