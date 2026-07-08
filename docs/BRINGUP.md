# New-game bring-up kit

Checklist for onboarding a title so Billy plays, fails fast, files a teachable wall, and the
human can teach it in Remix — without archaeology. Generalizes `billy/games/smw/STATUS.md`.

## Prerequisites

- ROM in `roms/` (gitignored), imported via `./emulator/setup_retro.sh`
- `.venv` active; all tests/scouting with `BILLY_HEADLESS=1`

## Checklist

| Step | What | Where |
|---|---|---|
| 1 | Import ROM → stable-retro integration id | `emulator/setup_retro.sh`, integration metadata |
| 2 | Probe RAM (or skip for pixel path) | `probe_<game>_ram.py` at repo root |
| 3 | Minimal `Game` adapter | `billy/games/<game>/game.py` — observe, boot, progress, level_key, dead |
| 4 | Reflex or pixel fallback | `reflex.py` or `billy/games/pixel/` / `shmup/` pattern |
| 5 | Route rank (or accept frontier exploration) | `Game.route_rank` override if non-SMB ordinals |
| 6 | Remix hooks (4 + capture) | `remix_goal`, `remix_win`, `remix_min_progress`, `remix_anchor_ok` on `Game` |
| 7 | Add to campaign | `remix.py` `CAMPAIGN` + `_ENDING` |
| 8 | Smoke run | `BILLY_HEADLESS=1 .venv/bin/python run.py --game <key> --attempts 3 --no-llm` |
| 9 | Remix teach smoke | `.venv/bin/python remix.py --only <key> --no-scout` (needs window) |

## Pixel-first fallback

When no RAM map exists, onboard via the pixel path (`--game pixel` or `shmup`). The Airstriker
adapter proved the engine runs with zero RAM knowledge. RAM mapping is an optimization, not a
prerequisite.

## Remix hooks (required for gauntlet membership)

Implement on your `Game` subclass — defaults suit progress-keyed platformers:

```python
def remix_goal(self, req) -> str: ...
def remix_win(self, obs, req, start_obs) -> bool: ...
def remix_min_progress(self) -> int: ...
def remix_anchor_ok(self, source) -> bool: ...
```

Optional overrides: `remix_approach_progress_window`, `remix_capture_ready`, `remix_on_ground`,
`remix_dropin_level_ok`, `remix_director_sections`, `stuck_death_threshold`.

No edits to `remix.py` should be needed after step 6.

## Per-system notes

### NES platformer (SMB family)
- Inherit `SmbGame` for Lost Levels (`smb_lost`) — reflex + perception carry over.
- `remix_director_sections()` enables approach capture with hazard sub-policies.

### NES top-down (Zelda)
- `remix_win` = screen-crossing (`map_location` advances).
- `remix_anchor_ok` always True — screen-keyed tapes.

### Genesis RPG (PSII)
- Progress = exploration credit (tiles seen), not spatial x.
- Menu walls: `remix_win` when menu closes and progress advances.
- Battle RAM deferred until overworld probe — `dead` stays False in town.

### SNES (SMW)
- See `billy/games/smw/STATUS.md` for scaffold status.
- `level_key` uses monotonic `events_triggered` counter.

## Acceptance

A brand-new title reaches **"Billy plays → fails fast → files a wall → human teaches in Remix"**
in under a day, documented start to finish. Verify with:

```bash
.venv/bin/python remix.py --list          # frontier + open walls
BILLY_HEADLESS=1 .venv/bin/python -m pytest -q tests/
```