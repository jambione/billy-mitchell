# Billy Mitchell 🕹️

An agentic NES game-player that **learns to beat levels and carries that learning forward to new
games**. Billy perceives the game by reading emulator RAM, plays through a simulated NES controller,
and **gets faster every attempt** by banking the exact solutions he discovers. He has the
personality of the real Billy Mitchell: cocky, boastful, never wrong, and quick to blame a "glitchy
cartridge" when he dies.

The emulator runs **in-process** via [stable-retro](https://github.com/Farama-Foundation/stable-retro)
— no external process, no file IPC, and **deterministic state cloning** so Billy can plan invisibly.

## The idea: discover once, replay forever

A local LLM is far too slow to react ~60×/sec, so Billy is **not** a frame-by-frame controller.
Instead he learns a **position-keyed policy** the first time he sees each hazard:

1. **Reflex** runs the routine play (run right, hop gaps, stomp enemies) every frame — no LLM.
2. At a hazard, Billy **micro-searches on a cloned copy of the game** (invisible to the live run) for
   a button sequence that *verifiably survives and makes progress*, and **caches it** keyed to where
   it happened — `(level, x)`.
3. On any later pass he **replays that exact sequence** — no search, no LLM. Each hazard solved once
   is solved forever, so later attempts only search the *new* frontier.
4. On a **death**, he searches backward from the last safe spot for a sequence that gets *past* the
   death, and banks it (learn-from-death) — this is what advances the frontier.
5. Because enemies move, a cached plan is **verified on a clone first**; if it's gone stale he
   live-searches with the enemy where it *actually* is now (replay-verify → live-search).

The LLM (Billy + Coach) is consulted only for genuinely novel/stuck moments and persona — it is out
of the hot loop.

```mermaid
flowchart TD
    obs[Observe RAM → Scene] --> reflex{Reflex: routine or hazard?}
    reflex -- routine --> act[Run right / hop / stomp]
    reflex -- hazard --> cache{Solution cached here?}
    cache -- yes --> verify{Verify on clone}
    verify -- survives --> replay[Replay exact sequence ⚡]
    verify -- stale --> search
    cache -- no --> search["Micro-search on a CLONE 🔍<br/>seeded by reflex spread + transferable Skills"]
    search -- found --> bank["Bank solution at (level, x)"] --> act
    search -- none --> llm[Billy LLM improvises]
    act --> obs
    death[💀 death] --> lfd["Learn-from-death:<br/>search a survivor past the death, bank it"] --> obs
```

## Two kinds of learning → cross-game transfer

- **SolutionCache** (`knowledge/cache.py`) — *exact* solutions, replayed deterministically. Keyed on
  the engine's generic `(level_key, progress)`, so the whole discover-once/replay-forever capability
  is **game-agnostic**.
- **Skill library** (`knowledge/skills.py`) — *abstract* tactics ("precise gap jump", "stomp from
  approach", "run-jump a tall obstacle") carried as embeddings. On a new game the cache is empty, but
  skills retrieved by situation-similarity **seed the search** with carried-forward tactics. Skills
  only widen the search set — they never blind-replay, so transfer can't cause a wrong action.
- **Shared platformer reflex** (`games/common/platformer.py`) — the whole side-scroller policy,
  parameterised by a per-game `PhysicsProfile`. A new NES platformer reuses it wholesale; e.g.
  **SMB2-Japan / Lost Levels** (`games/smb_lost/`) plays with *zero new reflex code*.

## Setup

```bash
./emulator/setup_retro.sh          # creates .venv, installs deps, imports the ROM
```
You must supply a legally-obtained `Super Mario Bros (USA).nes` at `roms/smb.nes` (gitignored). For
the second game, drop the SMB2-Japan ROM in `roms/` and re-run `python -m retro.import roms/`.
The LLM tiers are optional — run with `--no-llm` for the pure learning loop. To enable Billy/Coach,
run LM Studio on `localhost:1234` with a chat model + the `nomic-embed-text` embedder.

## Run

```bash
.venv/bin/python run.py --attempts 20                      # play + learn, watch the window
BILLY_HEADLESS=1 .venv/bin/python run.py --attempts 10 --no-llm   # fast headless benchmark
.venv/bin/python run.py --game smb_lost --seed-skills      # SMB2-Japan, seeded with SMB skills
```

Flags / env:
- `--no-llm` — pure learning loop (reflex + cache + search), no LLM. Great first smoke test.
- `--game smb|smb_lost` — which title. `--seed-skills` — seed transferable SMB tactics.
- `--fresh` — wipe learned solutions, skills, and lessons.
- `BILLY_HEADLESS=1` — no window (fast). `BILLY_TURBO=1` — no realtime pacing when windowed.
- `BILLY_REPEAT_LEVEL=1` — eval mode: end each attempt at the first clear so the **same** level
  repeats and the compounding curve is visible. `BILLY_MAX_FRAMES=N` — cap attempt length.

**Prove the learning compounds** (Billy clears 1-1 every attempt; the curve prints search↓/replay↑):
```bash
BILLY_HEADLESS=1 BILLY_REPEAT_LEVEL=1 BILLY_MAX_FRAMES=8000 .venv/bin/python -u run.py --attempts 10 --no-llm
```

## Optional: learned (RL) reflex tier

A PPO policy can replace the hand-crafted reflex as the fast Tier-1 controller, trained against a
Gymnasium wrapper over the *same* in-process emulator + RAM perception (`billy/rl/`). It's optional
(heavy deps) and **coexists** with everything else: the Director order stays cache → reflex →
micro-search → LLM, so the SolutionCache still owns deterministic hazard replay and the verified
search still handles lethal spots — RL just makes routine movement smarter (and falls back to the
hand-crafted reflex at hazards).

```bash
.venv/bin/pip install -r requirements-rl.txt        # torch + stable-baselines3 (cp314 wheels)
.venv/bin/python train_rl.py --timesteps 200000 --n-envs 4 --imitate 4000   # train (BC warm-start)
.venv/bin/python run.py --rl data/rl/ppo_smb        # play with the learned policy
```

## Tests

```bash
BILLY_HEADLESS=1 .venv/bin/python -m pytest -q tests/
```

## Layout

Three layers — `Game → System → Controller` — behind abstract contracts, so the engine is reusable
across consoles and titles. New system = a new `systems/<x>/`; new game = a new `games/<y>/`.

| Path | Layer | Role |
|------|-------|------|
| `billy/abstractions.py` | engine | Contracts: `Observation`, `Decision`, `Session`, `System`, `Game`, `ReflexPolicy` |
| `billy/director.py` | engine | Game-agnostic loop: cache-first replay → invisible micro-search → learn-from-death → LLM |
| `billy/knowledge/cache.py` | engine | `SolutionCache` — position-keyed exact solutions (the compounding policy) |
| `billy/knowledge/skills.py` | engine | `SkillLibrary` — embedding-retrieved transferable tactics (cross-game) |
| `billy/knowledge/store.py` | engine | Prose-lesson KB + embedding helpers (LLM strategy/narration) |
| `billy/agents/billy.py` · `coach.py` | engine | LLM strategist + analyst (off the hot loop) |
| `billy/metrics.py` · `commentary.py` · `persona.py` · `llm.py` | engine | Compounding metrics, Billy's voice, LLM client |
| `billy/systems/nes/retro_session.py` | system | In-process stable-retro transport: step, RAM, **state cloning**, invisible search |
| `billy/systems/nes/controller.py` · `system.py` | system | NES pad (button bits) + system wiring |
| `billy/games/common/platformer.py` | game | Shared NES-platformer reflex + `PhysicsProfile` + candidate builders |
| `billy/games/smb/{perception,reflexes,tuning,game}.py` | game | SMB: RAM→`Scene`, SMB profile, `SmbGame` |
| `billy/games/smb_lost/game.py` | game | SMB2-Japan — same engine, reuses SMB perception + the shared reflex |
| `run.py` | — | Entry point (picks the game, seeds skills, runs the engine) |
