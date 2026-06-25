# Billy Mitchell 🕹️

An agentic NES game-player that **learns to beat levels** — starting with Super Mario Bros in
FCEUX. "Billy" plays through a simulated NES controller, perceives the game by reading
emulator RAM, reasons with a **local LLM in LM Studio**, and gets better across attempts by
banking lessons in a knowledge base. He has the personality of the real Billy Mitchell:
cocky, boastful, never wrong, and quick to blame a "glitchy cartridge" when he dies.

## How it works

A local 7B model is far too slow to react frame-by-frame (Mario wants ~60 decisions/sec), so
Billy is **not** a frame-by-frame controller. The design is a **two-tier loop**:

- **Tier 1 — Reflex executor** (pure Python, every frame): runs right, hops gaps and enemies,
  and detects "interesting" events (stuck, death, level clear). No LLM.
- **Tier 2 — Billy (LLM)**: consulted *only* at decision points (e.g. when stuck). Reads the
  scene + retrieved lessons and returns a short controller plan + trash talk.
- **Tier 3 — Coach (LLM)**: after each attempt, distills one reusable lesson into the
  knowledge base (embedded with `nomic-embed-text` for retrieval).

Billy and FCEUX talk over **lock-step file IPC**: the Lua bridge publishes a state (2KB of
RAM) and blocks until Python sends an action plan, so the LLM's latency never drops a frame.
Instant `savestate`/`loadstate` makes every attempt a clean retry of the level start.

```
FCEUX + billy_bridge.lua  <--state.bin / action.bin-->  Python brain (Director)
   RAM read, joypad.set                                  perception · executor · Billy · Coach · KB
```

## Setup

1. **FCEUX**: `brew install fceux`
2. **ROM**: put a legally-obtained `Super Mario Bros (USA).nes` at `roms/smb.nes` (gitignored).
3. **LM Studio**: run it with the server on `localhost:1234` and load a chat model
   (defaults to `deepseek-coder-v2-lite-instruct`) plus the `nomic-embed-text` embedding model.
4. **Python**: `pip install -r requirements.txt` (only needs `requests`; Python 3.11+).

## Run

Two terminals:

```bash
# Terminal 1 — launch the emulator + bridge (loads SMB to the title screen)
./emulator/run_fceux.sh

# Terminal 2 — start Billy's brain (he presses Start himself and begins learning)
python run.py --attempts 20
```

Useful flags:
- `--no-llm` — pure Tier-1 reflex run (no Billy/Coach); good for a first smoke test.
- `--fresh` — wipe previously learned lessons.
- `BILLY_SPEED=turbo ./emulator/run_fceux.sh` — run faster-than-realtime for quicker learning.
- `BILLY_CHAT_MODEL=<id>` — use a different loaded LM Studio model.

## Develop / verify without the emulator

```bash
python -c "import tests.test_perception as t; [getattr(t,n)() for n in dir(t) if n.startswith('test_')]"
python scripts/dryrun.py          # full loop vs a FAKE emulator (no LLM)
python scripts/dryrun.py --llm    # same, but also drives Billy + Coach against LM Studio
```

`scripts/dryrun.py` speaks the exact same binary protocol as the Lua bridge, so it validates
the whole brain (IPC, perception, executor, Billy, Coach, KB, metrics) without FCEUX.

## Layout

Three layers — `Game → System → Controller` — behind abstract contracts, so the engine is
reusable across consoles and titles. Adding a new system = a new `systems/<x>/`; a new game =
a new `games/<y>/`; the engine never changes.

| Path | Layer | Role |
|------|-------|------|
| `billy/abstractions.py` | engine | The contracts: `Observation`, `Decision`, `Controller`, `System`, `Game`, `ReflexPolicy` |
| `billy/director.py` | engine | Game-agnostic loop: reflex ↔ Billy, danger-zone micro-search, checkpoints, records |
| `billy/agents/billy.py` · `coach.py` | engine | LLM strategist + analyst (operate on an `Observation`) |
| `billy/knowledge/store.py` · `metrics.py` · `commentary.py` · `persona.py` · `llm.py` · `ipc.py` | engine | KB, metrics, Billy's voice, LLM client, file-IPC transport |
| `billy/systems/nes/controller.py` | controller | NES pad: button bits + name↔mask encoding |
| `billy/systems/nes/system.py` · `bridge.lua` | system | NES transport + FCEUX launch + the in-emulator Lua bridge |
| `billy/games/smb/perception.py` | game | SMB RAM → `Scene` (ported MarI/O readers) |
| `billy/games/smb/reflexes.py` · `tuning.py` | game | SMB reflex policy (gap/pipe/stomp/air-steer) + physics constants |
| `billy/games/smb/game.py` | game | `SmbGame` binds it all: `observe`, `boot`, `make_reflex` |
| `run.py` | — | Entry point (picks the game, runs the engine) |
