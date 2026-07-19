# Billy Mitchell

Billy is an agentic retro game-player that discovers exact solutions once and replays them forever, transferring abstract tactics across titles via a game-agnostic engine.

## Language

### Perception & identity

**Observation**:
A single-frame snapshot of game state the engine can reason about: progress, level identity, death, and a game-specific raw scene.
_Avoid_: Frame dump, screen, state blob

**Progress**:
A monotonic within-level measure of how far Billy has advanced (platformers: usually horizontal position; other genres may use exploration credit).
_Avoid_: Score, x alone (when another coordinate also matters), completion %

**Level key**:
An ordered identity for “where we are in the campaign,” used to detect *clear* (advance) vs mere screen change. Comparable so “new key is further” is well-defined.
_Avoid_: Level name, stage string, ROM bank

**Level label**:
A human-readable tag for logs and files (e.g. `smw-12`), not used for ordinal clear detection.
_Avoid_: Level key

**Elevation**:
A secondary route coordinate so two paths at the same progress (high road vs low road) stay distinct.
_Avoid_: Y alone when it is not part of the route key

### Engine seam

**Session**:
The live emulator connection: step the game, read RAM/video, clone state for invisible planning.
_Avoid_: Emulator process, gym env (unless speaking of stable-retro specifically)

**Game**:
A title adapter that boots, observes, and supplies a reflex for one ROM/integration. The engine talks only to this contract, not to console details.
_Avoid_: System, platform, integration (when you mean the adapter)

**System**:
Console-level transport and controller mapping (NES, SNES, …) shared by multiple games.
_Avoid_: Game

**Seam**:
The intentional boundary where the engine stops and game-specific code begins (`Session` / `Game` / `Observation` / `ReflexPolicy`).
_Avoid_: Interface soup, “the API”

### Play loop

**Reflex**:
Cheap, every-frame routine play (run, hop, stomp) with no search and no LLM.
_Avoid_: Policy (when you mean the routine controller), AI, agent brain

**Hazard**:
A situation where routine reflex is not enough and Billy must search, replay a banked solution, or ask for help.
_Avoid_: Obstacle, enemy (too narrow), hard part

**Micro-search**:
Invisible planning on a **cloned** game state to find a short button sequence that survives and makes progress.
_Avoid_: Brute force, MCTS (unless that algorithm is actually in use), live rewind

**Learn-from-death**:
After a death, search backward from the last safe spot for a sequence that gets *past* the death site and bank it.
_Avoid_: Respawn retry, random restart

**Exact-replay**:
Replaying a stored button sequence verbatim from a matching state — never a fuzzy or embedding-chosen action stream as authority.
_Avoid_: Approximate replay, nearest-neighbor control, “similar situation” blind play

### Memory

**Solution cache**:
Exact verified button sequences keyed to where they work (level identity + progress bucket), the compounding per-hazard memory.
_Avoid_: Memory bank, embedding store, vector DB

**Reachback**:
Trying a high-reach cache entry a few progress buckets *behind* the current spot, clone-verified before trust.
_Avoid_: Nearby match, fuzzy cache hit

**Tape**:
A whole-trajectory input stream for a level or screen that can re-clear search-free after verify.
_Avoid_: Demo file (when you mean engine tape), movie, savestate

**Entry-state anchor**:
A savestate captured at level begin that a tape restores so moving hazards (e.g. lifts whose phase is fixed at load) reproduce.
_Avoid_: Checkpoint (engine march checkpoints are different), full-game save

**Skill** (maneuver):
A transferable candidate sequence distilled from verified wins; may **seed** search only — never blind exact-replay authority. Console-gated.
_Avoid_: Agent skill (Grok/Claude skill packs), ability, power-up

**Walkthrough guide**:
Ingested FAQ/text advice that seeds search candidates and LLM context; never authoritative without clone verify.
_Avoid_: Strategy guide as source of truth, hard-coded script

### Human & assist

**Teleop**:
Human takeover on the controller (e.g. live **T** in the watch window) that can bank demos into cache/tapes/skills.
_Avoid_: Manual mode, babysitting

**Demo request**:
Billy asking for a human segment when search and stuck-training keep missing.
_Avoid_: Help ticket (unless filed that way), bug report

### Cross-title

**Logical button**:
Console-agnostic control names (e.g. jump/run) mapped by each System to real pad bits so one reflex can drive NES and SNES.
_Avoid_: Raw bitmask, SNES B/Y in engine-level talk

**Platformer profile**:
Per-game physics priors (jump frames, etc.) that specialize the shared platformer reflex without forking the engine.
_Avoid_: Hardcoded SMB constants in the Director
