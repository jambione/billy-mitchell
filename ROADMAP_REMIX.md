# BILLY MITCHELL — ROADMAP: exponential learning through the Remix

**Handoff document** (written 2026-07-07 for an external agent to work from). Self-contained:
vision, current state, architecture you must not break, and phased work with acceptance
criteria. Read `CLAUDE.md` alongside this — it holds the run commands and invariants in full.
The older `ROADMAP.md` is historical context (most of its Phase 1 shipped); this doc is the
current plan.

---

## 1. Vision (the north star)

Billy is an agentic game-player that **learns exponentially on any game and any system he is
given**. The learning contract:

1. **Fail fast.** Billy plays himself, hits a wall, and recognizes quickly that he is stuck —
   minutes, not hours.
2. **Ask for help.** A stuck wall becomes a *teachable moment*: Billy files a demo request with
   a real drop-in savestate, and the human is dropped in via the **Remix** to cross it once.
3. **Keep everything.** One human crossing banks ALL FOUR carriers (exact cache entry,
   entry-anchored tape, distilled transferable skill, BC warm-start seed) so the lesson sticks,
   reproduces deterministically, and transfers.
4. **Compound.** Next attempt, Billy gets further before the NEXT wall. Search per level trends
   to zero; replay trends to 100%. Within a session: teach → re-scout → next wall.
5. **Outgrow the human.** Long-term, demos-per-world must FALL. Billy learns from his
   environment — recognizing obstacle/enemy patterns, forming goals, reaching them efficiently
   — until human help is the exception, not the loop.

The human's experience matters as much as Billy's: the Remix is the *fun* surface. Every new
game Billy takes on grows the Remix gauntlet. Teaching should feel like playing NES Remix —
challenge cards, a visible payoff (Billy replays your line on-screen), medals, and a scoreboard
of Billy's march toward finishing each game.

**End state: the best game player ever — one that learned the past (every taught line, every
banked trajectory) so thoroughly it stops needing a teacher.**

---

## 2. Current state (honest, as of 2026-07-12, `main` @ 0754da2)

> **Update 2026-07-12:** Phase 1 and Phase 2 below shipped in the four commits after this doc
> was first written (19d7f9d, 522a88e, e74e07b, 0754da2). Left the phase writeups intact as
> reference for *how* they were built; new work should start at Phase 3.

### What works
- **Game-agnostic engine**: talks only to `billy/abstractions.py` contracts (`Session`, `Game`,
  `Observation`, `ReflexPolicy`). New console = `billy/systems/<x>/`; new title = `billy/games/<y>/`.
  Proven across NES (SMB, SMB2J, Zelda), Genesis (Airstriker pixel-only, Phantasy Star II), and
  a SNES scaffold (SMW, awaiting ROM).
- **The compounding loop**: cache-hit → clone-verify → replay; else micro-search on a clone;
  else LLM. Death → learn-from-death. Tapes (whole-trajectory, entry-state-anchored) reproduce
  moving hazards deterministically.
- **Remix v2 — universal teach contract (Phase 1, ✅ shipped)**: `remix_goal` / `remix_win` /
  `remix_min_progress` / `remix_anchor_ok` plus approach-capture hooks
  (`remix_needs_approach_capture`, `remix_approach_progress_window`, `remix_on_ground`,
  `remix_capture_ready`, `remix_dropin_is_safe`, `remix_stabilize_dropin`,
  `remix_director_sections`) all live on `Game` with progress-keyed-platformer defaults in
  `billy/abstractions.py`. SMB, Zelda, PSII, smb_lost all join the gauntlet on hooks alone —
  `remix.py` itself is generic (`_prepare_approach` in `remix.py:470` takes any `game`). 42 remix
  tests green.
- **Fail-fast + ask-for-help (Phase 2, ✅ shipped)**: approach capture is game-agnostic (no more
  SMB-only `_prepare_smb_approach`); safe drop-in selection rejects cliff-lip/mid-air captures
  and backs off to earlier candidates or the level checkpoint (0754da2); a taught wall stops
  re-surfacing until a fresh death streak reopens it (`StuckTracker` remediation, 522a88e); the
  Director hands off Remix-taught BC demos by warping to the taught state and replaying the
  verified plan, so one human crossing keeps paying off every attempt (e74e07b); Billy tells the
  human what he needs via `data/remix_inbox.txt` + a macOS notification
  (`billy/stuck_trainer.py:412`).
- **Routes + strategist**: transition graph persisted (`data/routes.jsonl`), warp-preferring
  route planning for SMB, furthest-checkpoint resume.
- **Pixel perception exists**: `--game pixel` (SMB from pixels) and `--game shmup` (Airstriker,
  no RAM map at all — 100% pixel ship-tracking; tape-evolution cracked reactive learning).

### Where Billy actually is
| Game | Frontier | Goal | Open wall |
|---|---|---|---|
| smb | **4-2** (was 3-4) | 8-4 | 4-2 @ x≈322 (8 deaths) — queued in `remix_inbox.txt` now |
| smb_lost | 1-1, barely probed | 8-4 | never scouted seriously |
| zelda | start | level-9 | old wall: overworld #121 combat |
| psii | start | end | menu walls have `remix_win`; battle RAM still unmapped (town has no encounter perception yet) |
| smw | scaffold only | — | needs ROM (`billy/games/smw/STATUS.md`) |

### Honest limits
- Position-keyed cache entries re-search moving hazards each pass; only entry-anchored tapes
  fully solve that, and tapes need a valid level/screen-entry anchor.
- Phase 3's bring-up kit is half-done: `docs/BRINGUP.md` exists and PSII got menu-shaped Remix
  hooks, but PSII battle RAM is still unmapped and smb_lost hasn't been scouted seriously (it
  should mostly ride SMB's reflex + skills — untested claim).
- BC seeds are written (`billy/knowledge/demo_seed.py`) but nothing trains them automatically;
  `train_section.py` is a manual, SMB-shaped offline job — Phase 4's auto BC→PPO queue doesn't
  exist yet.
- `report.py` (Phase 6 v1, 2026-07-12) reads `data/metrics.jsonl` back into search↓/replay↑,
  frontier, and demos-per-world curves — but every attempt logged before this lands in an
  untagged "legacy" bucket, and cross-game A/B (curve 4) still needs a real comparative run.
- No fail-reel/medals/session-recap — Phase 5 not started.

---

## 3. Architecture primer + invariants (DO NOT BREAK)

Decision flow per hazard (`billy/director.py` `run_attempt`):
**cache-hit → verify on clone → replay** ⚡; else **micro-search on a clone (invisible) → bank** 🔍;
else **LLM**. On death → **learn-from-death**.

The invariants (full text in `CLAUDE.md` — each has been the cause of a real regression):
1. **Exact-replay only.** Embeddings/skills may seed search candidates; they must NEVER drive a
   blind replay. Every replay is clone-verified first.
2. **Don't cache non-progress.** A solution must survive AND advance (`_MIN_PROGRESS_PX`).
3. **Rollout `settle` is a post-candidate budget**, not a total.
4. **A plan is verified only UP TO a transition.** Never replay an unverified tail across a
   level_key change.
5. **Snapshot/learn/replay only on-ground.** Airborne states don't reproduce.
6. **Behaviour-preserving reflex refactors** must keep the 1-1 clear + identical compounding
   curve (regression guard).
7. **Tape anchors must be true level/screen entries.** A mid-level anchor on a per-level tape
   replays from the wrong place (Remix gates this in `_anchor_is_level_entry`; keep that
   property under any refactor). Safe backstop: the Director clone-verifies tapes before replay.

Run/test (always the venv):
```bash
BILLY_HEADLESS=1 .venv/bin/python -m pytest -q tests/
BILLY_HEADLESS=1 BILLY_REPEAT_LEVEL=1 BILLY_MAX_FRAMES=8000 .venv/bin/python -u run.py --attempts 10 --no-llm
.venv/bin/python remix.py --list        # frontier + open walls, read-only
.venv/bin/python report.py --open       # data/report.html: search/replay + frontier + demos-per-world curves
```

---

## 4. The roadmap

Phases are ordered by leverage. Each has acceptance criteria — build to those, not to vibes.

### Phase 1 — Universalize the Remix teach contract (unblocks every new game) ✅ SHIPPED (19d7f9d)
The Remix currently special-cases SMB and Zelda inside `remix.py` (`_goal_blurb`,
`_teach_params`, `_wall_cleared`, `_prepare_smb_approach`, `_anchor_is_level_entry` gating).
Move the per-game knowledge behind the `Game` contract so a new title plugs into the gauntlet
with zero `remix.py` edits.

- Add optional `Game` hooks (default implementations preserve today's behavior):
  - `remix_goal(req) -> str` — the human-readable challenge line.
  - `remix_win(obs, req, start_obs) -> bool` — what "crossed the wall" means (x-progress,
    screen change, battle won, menu escaped…).
  - `remix_min_progress() -> int` — the anti-trivial-win gate (Zelda's 48px rule generalized).
  - `remix_anchor_ok(source) -> bool` — is this drop-in a true level/screen entry for taping?
- Port SMB + Zelda onto the hooks; `remix.py` keeps only generic flow. Tests: existing 31 remix
  tests still pass unchanged in behavior; add one fake-game test proving a new `Game` joins the
  gauntlet without touching `remix.py`.
- **Acceptance:** adding a hypothetical game to `CAMPAIGN` + implementing the 4 hooks is the
  ONLY work needed to teach its walls.

### Phase 2 — Fail-fast + ask-for-help, everywhere ✅ SHIPPED (522a88e, e74e07b, 0754da2)
Make "Billy notices he's stuck and files a *good* demo request" a universal, fast behavior.

- **Universal approach capture.** Generalize `_prepare_smb_approach` (drive Billy headless to
  just-before-the-wall and snapshot on-ground) into a game-agnostic routine using `progress` +
  an `on_ground`-equivalent from the `Game` contract. Zelda/PSII walls should get real drop-ins
  automatically, not rely on hand-captured teleop states.
- **Tune time-to-wall.** Audit `STUCK_DEATH_THRESHOLD` and stuck-detection latency per game;
  target: a genuine wall is detected and filed within ~5 minutes of headless play.
- **Tell the human.** When a scout files a wall, surface it: write a one-line
  `data/remix_inbox.txt` and (macOS) fire a notification — "Billy needs you at zelda #121."
  The Remix `--no-scout` shortcut then teaches it immediately.
- **Acceptance:** on a fresh game with a working integration, Billy goes from "never played" to
  "filed a teachable wall with a valid drop-in state" with zero manual state capture.

### Phase 3 — New-game bring-up kit (any game, any system)
Codify what it takes to onboard a title so it's a checklist, not archaeology. The kit lives in
`docs/BRINGUP.md` + scaffolding tools.

- **The checklist** (generalize `billy/games/smw/STATUS.md`): ROM import → RAM probe
  (`probe_*_ram.py` pattern) → minimal `Game` (observe/boot/progress/level_key/dead) →
  reflex or pixel fallback → `route_rank` (or accept frontier exploration) → Remix hooks
  (Phase 1) → add to `CAMPAIGN` + `_ENDING`.
- **Pixel-first fallback.** When no RAM map exists, onboard via the pixel path (the shmup
  adapter proved the engine runs with zero RAM knowledge). RAM mapping becomes an optimization,
  not a prerequisite.
- **Immediate applications** (in order): finish **SMW** when the ROM lands; give **PSII** its
  battle/menu RAM + Remix hooks (its walls are menus and fights, not pits — a perfect test that
  Phase 1's `remix_win` generalizes); scout **smb_lost** seriously (it should mostly ride SMB's
  reflex + skills — measure the transfer).
- **Acceptance:** a brand-new NES/Genesis title reaches "Billy plays, fails fast, files a wall,
  human teaches it in Remix" in under a day of work, documented start to finish.

### Phase 4 — Learning WITHOUT the human (the demos must taper off)
Everything the human teaches should be leverage for Billy to teach himself more.

- **Auto BC→PPO queue.** Remix already writes BC seeds (`data/rl/demos/<game>/*.demo.json` +
  `.state`). Add an offline worker (cron or `--train-queue` mode) that runs
  `train_section.py --demo` on new seeds and registers the resulting sub-policy with the
  SectionController. Human teaches once; overnight, it becomes a robust policy.
- **Offline self-improvement pass.** Replay ALL banked solutions/tapes headless; distill
  patterns (`billy/knowledge/distill.py`) into higher-quality, console-gated skills; prune
  stale cache entries that repeatedly fail verify. Billy studies his own past — "learning the
  past" made literal.
- **Tape evolution beyond shmup.** The tape-evolution loop (search whole trajectories) cracked
  reactive play on Airstriker. Apply it to Zelda combat screens — evolve a surviving line
  instead of demanding a demo for every fight.
- **Death-pattern generalization.** Learn-from-death currently searches past the specific death.
  Add pattern recognition over stuck history: "this is a pit like the last 4 pits" → seed search
  with previously-successful pit skills FIRST. Obstacle/enemy taxonomy grows from experience.
- **Acceptance metric:** demos-per-world declines game-over-game (see Phase 6). Concretely: by
  SMB world 6+, Billy passes new walls with search+skills alone at a measurably higher rate
  than in worlds 1–3.

### Phase 5 — The fun layer (make teaching a game, not a chore)
- **Watch Billy fail first.** Before dropping the human in, replay Billy's best failing attempt
  at the wall on-screen (5–10s). Stakes, then challenge. (The state + deaths are already known.)
- **Medals + personal bests.** Time each crossing; award 🥉/🥈/🥇 vs your prior best on retries
  (the bank-your-best loop already exists — score it). Persist to `data/remix_medals.jsonl`.
- **Session recap.** End-of-session card: walls taught, Billy's frontier delta per game ("your
  3 lines moved Billy 3-4 → 4-2"), medals earned, total lines Billy owns.
- **Challenge variety.** Once Phase 1 lands, walls stop being only "get past x": Zelda fight
  screens, PSII battles/menus, SMW flight sections — the gauntlet naturally diversifies with
  every game onboarded.
- **Acceptance:** a Remix session has a visible arc — fail-reel → challenge → replay-payoff →
  medal → march — with zero extra setup by the human.

### Phase 6 — Measure the exponent (prove it, don't vibe it) 🟡 STARTED (2026-07-12)
`report.py` renders `data/metrics.jsonl` (now `game`-tagged, `billy/metrics.py` +
`director.py` `run_attempt`/`_run_attempt_evolve`) into `data/report.html`: search-vs-replay
and solved-frontier line charts per game + per top-attempted level, plus a demos-per-world
bar chart sourced straight from `data/rl/demos/<game>/` (so it's correct even for games with
no tagged attempts yet). Curves 1–3 below are covered; curve 4 (cross-game A/B) is not — it
needs an explicit comparative run, not just a report over existing logs. All 1,007
pre-existing metrics rows predate the `game` field and render under a "legacy (untagged)"
bucket rather than being guessed at.

- **Per-attempt telemetry** (`data/metrics.jsonl`): frames spent in search vs replay vs teleop,
  walls passed by tier (reflex/tape/cache/search/demo), frontier progress, demos requested.
  (Foundation exists: `billy/metrics.py` prints the per-attempt compounding table — persist it.)
- **The curves that matter:**
  1. search%↓ / replay%↑ per level over attempts (within-game compounding);
  2. time-to-frontier-advance per world (should shrink);
  3. **demos-per-world** (the outgrow-the-human metric — must trend down);
  4. cross-game: time-to-first-wall on a new title with vs without the skill library.
- **Dashboard**: a single static HTML report generated from metrics.jsonl (no live server
  needed). The Remix scoreboard links to it.
- **Acceptance:** after any change, one command answers "is Billy learning faster than last
  week?" with a graph, not an anecdote.

---

## 5. Campaign board (near-term concrete moves)

| Priority | Task | Where | Phase | Status |
|---|---|---|---|---|
| ~~1~~ | ~~Teach SMB 3-4 @ x=525~~ | `remix.py` | — | ✅ done — frontier now 4-2 |
| ~~2~~ | ~~Extract Remix per-game hooks into `Game` contract~~ | `remix.py`, `billy/abstractions.py` | 1 | ✅ done (19d7f9d) |
| ~~3~~ | ~~Generalize approach capture beyond SMB~~ | `remix.py:_prepare_approach` | 2 | ✅ done (0754da2) |
| 1 | Teach SMB 4-2 @ x≈322 (wall is queued NOW — run the Remix shortcut) | human + `remix.py` | — | open |
| 2 | PSII: map battle RAM, wire non-spatial `remix_win` for fights | `billy/games/psii` | 3 | menu half done; battle RAM unmapped |
| 3 | Scout smb_lost from scratch; measure reflex/skill transfer | `run.py --game smb_lost`, metrics | 3/6 | barely started |
| 4 | Auto-train queued BC seeds offline | new worker + `train_section.py` | 4 | not started |
| 5 | Zelda: tape-evolution on combat screens (hooks already exist) | `billy/games/zelda` | 1/4 | not started |
| 6 | Fail-reel + medals + recap | `remix.py` | 5 | not started |
| 7 | metrics.jsonl + curves report | `billy/director.py`, `report.py` | 6 | 🟡 v1 shipped (2026-07-12) — curve 4 (cross-game A/B) still open |
| 8 | SMW bring-up when ROM lands | `billy/games/smw/STATUS.md` | 3 | blocked on ROM |

---

## 6. Ground rules for whoever picks this up

- Always run in the venv; always `BILLY_HEADLESS=1` for tests/scouting. ROMs and `data/` are
  gitignored — never commit them.
- The regression guard is sacred: SMB 1-1 must clear every attempt with the same compounding
  curve after any reflex/engine change.
- Interactive paths (teach, replay-payoff) cannot run in CI — unit-test the logic with injected
  fakes (see `tests/test_remix.py` `_FakeSession`/`_ReplaySession` patterns) and say honestly
  what was and wasn't exercised end-to-end.
- Advice sources (guide, LLM, skills) SEED; verification DECIDES. Nothing replays blind. Ever.
- Keep the human's surface simple: two desktop shortcuts, one scoreboard, walls that explain
  themselves. Complexity budget goes into Billy's learning, not the human's setup.
