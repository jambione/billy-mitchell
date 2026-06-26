# Billy Mitchell — Next Steps

Roadmap as of the hazard-scoped RL sub-policy milestone (branch
`feat/pipe-entry-and-powerup-perception`, commit `9e364d6`).

## Where Billy stands now

- **Clears 1-1 and 1-2 every attempt** (score ~41k), via the compounding cache + reflex + invisible
  micro-search loop.
- **1-3:** crosses the tree-top platform-hop section (x≈215→730) using a **hazard-scoped RL
  sub-policy**. The crossing is verified on a clone, committed, and **banked in the cache** — so it
  replays like any other solution (`replay=4/search=4` on later passes) instead of re-searching.
- **Current 1-3 wall: x≈760, the moving-lift gap.** This is past the sub-policy's trained range
  (`goal_x=700`). It's the original "ride the lift" problem, now isolated as the next target.

Run it: `BILLY_HEADLESS=1 .venv/bin/python run.py --attempts 6 --no-llm --rl-sections`
(drop `BILLY_HEADLESS=1` to watch in a window). Train/eval the sub-policy: `train_section.py` /
`eval_section.py`. Models live under `data/rl/` (gitignored — reproduce from the train script).

## 1. Finish 1-3 — the lift-gap sub-policy (direct next step)

The section framework is fully parameterized, so this is a clean repeat of what already works:

1. **Capture a savestate** on the long platform just before the lift gap (~x=700), the same way
   `data/rl/states/smb_1_3_section.state` was made (drive Billy there, `session.clone_state()`).
2. **Train a second sub-policy** with `train_section.py --state <new.state> --goal-x 950` (goal past
   the lift, on the next solid ground). The lift (object id `0x25`) already appears in the RL
   observation's enemy channel, so the policy can learn to time boarding/riding/dismounting it.
   Watch `cross_rate` climb; if it stalls in the risk-averse idle optimum, add milestone bonuses at
   the lift-board and dismount x's (see `SectionEnv.milestones`).
3. **Register it**: add one `Section(label="1-3", x_lo=700, x_hi=860, goal_x=950, model_path=...)`
   to `default_smb_sections()` in `billy/rl/section_policy.py`.
4. **Verify** Billy crosses x=760 end-to-end and the frontier advances; the crossing should bank and
   compound. That should **clear 1-3** (reach the flagpole) if no further wall exists past the lift.

Risk: the lift is a *moving* platform — timing-sensitive. If a single savestate overfits, train from
a few savestates at different lift phases (the start-randomization trick generalized).

## 2. World 1-4 (Bowser castle) → reach 2-1

Clearing 1-3 lands Billy in 1-4: firebars, lava pits, the Bowser/axe finish. Expect new hazards the
reflex can't chain → likely one or two more section sub-policies (firebar timing, the bridge). Goal:
clear 1-4 to reach **2-1**, completing World 1.

## 3. Harden + automate the section framework (infra)

- **Auto-propose sub-policies:** when the stall-breaker marks a spot dead-end repeatedly, emit a
  "train a sub-policy here" signal (savestate + x-range) instead of just giving up. This closes the
  loop from "Billy is stuck" → "Billy trains himself past it."
- **Savestate capture CLI:** a small command to record a hazard savestate at Billy's current spot,
  so adding a section doesn't need a bespoke probe script.
- **Multi-savestate / domain randomization** baked into `SectionEnv` for robustness, beyond the
  current cruise-forward start randomization.

## 4. Cross-level / cross-game transfer (longer horizon)

- **Distill banked crossings into the Skill layer** (`knowledge/skills.py`) so a learned maneuver
  (e.g. "precise chained platform hops") seeds candidates at *similar* hazards in other levels/games,
  not just its exact bucket. This is the lever that turns per-spot wins into general competence.
- The shared platformer reflex already carries to SMB2-Japan with no new code; section sub-policies
  should transfer the same way once distilled.

## Honest open questions

- Does the lift gap (#1) need true moving-platform riding, or is there a static jump line? Confirm by
  probing the geometry + lift `0x25` trajectory before training (don't assume).
- Within-level compounding is still partial (moving enemies re-search each pass; see CLAUDE.md
  "Known limits"). Section crossings bank and replay, but enemy-dense spots still cost live search.
- Reaching 2-1 is several sub-policies of work; each is cheap *given* the framework, but the lift and
  Bowser are genuinely harder than the tree-top hops.
