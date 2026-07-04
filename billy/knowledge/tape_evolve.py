"""Tape evolution — search over WHOLE input trajectories, not local escapes.

The cache/learn-from-death tiers key on position and search a LOCAL escape just before a death.
That fails a reactive game (a shmup, a moving-enemy gauntlet) where dying is a POSITIONING
problem that began much earlier: the fix is to be somewhere else when the threat arrives, which
means changing earlier input. The emulator is deterministic from a fixed anchor, so a whole
input tape reproduces exactly — which makes a tape a searchable genome. This is a (1+λ)
hill-climb over that genome: mutate the best tape, keep any mutant that reaches a higher fitness
(survival + score), bank it, repeat. Compounding without any position key.

Pure and emulator-free: the caller supplies `evaluate(tape) -> fitness` (roll the tape from the
anchor on a clone, return progress at death/end). That keeps this unit-testable on a synthetic
fitness and keeps the emulator plumbing in the Director.
"""
from __future__ import annotations

import random
from collections.abc import Callable

from ..abstractions import Step

Tape = list


class TapeEvolver:
    """(1+λ) evolutionary hill-climb over an input tape.

    `moves` is the game's mutation vocabulary — the button masks a movement window may take
    (e.g. fire+left, fire+right, fire+up, fire). `slot` is the frame length of a freshly minted
    step (match the reflex's commit chunk so mutated and recorded tapes splice cleanly)."""

    def __init__(self, moves: list[int], *, slot: int = 6, max_steps: int = 400, seed: int = 0):
        if not moves:
            raise ValueError("TapeEvolver needs a non-empty move vocabulary")
        self.moves = list(moves)
        self.slot = slot
        self.max_steps = max_steps
        self.rng = random.Random(seed)

    # --- mutation -------------------------------------------------------------------------
    def mutate(self, tape: Tape) -> Tape:
        """One random edit: EXTEND the tail (survive further — the key move for a base that
        dies), RETARGET a window (be elsewhere when a threat arrives), or INSERT a burst (shift
        the timing of everything after it)."""
        t = [Step(s.frames, s.buttons) for s in tape]
        r = self.rng.random()
        if not t or r < 0.45:
            for _ in range(self.rng.randint(2, 8)):
                if len(t) >= self.max_steps:
                    break
                t.append(Step(self.slot, self.rng.choice(self.moves)))
        elif r < 0.85:
            i = self.rng.randrange(len(t))
            j = min(len(t), i + self.rng.randint(1, 6))
            mv = self.rng.choice(self.moves)
            for k in range(i, j):
                t[k] = Step(t[k].frames, mv)
        elif len(t) < self.max_steps:
            i = self.rng.randrange(len(t) + 1)
            t.insert(i, Step(self.slot, self.rng.choice(self.moves)))
        return t

    # --- the hill-climb -------------------------------------------------------------------
    def evolve(self, base: Tape, evaluate: Callable[[Tape], int], *,
               rounds: int, mutants: int) -> tuple[Tape, int, int]:
        """Improve `base` for `rounds` generations of `mutants` each. Returns
        (best_tape, best_fitness, evaluations). The base itself is evaluated once so a round
        that finds nothing better can't regress."""
        best = [Step(s.frames, s.buttons) for s in base]
        best_fit = evaluate(best)
        evals = 1
        for _ in range(rounds):
            improved = False
            for _ in range(mutants):
                cand = self.mutate(best)
                fit = evaluate(cand)
                evals += 1
                if fit > best_fit:
                    best, best_fit, improved = cand, fit, True
            if not improved:
                # Stuck at this optimum — widen by mutating harder next round (the rng keeps
                # exploring; no state to reset). Continue to spend the budget.
                continue
        return best, best_fit, evals
