"""TapeEvolver on a synthetic fitness (no emulator).

The evolver must (1) never regress below the base, and (2) actually climb toward a target the
fitness rewards — here, a tape whose steps match a 'safe' move and that is long enough. That
mirrors the real signal: survive longer (extend) while picking the right moves (retarget)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.abstractions import Step
from billy.knowledge.tape_evolve import TapeEvolver

SAFE = 0b0001        # the "good" move; others are "bad"
BAD = 0b0010
MOVES = [SAFE, BAD, 0b0100]


def _fitness(tape):
    """Reward length (survival) but only while steps are the SAFE move — a wrong move 'kills'
    the run there, so fitness = count of leading SAFE steps."""
    n = 0
    for s in tape:
        if s.buttons != SAFE:
            break
        n += 1
    return n


def test_evolve_never_regresses():
    base = [Step(6, SAFE), Step(6, SAFE)]           # base fitness 2
    ev = TapeEvolver(MOVES, seed=1)
    best, fit, evals = ev.evolve(base, _fitness, rounds=5, mutants=8)
    assert fit >= _fitness(base)
    assert evals >= 1


def test_evolve_climbs_toward_longer_safe_runs():
    base = [Step(6, SAFE)]                           # base fitness 1
    ev = TapeEvolver(MOVES, seed=3)
    best, fit, evals = ev.evolve(base, _fitness, rounds=40, mutants=12)
    # extending the tail with SAFE moves must lengthen the safe run well past the base
    assert fit > 3, f"evolver did not climb (fit={fit})"
    assert best[:fit] == [Step(s.frames, SAFE) for s in best[:fit]]


def test_mutate_respects_max_steps():
    ev = TapeEvolver(MOVES, slot=6, max_steps=10, seed=0)
    tape = [Step(6, SAFE)] * 10
    for _ in range(50):
        tape = ev.mutate(tape)
        assert len(tape) <= 10


def test_empty_move_vocab_rejected():
    try:
        TapeEvolver([])
    except ValueError:
        return
    assert False, "empty move vocabulary must raise"
