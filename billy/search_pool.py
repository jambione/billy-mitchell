"""Parallel micro-search — N emulator workers evaluate candidates concurrently.

stable-retro allows ONE emulator per process, so parallel candidate evaluation means worker
subprocesses, each hosting its own headless session for the same game. A job ships the cloned
state bytes plus a chunk of candidate plans; workers restore the state and run the SAME
`rollout_candidate` code the serial path uses, returning (survived, reached, end_x, end_y)
per candidate. Scoring / route-awareness / banking all stay in the parent — workers are pure
evaluators.

Honesty notes:
 - OFF by default (BILLY_PARALLEL_SEARCH=<n_workers> to enable). Workers coast candidates
   with a FRESH reflex instance; the serial path coasts with the live reflex mid-attempt,
   whose small internal state (stuck counters, last scene) can occasionally pick a different
   coast plan — so scores can differ from serial in rare edge cases. Anything committed still
   passes the same survive+advance gates, so correctness is unaffected; the regression-guard
   curve is only guaranteed with the flag off.
 - Worth it at wide candidate sets (skill-seeded searches, the ~40-candidate expanded grid);
   below `MIN_CANDIDATES` the pickling/IPC overhead beats the win and the serial path runs.
 - Any worker failure falls back to serial for that search (never crashes an attempt).
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os

from .abstractions import Plan

MIN_CANDIDATES = 6      # below this, serial is faster than shipping state to workers

_G: dict = {}           # per-worker singletons (session/game/reflex), set by _init_worker


def _init_worker(factory: str, env: dict) -> None:
    os.environ.update(env)
    os.environ["BILLY_HEADLESS"] = "1"
    os.environ["BILLY_TURBO"] = "1"
    mod_name, cls_name = factory.split(":")
    game = getattr(importlib.import_module(mod_name), cls_name)()
    session = game.system.connect()
    session.wait_until_live()
    _G["game"] = game
    _G["session"] = session
    _G["reflex"] = game.make_reflex()

    def observe():
        st = session.read_state()
        return game.observe(st.frame, st.ram, getattr(st, "rgb", None))

    _G["observe"] = observe


def _eval_chunk(job: tuple) -> list[tuple]:
    """Evaluate a chunk of candidate plans from one cloned state. Runs in a worker."""
    from .director import rollout_candidate

    state, plans, settle, min_progress = job
    session, game, reflex, observe = (_G["session"], _G["game"],
                                      _G["reflex"], _G["observe"])
    results: list[tuple] = []
    with session.search_mode():
        for plan in plans:
            session.restore(state)
            observe()
            results.append(rollout_candidate(session, observe, reflex, game,
                                             plan, settle, min_progress=min_progress))
    return results


def split_chunks(items: list, n: int) -> list[list]:
    """Round-robin-free contiguous split preserving order across concatenation."""
    n = max(1, min(n, len(items)))
    size, rem = divmod(len(items), n)
    out, i = [], 0
    for k in range(n):
        take = size + (1 if k < rem else 0)
        out.append(items[i:i + take])
        i += take
    return [c for c in out if c]


class SearchPool:
    """Lazy pool of emulator workers. Created once per Director; workers boot on first use."""

    def __init__(self, game, n_workers: int) -> None:
        cls = type(game)
        self._factory = f"{cls.__module__}:{cls.__qualname__}"
        self._env = {k: v for k, v in os.environ.items() if k.startswith("BILLY_")}
        self.n_workers = max(1, n_workers)
        self._pool: mp.pool.Pool | None = None

    def _ensure(self) -> mp.pool.Pool:
        if self._pool is None:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(self.n_workers, initializer=_init_worker,
                                  initargs=(self._factory, self._env))
            print(f"[search-pool] {self.n_workers} emulator workers up "
                  f"(BILLY_PARALLEL_SEARCH)")
        return self._pool

    def evaluate(self, state: bytes, plans: list[Plan], settle: int,
                 min_progress: int) -> list[tuple] | None:
        """Evaluate all candidates in parallel; results align with `plans` order.
        Returns None on any failure — the caller falls back to the serial path."""
        if len(plans) < MIN_CANDIDATES:
            return None
        try:
            pool = self._ensure()
            chunks = split_chunks(plans, self.n_workers)
            jobs = [(state, chunk, settle, min_progress) for chunk in chunks]
            per_chunk = pool.map(_eval_chunk, jobs)
            flat: list[tuple] = []
            for res in per_chunk:
                flat.extend(res)
            return flat if len(flat) == len(plans) else None
        except Exception as e:
            print(f"[search-pool] parallel evaluate failed ({type(e).__name__}: {e}) — "
                  f"serial fallback")
            self.close()
            return None

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
