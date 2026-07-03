"""Phantasy Star II reflex — four-way exploration for a top-down RPG.

Field mode: hold a heading; when the party stops making ground (wall), rotate. A stuck spot
occasionally gets an interact tap (logical A = physical C: talk / open a door / advance a
scripted dialog) — but A with nothing ahead opens the command menu, so any observed open
menu is immediately cancelled with B. No battle mode yet: battle-flag/party RAM is deferred
until exploration reaches the overworld (see games/psii/ram_map.py).
"""
from __future__ import annotations

from ...abstractions import Decision, Observation, Plan, ReflexPolicy, Step
from ...systems.genesis import controller as C

_HEADINGS = (C.DOWN, C.LEFT, C.UP, C.RIGHT)


class PsiiReflex(ReflexPolicy):
    def __init__(self) -> None:
        self._heading = 0
        self._last_tile: tuple | None = None
        self._stalls = 0

    def reset(self, obs: Observation) -> None:
        self._heading = 0
        self._last_tile = None
        self._stalls = 0

    def note_level_advance(self, obs: Observation) -> None:
        self._last_tile = None
        self._stalls = 0

    def step(self, obs: Observation) -> Decision:
        s = obs.raw
        if s is not None and s.menu_open:
            return Decision(plan=[Step(8, C.B), Step(20, 0)], note="close menu")
        tile = s.tile if s is not None else None
        if tile is not None and tile == self._last_tile:
            self._stalls += 1
            if self._stalls % 4 == 3:
                # Third strike on a wall: try interacting (door / NPC / dialog advance).
                return Decision(plan=[Step(8, C.A), Step(20, 0)], note="interact")
            self._heading = (self._heading + 1) % 4      # rotate off the wall
        else:
            self._stalls = 0
        self._last_tile = tile
        return Decision(plan=[Step(24, _HEADINGS[self._heading])], note="explore")

    def advance_plan(self, obs: Observation) -> Plan:
        return [Step(16, _HEADINGS[self._heading])]

    def danger_candidates(self, obs: Observation) -> list[Plan]:
        out: list[Plan] = [[Step(32, d)] for d in _HEADINGS]
        out.append([Step(8, C.A), Step(24, 0)])                    # interact / advance text
        out.append([Step(8, C.B), Step(24, 0)])                    # cancel / close
        for d in _HEADINGS:
            out.append([Step(16, d), Step(8, C.A), Step(16, 0)])   # step up + interact
        return out

    def expanded_candidates(self, obs: Observation) -> list[Plan]:
        out = self.danger_candidates(obs)
        for a in _HEADINGS:
            for b in _HEADINGS:
                if a != b:
                    out.append([Step(24, a), Step(24, b)])          # two-leg dodges/corners
        return out
