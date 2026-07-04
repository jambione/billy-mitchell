"""Phantasy Star II perception — RAM bytes to a Scene the reflex/engine can reason about."""
from __future__ import annotations

from dataclasses import dataclass

from . import ram_map as M


@dataclass
class Scene:
    frame: int
    x: int               # lead character map position, pixels
    y: int
    place: tuple         # comparable place identity (not semantically decoded)
    outdoor: bool
    menu_open: bool

    @property
    def in_play(self) -> bool:
        # Transition fades zero the place bytes — a (0, 0) place is BETWEEN maps, not a map
        # (banking it would flap level_key through every doorway). Coord sanity bounds out
        # garbage frames.
        return (self.place[0] != 0 or self.place[1] != 0) \
            and 0 <= self.x < 4096 and 0 <= self.y < 4096

    @property
    def tile(self) -> tuple[int, int]:
        """Coarse 16px map tile — the unit of exploration credit."""
        return (self.x // 16, self.y // 16)

    def summary(self) -> str:
        where = "outdoors" if self.outdoor else "indoors"
        return (f"PSII place {self.place} {where} at tile {self.tile}"
                f"{', menu open' if self.menu_open else ''}")


def build_scene(ram: bytes, frame: int) -> Scene:
    return Scene(
        frame=frame,
        x=M.u16(ram, M.PLAYER_X_HI),
        y=M.u16(ram, M.PLAYER_Y_HI),
        place=(ram[M.PLACE_A], ram[M.PLACE_B], ram[M.OUTDOOR]),
        outdoor=bool(ram[M.OUTDOOR]),
        menu_open=bool(ram[M.MENU_OPEN]),
    )
