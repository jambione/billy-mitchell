"""RAM perception for The Legend of Zelda (NES PRG0).

Addresses mirror stable-retro's experimental LegendOfZeldaPRG0-Nes integration
(data.json). All decoding lives here so it is unit-testable against captured RAM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .items import GroundItem, nearest_item, read_ground_items
from .tuning import SCREEN_EDGE_HI, SCREEN_EDGE_LO

# Normal play + scroll/transition modes (4/10/16) and dungeon (9/12).
IN_PLAY_MODES = (4, 5, 7, 9, 10, 11, 12, 16)
CAVE_INTERIOR_MODES = (11, 16)
DUNGEON_MODES = (9, 12)
N_ENEMIES = 6


def _u8(ram: bytes, addr: int) -> int:
    return ram[addr]


@dataclass(frozen=True)
class Enemy:
    slot: int
    x: int
    y: int
    enemy_type: int


@dataclass
class Scene:
    frame: int
    link_x: int
    link_y: int
    direction: int
    game_mode: int
    current_level: int      # 0 = overworld, 1-8 = dungeon
    map_location: int
    next_location: int
    health: int             # hearts remaining (lower nibble of 0x0676)
    max_hearts: int
    partial_heart: int
    triforce_pieces: int
    sword_level: int
    rupees: int
    keys: int
    bombs: int
    scrolling: bool
    visited_screens: tuple[int, ...]
    cave_mouths: tuple[tuple[int, int], ...] = ()   # link-space targets from vision
    enemies: list[Enemy] = field(default_factory=list)
    items: list[GroundItem] = field(default_factory=list)
    ram: bytes = b""
    dungeon: object = None   # DungeonState | None when in_dungeon
    # High object slots (7..15) of the NES object table — where monster SHOTS (octorok rocks,
    # arrows) live. Kept raw as (slot, x, y); the reflex tracks them frame-to-frame to find the
    # ones moving fast+straight toward Link (an incoming projectile) and dodge. See
    # probe_zelda_projectiles.py for the mapping.
    objects: tuple[tuple[int, int, int], ...] = ()

    def object_positions(self) -> dict[int, tuple[int, int]]:
        """{slot: (x, y)} for the active high-slot objects (candidate projectiles)."""
        return {slot: (x, y) for slot, x, y in self.objects}

    @property
    def realm(self) -> str:
        return "overworld" if self.current_level == 0 else f"dungeon-{self.current_level}"

    @property
    def room_label(self) -> str:
        return f"{self.realm} #{self.map_location}"

    @property
    def in_play(self) -> bool:
        return self.game_mode in IN_PLAY_MODES and self.health > 0

    @property
    def in_cave(self) -> bool:
        return self.game_mode in CAVE_INTERIOR_MODES

    @property
    def is_dying(self) -> bool:
        return self.health == 0

    @property
    def full_health(self) -> bool:
        """Every heart full — the state in which Link's stab fires a screen-length sword beam.

        The heart byte (0x066F) stores the filled-heart count 0-indexed (3/3 hearts → health 2,
        max_hearts 3), and 0x0670 the current heart's partial fill (0xFF = full). So 'completely
        full' is the top heart count AND a full partial."""
        return self.health >= self.max_hearts - 1 and self.partial_heart >= 0xFF

    @property
    def in_dungeon(self) -> bool:
        return self.current_level > 0 or self.game_mode in DUNGEON_MODES

    @property
    def at_right_edge(self) -> bool:
        return self.link_x >= SCREEN_EDGE_HI

    @property
    def at_left_edge(self) -> bool:
        return self.link_x <= SCREEN_EDGE_LO

    @property
    def at_top_edge(self) -> bool:
        return self.link_y <= 48

    @property
    def at_bottom_edge(self) -> bool:
        return self.link_y >= 200

    def enemy_ahead(self, within: int = 48) -> bool:
        return self.nearest_enemy(within=within) is not None

    def nearest_enemy(self, within: int = 48) -> tuple[int, int] | None:
        """Return (dx, dy) to the closest on-screen enemy within `within` pixels."""
        best: tuple[int, int] | None = None
        best_dist = within + 1
        for e in self.enemies:
            dx = e.x - self.link_x
            dy = e.y - self.link_y
            dist = abs(dx) + abs(dy)
            if dist < best_dist:
                best_dist = dist
                best = (dx, dy)
        return best

    def nearest_ground_item(self, within: int = 96) -> tuple[int, int, GroundItem] | None:
        return nearest_item(self.items, self.link_x, self.link_y, within=within)

    def enemy_count(self) -> int:
        return len(self.enemies)

    def item_count(self) -> int:
        return len(self.items)

    def objective_score(self) -> int:
        """Monotonic frontier signal for cache / metrics (higher = more progress).

        Uses max_hearts (not current health) so combat damage does not shrink progress
        and break learn-from-death runway on east-march screens."""
        visited = len(self.visited_screens)
        score = (
            visited * 512
            + self.triforce_pieces * 2048
            + self.sword_level * 1024
            + self.keys * 128
            + self.max_hearts * 16
            + (10 if self.in_dungeon else 0)
            + self.rupees * 2
            + self.map_location
            + self.link_x
        )
        if self.in_cave:
            score += 256 + (220 - self.link_y)
        return score

    def summary(self) -> str:
        from .curiosity import next_curious_dest, requires_start_cave_inspection
        from .walkthrough import phase_summary

        sword = ("none", "wood", "white", "magic")[min(self.sword_level, 3)]
        visited = set(self.visited_screens)
        if requires_start_cave_inspection(
                self.map_location, visited, sword_level=self.sword_level,
                in_cave=self.in_cave):
            curious_bit = " inspecting-nw-cave"
        elif self.cave_mouths:
            cx, cy = self.cave_mouths[0]
            curious_bit = f" cave■@({cx},{cy})"
        elif (curious := next_curious_dest(self.map_location, visited)) is not None:
            curious_bit = f" curious→#{curious}"
        else:
            curious_bit = ""
        faq = phase_summary(
            map_location=self.map_location,
            sword_level=self.sword_level,
            max_hearts=self.max_hearts,
            visited=visited,
            in_cave=self.in_cave,
        )
        from .dungeon import dungeon_summary
        dung = dungeon_summary(self.dungeon) if self.dungeon else ""
        dung_bit = f" {dung}" if dung else ""
        return (f"{self.room_label} link=({self.link_x},{self.link_y}) "
                f"hearts={self.health}/{self.max_hearts} rupees={self.rupees} "
                f"sword={sword} triforce={self.triforce_pieces}/8 "
                f"visited={len(self.visited_screens)} enemies={self.enemy_count()} "
                f"items={self.item_count()} {faq}{dung_bit}{curious_bit}")

    def ascii_view(self) -> str:
        """Tiny 9x7 grid centred on Link for the LLM."""
        cols, rows = 9, 7
        grid = [["." for _ in range(cols)] for _ in range(rows)]
        cx, cy = cols // 2, rows // 2
        grid[cy][cx] = "L"
        for mx, my in self.cave_mouths:
            ex = cx + (mx - self.link_x) // 16
            ey = cy + (my - self.link_y) // 16
            if 0 <= ex < cols and 0 <= ey < rows:
                grid[ey][ex] = "#"
        for item in self.items:
            ex = cx + (item.x - self.link_x) // 16
            ey = cy + (item.y - self.link_y) // 16
            if 0 <= ex < cols and 0 <= ey < rows:
                grid[ey][ex] = "I"
        for e in self.enemies:
            ex = cx + (e.x - self.link_x) // 16
            ey = cy + (e.y - self.link_y) // 16
            if 0 <= ex < cols and 0 <= ey < rows and grid[ey][ex] == ".":
                grid[ey][ex] = "E"
        return "\n".join("".join(row) for row in grid)


def _read_enemies(ram: bytes, drops: list[GroundItem]) -> list[Enemy]:
    drop_slots = {item.slot - 1 for item in drops}
    types = [_u8(ram, 848 + i) for i in range(N_ENEMIES)]
    out: list[Enemy] = []
    for i in range(N_ENEMIES):
        if i in drop_slots:
            continue
        etype = types[i]
        if etype == 0:
            continue
        x = _u8(ram, 113 + i)
        y = _u8(ram, 133 + i)
        if x == 0 and y == 0:
            continue
        out.append(Enemy(slot=i + 1, x=x, y=y, enemy_type=etype))
    return out


# NES object table: X at 0x70+slot, Y at 0x84+slot (slot 0 = Link, 1..6 = the enemies above).
# Monster SHOTS occupy the HIGH slots (7..15) — see probe_zelda_projectiles.py.
_OBJ_X_BASE, _OBJ_Y_BASE = 0x70, 0x84


def _read_objects(ram: bytes) -> tuple[tuple[int, int, int], ...]:
    """Active high-slot objects (slot, x, y) — candidate projectiles the reflex tracks by velocity."""
    out: list[tuple[int, int, int]] = []
    for slot in range(7, 16):
        x = _u8(ram, _OBJ_X_BASE + slot)
        y = _u8(ram, _OBJ_Y_BASE + slot)
        if x == 0 and y == 0:
            continue
        out.append((slot, x, y))
    return tuple(out)


def _visited_screens(ram: bytes) -> tuple[int, ...]:
    hist = tuple(_u8(ram, 1569 + i) for i in range(5))
    return tuple(s for s in hist if s)


def build_scene(ram: bytes, frame: int = 0, rgb=None) -> Scene:
    hearts_byte = _u8(ram, 1647)
    health = hearts_byte & 0x0F
    max_hearts = ((hearts_byte >> 4) & 0x0F) + 1
    cave_mouths: tuple[tuple[int, int], ...] = ()
    if rgb is not None:
        from .vision import detect_cave_mouths
        cave_mouths = tuple(detect_cave_mouths(rgb))

    items = read_ground_items(ram)
    current_level = _u8(ram, 16)
    keys_held = _u8(ram, 1646)
    dungeon = None
    if current_level > 0:
        from .dungeon import read_dungeon_state
        dungeon = read_dungeon_state(ram, room_id=_u8(ram, 235), keys_held=keys_held)

    return Scene(
        frame=frame,
        link_x=_u8(ram, 112),
        link_y=_u8(ram, 132),
        direction=_u8(ram, 152),
        game_mode=_u8(ram, 18),
        current_level=current_level,
        map_location=_u8(ram, 235),
        next_location=_u8(ram, 236),
        health=health,
        max_hearts=max_hearts,
        partial_heart=_u8(ram, 1648),
        triforce_pieces=_u8(ram, 1649),
        sword_level=_u8(ram, 1623),
        rupees=_u8(ram, 1645),
        keys=keys_held,
        bombs=_u8(ram, 1624),
        scrolling=_u8(ram, 232) != 255,
        visited_screens=_visited_screens(ram),
        cave_mouths=cave_mouths,
        enemies=_read_enemies(ram, items),
        items=items,
        ram=ram,
        dungeon=dungeon,
        objects=_read_objects(ram),
    )