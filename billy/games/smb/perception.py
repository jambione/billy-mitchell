"""RAM perception: turn 2KB of NES work RAM into a structured Scene.

The address map and the tile/sprite readers are ported from SethBling's MarI/O (the proven,
canonical way to read Super Mario Bros state). All decoding lives here in Python so it's unit
testable against captured RAM dumps — the Lua bridge stays dumb.

References (SMB work-RAM map):
  0x0057 player x-velocity (signed)        0x000E player state (0x06/0x0B = dying)
  0x006D player x page    0x0086 player x  0x001D player float/air state (0=on ground)
  0x03B8 player y (screen) 0x00B5 player y viewport (>1 => fell into a pit)
  0x0756 player size (0 small,1 big,2 fire) 0x075A lives  0x075E coins
  0x075F world  0x075C stage  0x0760 area   0x07F8-0x07FA time   0x07DD-0x07E2 score
  enemies: alive 0x000F+slot, x page 0x006E+slot, x 0x0087+slot, y 0x00CF+slot  (slots 0-4)
  level tiles: 0x0500 region, read via tile_at() (page*13*16 + suby*16 + subx)
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Grid window (in 16px tiles) rendered around Mario for the LLM/debug view.
GRID_ROWS = 13          # the 13 tile rows of the play area (suby 0..12)
COL_BEHIND = 2          # tiles shown behind Mario
COL_AHEAD = 10          # tiles shown ahead of Mario
DYING_STATES = (0x06, 0x0B)
POWERUP_ID = 0x2E       # Enemy_ID of a power-up object (mushroom/flower) — a collectible, NOT a threat
N_OBJECT_SLOTS = 6      # SMB has 6 object slots (0-4 enemies, 5 = special/power-up slot)


def _u8(ram: bytes, addr: int) -> int:
    return ram[addr]


def _s8(ram: bytes, addr: int) -> int:
    v = ram[addr]
    return v - 256 if v >= 128 else v


def _digits(ram: bytes, start: int, n: int) -> int:
    """Decode SMB's one-decimal-digit-per-byte fields (time, score)."""
    val = 0
    for i in range(n):
        val = val * 10 + (ram[start + i] % 10)
    return val


def tile_at(ram: bytes, x_world: int, y_world: int) -> int:
    """Tile id at an absolute level position (0 = empty). Ported from MarI/O getTile."""
    page = (x_world // 256) % 2
    sub_x = (x_world % 256) // 16
    sub_y = (y_world - 32) // 16
    if sub_y < 0 or sub_y > 12:
        return 0
    addr = 0x500 + page * 13 * 16 + sub_y * 16 + sub_x
    return ram[addr] if 0 <= addr < len(ram) else 0


@dataclass(frozen=True)
class Enemy:
    slot: int
    x: int  # absolute level x
    y: int  # screen y


@dataclass
class Scene:
    frame: int
    # progress / identity
    mario_x: int
    mario_y: int
    x_speed: int
    world: int
    stage: int
    area: int
    lives: int
    coins: int
    score: int
    time: int
    size: int          # 0 small, 1 big, 2 fire
    player_state: int
    float_state: int   # 0 = on ground
    y_viewport: int
    enemies: list[Enemy] = field(default_factory=list)
    powerups: list[Enemy] = field(default_factory=list)   # collectible power-ups (mushroom/flower)
    tiles: list[list[int]] = field(default_factory=list)  # GRID_ROWS x window, 1=solid
    ram: bytes = b""   # raw snapshot, kept so gap checks probe the true level

    # --- derived flags ------------------------------------------------------------------
    @property
    def on_ground(self) -> bool:
        return self.float_state == 0

    @property
    def is_dying(self) -> bool:
        return self.player_state in DYING_STATES or self.y_viewport > 1

    @property
    def in_play(self) -> bool:
        """False on death / level-transition / reset frames, where RAM positions read garbage
        (world-x and world/stage flip to 0xFF -> the old x=65535 and '256-256' bugs). On those
        frames the engine must NOT trust mario_x / world / stage."""
        if self.world > 7 or self.stage > 3:        # implausible level id -> transition frame
            return False
        if self.mario_x >= 0x4000:                  # no single area is this long -> overflow read
            return False
        return True

    @property
    def world_stage(self) -> str:
        return f"{self.world + 1}-{self.stage + 1}"

    def pipe_entry_spot(self, max_tiles: int = 2) -> int | None:
        """Mario can enter a vertical pipe here (1-2 exit, warp zone → 4/5/6). Returns px to
        align (0 = centred); None if no pipe mouth detected. x may stall while ducking — not stuck."""
        if not self.on_ground:
            return None
        above = self.block_above_ahead(max_tiles=max_tiles)
        if above is not None:
            return above
        wall = self.obstacle_ahead(max_tiles=1)
        if wall and wall[0] <= 16 and wall[1] >= 2:
            return wall[0]
        return None

    def block_above_ahead(self, max_tiles: int = 2) -> int | None:
        """A floating, bonkable block (? block or brick) just ahead: a solid tile up at head
        height with open space beneath it (so it's not a wall/pipe). Returns dist_px or None.
        ID-free so it survives ROM/tile-map quirks. Bonking yields coins / power-ups."""
        for dt in range(0, max_tiles + 1):
            x = self.mario_x + dt * 16
            below_open = (tile_at(self.ram, x, self.mario_y) == 0
                          and tile_at(self.ram, x, self.mario_y - 16) == 0)
            block_up = (tile_at(self.ram, x, self.mario_y - 32) != 0
                        or tile_at(self.ram, x, self.mario_y - 48) != 0)
            if below_open and block_up:
                return max(0, dt * 16 - (self.mario_x % 16))
        return None

    def obstacle_ahead(self, max_tiles: int = 3) -> tuple[int, int] | None:
        """A wall/pipe to jump OVER: solid tile(s) at Mario's body height just ahead.
        Returns (dist_px, height_tiles). None if the path is clear. Body-height probing
        ignores the floating '?' blocks above his head (those don't block movement)."""
        for dt in range(1, max_tiles + 1):
            x = self.mario_x + dt * 16
            if tile_at(self.ram, x, self.mario_y) != 0:  # solid at body level => a wall
                height = 0
                for k in range(4):
                    if tile_at(self.ram, x, self.mario_y - k * 16) != 0:
                        height += 1
                    else:
                        break
                return (max(0, dt * 16 - (self.mario_x % 16)), height)
        return None

    def enemy_ahead(self, within: int = 48) -> bool:
        """Any living enemy just in front of Mario within `within` pixels."""
        return self.nearest_enemy_ahead(within) is not None

    def nearest_enemy_ahead(self, within: int = 64, y_above: int = 24,
                            y_below: int = 72) -> int | None:
        """Distance (px) to the closest stompable enemy ahead. The vertical band leans
        DOWNWARD (you stomp from above), so it also catches Koopas on lower ground while
        Mario is up on a platform. None if clear."""
        ds = [e.x - self.mario_x for e in self.enemies
              if 0 < (e.x - self.mario_x) <= within and -y_above <= (e.y - self.mario_y) <= y_below]
        return min(ds) if ds else None

    def nearest_enemy(self, within: int = 72, y_above: int = 24,
                      y_below: int = 72) -> tuple[int, int] | None:
        """The closest stompable enemy ahead as (dx, dy) relative to Mario. dy>0 means the
        enemy is BELOW him (e.g. on lower ground while Mario's on a platform). None if clear."""
        cands = [(e.x - self.mario_x, e.y - self.mario_y) for e in self.enemies
                 if 0 < (e.x - self.mario_x) <= within and -y_above <= (e.y - self.mario_y) <= y_below]
        return min(cands) if cands else None

    def nearest_powerup(self, within: int = 80) -> tuple[int, int] | None:
        """Closest collectible power-up as (dx, dy) relative to Mario — ahead or just behind (a
        mushroom can bounce back past him), within reach. None if there's nothing to grab."""
        cands = [(p.x - self.mario_x, p.y - self.mario_y) for p in self.powerups
                 if -40 <= (p.x - self.mario_x) <= within]
        return min(cands, key=lambda d: abs(d[0])) if cands else None

    def air_landing_target(self) -> int | None:
        """Absolute x Mario should aim to land on while airborne — the nearest enemy to stomp.
        None when there's nothing to aim at (just carry forward)."""
        near = self.nearest_enemy(within=96, y_above=56, y_below=96)
        return self.mario_x + near[0] if near is not None else None

    def gap_ahead(self, lookahead_tiles: int = 3) -> bool:
        """Is the floor missing in the next few tiles (a pit to jump)?"""
        return self.gap_info(max_tiles=lookahead_tiles) is not None

    def _is_pit_column(self, x: int, depth: int = 4) -> bool:
        """True only for a real DEATH pit: empty floor with no ground for several rows below.
        This distinguishes a lethal pit from a harmless step-down off a pipe/block (which has
        solid ground a row or two beneath)."""
        floor_y = self.mario_y + 16
        for k in range(depth + 1):
            if tile_at(self.ram, x, floor_y + k * 16) != 0:
                return False
        return True

    def gap_info(self, max_tiles: int = 8) -> tuple[int, int] | None:
        """Geometry of the next death pit ahead, for precise jump timing.

        Returns (dist_px, width_tiles): pixels from Mario to the pit's near edge, and how
        many tiles wide it is. None if there's solid ground ahead (including elevated terrain
        you can simply walk off). Probes the true level via RAM, scanning downward so pipe
        edges aren't mistaken for pits.
        """
        first_gap = next((dt for dt in range(1, max_tiles + 1)
                          if self._is_pit_column(self.mario_x + dt * 16)), None)
        if first_gap is None:
            return None
        width = 0
        for dt in range(first_gap, max_tiles + 1):
            if self._is_pit_column(self.mario_x + dt * 16):
                width += 1
            else:
                break
        dist_px = first_gap * 16 - (self.mario_x % 16)
        return (max(0, dist_px), width)

    # --- views for the LLM / logs -------------------------------------------------------
    def ascii_view(self) -> str:
        """Small ASCII map: '#' solid, ' ' empty, 'M' Mario, 'E' enemy."""
        rows = [list(" " if t == 0 else "#" for t in row) for row in self.tiles]
        m_col = COL_BEHIND
        m_row = max(0, min(GRID_ROWS - 1, (self.mario_y - 32) // 16))
        for e in self.enemies:
            col = round((e.x - self.mario_x) / 16) + COL_BEHIND
            row = max(0, min(GRID_ROWS - 1, (e.y - 32) // 16))
            if 0 <= col < (COL_BEHIND + COL_AHEAD + 1):
                rows[row][col] = "E"
        if 0 <= m_col < (COL_BEHIND + COL_AHEAD + 1):
            rows[m_row][m_col] = "M"
        return "\n".join("".join(r) for r in rows)

    def summary(self) -> str:
        """One compact, token-cheap line of game facts for the LLM prompt."""
        sz = {0: "small", 1: "big", 2: "fire"}.get(self.size, "?")
        ens = ", ".join(f"+{e.x - self.mario_x}px" for e in self.enemies) or "none"
        return (f"world {self.world_stage} x={self.mario_x} y={self.mario_y} "
                f"vx={self.x_speed} {sz} lives={self.lives} time={self.time} "
                f"on_ground={self.on_ground} enemies_ahead=[{ens}]")


def build_scene(ram: bytes, frame: int) -> Scene:
    """Decode a full Scene from a 2KB RAM snapshot."""
    mario_x = _u8(ram, 0x6D) * 256 + _u8(ram, 0x86)
    mario_y = _u8(ram, 0x03B8) + 16

    enemies: list[Enemy] = []
    powerups: list[Enemy] = []
    for slot in range(N_OBJECT_SLOTS):
        if _u8(ram, 0x0F + slot) == 0:
            continue
        eid = _u8(ram, 0x16 + slot)                      # Enemy_ID (object type)
        ex = _u8(ram, 0x6E + slot) * 256 + _u8(ram, 0x87 + slot)
        ey = _u8(ram, 0xCF + slot) + 24
        if eid == POWERUP_ID:                            # a mushroom/flower to COLLECT, not avoid
            powerups.append(Enemy(slot=slot, x=ex, y=ey))
        elif slot < 5:                                   # hostiles occupy slots 0-4 (5 is special)
            enemies.append(Enemy(slot=slot, x=ex, y=ey))

    # Tile window around Mario: GRID_ROWS rows x (COL_BEHIND+COL_AHEAD+1) cols, 1=solid.
    tiles: list[list[int]] = []
    for row in range(GRID_ROWS):
        y_world = 32 + row * 16
        line = []
        for col in range(-COL_BEHIND, COL_AHEAD + 1):
            x_world = mario_x + col * 16
            line.append(1 if tile_at(ram, x_world, y_world) != 0 else 0)
        tiles.append(line)

    scene = Scene(
        frame=frame,
        mario_x=mario_x,
        mario_y=mario_y,
        x_speed=_s8(ram, 0x57),
        world=_u8(ram, 0x075F),
        stage=_u8(ram, 0x075C),
        area=_u8(ram, 0x0760),
        lives=_u8(ram, 0x075A),
        coins=_u8(ram, 0x075E),
        score=_digits(ram, 0x07DD, 6) * 10,  # SMB score is stored without the trailing 0
        time=_digits(ram, 0x07F8, 3),
        size=_u8(ram, 0x0756),
        player_state=_u8(ram, 0x000E),
        float_state=_u8(ram, 0x001D),
        y_viewport=_u8(ram, 0x00B5),
        enemies=enemies,
        powerups=powerups,
        tiles=tiles,
        ram=ram,
    )
    return scene
