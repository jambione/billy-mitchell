"""SMW perception: WRAM -> Scene implementing the shared PlatformerView surface.

Honest scope for the scaffold: player/sprite state comes straight from the RAM map; the
TILE-based queries (gap_info / obstacle_ahead / block_above_ahead) return None until the
Layer-1 map16 reads are verified against a live ROM — the reflex then degrades to cruise +
micro-search + learn-from-death, which is exactly how Billy cracks unknown geometry anyway.
Verify order on first live boot: positions -> mode/dying -> sprites -> then wire tiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import ram_map as R


def _u8(ram: bytes, addr: int) -> int:
    return ram[addr] if addr < len(ram) else 0


def _u16(ram: bytes, addr: int) -> int:
    return _u8(ram, addr) | (_u8(ram, addr + 1) << 8)


@dataclass
class SmwScene:
    frame: int
    mario_x: int
    mario_y: int
    on_ground_flag: bool
    in_air: bool
    dying: bool
    powerup: int
    lives: int
    coins: int
    score: int
    game_mode: int
    translevel: int
    end_level_timer: int
    events_triggered: int
    enemies: list = field(default_factory=list)   # [(dx, dy)] relative to the player
    size: int = 0                                  # PlatformerView: 0 small, 1+ powered

    # --- PlatformerView surface -----------------------------------------------------------
    @property
    def on_ground(self) -> bool:
        return self.on_ground_flag and not self.in_air

    @property
    def is_dying(self) -> bool:
        return self.dying

    @property
    def in_play(self) -> bool:
        return self.game_mode == R.GAME_MODE_LEVEL

    @property
    def level_cleared_now(self) -> bool:
        return self.end_level_timer > 0

    def nearest_enemy(self, within: int = 72, y_above: int = 24,
                      y_below: int = 24) -> tuple[int, int] | None:
        best = None
        for dx, dy in self.enemies:
            if 0 < dx <= within and -y_above <= dy <= y_below:
                if best is None or dx < best[0]:
                    best = (dx, dy)
        return best

    def enemy_ahead(self, within: int = 48) -> bool:
        return self.nearest_enemy(within=within) is not None

    # Tile-based queries: None until map16 reads are ROM-verified (see module docstring).
    def gap_info(self, max_tiles: int = 8):
        return None

    def obstacle_ahead(self, max_tiles: int = 3):
        return None

    def block_above_ahead(self, max_tiles: int = 2):
        return None

    def air_landing_target(self):
        return None

    def nearest_powerup(self, within: int = 80):
        return None

    def pipe_entry_spot(self, max_tiles: int = 2):
        return None

    def enemy_count(self) -> int:
        return len(self.enemies)

    def summary(self) -> str:
        power = {0: "small", 1: "big", 2: "cape", 3: "fire"}.get(self.powerup, "?")
        near = self.nearest_enemy()
        enemy = f", enemy {near[0]}px ahead" if near else ""
        air = "airborne" if not self.on_ground else "on ground"
        return (f"SMW level {self.translevel} x={self.mario_x} {air}, {power} Mario, "
                f"{self.coins} coins{enemy}")

    def ascii_view(self) -> str:
        return ""   # tile map pending ROM verification


def build_scene(ram: bytes, frame: int, rgb=None) -> SmwScene:
    px = _u16(ram, R.PLAYER_X)
    py = _u16(ram, R.PLAYER_Y)
    enemies: list[tuple[int, int]] = []
    for slot in range(R.SPRITE_COUNT):
        if _u8(ram, R.SPRITE_STATUS + slot) < 8:
            continue   # empty/dead slot
        sx = _u8(ram, R.SPRITE_X_LO + slot) | (_u8(ram, R.SPRITE_X_HI + slot) << 8)
        sy = _u8(ram, R.SPRITE_Y_LO + slot) | (_u8(ram, R.SPRITE_Y_HI + slot) << 8)
        enemies.append((sx - px, sy - py))
    powerup = _u8(ram, R.POWERUP)
    return SmwScene(
        frame=frame,
        mario_x=px,
        mario_y=py,
        on_ground_flag=_u8(ram, R.ON_GROUND) != 0,
        in_air=_u8(ram, R.PLAYER_IN_AIR) != 0,
        dying=_u8(ram, R.PLAYER_STATE) == 9,
        powerup=powerup,
        lives=_u8(ram, R.LIVES),
        coins=_u8(ram, R.COINS),
        score=(_u8(ram, R.SCORE) | (_u8(ram, R.SCORE + 1) << 8)
               | (_u8(ram, R.SCORE + 2) << 16)) * 10,
        game_mode=_u8(ram, R.GAME_MODE),
        translevel=_u8(ram, R.TRANSLEVEL),
        end_level_timer=_u8(ram, R.END_LEVEL_TIMER),
        events_triggered=_u8(ram, R.EVENTS_TRIGGERED),
        enemies=enemies,
        size=min(powerup, 1),
    )
