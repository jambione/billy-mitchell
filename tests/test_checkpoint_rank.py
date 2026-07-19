"""Guide-anchored checkpoint frontier (route_rank) + rank-aware persistence.

Covers the Zelda-stagnation fix: screen-progressing games rank their cross-session checkpoint by
"how far along the guide" instead of an ordinal level key, and `_persist_checkpoint` ratchets on
that rank. SMB's ordinal path stays byte-identical (regression)."""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy import config  # noqa: E402
from billy.director import Director  # noqa: E402
from billy.games.zelda import ZeldaGame  # noqa: E402
from billy.games.zelda.perception import Scene  # noqa: E402
from billy.games.zelda.walkthrough import (  # noqa: E402
    LEVEL_1_SCREEN,
    SEA_EAST_SCREEN,
    START_SCREEN,
)


def _scene(**over) -> Scene:
    base = dict(
        frame=0, link_x=120, link_y=120, direction=0, game_mode=4, current_level=0,
        map_location=START_SCREEN, next_location=START_SCREEN, health=3, max_hearts=3,
        partial_heart=0, triforce_pieces=0, sword_level=1, rupees=0, keys=0, bombs=0,
        scrolling=False, visited_screens=(START_SCREEN,),
    )
    base.update(over)
    return Scene(**base)


def _obs(scene: Scene):
    return SimpleNamespace(raw=scene, progress=scene.objective_score(),
                           level_key=(scene.realm, scene.map_location),
                           level_label=scene.room_label)


# --- Zelda route_rank: monotonic "how far along the guide" --------------------------------------

def test_route_rank_rises_on_sword_pickup():
    game = ZeldaGame()
    before = game.route_rank(_obs(_scene(sword_level=0)))
    after = game.route_rank(_obs(_scene(sword_level=1)))
    assert after > before


def test_route_rank_rises_on_visited_gain_same_phase():
    game = ZeldaGame()
    # Both are "east_to_sea" (sword in hand, grid row 8, west of the sea, sea not yet visited).
    one = game.route_rank(_obs(_scene(map_location=START_SCREEN, visited_screens=(START_SCREEN,))))
    two = game.route_rank(_obs(_scene(map_location=START_SCREEN + 1,
                                      visited_screens=(START_SCREEN, START_SCREEN + 1))))
    assert two > one


def test_route_rank_rises_on_phase_advance():
    game = ZeldaGame()
    east = game.route_rank(_obs(_scene(map_location=START_SCREEN, visited_screens=(START_SCREEN,))))
    # Sea + Level 1 both visited → past east_to_sea and level_1_approach → "explore" (higher phase).
    explore = game.route_rank(_obs(_scene(
        map_location=LEVEL_1_SCREEN,
        visited_screens=(START_SCREEN, SEA_EAST_SCREEN, LEVEL_1_SCREEN))))
    assert explore > east
    assert explore >= 3 * 1_000_000   # phase ordinal for "explore"


def test_route_rank_default_is_none_for_ordinal_games():
    # Base Game.route_rank returns None; SMB inherits it (its level_key IS ordinal).
    from billy.games.smb import SmbGame
    assert SmbGame().route_rank(_obs(_scene())) is None


# --- Zelda checkpoint_ready ---------------------------------------------------------------------

def test_checkpoint_ready_on_overworld_screen():
    assert ZeldaGame().checkpoint_ready(_obs(_scene(game_mode=4))) is True


def test_checkpoint_not_ready_in_cave():
    # game_mode 11 is a cave interior (CAVE_INTERIOR_MODES) — reflex owns those, don't checkpoint.
    assert ZeldaGame().checkpoint_ready(_obs(_scene(game_mode=11))) is False


def test_checkpoint_not_ready_when_not_in_play():
    # game_mode 0 (title/death) isn't an in-play mode.
    assert ZeldaGame().checkpoint_ready(_obs(_scene(game_mode=0))) is False


# --- rank-aware _persist_checkpoint (director) --------------------------------------------------

class _FakeSession:
    def clone_state(self) -> bytes:
        return b"SAVESTATE"


class _RankGame:
    """route_rank reads a value stashed on the obs, so the test controls the frontier directly."""
    def route_rank(self, obs) -> int | None:
        return obs.rank


class _OrdinalGame:
    def route_rank(self, obs) -> int | None:
        return None


class _Stub:
    """Duck-typed Director just for _persist_checkpoint (no emulator / real session)."""
    _checkpoint_paths = Director._checkpoint_paths
    _persist_checkpoint = Director._persist_checkpoint

    def __init__(self, game):
        self.game = game
        self.session = _FakeSession()

    def _game_id(self) -> str:
        return "checkpoint_test"


def _rank_obs(rank: int, label: str):
    return SimpleNamespace(level_key=("overworld", 100 + rank), level_label=label,
                           progress=rank, rank=rank)


def _read_meta(tmp_path):
    return json.loads((tmp_path / "checkpoint_test" / "furthest.json").read_text())


def test_persist_ratchets_on_rank(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINTS_DIR", tmp_path)
    stub = _Stub(_RankGame())

    stub._persist_checkpoint(_rank_obs(10, "screen-a"))
    meta = _read_meta(tmp_path)
    assert meta["label"] == "screen-a" and meta["rank"] == 10

    # Higher rank advances the frontier.
    stub._persist_checkpoint(_rank_obs(25, "screen-b"))
    assert _read_meta(tmp_path)["label"] == "screen-b"

    # Equal or lower rank does NOT overwrite.
    stub._persist_checkpoint(_rank_obs(25, "screen-b-again"))
    stub._persist_checkpoint(_rank_obs(5, "backtrack"))
    assert _read_meta(tmp_path)["label"] == "screen-b"


def test_persist_ordinal_game_unchanged(tmp_path, monkeypatch):
    """route_rank None → ordinal level_key ratchet, and no `rank` field written (SMB path)."""
    monkeypatch.setattr(config, "CHECKPOINTS_DIR", tmp_path)
    stub = _Stub(_OrdinalGame())

    a = SimpleNamespace(level_key=[1, 1, 0], level_label="1-1", progress=40)
    b = SimpleNamespace(level_key=[1, 2, 0], level_label="1-2", progress=40)
    back = SimpleNamespace(level_key=[1, 1, 0], level_label="1-1-again", progress=40)

    stub._persist_checkpoint(a)
    meta = _read_meta(tmp_path)
    assert meta["label"] == "1-1" and "rank" not in meta

    stub._persist_checkpoint(b)               # higher ordinal key advances
    assert _read_meta(tmp_path)["label"] == "1-2"

    stub._persist_checkpoint(back)            # lower key is ignored
    assert _read_meta(tmp_path)["label"] == "1-2"
