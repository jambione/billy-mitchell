"""Tests for the RL featurizer + action set. Torch-free, so they run in the default suite;
the env/policy themselves are exercised by the smoke tests in train_rl.py / run.py --rl.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.games.smb.perception import build_scene  # noqa: E402
from billy.rl import features  # noqa: E402
from billy.systems.nes import controller  # noqa: E402


def test_obs_dim_matches_featurized_length():
    scene = build_scene(bytes(0x800), 0)
    vec = features.featurize(scene)
    assert vec.shape == (features.OBS_DIM,)
    assert str(vec.dtype) == "float32"
    assert (vec >= -1.0).all() and (vec <= 1.0).all()   # normalized, bounded


def test_action_masks_are_valid_controller_masks():
    assert features.N_ACTIONS == len(features.ACTION_MASKS) == len(features.ACTION_NAMES)
    assert features.ACTION_MASKS[0] == 0   # NOOP
    # 'right'+'B' is the run-right combo
    assert features.ACTION_MASKS[2] == controller.mask(controller.RIGHT, controller.B)
    for m in features.ACTION_MASKS:
        assert 0 <= m <= 0xFF


def test_featurize_reflects_enemy_distance():
    ram = bytearray(0x800)
    ram[0x6D], ram[0x86] = 0, 50          # mario x = 50
    ram[0x03B8] = 100
    ram[0x0F] = 1                          # enemy slot 0 alive
    ram[0x6E], ram[0x87] = 0, 80          # enemy x = 80 (30px ahead)
    ram[0xCF] = 100
    vec = features.featurize(build_scene(bytes(ram), 0))
    # the first enemy dx slot sits right after the tile block + 4 mario scalars
    enemy0_dx = vec[features._N_TILES + 4]
    assert enemy0_dx > 0   # enemy is ahead (positive dx)
