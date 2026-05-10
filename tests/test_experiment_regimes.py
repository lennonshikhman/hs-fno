import numpy as np
from hsno.data.datasets import sample_params, regime_config


def test_held_out_delay_sampling_exceeds_training_range():
    ranges = {"tau": [0.1, 0.2], "D": [0.01, 0.02]}
    p = sample_params(np.random.default_rng(0), ranges, "held_out_delay")
    assert p["tau"] > 0.2
    assert 0.01 <= p["D"] <= 0.02


def test_resolution_transfer_config_increases_grid():
    cfg = {"regime": "in_distribution", "data": {"n_test": 1, "nx": 16, "dim": 1}}
    rcfg = regime_config(cfg, "resolution_transfer")
    assert rcfg["data"]["nx"] > cfg["data"]["nx"]


def test_nonfinite_trajectory_cache_is_not_reused(tmp_path):
    from hsno.data.datasets import _cached_trajectories_are_finite

    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, fields=np.array([np.inf], dtype=np.float32), static=np.array([0.0], dtype=np.float32))
    good = tmp_path / "good.npz"
    np.savez_compressed(good, fields=np.array([1.0], dtype=np.float32), static=np.array([0.0], dtype=np.float32))

    assert not _cached_trajectories_are_finite(bad)
    assert _cached_trajectories_are_finite(good)
