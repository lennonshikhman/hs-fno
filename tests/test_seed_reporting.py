import pandas as pd

from run_all_experiments import _configured_seeds
from hsno.utils.seedstats import bootstrap_mean_ci, grouped_bootstrap_table, seed_level_values


def test_default_configured_seeds_are_ten_replicates():
    cfg = {"seed": 7, "seeds": list(range(7, 17))}
    assert _configured_seeds(cfg, None) == list(range(7, 17))
    assert len(_configured_seeds(cfg, None)) == 10
    assert _configured_seeds(cfg, [1, 2]) == [1, 2]


def test_bootstrap_table_uses_seed_level_units():
    df = pd.DataFrame(
        [
            {"seed": 1, "model": "a", "regime": "id", "rollout_rel_l2": 1.0},
            {"seed": 1, "model": "a", "regime": "ood", "rollout_rel_l2": 3.0},
            {"seed": 2, "model": "a", "regime": "id", "rollout_rel_l2": 5.0},
            {"seed": 2, "model": "a", "regime": "ood", "rollout_rel_l2": 7.0},
        ]
    )
    seed_means = seed_level_values(df, "rollout_rel_l2")
    assert seed_means.tolist() == [2.0, 6.0]
    table = grouped_bootstrap_table(df, ["model"], ["rollout_rel_l2"], n_boot=100)
    assert table.loc[0, "n_seeds"] == 2
    assert table.loc[0, "rollout_rel_l2_mean"] == 4.0
    assert "rollout_rel_l2_ci95_low" in table.columns
    assert "rollout_rel_l2_ci95_high" in table.columns


def test_bootstrap_mean_ci_is_deterministic():
    first = bootstrap_mean_ci([1.0, 2.0, 3.0], n_boot=1000)
    second = bootstrap_mean_ci([1.0, 2.0, 3.0], n_boot=1000)
    assert first == second
    assert first[0] <= 2.0 <= first[1]
