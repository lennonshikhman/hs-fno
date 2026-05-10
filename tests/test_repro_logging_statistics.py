from __future__ import annotations

import json

import pandas as pd

from analysis.statistics import paired_hsfno_vs_baselines, rank_based_summary, write_statistics_outputs
from hsno.utils.repro import stable_config_hash, write_config_lock


def test_config_lock_contains_reproducibility_metadata(tmp_path):
    cfg = {"benchmark": "b", "seed": 7, "data": {"history_steps": 4}}
    lock = write_config_lock(tmp_path / "config.lock.json", cfg=cfg, seed=7, args={"models": ["hs_fno"]}, run_id="r1")
    assert (tmp_path / "config.lock.json").exists()
    data = json.loads((tmp_path / "config.lock.json").read_text())
    assert data["seed"] == 7
    assert data["config_hash"] == stable_config_hash(cfg)
    assert "python_version" in data["environment"]
    assert "package_versions" in data["environment"]
    assert lock["run_id"] == "r1"


def test_statistics_outputs_use_matched_units_and_claim_files(tmp_path):
    rows = []
    for seed in [1, 2]:
        for traj in [0, 1]:
            rows.append({"seed": seed, "benchmark": "b", "regime": "id", "trajectory_id": traj, "rollout_step": 1, "model": "hs_fno", "rollout_rel_l2": 0.1 * seed})
            rows.append({"seed": seed, "benchmark": "b", "regime": "id", "trajectory_id": traj, "rollout_step": 1, "model": "current_state", "rollout_rel_l2": 0.2 * seed})
    df = pd.DataFrame(rows)
    pair = paired_hsfno_vs_baselines(df, n_boot=100)
    assert "current_state" in set(pair["baseline"])
    assert pair.loc[pair["baseline"] == "current_state", "paired_difference_baseline_minus_hsfno"].iloc[0] > 0
    ranks = rank_based_summary(df)
    assert ranks.iloc[0]["model"] == "hs_fno"
    paths = write_statistics_outputs(df, tmp_path, n_boot=100)
    assert paths["main_model_summary"].exists()
    assert paths["pairwise"].exists()
    assert paths["rank"].exists()
    assert paths["claims"].exists()
