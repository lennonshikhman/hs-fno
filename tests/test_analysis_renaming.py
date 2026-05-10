import pandas as pd

from analyze_all_results import _load_metrics, build_report


def test_analysis_remaps_legacy_model_names(tmp_path):
    path = tmp_path / "all_metrics.csv"
    pd.DataFrame(
        [
            {"seed": 7, "benchmark": "b", "model": "ablation_hsno_fno_backbone", "regime": "in_distribution", "rollout_rel_l2": 0.1},
            {"seed": 7, "benchmark": "b", "model": "hsno", "regime": "in_distribution", "rollout_rel_l2": 0.2},
            {"seed": 7, "benchmark": "b", "model": "ablation_hsno_transformer_backbone", "regime": "in_distribution", "rollout_rel_l2": 0.3},
        ]
    ).to_csv(path, index=False)
    df = _load_metrics(path)
    assert {"hs_fno", "hsno_unet", "hs_transformer"}.issubset(set(df["model"]))
    assert "ablation_hsno_fno_backbone" not in set(df["model"])


def test_analysis_report_includes_seed_bootstrap_ci(tmp_path):
    path = tmp_path / "all_metrics.csv"
    rows = []
    for seed, hs, baseline in [(7, 0.1, 0.2), (8, 0.2, 0.4)]:
        rows.extend(
            [
                {"seed": seed, "benchmark": "b", "model": "hs_fno", "regime": "in_distribution", "rollout_rel_l2": hs},
                {"seed": seed, "benchmark": "b", "model": "current_state", "regime": "in_distribution", "rollout_rel_l2": baseline},
            ]
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    report = build_report(_load_metrics(path), path)
    assert "Seeds (2): 7, 8" in report
    assert "ci95_low" in report
    assert "ci95_low_%_improvement" in report
    assert "10,000 resamples" in report
