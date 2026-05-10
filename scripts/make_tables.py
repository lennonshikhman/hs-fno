from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from hsno.utils.naming import rename_model_series
from hsno.utils.seedstats import BOOTSTRAP_SAMPLES, grouped_bootstrap_table


def main():
    ap = argparse.ArgumentParser(description="Create bootstrap-CI benchmark/model/regime summary tables from all_metrics.csv")
    ap.add_argument("--metrics", default="outputs/metrics/all_metrics.csv")
    ap.add_argument("--out-dir", default="outputs/metrics")
    args = ap.parse_args()
    df = pd.read_csv(args.metrics)
    if "model" in df.columns:
        df["model"] = rename_model_series(df["model"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metric_cols = [
        c
        for c in df.columns
        if c.endswith("rel_l2")
        or c.startswith("rollout_step_")
        or c in {"inference_time_per_step", "speedup_vs_solver", "peak_memory_mb", "amplitude_error", "dominant_frequency_error", "phase_drift"}
    ]
    tables = {
        "table_by_benchmark_model.csv": grouped_bootstrap_table(df, ["benchmark", "model"], metric_cols, n_boot=BOOTSTRAP_SAMPLES),
        "table_by_regime_model.csv": grouped_bootstrap_table(df, ["regime", "model"], metric_cols, n_boot=BOOTSTRAP_SAMPLES),
        "table_by_benchmark_regime.csv": grouped_bootstrap_table(df, ["benchmark", "regime"], metric_cols, n_boot=BOOTSTRAP_SAMPLES),
    }
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    print(tables["table_by_benchmark_model.csv"].to_string(index=False))


if __name__ == "__main__":
    main()
