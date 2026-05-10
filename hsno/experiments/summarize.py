from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hsno.utils.io import read_jsonl, write_jsonl
from hsno.utils.naming import rename_model_series
from hsno.utils.plotting import save_basic_plots
from hsno.utils.seedstats import BOOTSTRAP_SAMPLES, grouped_bootstrap_table


def _metric_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"seed", "train_history_steps", "eval_nx", "trajectory_id", "window_id", "rollout_step", "history_steps", "history_resolution", "train_nx"}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def _collect_raw_logs(output_dir: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted((Path(output_dir) / "logs").glob("*_raw_metrics.jsonl")):
        rows.extend(read_jsonl(path))
    raw = pd.DataFrame(rows)
    if len(raw) and "model" in raw.columns:
        raw["model"] = rename_model_series(raw["model"])
    return raw


def _write_results(raw: pd.DataFrame, summary: pd.DataFrame, output_dir: str | Path) -> None:
    root = Path(output_dir) / "results"
    root.mkdir(parents=True, exist_ok=True)
    if len(raw):
        raw.to_csv(root / "raw_metrics.csv", index=False)
        write_jsonl(root / "raw_metrics.jsonl", raw.to_dict(orient="records"))
    if len(summary):
        summary.to_csv(root / "run_summary.csv", index=False)
        (root / "run_summary.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")


def _write_multiseed_outputs(raw: pd.DataFrame, output_dir: str | Path) -> None:
    if raw.empty:
        return
    out = Path(output_dir) / "results" / "main_multiseed"
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "raw_metrics.csv", index=False)
    metric_cols = [c for c in ["one_step_rel_l2", "history_rel_l2", "rollout_rel_l2"] if c in raw.columns]
    grouped_bootstrap_table(raw, ["model"], metric_cols, n_boot=BOOTSTRAP_SAMPLES).to_csv(out / "summary_by_model.csv", index=False)
    grouped_bootstrap_table(raw, ["benchmark", "regime"], metric_cols, n_boot=BOOTSTRAP_SAMPLES).to_csv(out / "summary_by_benchmark_regime.csv", index=False)
    grouped_bootstrap_table(raw, ["seed"], metric_cols, n_boot=BOOTSTRAP_SAMPLES).to_csv(out / "summary_by_seed.csv", index=False)


def _write_tables(raw: pd.DataFrame, output_dir: str | Path) -> None:
    if raw.empty:
        return
    out = Path(output_dir) / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    metrics = [m for m in ["one_step_rel_l2", "history_rel_l2", "rollout_rel_l2"] if m in raw.columns]
    main = grouped_bootstrap_table(raw, ["benchmark", "regime", "model"], metrics, n_boot=BOOTSTRAP_SAMPLES)
    main.to_csv(out / "table_main_with_uncertainty.csv", index=False)
    for metric, name in [("rollout_rel_l2", "rollout"), ("one_step_rel_l2", "one_step"), ("history_rel_l2", "history")]:
        if metric in raw.columns:
            grouped_bootstrap_table(raw, ["benchmark", "regime", "model"], [metric], n_boot=BOOTSTRAP_SAMPLES).to_csv(out / f"table_{name}_by_benchmark_regime_all_models.csv", index=False)
    grouped_bootstrap_table(raw, ["regime", "model"], metrics, n_boot=BOOTSTRAP_SAMPLES).to_csv(out / "table_per_regime_aggregate.csv", index=False)
    grouped_bootstrap_table(raw, ["benchmark", "model"], metrics, n_boot=BOOTSTRAP_SAMPLES).to_csv(out / "table_per_benchmark_aggregate.csv", index=False)


def _write_long_rollout(raw: pd.DataFrame, output_dir: str | Path) -> None:
    if raw.empty or "rollout_step" not in raw.columns:
        return
    out = Path(output_dir) / "results" / "long_rollout"
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "raw_rollout_steps.csv", index=False)
    grouped_bootstrap_table(raw, ["model", "benchmark", "rollout_step"], ["rollout_rel_l2"], n_boot=BOOTSTRAP_SAMPLES).to_csv(out / "summary_by_step.csv", index=False)


def summarize(metrics, output_dir):
    out = Path(output_dir) / 'metrics'
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics)
    if len(df) and "model" in df.columns:
        df["model"] = rename_model_series(df["model"])
    df.to_csv(out / 'all_metrics.csv', index=False)
    df.to_json(out / 'all_metrics.json', orient='records', indent=2)
    raw = _collect_raw_logs(output_dir)
    _write_results(raw, df, output_dir)
    _write_multiseed_outputs(raw if len(raw) else df, output_dir)
    _write_tables(raw if len(raw) else df, output_dir)
    _write_long_rollout(raw, output_dir)
    if len(df):
        metric_cols = _metric_columns(df)
        grouped_bootstrap_table(df, ['benchmark'], metric_cols, n_boot=BOOTSTRAP_SAMPLES).to_csv(out / 'summary_by_benchmark.csv', index=False)
        grouped_bootstrap_table(df, ['model'], metric_cols, n_boot=BOOTSTRAP_SAMPLES).to_csv(out / 'summary_by_model.csv', index=False)
        grouped_bootstrap_table(df, ['benchmark', 'model', 'regime'], metric_cols, n_boot=BOOTSTRAP_SAMPLES).to_csv(out / 'summary_by_benchmark_model_regime.csv', index=False)
    save_basic_plots(df, Path(output_dir) / 'plots')
    return df
