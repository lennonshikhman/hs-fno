#!/usr/bin/env python
"""Generate a ChatGPT-ready statistical analysis report for HS-FNO outputs.

Run this after ``python run_all_experiments.py`` has produced
``outputs/metrics/all_metrics.csv``. The script prints a long-form Markdown
report that can be pasted into ChatGPT to help interpret results and draft paper
text. It also saves the report to ``outputs/metrics/results_for_chatgpt.md`` by
default.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from textwrap import indent

import numpy as np
import pandas as pd
from scipy import stats

from hsno.utils.naming import MODEL_RENAME, ABLATION_MODELS, CORE_MODELS, rename_model_series
from hsno.utils.seedstats import BOOTSTRAP_SAMPLES, bootstrap_mean_ci, seed_level_values
from analysis.statistics import write_statistics_outputs

LOWER_IS_BETTER = {
    "one_step_rel_l2",
    "history_rel_l2",
    "rollout_rel_l2",
    "semiflow_error",
    "amplitude_error",
    "dominant_frequency_error",
    "phase_drift",
    "inference_time_per_step",
    "peak_memory_mb",
    "output_dimension",
}
HIGHER_IS_BETTER = {"speedup_vs_solver"}
PRIMARY_METRICS = ["one_step_rel_l2", "history_rel_l2", "rollout_rel_l2"]
BASELINE_MODELS = ["current_state", "lag_stack", "history2history", "temporal_unet", "convlstm", "temporal_transformer"]
HSNO_MODELS = ["hs_fno", "hsno_unet", "hs_transformer"]
REGIME_ORDER = ["in_distribution", "held_out_delay", "held_out_parameter", "resolution_transfer"]


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    count: int


def _fmt(x: float | int | str | None, digits: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return "NA"
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def _metric_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in LOWER_IS_BETTER or c in HIGHER_IS_BETTER or c.startswith("rollout_step_"):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return cols


def _summary(values: pd.Series) -> MetricSummary:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) == 0:
        return MetricSummary(np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    return MetricSummary(float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, float(v.median()), float(v.quantile(0.25)), float(v.quantile(0.75)), int(len(v)))


def _direction(metric: str) -> str:
    return "higher" if metric in HIGHER_IS_BETTER else "lower"


def _best_rows(df: pd.DataFrame, metric: str, group_cols: list[str]) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame()
    ascending = metric not in HIGHER_IS_BETTER
    rows = []
    groupby_arg = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, group in df.dropna(subset=[metric]).groupby(groupby_arg, dropna=False):
        cell = group.groupby("model", dropna=False)[metric].mean().reset_index()
        if cell.empty:
            continue
        best = cell.sort_values(metric, ascending=ascending).iloc[0]
        record = {col: val for col, val in zip(group_cols, keys if isinstance(keys, tuple) else (keys,))}
        record.update({"best_model": best["model"], f"mean_{metric}": best[metric]})
        rows.append(record)
    return pd.DataFrame(rows)


def _paired_model_frame(df: pd.DataFrame, metric: str, model_a: str, model_b: str) -> pd.DataFrame:
    key_cols = [c for c in ["benchmark", "regime", "seed"] if c in df.columns]
    if not key_cols or metric not in df.columns:
        return pd.DataFrame()
    sub = df[df["model"].isin([model_a, model_b])][key_cols + ["model", metric]].dropna(subset=[metric])
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index=key_cols, columns="model", values=metric, aggfunc="mean").dropna(subset=[model_a, model_b])


def _cohen_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return np.nan
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else np.nan


def _wilcoxon_p(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[np.abs(diff) > 1e-12]
    if len(diff) < 2:
        return np.nan
    return float(stats.wilcoxon(diff).pvalue)


def _sign_test_p(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    pos = int((diff > 0).sum())
    neg = int((diff < 0).sum())
    n = pos + neg
    if n == 0:
        return np.nan
    return float(stats.binomtest(min(pos, neg), n=n, p=0.5, alternative="two-sided").pvalue)


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return np.nan
    return float(num / den)


def _markdown_table(df: pd.DataFrame, max_rows: int = 40, digits: int = 4) -> str:
    """Render a small Markdown table without pandas' optional tabulate dependency."""
    if df.empty:
        return "_No rows available._"
    show = df.head(max_rows).copy()
    headers = [str(c) for c in show.columns]
    rows: list[list[str]] = []
    for _, row in show.iterrows():
        rendered = []
        for c in show.columns:
            value = row[c]
            if pd.api.types.is_numeric_dtype(show[c]):
                rendered.append(_fmt(value, digits))
            else:
                rendered.append("NA" if pd.isna(value) else str(value))
        rows.append(rendered)
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(w) for v, w in zip(values, widths)) + " |"

    lines = [fmt_row(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(fmt_row(row) for row in rows)
    if len(df) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def _section(title: str, body: str) -> str:
    return f"\n## {title}\n\n{body.strip()}\n"


def _load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}. Run run_all_experiments.py first.")
    df = pd.read_csv(path)
    if "model" in df.columns:
        df["model"] = rename_model_series(df["model"])
    required = {"benchmark", "model", "regime"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Metrics file is missing required columns: {missing}")
    return df


def _overview(df: pd.DataFrame, metrics_path: Path) -> str:
    metric_cols = _metric_columns(df)
    lines = [
        f"Metrics file: `{metrics_path}`",
        f"Rows: {len(df)}",
        f"Benchmarks ({df['benchmark'].nunique()}): {', '.join(map(str, sorted(df['benchmark'].dropna().unique())))}",
        f"Models ({df['model'].nunique()}): {', '.join(map(str, sorted(df['model'].dropna().unique())))}",
        f"Legacy model names remapped: {', '.join(f'{k}->{v}' for k, v in MODEL_RENAME.items())}",
        f"Regimes ({df['regime'].nunique()}): {', '.join(map(str, sorted(df['regime'].dropna().unique())))}",
        f"Seeds ({df['seed'].nunique() if 'seed' in df.columns else 1}): {', '.join(map(str, sorted(df['seed'].dropna().unique()))) if 'seed' in df.columns else 'legacy metrics without seed column'}",
        f"Bootstrap confidence intervals: 95% percentile CIs with {BOOTSTRAP_SAMPLES:,} resamples over seed-level means.",
        f"Numeric metric columns detected ({len(metric_cols)}): {', '.join(metric_cols)}",
        "Metric convention: lower is better for error/time/memory metrics; higher is better for `speedup_vs_solver`.",
    ]
    if df.isna().any().any():
        na_counts = df.isna().sum().loc[lambda s: s > 0].sort_values(ascending=False)
        lines.append("Missing values by column:\n" + indent(na_counts.to_string(), "  "))
    return "\n".join(lines)


def _overall_rankings(df: pd.DataFrame) -> str:
    parts = []
    for metric in PRIMARY_METRICS:
        if metric not in df.columns:
            continue
        rows = []
        for model, group in df.groupby("model", dropna=False):
            vals = seed_level_values(group, metric)
            arr = vals.dropna().to_numpy(dtype=float)
            if arr.size == 0:
                continue
            ci_low, ci_high = bootstrap_mean_ci(arr)
            rows.append({
                "model": model,
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "median": float(np.median(arr)),
                "seed_units": int(arr.size),
            })
        tab = pd.DataFrame(rows)
        if tab.empty:
            continue
        tab = tab.sort_values("mean", ascending=metric not in HIGHER_IS_BETTER)
        tab.insert(0, "rank", range(1, len(tab) + 1))
        parts.append(f"### Overall ranking by `{metric}` ({_direction(metric)} is better)\n\n{_markdown_table(tab)}")
    return "\n\n".join(parts) if parts else "No primary metric columns found."


def _benchmark_regime_rankings(df: pd.DataFrame) -> str:
    parts = []
    for metric in PRIMARY_METRICS:
        if metric not in df.columns:
            continue
        best = _best_rows(df, metric, ["benchmark", "regime"])
        if best.empty:
            continue
        counts = best["best_model"].value_counts().rename_axis("model").reset_index(name="wins")
        parts.append(f"### Best model counts for `{metric}` across benchmark/regime cells\n\n{_markdown_table(counts)}")
        wide = best.sort_values(["benchmark", "regime"])
        parts.append(f"### Per-cell winners for `{metric}`\n\n{_markdown_table(wide, max_rows=80)}")
    return "\n\n".join(parts) if parts else "No benchmark/regime rankings available."


def _hsno_vs_baselines(df: pd.DataFrame) -> str:
    hsno = "hs_fno" if "hs_fno" in set(df["model"]) else next((m for m in HSNO_MODELS if m in set(df["model"])), None)
    if hsno is None:
        return "No HS-FNO/history-space model found for paired comparisons."
    rows = []
    for metric in PRIMARY_METRICS:
        if metric not in df.columns:
            continue
        for baseline in [m for m in BASELINE_MODELS if m in set(df["model"])] + [m for m in sorted(df["model"].unique()) if m in set(ABLATION_MODELS) or str(m).startswith("hsno_unet_")]:
            if baseline == hsno:
                continue
            paired = _paired_model_frame(df, metric, hsno, baseline)
            if paired.empty:
                continue
            # Positive improvement means HS-FNO is better for lower-is-better metrics.
            if metric in HIGHER_IS_BETTER:
                improvement = 100.0 * (paired[hsno] - paired[baseline]) / paired[baseline].replace(0, np.nan)
                diff = paired[hsno].to_numpy() - paired[baseline].to_numpy()
            else:
                improvement = 100.0 * (paired[baseline] - paired[hsno]) / paired[baseline].replace(0, np.nan)
                diff = paired[baseline].to_numpy() - paired[hsno].to_numpy()
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"{hsno} vs {baseline}",
                    "paired_seed_cells": int(len(paired)),
                    "mean_%_improvement": float(np.nanmean(improvement)),
                    "ci95_low_%_improvement": bootstrap_mean_ci(improvement)[0],
                    "ci95_high_%_improvement": bootstrap_mean_ci(improvement)[1],
                    "median_%_improvement": float(np.nanmedian(improvement)),
                    "wins": int((improvement > 0).sum()),
                    "losses": int((improvement < 0).sum()),
                    "cohen_dz_on_diff": _cohen_dz(diff),
                    "wilcoxon_p": _wilcoxon_p(diff),
                    "sign_test_p": _sign_test_p(diff),
                }
            )
    if not rows:
        return "No paired HS-FNO-vs-baseline comparisons could be formed."
    out = pd.DataFrame(rows).sort_values(["metric", "mean_%_improvement"], ascending=[True, False])
    caveat = (
        "HS-FNO (`hs_fno`) is the headline model; generic `hsno_unet` is a backbone variant/baseline. "
        f"Intervals are 95% percentile bootstrap CIs with {BOOTSTRAP_SAMPLES:,} resamples over paired seed-level benchmark/regime cells. "
        "Interpret p-values cautiously and prioritize effect sizes plus confidence intervals."
    )
    return caveat + "\n\n" + _markdown_table(out, max_rows=120)


def _ood_degradation(df: pd.DataFrame) -> str:
    metric = "rollout_rel_l2" if "rollout_rel_l2" in df.columns else ("history_rel_l2" if "history_rel_l2" in df.columns else None)
    if metric is None:
        return "No rollout/history error metric found for OOD degradation analysis."
    rows = []
    grouped = df.groupby(["benchmark", "model", "regime"])[metric].mean().reset_index()
    for (benchmark, model), sub in grouped.groupby(["benchmark", "model"]):
        values = dict(zip(sub["regime"], sub[metric]))
        base = values.get("in_distribution")
        for regime in [r for r in REGIME_ORDER if r != "in_distribution"]:
            if regime in values and base is not None:
                rows.append(
                    {
                        "benchmark": benchmark,
                        "model": model,
                        "regime": regime,
                        "id_error": base,
                        "ood_error": values[regime],
                        "ood/id_ratio": _safe_ratio(values[regime], base),
                        "absolute_delta": values[regime] - base,
                    }
                )
    if not rows:
        return "No in-distribution plus OOD regime pairs were found."
    tab = pd.DataFrame(rows)
    model_summary = tab.groupby(["model", "regime"])[["ood/id_ratio", "absolute_delta"]].agg(["mean", "median", "count"]).reset_index()
    worst = tab.sort_values("ood/id_ratio", ascending=False).head(20)
    return (
        f"OOD degradation uses `{metric}`. Ratio > 1 means worse than in-distribution.\n\n"
        f"### Mean/median degradation by model and regime\n\n{_markdown_table(model_summary, max_rows=80)}\n\n"
        f"### Worst individual degradation cells\n\n{_markdown_table(worst, max_rows=20)}"
    )


def _rollout_curve_analysis(df: pd.DataFrame) -> str:
    step_cols = sorted([c for c in df.columns if c.startswith("rollout_step_") and c.endswith("_rel_l2")], key=lambda c: int(c.split("_")[2]))
    if not step_cols:
        return "No per-step rollout columns were found."
    rows = []
    for model, group in df.groupby("model"):
        means = group[step_cols].mean(numeric_only=True)
        if means.isna().all():
            continue
        first = means.iloc[0]
        last = means.iloc[-1]
        rows.append(
            {
                "model": model,
                "step1_mean": first,
                "final_step_mean": last,
                "final/step1_ratio": _safe_ratio(last, first),
                "mean_slope_per_step": float(np.polyfit(np.arange(1, len(means) + 1), means.to_numpy(dtype=float), 1)[0]) if len(means) > 1 else 0.0,
            }
        )
    if not rows:
        return "Per-step rollout columns exist but contain no numeric data."
    return _markdown_table(pd.DataFrame(rows).sort_values("final_step_mean"), max_rows=80)


def _efficiency_analysis(df: pd.DataFrame) -> str:
    needed = [c for c in ["one_step_rel_l2", "rollout_rel_l2", "inference_time_per_step", "param_count", "peak_memory_mb", "speedup_vs_solver"] if c in df.columns]
    if not needed:
        return "No efficiency columns found."
    tab = df.groupby("model")[needed].mean(numeric_only=True).reset_index()
    score_metric = "rollout_rel_l2" if "rollout_rel_l2" in tab.columns else needed[0]
    if "inference_time_per_step" in tab.columns:
        tab["error_x_time"] = tab[score_metric] * tab["inference_time_per_step"]
    if "param_count" in tab.columns:
        tab["error_x_mparams"] = tab[score_metric] * (tab["param_count"] / 1e6)
    sort_col = "error_x_time" if "error_x_time" in tab.columns else score_metric
    return _markdown_table(tab.sort_values(sort_col), max_rows=80)


def _ablation_analysis(df: pd.DataFrame) -> str:
    if "model" not in df.columns:
        return "No model column found."
    ablation_names = set(ABLATION_MODELS) | {"hsno_unet_no_shift", "hsno_unet_rollout_semiflow"}
    ablations = df[df["model"].astype(str).isin(ablation_names)]
    if ablations.empty:
        return "No HS-FNO ablation rows were found."
    metric = "rollout_rel_l2" if "rollout_rel_l2" in df.columns else "history_rel_l2"
    if metric not in df.columns:
        return "Ablation rows exist, but no rollout/history metric is available."
    rows = []
    for ablation in sorted(ablations["model"].unique()):
        paired = _paired_model_frame(df, metric, "hs_fno", ablation) if "hs_fno" in set(df["model"]) else pd.DataFrame()
        if paired.empty:
            rows.append({"ablation": ablation, "paired_cells": 0, "mean_delta_vs_hs_fno": np.nan, "mean_%_change_vs_hs_fno": np.nan, "median_%_change_vs_hs_fno": np.nan})
            continue
        delta = paired[ablation] - paired["hs_fno"]
        pct = 100.0 * delta / paired["hs_fno"].replace(0, np.nan)
        rows.append(
            {
                "ablation": ablation,
                "paired_cells": int(len(paired)),
                "mean_delta_vs_hs_fno": float(delta.mean()),
                "mean_%_change_vs_hs_fno": float(pct.mean()),
                "median_%_change_vs_hs_fno": float(pct.median()),
            }
        )
    note = f"Ablation deltas use `{metric}`; positive means the ablation is worse than headline HS-FNO."
    return note + "\n\n" + _markdown_table(pd.DataFrame(rows).sort_values("mean_%_change_vs_hs_fno"), max_rows=80)


def _benchmark_diagnostics(df: pd.DataFrame) -> str:
    diagnostic_cols = [
        c
        for c in df.columns
        if c.endswith("_error")
        or c.endswith("_rel_l2")
        or c in {"rd_oscillation_index", "phase_drift", "dominant_frequency_error", "amplitude_error"}
    ]
    diagnostic_cols = [c for c in diagnostic_cols if c not in PRIMARY_METRICS and pd.api.types.is_numeric_dtype(df[c])]
    if not diagnostic_cols:
        return "No benchmark-specific diagnostic columns found."
    rows = []
    for benchmark, group in df.groupby("benchmark"):
        cols = [c for c in diagnostic_cols if group[c].notna().any()]
        for col in cols:
            s = _summary(group[col])
            rows.append({"benchmark": benchmark, "diagnostic": col, "mean": s.mean, "median": s.median, "iqr": s.q75 - s.q25, "count": s.count})
    return _markdown_table(pd.DataFrame(rows).sort_values(["benchmark", "diagnostic"]), max_rows=120)


def _paper_guidance(df: pd.DataFrame) -> str:
    guidance = []
    if "rollout_rel_l2" in df.columns:
        overall = df.groupby("model")["rollout_rel_l2"].mean().sort_values()
        if len(overall):
            guidance.append(f"Best mean rollout model: `{overall.index[0]}` with mean rollout relative L2 {_fmt(overall.iloc[0])}.")
            if "hs_fno" in overall.index:
                guidance.append(f"Headline HS-FNO mean rollout relative L2: {_fmt(overall.loc['hs_fno'])}; rank {int(overall.index.get_loc('hs_fno') + 1)} of {len(overall)}.")
            if "hsno_unet" in overall.index:
                guidance.append(f"Generic hsno_unet mean rollout relative L2: {_fmt(overall.loc['hsno_unet'])}; treat it as a backbone variant, not the main method.")
    if "history_rel_l2" in df.columns:
        winners = _best_rows(df, "history_rel_l2", ["benchmark", "regime"])
        if not winners.empty:
            win_counts = winners["best_model"].value_counts()
            guidance.append("History-space winner counts: " + ", ".join(f"{m}={n}" for m, n in win_counts.items()) + ".")
    guidance.extend(
        [
            "Use HS-FNO as the headline model and generic history-space neural operators as the broader family/ablations.",
            "When drafting paper claims, separate in-distribution accuracy, OOD robustness, and efficiency; the best model for one may not dominate the others.",
            "Do not emphasize semiflow_error, dominant_frequency_error, amplitude, or memory diagnostics as headline claims unless they have been separately debugged/normalized.",
            f"Report 95% bootstrap confidence intervals using {BOOTSTRAP_SAMPLES:,} resamples over the 10 default seed replicates; avoid headline claims whose CIs overlap materially.",
            "For tables, report seed-aggregated mean, std, and 95% CI plus paired HS-FNO-vs-baseline percent improvements; for figures, use rollout curves, OOD degradation ratios, and efficiency tradeoffs.",
        ]
    )
    return "\n".join(f"- {g}" for g in guidance)


def build_report(df: pd.DataFrame, metrics_path: Path) -> str:
    title = "# ChatGPT-Ready HS-FNO Results Analysis\n"
    preamble = (
        "Paste this report into ChatGPT and ask it to help interpret the experimental results, "
        "identify likely paper claims, draft result paragraphs, and suggest additional tables/figures. "
        "All statistics below are computed from the existing metrics table; they do not rerun models.\n"
    )
    sections = [
        _section("Dataset and Metrics Overview", _overview(df, metrics_path)),
        _section("Overall Model Rankings", _overall_rankings(df)),
        _section("Benchmark/Regime Winners", _benchmark_regime_rankings(df)),
        _section("Core Comparison: HS-FNO vs Baselines", _hsno_vs_baselines(df)),
        _section("OOD and Resolution-Transfer Degradation", _ood_degradation(df)),
        _section("Rollout Error Growth", _rollout_curve_analysis(df)),
        _section("Efficiency and Pareto-Style Tradeoffs", _efficiency_analysis(df)),
        _section("Ablation Analysis", _ablation_analysis(df)),
        _section("Benchmark-Specific Diagnostics", _benchmark_diagnostics(df)),
        _section("Paper-Writing Guidance and Caveats", _paper_guidance(df)),
    ]
    return title + "\n" + preamble + "".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a ChatGPT-ready statistical analysis of HS-FNO experiment outputs.")
    parser.add_argument("--metrics", default="outputs/metrics/all_metrics.csv", help="Path to all_metrics.csv produced by run_all_experiments.py")
    parser.add_argument("--save", default="outputs/metrics/results_for_chatgpt.md", help="Where to save the generated report; use empty string to disable saving")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    df = _load_metrics(metrics_path)
    report = build_report(df, metrics_path)
    try:
        output_dir = metrics_path.parents[1] if metrics_path.parent.name == "metrics" else Path("outputs")
        write_statistics_outputs(df, output_dir)
    except Exception as exc:
        report += f"\n## Statistics Artifact Warning\n\nCould not write structured statistics artifacts: {exc}\n"
    print(report)
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
