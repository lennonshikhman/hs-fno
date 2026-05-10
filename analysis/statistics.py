from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from hsno.utils.seedstats import BOOTSTRAP_SAMPLES, bootstrap_mean_ci

MAIN_MODELS = ["current_state", "lag_stack", "history2history", "temporal_unet", "convlstm", "temporal_transformer", "hs_fno"]
PRIMARY_METRIC = "rollout_rel_l2"


def _finite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def standard_summaries(df: pd.DataFrame, group_cols: list[str], metric: str = PRIMARY_METRIC, n_boot: int = BOOTSTRAP_SAMPLES) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        vals = _finite(g[metric]) if metric in g else pd.Series(dtype=float)
        lo, hi = bootstrap_mean_ci(vals.to_numpy(), n_boot=n_boot) if len(vals) else (np.nan, np.nan)
        q25, q75 = (np.percentile(vals, [25, 75]) if len(vals) else (np.nan, np.nan))
        row = dict(zip(group_cols, key))
        row.update({
            "metric": metric,
            "mean": float(vals.mean()) if len(vals) else np.nan,
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "stderr": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "median": float(vals.median()) if len(vals) else np.nan,
            "iqr": float(q75 - q25) if len(vals) else np.nan,
            "ci95_low": lo,
            "ci95_high": hi,
            "n_rows": int(len(vals)),
            "n_seeds": int(g["seed"].nunique()) if "seed" in g else 0,
            "n_trajectories": int(g["trajectory_id"].nunique()) if "trajectory_id" in g else 0,
            "n_benchmark_regime_cells": int(g[["benchmark", "regime"]].drop_duplicates().shape[0]) if {"benchmark", "regime"}.issubset(g.columns) else 0,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def matched_units(df: pd.DataFrame, metric: str = PRIMARY_METRIC, unit_cols: Iterable[str] | None = None) -> pd.DataFrame:
    unit_cols = list(unit_cols or ["benchmark", "regime", "seed", "trajectory_id", "rollout_step"])
    unit_cols = [c for c in unit_cols if c in df.columns]
    return df.groupby(unit_cols + ["model"], dropna=False)[metric].mean().reset_index()


def paired_hsfno_vs_baselines(df: pd.DataFrame, metric: str = PRIMARY_METRIC, n_boot: int = BOOTSTRAP_SAMPLES) -> pd.DataFrame:
    m = matched_units(df, metric)
    if m.empty or "hs_fno" not in set(m["model"]):
        return pd.DataFrame()
    unit_cols = [c for c in m.columns if c not in {"model", metric}]
    wide = m.pivot_table(index=unit_cols, columns="model", values=metric, aggfunc="mean")
    rows = []
    rng = np.random.default_rng(12345)
    for baseline in [c for c in wide.columns if c != "hs_fno"]:
        pair = wide[["hs_fno", baseline]].dropna()
        if pair.empty:
            continue
        diff = pair[baseline] - pair["hs_fno"]
        pct = diff / (pair[baseline].abs() + 1e-12) * 100.0
        boot = []
        arr = diff.to_numpy()
        pct_arr = pct.to_numpy()
        for _ in range(int(n_boot)):
            idx = rng.integers(0, len(arr), len(arr))
            boot.append([arr[idx].mean(), pct_arr[idx].mean()])
        boot = np.asarray(boot)
        try:
            wil = stats.wilcoxon(pair["hs_fno"], pair[baseline], zero_method="wilcox", alternative="less")
            pvalue = float(wil.pvalue)
        except ValueError:
            pvalue = np.nan
        rows.append({
            "baseline": baseline,
            "metric": metric,
            "n_matched_units": int(len(pair)),
            "paired_difference_baseline_minus_hsfno": float(diff.mean()),
            "paired_difference_ci95_low": float(np.percentile(boot[:, 0], 2.5)),
            "paired_difference_ci95_high": float(np.percentile(boot[:, 0], 97.5)),
            "paired_percent_improvement": float(pct.mean()),
            "paired_percent_improvement_ci95_low": float(np.percentile(boot[:, 1], 2.5)),
            "paired_percent_improvement_ci95_high": float(np.percentile(boot[:, 1], 97.5)),
            "wilcoxon_pvalue_matched_units": pvalue,
            "exploratory": False,
        })
    # Explicit benchmark-regime-cell exploratory analysis; do not present as independent seed inference.
    cell = df.groupby(["benchmark", "regime", "model"], dropna=False)[metric].mean().reset_index()
    cell_wide = cell.pivot_table(index=["benchmark", "regime"], columns="model", values=metric, aggfunc="mean")
    for baseline in [c for c in cell_wide.columns if c != "hs_fno"]:
        pair = cell_wide[["hs_fno", baseline]].dropna()
        if len(pair) < 2:
            continue
        try:
            pvalue = float(stats.wilcoxon(pair["hs_fno"], pair[baseline], zero_method="wilcox", alternative="less").pvalue)
        except ValueError:
            pvalue = np.nan
        rows.append({
            "baseline": baseline,
            "metric": metric,
            "n_matched_units": int(len(pair)),
            "paired_difference_baseline_minus_hsfno": float((pair[baseline] - pair["hs_fno"]).mean()),
            "paired_percent_improvement": float(((pair[baseline] - pair["hs_fno"]) / (pair[baseline].abs() + 1e-12) * 100).mean()),
            "wilcoxon_pvalue_matched_units": pvalue,
            "exploratory": True,
        })
    return pd.DataFrame(rows)


def rank_based_summary(df: pd.DataFrame, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    if not {"benchmark", "regime", "model", metric}.issubset(df.columns):
        return pd.DataFrame()
    cell = df.groupby(["benchmark", "regime", "model"], dropna=False)[metric].mean().reset_index()
    cell["rank"] = cell.groupby(["benchmark", "regime"])[metric].rank(method="average", ascending=True)
    out = cell.groupby("model").agg(mean_rank=("rank", "mean"), mean_reciprocal_rank=("rank", lambda x: float((1.0 / x).mean())), cells=("rank", "count")).reset_index()
    # Friedman test is only valid for complete repeated-measures cells.
    wide = cell.pivot_table(index=["benchmark", "regime"], columns="model", values=metric, aggfunc="mean").dropna(axis=1)
    friedman_p = np.nan
    if wide.shape[0] >= 2 and wide.shape[1] >= 3:
        try:
            friedman_p = float(stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns]).pvalue)
        except Exception:
            friedman_p = np.nan
    out["friedman_pvalue_complete_cells"] = friedman_p
    return out.sort_values("mean_rank")


def robustness_rankings(df: pd.DataFrame, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    rows = []
    if metric not in df:
        return pd.DataFrame()
    raw = df.groupby("model")[metric].mean().sort_values()
    med = df.groupby("model")[metric].median().sort_values()
    rows.extend({"aggregation": "raw_mean", "model": m, "rank": i + 1, "score": v} for i, (m, v) in enumerate(raw.items()))
    rows.extend({"aggregation": "median", "model": m, "rank": i + 1, "score": v} for i, (m, v) in enumerate(med.items()))
    if {"benchmark", "regime"}.issubset(df.columns):
        cell = df.groupby(["benchmark", "regime", "model"])[metric].mean().reset_index()
        cell["cell_mean"] = cell.groupby(["benchmark", "regime"])[metric].transform("mean")
        cell["normalized_error"] = cell[metric] / (cell["cell_mean"].abs() + 1e-12)
        norm = cell.groupby("model")["normalized_error"].mean().sort_values()
        cell["rank_score"] = cell.groupby(["benchmark", "regime"])[metric].rank(ascending=True)
        rank = cell.groupby("model")["rank_score"].mean().sort_values()
        rows.extend({"aggregation": "per_cell_normalized", "model": m, "rank": i + 1, "score": v} for i, (m, v) in enumerate(norm.items()))
        rows.extend({"aggregation": "rank_based", "model": m, "rank": i + 1, "score": v} for i, (m, v) in enumerate(rank.items()))
    return pd.DataFrame(rows)


def claim_checks(df: pd.DataFrame, metric: str = PRIMARY_METRIC) -> dict:
    checks = {}
    robust = robustness_rankings(df, metric)
    for agg, group in robust.groupby("aggregation") if not robust.empty else []:
        top = group.sort_values("rank").iloc[0]["model"]
        checks[f"hs_fno_best_under_{agg}"] = {"passed": bool(top == "hs_fno"), "top_model": str(top)}
    pair = paired_hsfno_vs_baselines(df, metric, n_boot=min(1000, BOOTSTRAP_SAMPLES))
    checks["hs_fno_beats_baselines_paired_mean"] = {
        "passed": bool((pair[~pair.get("exploratory", False).astype(bool)]["paired_difference_baseline_minus_hsfno"] > 0).all()) if not pair.empty else False,
        "baselines_checked": pair[~pair.get("exploratory", False).astype(bool)]["baseline"].tolist() if not pair.empty else [],
    }
    checks["warning"] = "Benchmark-regime cells are analyzed separately from independent seed/trajectory units; exploratory cell tests are flagged."
    return checks


def write_statistics_outputs(df: pd.DataFrame, output_dir: str | Path, metric: str = PRIMARY_METRIC, n_boot: int = BOOTSTRAP_SAMPLES) -> dict[str, Path]:
    out = Path(output_dir) / "results" / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "main_model_summary": out / "main_model_summary.csv",
        "pairwise": out / "pairwise_hsfno_vs_baselines.csv",
        "rank": out / "rank_based_summary.csv",
        "claims": out / "claim_checks.json",
    }
    standard_summaries(df, ["model"], metric, n_boot=n_boot).to_csv(paths["main_model_summary"], index=False)
    paired_hsfno_vs_baselines(df, metric, n_boot=n_boot).to_csv(paths["pairwise"], index=False)
    rank_based_summary(df, metric).to_csv(paths["rank"], index=False)
    paths["claims"].write_text(json.dumps(claim_checks(df, metric), indent=2), encoding="utf-8")
    return paths
