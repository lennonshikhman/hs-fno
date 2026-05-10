from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_RANDOM_SEED = 12345


@dataclass(frozen=True)
class BootstrapSummary:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    count: int
    seed_count: int


def bootstrap_mean_ci(values, n_boot: int = BOOTSTRAP_SAMPLES, alpha: float = BOOTSTRAP_ALPHA, random_seed: int = BOOTSTRAP_RANDOM_SEED) -> tuple[float, float]:
    """Return percentile bootstrap CI for the mean of finite values."""
    arr = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(random_seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def seed_level_values(df: pd.DataFrame, metric: str, group_cols: list[str] | tuple[str, ...] | None = None) -> pd.Series:
    """Return seed-level metric means for bootstrap units.

    If a metrics table has a ``seed`` column, rows are first averaged within
    each seed (and optional grouping columns) so the bootstrap samples
    independent seed replicates rather than per-regime rows. Without a seed
    column, this falls back to row-level values for backward compatibility with
    legacy one-seed metrics files.
    """
    group_cols = list(group_cols or [])
    if metric not in df.columns:
        return pd.Series(dtype=float)
    cols = [c for c in group_cols if c in df.columns]
    if "seed" in df.columns:
        cols = cols + ["seed"]
    values = pd.to_numeric(df[metric], errors="coerce")
    work = df.loc[values.notna(), cols].copy()
    work[metric] = values[values.notna()].to_numpy()
    if not len(work):
        return pd.Series(dtype=float)
    if cols:
        return work.groupby(cols, dropna=False)[metric].mean().reset_index()[metric]
    return work[metric]


def bootstrap_summary(df: pd.DataFrame, metric: str, group_cols: list[str] | tuple[str, ...] | None = None, n_boot: int = BOOTSTRAP_SAMPLES) -> BootstrapSummary:
    vals = seed_level_values(df, metric, group_cols)
    arr = np.asarray(vals.dropna(), dtype=float)
    if arr.size == 0:
        return BootstrapSummary(np.nan, np.nan, np.nan, np.nan, 0, 0)
    ci_low, ci_high = bootstrap_mean_ci(arr, n_boot=n_boot)
    seed_count = int(df["seed"].nunique()) if "seed" in df.columns else int(arr.size)
    return BootstrapSummary(float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0, ci_low, ci_high, int(arr.size), seed_count)


def grouped_bootstrap_table(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str], n_boot: int = BOOTSTRAP_SAMPLES) -> pd.DataFrame:
    rows: list[dict] = []
    if df.empty or not metric_cols:
        return pd.DataFrame(columns=group_cols)
    groupby_arg = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, group in df.groupby(groupby_arg, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_rows"] = int(len(group))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else 1
        for metric in metric_cols:
            if metric not in group.columns:
                continue
            vals = seed_level_values(group, metric)
            arr = np.asarray(vals.dropna(), dtype=float)
            if arr.size == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_ci95_low"] = np.nan
                row[f"{metric}_ci95_high"] = np.nan
                continue
            ci_low, ci_high = bootstrap_mean_ci(arr, n_boot=n_boot)
            row[f"{metric}_mean"] = float(arr.mean())
            row[f"{metric}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)
