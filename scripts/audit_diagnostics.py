#!/usr/bin/env python
from __future__ import annotations
import argparse, json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

DIAGNOSTICS = [
    "amplitude_error", "dominant_frequency_error", "phase_drift", "rd_oscillation_index",
    "rd_threshold_crossing_error", "epidemic_peak_error", "epidemic_attack_rate_error",
    "neural_field_pattern_rel_l2", "wave_energy_error", "memory_mean_state_error", "semiflow_error",
]


def audit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, report = [], {"excluded_from_headline": [], "metrics": {}}
    for col in [c for c in DIAGNOSTICS if c in df.columns]:
        vals = pd.to_numeric(df[col], errors="coerce")
        finite = vals[np.isfinite(vals)]
        flags = []
        if vals.isna().any(): flags.append("contains_nan")
        if len(finite) == 0: flags.append("no_finite_values")
        if len(finite) and float(finite.std()) < 1e-12: flags.append("degenerate_constant")
        if len(finite) and (finite < 0).any(): flags.append("negative_values")
        unreliable = bool(flags)
        if unreliable:
            report["excluded_from_headline"].append(col)
        report["metrics"][col] = {"flags": flags, "unreliable": unreliable}
        rows.append({"diagnostic": col, "mean": float(finite.mean()) if len(finite) else np.nan, "std": float(finite.std()) if len(finite) else np.nan, "finite_count": int(len(finite)), "nan_count": int(vals.isna().sum()), "unreliable": unreliable, "flags": ";".join(flags)})
    return pd.DataFrame(rows), report


def main():
    ap = argparse.ArgumentParser(description="Audit reliability of benchmark diagnostic metrics.")
    ap.add_argument("--metrics", default="outputs/results/raw_metrics.csv")
    ap.add_argument("--output-dir", default="outputs/results/diagnostics")
    args = ap.parse_args()
    df = pd.read_csv(args.metrics)
    summary, report = audit(df)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "diagnostic_summary.csv", index=False)
    (out / "diagnostic_reliability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote diagnostic audit under {out}")

if __name__ == "__main__":
    main()
