#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Summarize learned-model runtime against reference solver timing metadata.")
    ap.add_argument("--metrics", default="outputs/results/raw_metrics.csv")
    ap.add_argument("--output-dir", default="outputs/results/solver_comparison")
    args = ap.parse_args()
    df = pd.read_csv(args.metrics)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["benchmark", "model", "regime", "seed", "solver_time_per_step", "inference_time_per_step", "speedup_vs_solver", "rollout_rel_l2"] if c in df]
    runtime = df[cols].groupby([c for c in ["benchmark", "model", "regime"] if c in cols]).mean(numeric_only=True).reset_index()
    runtime.to_csv(out / "solver_runtime.csv", index=False)
    if {"rollout_rel_l2", "inference_time_per_step"}.issubset(df.columns):
        tta = runtime.copy()
        tta["time_to_accuracy_proxy"] = tta["inference_time_per_step"] * tta["rollout_rel_l2"]
        tta.to_csv(out / "time_to_accuracy.csv", index=False)
    print(f"Wrote solver comparison under {out}")

if __name__ == "__main__":
    main()
