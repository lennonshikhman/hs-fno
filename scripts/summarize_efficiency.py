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
    ap = argparse.ArgumentParser(description="Create reproducible efficiency table from raw metrics.")
    ap.add_argument("--metrics", default="outputs/results/raw_metrics.csv")
    ap.add_argument("--output-dir", default="outputs/results/efficiency")
    args = ap.parse_args()
    df = pd.read_csv(args.metrics)
    cols = [c for c in ["model", "benchmark", "regime", "seed", "inference_time_per_step", "peak_memory_mb", "param_count", "output_dimension"] if c in df]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    summary = df[cols].groupby([c for c in ["model", "benchmark", "regime"] if c in cols]).agg({c: "mean" for c in cols if c not in {"model", "benchmark", "regime", "seed"}}).reset_index()
    summary.to_csv(out / "runtime_benchmark.csv", index=False)
    print(f"Wrote {out / 'runtime_benchmark.csv'}")

if __name__ == "__main__":
    main()
