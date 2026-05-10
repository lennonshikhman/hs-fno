#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from hsno.utils.plotting import save_basic_plots


def main():
    ap = argparse.ArgumentParser(description="Regenerate saved-result plots without retraining.")
    ap.add_argument("--metrics", default="outputs/results/raw_metrics.csv")
    ap.add_argument("--output-dir", default="figures/generated")
    args = ap.parse_args()
    df = pd.read_csv(args.metrics)
    out = Path(args.output_dir)
    save_basic_plots(df, out)
    print(f"Wrote plots under {out}")

if __name__ == "__main__":
    main()
