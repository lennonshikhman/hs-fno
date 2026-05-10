#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from analysis.statistics import claim_checks


def main():
    ap = argparse.ArgumentParser(description="Write HS-FNO claim-evidence checks from saved raw metrics.")
    ap.add_argument("--metrics", default="outputs/results/raw_metrics.csv")
    ap.add_argument("--output-dir", default="outputs/results/claim_checks")
    args = ap.parse_args()
    df = pd.read_csv(args.metrics)
    checks = claim_checks(df)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    import json
    (out / "claim_evidence_report.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    pd.json_normalize(checks).to_csv(out / "claim_evidence_report.csv", index=False)
    print(f"Wrote claim checks under {out}")

if __name__ == "__main__":
    main()
