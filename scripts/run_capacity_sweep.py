#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from hsno.utils.config import load_config
from hsno.utils.naming import CORE_MODELS


def _estimate_params(model: str, width: int, depth: int, channels: int, history_steps: int, cond_dim: int, static_channels: int) -> int:
    """Fast capacity proxy for planning; exact counts are logged by real runs."""
    in_ch = history_steps * channels + cond_dim + static_channels
    out_ch = history_steps * channels if model == "history2history" else channels
    if model == "temporal_transformer":
        return int(depth * (12 * width * width + 4 * width) + (in_ch + out_ch) * width)
    if model == "convlstm":
        return int(depth * (4 * (channels + cond_dim + width) * width * 3 + 4 * width) + width * out_ch)
    spectral_multiplier = 8 if model == "hs_fno" else 3
    return int((in_ch * width + out_ch * width) + depth * spectral_multiplier * width * width * 16)


def _exact_param_count(base, model, width, depth):
    from hsno.experiments.common import make_model
    from hsno.training.metrics import parameter_count
    cfg = deepcopy(base)
    cfg.setdefault("models", {}).setdefault("per_model", {}).setdefault(model, {})
    cfg["models"]["per_model"][model].update({"width": int(width), "depth": int(depth)})
    m = make_model(cfg, model, cond_dim=len(cfg["params"]) + 1, static_channels=cfg.get("static_channels", 0))
    return parameter_count(m)


def plan_capacity_sweep(config: Path, output_dir: Path, exact: bool = False) -> pd.DataFrame:
    cfg = load_config(config, quick=True)
    budgets = cfg.get("capacity_sweep", {}).get("budgets", [100000, 300000, 1000000, 3000000, 10000000])
    models = cfg.get("capacity_sweep", {}).get("models", CORE_MODELS)
    widths = cfg.get("capacity_sweep", {}).get("candidate_widths", [16, 32, 64])
    depths = cfg.get("capacity_sweep", {}).get("candidate_depths", [1, 2, 3, 4])
    rows = []
    for model in models:
        candidates = []
        for w in widths:
            for d in depths:
                try:
                    params = _exact_param_count(cfg, model, w, d) if exact else _estimate_params(model, int(w), int(d), int(cfg["channels"]), int(cfg["data"]["history_steps"]), len(cfg["params"]) + 1, int(cfg.get("static_channels", 0)))
                    candidates.append((w, d, params))
                except Exception:
                    continue
        for budget in budgets:
            if not candidates:
                continue
            w, d, params = min(candidates, key=lambda x: abs(x[2] - budget))
            rows.append({"model": model, "target_budget": budget, "width": w, "depth": d, "actual_param_count" if exact else "estimated_param_count": params, "count_source": "exact_model" if exact else "fast_formula", "relative_budget_error": abs(params - budget) / budget})
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "capacity_plan.csv", index=False)
    return df


def pareto_from_raw(raw_path: Path, output_dir: Path) -> pd.DataFrame:
    if not raw_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(raw_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "raw_metrics.csv", index=False)
    needed = [c for c in ["model", "param_count", "inference_time_per_step", "peak_memory_mb", "rollout_rel_l2", "one_step_rel_l2", "history_rel_l2"] if c in df]
    if not needed:
        return pd.DataFrame()
    agg = df.groupby(["model", "param_count"], dropna=False).agg({c: "mean" for c in needed if c not in {"model", "param_count"}}).reset_index()
    agg["pareto_rollout_vs_params"] = False
    for _, g in agg.groupby("model"):
        best = float("inf")
        for idx, row in g.sort_values("param_count").iterrows():
            if row.get("rollout_rel_l2", float("inf")) <= best:
                agg.loc[idx, "pareto_rollout_vs_params"] = True
                best = row.get("rollout_rel_l2", best)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_dir / "pareto_summary.csv", index=False)
    return agg


def main():
    ap = argparse.ArgumentParser(description="Plan capacity-matched HS-FNO baseline sweeps and summarize Pareto frontiers from existing raw metrics.")
    ap.add_argument("--config", default="configs/delayed_reaction_diffusion.yaml")
    ap.add_argument("--output-dir", default="outputs/results/capacity_sweep")
    ap.add_argument("--raw", default="outputs/results/raw_metrics.csv")
    ap.add_argument("--exact-model-counts", action="store_true", help="Instantiate every candidate to log exact parameter counts; slower than the default formula planner.")
    args = ap.parse_args()
    out = Path(args.output_dir)
    plan_capacity_sweep(Path(args.config), out, exact=args.exact_model_counts)
    pareto_from_raw(Path(args.raw), out)
    print(f"Wrote capacity artifacts under {out}")


if __name__ == "__main__":
    main()
