from __future__ import annotations
import argparse
from pathlib import Path
import torch
import pandas as pd
from hsno.experiments.common import prepare_data, make_model, device_from_config
from hsno.training.evaluator import Evaluator
from hsno.utils.config import load_config
from hsno.utils.naming import canonical_model_name


def main():
    ap = argparse.ArgumentParser(description="Evaluate one trained checkpoint on all concrete regimes")
    ap.add_argument("config")
    ap.add_argument("--model", default="hs_fno")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=None, help="Seed associated with the checkpoint/data; defaults to config seed")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.model = canonical_model_name(args.model, warn=True)
    cfg = load_config(args.config, quick=False)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    train, _, regimes = prepare_data(cfg, overwrite=False)
    sample_static_channels = int(train[0]["static"].shape[0]) if len(train) else cfg.get("static_channels", 0)
    model = make_model(cfg, args.model, cond_dim=len(train.cond_keys), static_channels=sample_static_channels)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    device = device_from_config(cfg)
    rows = []
    for regime, dataset in regimes.items():
        ev = Evaluator(model, dataset, device).evaluate(cfg["benchmark"], args.model, cfg["training"].get("rollout_steps", 3))
        ev["regime"] = regime
        ev["seed"] = int(cfg.get("seed", 0))
        sample_shape = dataset[0]["history"].shape if len(dataset) else None
        ev["eval_nx"] = int(sample_shape[-1] if len(sample_shape) == 3 else sample_shape[-2]) if sample_shape is not None else cfg["data"]["nx"]
        rows.append(ev)
    out = Path(cfg["output_dir"]) / "metrics" / f"{cfg['benchmark']}_{args.model}_seed{int(cfg.get('seed', 0))}_evaluation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
