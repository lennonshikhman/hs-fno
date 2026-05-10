#!/usr/bin/env python
from __future__ import annotations
import argparse
from copy import deepcopy
from hsno.utils.config import load_config
from hsno.utils.seed import set_seed
from hsno.experiments.run_benchmark import run_benchmark
from hsno.experiments.summarize import summarize


def _seeds(cfg: dict, cli_seeds: list[int] | None) -> list[int]:
    if cli_seeds:
        return [int(s) for s in cli_seeds]
    return [int(s) for s in cfg.get("seeds", [cfg.get("seed", 7)])]


ap = argparse.ArgumentParser()
ap.add_argument('config')
ap.add_argument('--quick', action='store_true')
ap.add_argument('--overwrite', action='store_true')
ap.add_argument('--seeds', nargs='*', type=int, default=None)
args = ap.parse_args()
base_cfg = load_config(args.config, quick=args.quick)
metrics = []
for seed in _seeds(base_cfg, args.seeds):
    cfg = deepcopy(base_cfg)
    cfg['seed'] = seed
    set_seed(seed)
    metrics.extend(run_benchmark(cfg, args.overwrite))
summarize(metrics, base_cfg.get('output_dir', 'outputs'))
