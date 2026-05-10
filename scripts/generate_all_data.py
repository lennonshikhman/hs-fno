#!/usr/bin/env python
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import argparse
from hsno.utils.config import load_config
from hsno.data.datasets import generate_trajectories


ap = argparse.ArgumentParser(description="Generate seed-specific benchmark datasets")
ap.add_argument('--quick', action='store_true')
ap.add_argument('--overwrite', action='store_true')
ap.add_argument('--seeds', nargs='*', type=int, default=None)
args = ap.parse_args()
for p in Path('configs').glob('*.yaml'):
    if p.name == 'defaults.yaml':
        continue
    base_cfg = load_config(p, quick=args.quick)
    seeds = args.seeds if args.seeds else base_cfg.get('seeds', [base_cfg.get('seed', 7)])
    for seed in seeds:
        cfg = deepcopy(base_cfg)
        cfg['seed'] = int(seed)
        generate_trajectories(cfg, 'outputs/data', overwrite=args.overwrite)
        print(f'{p} seed={seed}')
