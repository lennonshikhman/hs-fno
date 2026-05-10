#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from hsno.utils.config import load_config
from hsno.utils.seed import set_seed
from hsno.experiments.run_benchmark import run_benchmark
from hsno.experiments.summarize import summarize
from hsno.utils.naming import ABLATION_MODELS, CORE_MODELS, canonical_model_name

BENCHMARK_CONFIGS = [
    'delayed_reaction_diffusion.yaml',
    'epidemic_delay.yaml',
    'nonlocal_neural_field.yaml',
    'delayed_wave.yaml',
    'distributed_memory.yaml',
]


def _configured_seeds(cfg: dict, cli_seeds: list[int] | None) -> list[int]:
    if cli_seeds:
        return [int(s) for s in cli_seeds]
    seeds = cfg.get('seeds')
    if seeds is None:
        seeds = [cfg.get('seed', 7)]
    return [int(s) for s in seeds]


def main():
    ap = argparse.ArgumentParser(description='Run HS-FNO / history-space benchmark suite')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--output-dir', default='outputs')
    ap.add_argument('--models', nargs='*', default=None, help='Optional explicit model list, e.g. --models hs_fno current_state')
    ap.add_argument('--seeds', nargs='*', type=int, default=None, help='Optional seed list; defaults to the 10 seeds in configs/defaults.yaml')
    ap.add_argument('--rollout-steps', type=int, default=None, help='Override evaluation/training rollout horizon K for this run')
    args = ap.parse_args()
    metrics_file = Path(args.output_dir) / 'metrics' / 'all_metrics.csv'
    if metrics_file.exists() and not args.overwrite:
        print(f'{metrics_file} exists; completed per-seed/model eval logs will be reused and the summary will be refreshed. Use --overwrite to retrain/regenerate everything.')
    metrics = []
    configs = BENCHMARK_CONFIGS
    minimal = bool(os.environ.get('HSNO_QUICK_TEST_MINIMAL'))
    if minimal:
        configs = BENCHMARK_CONFIGS[:1]
    requested_models = [canonical_model_name(m, warn=True) for m in args.models] if args.models else None
    for name in tqdm(configs, desc='benchmarks'):
        base_cfg = load_config(Path('configs') / name, quick=args.quick)
        base_cfg['output_dir'] = args.output_dir
        if args.rollout_steps is not None:
            base_cfg['training']['rollout_steps'] = int(args.rollout_steps)
        if requested_models is not None:
            base_cfg['models']['selected'] = requested_models
            base_cfg['ablations']['enabled'] = []
        else:
            base_cfg['models']['selected'] = CORE_MODELS
            base_cfg['ablations']['enabled'] = ABLATION_MODELS
        if minimal:
            base_cfg['models']['selected'] = ['current_state', 'hs_fno']
            base_cfg['ablations']['enabled'] = []
            base_cfg['data'].update({'n_train': 1, 'n_val': 1, 'n_test': 1, 'nx': 8, 'total_time': 0.2, 'dt': 0.02, 'dt_save': 0.1, 'history_steps': 3})
            base_cfg['training'].update({'epochs': 1, 'batch_size': 2, 'rollout_steps': 1})
        seeds = _configured_seeds(base_cfg, args.seeds)
        if minimal and args.seeds is None:
            seeds = seeds[:1]
        for seed in tqdm(seeds, desc=f'{Path(name).stem} seeds', leave=False):
            cfg = deepcopy(base_cfg)
            cfg['seed'] = int(seed)
            cfg['seeds'] = seeds
            set_seed(int(seed))
            metrics.extend(run_benchmark(cfg, args.overwrite))
    df = summarize(metrics, args.output_dir)
    columns = [c for c in ['seed', 'benchmark', 'model', 'regime', 'one_step_rel_l2', 'history_rel_l2', 'rollout_rel_l2'] if c in df.columns]
    print(df[columns].to_string(index=False))


if __name__ == '__main__':
    main()
