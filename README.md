# HS-FNO: History-Space Fourier Neural Operator for Non-Markovian Partial Differential Equations

This repository contains the reference PyTorch implementation and reproducibility workflow for the paper **“HS-FNO: History-Space Fourier Neural Operator for Non-Markovian Partial Differential Equations.”**

HS-FNO is designed for delay and memory-dependent PDEs whose future evolution cannot be represented as a Markovian map of the instantaneous state alone. The codebase implements the history-space formulation used in the paper: the model lifts trajectories to a history field, predicts only the newly exposed future slice or segment, and advances the history window with an exact deterministic shift-append transport.

The repository is intended to support:

- reproduction of the experiments reported in the preprint;
- comparison against current-state, lag-stack, history-to-history, recurrent, U-Net, and transformer baselines;
- ablation studies for HS-FNO design choices; and
- generation of paper-ready metrics, tables, plots, and reproducibility logs.

## Repository contents

```text
configs/                 # Shared defaults and per-benchmark experiment configs
hsno/                    # Data, solver, model, training, evaluation, and utility code
scripts/                 # Single-purpose experiment, analysis, plotting, and audit scripts
tests/                   # Unit and smoke tests for solvers, data construction, models, and logging
run_all_experiments.py   # Main multi-benchmark experiment runner
analyze_all_results.py   # Statistical report generator for completed experiments
Makefile                 # Convenience targets for paper-result workflows
```

## Method overview

For delay PDEs, the instantaneous state `u(t, x)` is generally non-Markovian: its future can depend on a past trajectory segment. HS-FNO instead works with the history state

```text
u_t(theta, x) = u(t + theta, x),   theta in [-tau, 0].
```

The main model, `hs_fno`, learns an exposed-future predictor on this history-space domain. At each step it:

1. encodes the current history window;
2. applies Fourier neural operator layers over the history-space representation;
3. predicts the next exposed future slice or segment; and
4. updates the window using deterministic `shift_append` history transport.

This repository also includes non-headline history-space variants and conventional baselines to support the comparisons and ablations described in the preprint.

## Implemented benchmarks

Benchmark configuration files live in `configs/`:

| Config | Problem family |
| --- | --- |
| `delayed_reaction_diffusion.yaml` | Delayed reaction-diffusion PDE |
| `epidemic_delay.yaml` | Epidemic latency PDE |
| `nonlocal_neural_field.yaml` | Pairwise-delay neural field |
| `delayed_wave.yaml` | Delayed wave equation |
| `distributed_memory.yaml` | Distributed-memory PDE |

The suite includes method-of-steps reference solvers with interpolation delay buffers, smooth random initial histories, trajectory-level train/validation/test splits, sliding history-window supervised examples, in-distribution and out-of-distribution evaluation regimes, and physical-space rollout metrics after decoding with train-set normalizers.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The defaults are CPU-runnable. CUDA is used automatically when a compatible PyTorch installation and GPU are available.

## Quickstart

Run a smoke version of all five benchmarks and the major model classes:

```bash
python run_all_experiments.py --quick --overwrite
```

Run only the headline HS-FNO model in quick mode:

```bash
python run_all_experiments.py --quick --models hs_fno --overwrite
```

Run one benchmark directly:

```bash
python scripts/train_one.py configs/delayed_reaction_diffusion.yaml --quick --overwrite
```

Quick mode uses smaller grids, fewer trajectories, and one training epoch per model. It is intended for installation checks and workflow validation, not for drawing scientific conclusions.

## Full experiment suite

The default full suite uses 10 independent seeds (`7` through `16`) and the model list configured in `configs/defaults.yaml`.

```bash
python run_all_experiments.py
```

Useful variants:

```bash
# Use a custom seed subset
python run_all_experiments.py --seeds 7 8 9

# Run only HS-FNO
python run_all_experiments.py --models hs_fno

# Run a compact core comparison
python run_all_experiments.py --models current_state lag_stack history2history hs_fno
```

Use `--overwrite` when you intentionally want to regenerate seed-specific datasets, checkpoints, evaluation logs, and summary metrics under `outputs/`.

## Reproducibility workflow

By default, `run_all_experiments.py` resumes completed work where possible. It reuses per-seed/per-model evaluation logs under `outputs/logs/*_seed*_eval.json`; when a checkpoint exists but evaluation is missing or incomplete, it loads the checkpoint and finishes evaluation instead of retraining.

Generated artifacts include:

```text
outputs/
├── data/                 # Seed-specific generated NPZ trajectories and JSON metadata
├── checkpoints/          # Seed-specific model state dicts
├── metrics/              # Run-level all_metrics.csv/json and bootstrap-CI summaries
├── results/
│   ├── raw_metrics.csv   # Per-run/per-trajectory/per-rollout-step records
│   ├── raw_metrics.jsonl
│   ├── run_summary.csv
│   ├── main_multiseed/   # Multi-seed summaries by model, benchmark, regime, and seed
│   ├── statistics/       # Structured HS-FNO pairwise/rank/claim statistics
│   ├── tables/           # Baseline comparison tables with uncertainty
│   ├── hparams/          # Tuning/search-space logs
│   └── runs/*/config.lock.json
├── plots/                # Rollout, prediction, delay, resolution, and efficiency plots
└── logs/                 # Copied configs, train logs, eval logs, and raw metric logs
```

For reproducibility, the workflow records resolved configs, command-line arguments, git commit, package versions, Python/CUDA/device metadata, seeds, and config hashes in machine-readable run logs. Splits are by trajectory rather than by history window, and normalization statistics are fit only on training trajectories.

## Paper tables, plots, and statistical summaries

After experiments have produced `outputs/metrics/all_metrics.csv`, generate the main Markdown statistical report with:

```bash
python analyze_all_results.py
```

The report is written to `outputs/metrics/results_for_chatgpt.md` by default. It treats `hs_fno` as the headline model and summarizes rankings, benchmark/regime winners, paired HS-FNO-vs-baseline comparisons, OOD degradation, rollout growth, efficiency tradeoffs, and ablations. Primary rankings and paired comparisons use seed-aggregated means with 95% percentile bootstrap confidence intervals over the default 10 seeds.

Convenience targets are available for common paper workflows:

```bash
make analyze
make plots
make all-paper-results
```

`make all-paper-results` launches the main multi-seed runs, capacity sweeps, ablations, analysis, and plotting. Expect the full workflow to be substantially more expensive than quick mode.

## Tests

Run the test suite with:

```bash
python -m pytest -q
```

Tests cover shift-append behavior, delay interpolation, rollout target construction, OOD/resolution regime construction, normalization decoding, model naming, analysis renaming, tensor shapes, reproducibility logging, and tiny solver smoke runs.

## Citation

If you use this repository in academic work, please cite the arXiv/SSRN preprint:

```bibtex
@misc{shikhman2026hsfno,
      title={HS-FNO: History-Space Fourier Neural Operator for Non-Markovian Partial Differential Equations}, 
      author={Lennon J. Shikhman},
      year={2026},
      eprint={2605.09523},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.09523}, 
}
```

The BibTeX entry above is a temporary citation stub; replace it with the final arXiv or SSRN metadata when preparing a manuscript.

## Notes for contributors

- Keep `hs_fno` as the headline model name in configs, metrics, and analysis outputs.
- Prefer adding new experiments through YAML configs and reusable scripts so that runs remain auditable.
- Regenerate tables and figures from `outputs/results/raw_metrics.csv` or `.jsonl` rather than hand-entering values.
- Do not treat quick-mode metrics as paper evidence; use the multi-seed workflows for scientific claims.
