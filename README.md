# HS-FNO: History-Space Fourier Neural Operator for Non-Markovian Partial Differential Equations

This repository is a complete, CPU-runnable PyTorch implementation of a reproducibility-oriented experimental suite for **HS-FNO (History-Space Fourier Neural Operator)**. Delay PDEs are non-Markovian on the instantaneous state `u(t,x)` but Markovian on the history state `u_t(theta,x)=u(t+theta,x)`. The main model, **HS-FNO**, lifts delay PDE states to history fields, parameterizes the exposed-future predictor with a Fourier neural operator over the history-space domain, predicts only the newly exposed future slice/segment, and updates the history window using exact deterministic `shift_append` transport.

The broader family can still be described as history-space neural operators. Generic HSNO/U-Net, transformer, lag-stack, current-state, and history-to-history variants are retained as baselines or ablations; they are not the headline method.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is recommended. The code uses PyTorch, NumPy, SciPy-compatible dependencies, matplotlib, pandas, tqdm, PyYAML, and pytest. CUDA is used automatically when available, but all defaults are CPU-compatible.

## Quickstart

Run the smoke version of all five benchmarks and major models:

```bash
python run_all_experiments.py --quick --overwrite
```

The full default suite is launched with 10 independent seeds (`7` through `16`):

```bash
python run_all_experiments.py
```

Override the default seed set when needed:

```bash
python run_all_experiments.py --seeds 7 8 9
```

Run only the headline method:

```bash
python run_all_experiments.py --models hs_fno
```

Run a smaller core comparison:

```bash
python run_all_experiments.py --models current_state lag_stack history2history hs_fno
```

Use `--overwrite` to regenerate seed-specific datasets/checkpoints in `outputs/`.

## Run one benchmark

```bash
python scripts/train_one.py configs/delayed_reaction_diffusion.yaml --quick --overwrite
```

Available benchmark configs are in `configs/`:

- `delayed_reaction_diffusion.yaml`
- `epidemic_delay.yaml`
- `nonlocal_neural_field.yaml`
- `delayed_wave.yaml`
- `distributed_memory.yaml`

## Implemented methodology

- Method-of-steps reference solvers with interpolation delay buffers.
- Delayed reaction-diffusion, epidemic latency, pairwise-delay neural field, delayed wave, and distributed-memory PDEs.
- Smooth random initial histories and trajectory-level train/validation/test splits.
- Sliding history-window supervised examples on each trajectory's delay grid, including true k-step rollout targets.
- **HS-FNO (`hs_fno`)**: a History-Space Fourier Neural Operator that predicts only the exposed future slice and applies exact shift-append history transport.
- Baselines/ablations: current-state, lag-stack, unconstrained history-to-history, spatial ConvLSTM, temporal U-Net, temporal transformer, `hsno_unet`, `hs_transformer`, no-shift variants, conditioning variants, history-resolution variants, and optional rollout/semiflow objectives.
- Normalized-space training losses, 100-epoch full-run defaults with patience-based early stopping, optional gradient clipping/weight decay, and physical-space one-step/history/rollout relative metrics computed after decoding with train-set normalizers.
- Concrete in-distribution, held-out-delay, held-out-parameter, and resolution-transfer evaluation regimes.

## Resume behavior

By default, `run_all_experiments.py` reuses completed per-seed/per-model evaluation logs under `outputs/logs/*_seed*_eval.json`. If a seed-specific checkpoint exists but the evaluation log is missing or incomplete, the runner loads the checkpoint and finishes evaluation instead of retraining. Legacy pre-HS-FNO result names are remapped in metrics/reporting where compatible. Use `--overwrite` when you intentionally want to regenerate datasets, checkpoints, evaluation logs, and summary metrics from scratch.

## Analyze completed results for paper writing

After `run_all_experiments.py` has produced `outputs/metrics/all_metrics.csv`, generate a ChatGPT-ready statistical report with:

```bash
python analyze_all_results.py
```

The script remaps legacy model names to the HS-FNO naming scheme, treats `hs_fno` as the headline model, prints a long Markdown report, and saves it to `outputs/metrics/results_for_chatgpt.md` by default. It summarizes model rankings, benchmark/regime winners, paired HS-FNO-vs-baseline comparisons, OOD degradation, rollout growth, efficiency tradeoffs, ablations, benchmark-specific diagnostics, and paper-writing caveats. Primary rankings and paired comparisons report seed-aggregated means with 95% percentile bootstrap confidence intervals using 10,000 resamples over the default 10 seeds. Do not emphasize `semiflow_error`, dominant-frequency error, or amplitude/memory diagnostics as headline claims unless separately debugged/normalized.

## Outputs

Running experiments creates:

```text
outputs/
├── data/                 # seed-specific generated NPZ trajectories and JSON metadata
├── checkpoints/          # seed-specific model state dicts, e.g. delayed_wave_hs_fno_seed7.pt
├── metrics/              # run-level all_metrics.csv/json plus bootstrap-CI summary CSVs
├── results/
│   ├── raw_metrics.csv   # per-run/per-trajectory/per-rollout-step records
│   ├── raw_metrics.jsonl
│   ├── run_summary.csv
│   ├── main_multiseed/   # multi-seed summaries by model/benchmark-regime/seed
│   ├── statistics/       # structured HS-FNO pairwise/rank/claim statistics
│   ├── tables/           # all-baseline tables with uncertainty
│   ├── hparams/          # structured tuning/search-space logs
│   └── runs/*/config.lock.json
├── plots/                # rollout, prediction, delay, resolution, efficiency plots
└── logs/                 # copied configs, train logs, eval logs, raw metric logs
```

## Config overview

`configs/defaults.yaml` controls shared data sizes, training defaults, the default 10-seed list, per-model hyperparameters, model lists, ablations, and quick-mode overrides. Full defaults use `main_model: hs_fno`, seeds `[7, 8, 9, 10, 11, 12, 13, 14, 15, 16]`, 100 epochs with early-stopping patience of 20, and explicit per-model settings under `models.per_model`. Benchmark YAML files define equation parameters and solver options.

## Runtime notes

`--quick` uses small grids, few trajectories, and one epoch per model while retaining the configured seed list unless you pass `--seeds`. It is intended to exercise every benchmark and the main model classes. Full defaults are more in-depth (10 seeds, 100 epochs, patience 20, larger per-model widths/depths, coordinate/FiLM options); reduce selected models/ablations or pass a shorter `--seeds` list if you need a shorter full run.

## Tests

```bash
python -m pytest -q
```

Tests cover shift-append behavior, delay interpolation, rollout target construction, OOD/resolution regime construction, normalization decoding, model names, analysis renaming, model shapes, and tiny solver smoke runs.

## Reproducibility

- By default, experiments run across 10 independent seeds (`7` through `16`); override them with `--seeds`.
- Random seeds are set through `hsno.utils.seed.set_seed`, and seed-specific artifacts include `_seed<value>` in their filenames.
- Config copies are saved under `outputs/logs/`, and every evaluated run/regime gets a machine-readable `outputs/results/runs/<run_id>/config.lock.json` with the resolved config, command-line arguments, git commit, package versions, Python/CUDA/device metadata, seed, and config hash.
- Normalization statistics are fit only on training trajectories and reused for validation/test.
- Splits are by trajectory, never by window.
- Paper tables and figures should be regenerated from `outputs/results/raw_metrics.csv`/`.jsonl` rather than hand-entered values. Use `make analyze`, `make plots`, or `make all-paper-results` for reproducibility workflows.
