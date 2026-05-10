# HS-FNO: History-Space Fourier Neural Operator for Non-Markovian PDEs

This repository contains the PyTorch implementation and experimental code for **HS-FNO (History-Space Fourier Neural Operator)**, a neural-operator surrogate for delay and memory-driven partial differential equations.

Delay and memory-driven PDEs are generally not Markovian in the instantaneous field \(u(t,x)\). Their natural state is the history segment

\[
u_t(\theta,x)=u(t+\theta,x), \qquad \theta\in[-\tau,0].
\]

HS-FNO uses this structure directly. Instead of learning a full history-to-history map, the model predicts only the newly exposed future slice and updates the remaining history window using exact deterministic shift-append transport. This separates the learned part of the update from the part already determined by the previous history state.

The repository includes benchmark generation, reference solvers, model training, evaluation scripts, baselines, ablations, and tools for reproducing the reported tables and figures.

## Repository overview

HS-FNO is evaluated on five delay and memory-driven PDE benchmark families:

- delayed reaction-diffusion,
- spatial epidemiology with delayed infectiousness,
- nonlocal neural-field dynamics with delayed coupling,
- delayed wave dynamics,
- distributed-memory PDE closures.

The main comparison includes:

- current-state neural operator,
- lag-stack neural operator,
- unconstrained history-to-history operator,
- ConvLSTM,
- temporal U-Net,
- temporal transformer,
- HS-FNO variants and ablations.

The primary metrics are one-step relative error, history-space relative error, rollout relative error, efficiency, and robustness under held-out delay, held-out parameter, and resolution-transfer regimes.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

Python 3.10+ is recommended.

The code uses PyTorch, NumPy, SciPy-compatible dependencies, matplotlib, pandas, tqdm, PyYAML, and pytest. CUDA is used automatically when available, but the default settings are CPU-compatible.

## Quickstart

Run a small smoke-test version of the benchmark suite:

```bash
python run_all_experiments.py --quick --overwrite
```

Run the full default experiment suite:

```bash
python run_all_experiments.py
```

The default full run uses ten independent seeds:

```text
7, 8, 9, 10, 11, 12, 13, 14, 15, 16
```

Run a smaller seed set:

```bash
python run_all_experiments.py --seeds 7 8 9
```

Run only HS-FNO:

```bash
python run_all_experiments.py --models hs_fno
```

Run the core comparison:

```bash
python run_all_experiments.py --models current_state lag_stack history2history hs_fno
```

Use `--overwrite` to regenerate datasets, checkpoints, evaluation logs, and summary metrics.

## Running a single benchmark

Each benchmark has a YAML config in `configs/`.

Example:

```bash
python scripts/train_one.py configs/delayed_reaction_diffusion.yaml --quick --overwrite
```

Available benchmark configs:

```text
configs/delayed_reaction_diffusion.yaml
configs/epidemic_delay.yaml
configs/nonlocal_neural_field.yaml
configs/delayed_wave.yaml
configs/distributed_memory.yaml
```

## Method summary

HS-FNO represents the delay-PDE state as a discrete history tensor. Given a history window, the model predicts only the unknown future slice. The next history state is then assembled by shifting the known history forward and appending the predicted slice.

This update has the form:

[
u_{t+\Delta t}(\theta,\cdot)
============================

u_t(\theta+\Delta t,\cdot),
\qquad \theta \in [-\tau,-\Delta t],
]

with only the newly exposed slice requiring prediction.

The main model uses a Fourier neural operator over the history-space domain. For one-dimensional spatial problems, this is a field over ((\theta,x)). For two-dimensional spatial problems, this is a field over ((\theta,x_1,x_2)).

## Implemented components

The repository includes:

* method-of-steps reference solvers with interpolation delay buffers,
* delayed reaction-diffusion, epidemic delay, neural-field, delayed-wave, and distributed-memory benchmark generators,
* smooth random initial-history generation,
* trajectory-level train/validation/test splits,
* sliding history-window supervised examples,
* k-step rollout targets,
* HS-FNO with exact shift-append history transport,
* current-state, lag-stack, history-to-history, ConvLSTM, U-Net, and transformer baselines,
* no-shift, conditioning, history-resolution, and backbone ablations,
* normalized-space training and physical-space evaluation,
* one-step, history-space, rollout, efficiency, and robustness metrics,
* in-distribution, held-out-delay, held-out-parameter, and resolution-transfer regimes.

## Configuration

Shared defaults are defined in:

```text
configs/defaults.yaml
```

This file controls:

* data sizes,
* seed lists,
* model lists,
* training defaults,
* early stopping,
* per-model hyperparameters,
* quick-mode overrides,
* ablation settings.

Benchmark-specific YAML files define equation parameters, solver settings, delays, boundary conditions, and evaluation regimes.

## Outputs

Running experiments creates an `outputs/` directory:

```text
outputs/
├── data/                 # generated NPZ trajectories and metadata
├── checkpoints/          # model checkpoints
├── logs/                 # training and evaluation logs
├── metrics/              # aggregate metric CSV/JSON files
├── plots/                # generated plots
└── results/
    ├── raw_metrics.csv
    ├── raw_metrics.jsonl
    ├── run_summary.csv
    ├── main_multiseed/
    ├── statistics/
    ├── tables/
    ├── hparams/
    └── runs/*/config.lock.json
```

Each evaluated run stores a resolved configuration and metadata record, including command-line arguments, package versions, Python/CUDA/device information, seed, and config hash.

## Reproducing results

After running the experiment suite, aggregate the results with:

```bash
python analyze_all_results.py
```

This produces summary tables and statistics under:

```text
outputs/metrics/
outputs/results/
```

The primary paper tables and figures should be regenerated from:

```text
outputs/results/raw_metrics.csv
outputs/results/raw_metrics.jsonl
```

rather than entered manually.

Useful workflows include:

```bash
make analyze
make plots
make all-paper-results
```

if the provided `Makefile` is available.

## Resume behavior

The experiment runner reuses completed per-seed and per-model evaluation logs when possible. If a checkpoint exists but the corresponding evaluation log is missing or incomplete, the runner loads the checkpoint and completes evaluation instead of retraining.

Use:

```bash
python run_all_experiments.py --overwrite
```

when you want to regenerate artifacts from scratch.

## Tests

Run the test suite with:

```bash
python -m pytest -q
```

The tests cover:

* shift-append behavior,
* delay interpolation,
* rollout target construction,
* OOD and resolution-transfer regime construction,
* normalization and decoding,
* model naming,
* model output shapes,
* analysis utilities,
* small solver smoke runs.

## Data notes

The synthetic PDE benchmark datasets are generated by the scripts in this repository.

If using external traffic datasets for the optional real-world sanity check, this repository does not need to redistribute the raw METR-LA or PEMS-BAY files. Users should obtain those datasets from their public benchmark sources and place them in the expected local data directory before running the corresponding scripts.

## Reproducibility notes

* Default experiments use ten independent seeds.
* Random seeds are set through `hsno.utils.seed.set_seed`.
* Seed-specific artifacts include `_seed<value>` in their filenames.
* Normalization statistics are fit only on training trajectories.
* Dataset splits are by trajectory, not by sliding window.
* Resolved configs are saved for each evaluated run.
* Metrics are computed after decoding predictions back to physical space.

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@misc{shikhman2026hsfno,
  title        = {HS-FNO: History-Space Fourier Neural Operator for Non-Markovian Partial Differential Equations},
  author       = {Shikhman, Lennon J.},
  year         = {2026},
  note         = {Code available at https://github.com/lennonshikhman/hs-fno}
}
```

## License

See `LICENSE` for licensing information.
