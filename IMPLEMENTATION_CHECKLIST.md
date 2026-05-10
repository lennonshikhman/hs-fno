# HSNO Implementation Checklist

This checklist audits the current repository against the original requested feature set for the paper-style “History-Space Neural Operators for Non-Markovian Partial Differential Equations” codebase.

Ordering is built in as:

1. **DONE**
2. **PARTIAL**
3. **DNE** (does not exist)

Within each status group, items are ordered by dependency: foundational infrastructure first, then data/solvers, then models/training, then evaluation/plots/tests/docs.

## DONE

| Feature | Dependency order | Current evidence / note |
|---|---:|---|
| Required repository scaffold exists | 1 | Top-level runner, configs, package modules, scripts, outputs placeholder, and tests are present. |
| `run_all_experiments.py` entrypoint exists | 2 | The runner defines the five benchmark config files and a main loop over them. |
| Config loading is wired into the runner | 3 | The runner calls `load_config` for every benchmark config. |
| `--quick` CLI flag exists | 4 | The runner defines `--quick`, and `configs/defaults.yaml` includes quick overrides. |
| Dataset generation reuses existing generated data unless overwrite is requested | 5 | `generate_trajectories` returns the existing NPZ path when present and `overwrite=False`. |
| Full end-to-end experiment protocol now orchestrates concrete regimes | 6 | `run_benchmark` now evaluates separate in-distribution, held-out-delay, held-out-parameter, and resolution-transfer datasets and includes configured ablation runs. |
| Quick mode includes the major smoke model families | 7 | Defaults now select current-state, lag-stack, history-to-history, ConvLSTM, temporal-transformer, and HSNO, with quick data/training overrides. |
| Basic trajectory-level train/validation/test split exists | 8 | `split_indices` splits trajectory indices, and `prepare_data` builds train/validation/test datasets from those indices. |
| Multiple trajectories can be generated per benchmark | 9 | `generate_trajectories` loops over `n_train + n_val + n_test`. |
| Parameter sampling exists | 10 | `sample_params` samples uniformly from two-element ranges. |
| Sliding history-window examples exist | 11 | `make_history_windows` builds input history, shifted target history, and exposed future segment windows. |
| Train-only field normalization stats are reused for validation/test | 12 | `prepare_data` fits the normalizer on train and passes it to validation/test datasets. |
| Channelwise field normalization exists | 13 | `ChannelNormalizer` computes per-channel mean/std and encodes/decodes tensors. |
| Conditioning normalization exists | 14 | `HistoryWindowDataset` computes conditioning mean/std and reuses supplied conditioning stats. |
| `DelayBuffer.lookup` provides linear interpolation | 15 | The buffer clamps outside its time range and linearly interpolates between bracketing states. |
| Centered finite-difference Laplacian helpers exist | 16 | 1D and 2D finite-difference Laplacians are implemented. |
| Configurable solver diffusion choices are honored | 17 | Local solvers now route diffusion through a shared operator that honors finite-difference or spectral scheme choices where applicable. |
| Fourier spectral derivatives are wired into solvers | 18 | 1D/2D Fourier Laplacians are available and used for periodic problems when `scheme: spectral` is selected. |
| Boundary-compatible smooth initial histories exist | 19 | `smooth_history` now selects Fourier, sine, or cosine bases for periodic, Dirichlet, or Neumann-style boundaries. |
| Improved nonnegative density history handling exists | 20 | Density-like histories are rescaled into a positive interval before simulation, with solver clipping metadata retained where applicable. |
| Scalar-delay history windows use per-trajectory delay grids | 21 | `make_tau_history_windows` samples `current_time + theta_j` with `theta_j=-tau+j*tau/M` rather than relying on consecutive saved frames. |
| Delay-field summaries are supplied as neural-field static channels | 22 | The neural-field solver returns static channels summarizing pairwise delay and kernel rows for model conditioning. |
| Saved trajectories include negative initial history times | 23 | Solvers now save initial history states and times along with positive-time rollout states for method-of-steps datasets. |
| d=2 reaction-diffusion smoke support exists | 24 | The solver supports 2D histories and the smoke tests include a 2D reaction-diffusion case. |
| d=2 distributed-memory smoke support exists | 25 | The solver supports 2D histories and the smoke tests include a 2D distributed-memory case. |
| Method-of-steps buffers keep sorted initial-history and positive-time states | 26 | `DelayBuffer` now preserves sorted times and supports seeded histories on `[-tau,0]` followed by appended states. |
| Pairwise source delay lookup is vectorized and exported as metadata | 27 | `DelayBuffer.lookup_pairwise_sources` vectorizes source-node interpolation, and neural-field metadata records pairwise delay/kernel matrices. |
| `Delta t` conditioning is included | 28 | Dataset conditioning vectors append `delta_t`, and model construction derives `cond_dim` from dataset conditioning keys. |
| Delayed reaction-diffusion solver file/class exists | 29 | `ReactionDiffusionSolver` simulates a delayed logistic reaction-diffusion system. |
| Epidemic latency solver file/class exists | 30 | `EpidemicDelaySolver` simulates scalar infected density with static susceptibility. |
| Static epidemic susceptibility is supplied as a static channel | 31 | The epidemic solver returns `static=S`, and datasets save static arrays. |
| Nonlocal neural-field solver file/class exists | 32 | `NeuralFieldSolver` builds a distance kernel and uses pairwise delayed lookups. |
| Neural-field kernel rows are normalized | 33 | The kernel is divided by row sums. |
| Neural-field activation is configurable between tanh/sigmoid | 34 | The solver chooses `np.tanh` or sigmoid based on `sigma`. |
| Delayed wave solver file/class exists | 35 | `DelayedWaveSolver` evolves `(u, v)` as a first-order system. |
| Wave force options exist | 36 | Linear, tanh, and cubic force forms are implemented. |
| Distributed-memory solver file/class exists | 37 | `DistributedMemorySolver` approximates a memory term over history states. |
| Distributed-memory kernel options exist | 38 | Uniform, gamma-like, and exponential/default kernels are implemented. |
| Core `shift_append(history, predicted, step_slices)` exists | 39 | `shift_append` validates dimensions, shifts known history, and appends predicted slices. |
| `shift_append` supports 1D histories | 40 | Implementation is dimension-generic and tests cover `[B,H,C,nx]`. |
| `shift_append` supports 2D histories | 41 | Implementation is dimension-generic and tests cover `[B,H,C,nx,ny]`. |
| `shift_append` supports one-step predicted slices | 42 | `[B,C,...]` predictions are accepted and unsqueezed. |
| `shift_append` supports multi-slice predicted segments | 43 | `[B,m,C,...]` predictions are validated against `step_slices=m`. |
| HSNO base model exists | 44 | `HSNO` predicts future slices and applies exact shift-append. |
| HSNO predicts only newly exposed future segment | 45 | `FutureSliceOperator` outputs `step_slices * channels`, then `HSNO.forward` shift-appends it. |
| Current-state baseline exists | 46 | `CurrentStateNO` uses only the final history slice. |
| Lag-stack baseline exists | 47 | `LagStackNO` uses the last `lags` history slices. |
| Unconstrained history-to-history baseline exists | 48 | `HistoryToHistoryNO` predicts a full history directly. |
| HSNO-without-shift ablation is configured | 49 | `hsno_no_shift` maps to `HistoryToHistoryNO`, and default ablations include it under an ablation label. |
| ConvLSTM baseline preserves spatial structure | 50 | The ConvLSTM baseline now uses convolutional recurrent gates over the grid instead of spatial averaging and broadcasting. |
| Transformer-over-history baseline preserves spatial structure | 51 | The transformer baseline now treats every grid point as a temporal token sequence and returns spatially varying futures. |
| FNO backbone uses learned spectral convolutions | 52 | `FNOStyle` now contains FFT-based 1D/2D spectral convolution blocks with learned complex low-mode weights. |
| U-Net backbone has encoder/decoder skip connections | 53 | `ConvUNet` now implements downsampling, upsampling, and skip-connected convolutional blocks for 1D/2D grids. |
| HSNO backbone factory supports conv, FNO, and transformer backbones | 54 | `make_backbone` now dispatches to U-Net/conv, FNO, and spatial-transformer backbones. |
| HSNO backbone ablations include transformer comparison | 55 | Default ablations include conv-vs-FNO-vs-transformer HSNO backbone variants. |
| Data loss over history space exists | 56 | `data_loss` computes MSE between predicted and target histories. |
| Loss weights are configurable | 57 | Training config includes rollout/semiflow weights, and the trainer reads them. |
| Normalized-space training is used | 58 | Dataset items are normalized before training loss computation. |
| Basic training loop exists | 59 | `Trainer.fit` iterates dataloaders, optimizes model parameters, validates, and saves checkpoints. |
| Early-stopping training defaults are configured | 59.1 | Full configs use 100 epochs with patience-based early stopping, min-delta tracking, best-checkpoint restoration, weight decay, and optional gradient clipping. |
| Physical one-step and history relative metrics are computed | 60 | The evaluator decodes normalized histories before computing one-step and history relative L2 values. |
| Parameter count is reported | 61 | The evaluator includes `param_count`. |
| Inference time per step is reported | 62 | The evaluator calls an inference timing helper. |
| Output dimension is reported | 63 | The evaluator reports `target_history.numel()`. |
| Metrics CSV/JSON and summary CSVs are written | 64 | `summarize` writes `all_metrics.csv`, `all_metrics.json`, and grouped summaries. |
| Output directories for plots are created | 65 | Plotting creates the requested plot subdirectories. |
| Config copies and training logs are saved | 66 | `run_benchmark` saves copied configs and model training histories under logs. |
| Basic overwrite protection exists for metrics/checkpoints | 67 | Existing metrics and checkpoints raise unless `--overwrite` is supplied. |
| `scripts/generate_all_data.py` exists | 68 | Script iterates benchmark configs and generates data. |
| `scripts/train_one.py` exists | 69 | Script runs one benchmark config. |
| `test_shift_append.py` covers core shifting behavior | 70 | Tests cover `m=1`, `m=2`, 1D, and 2D. |
| `test_delay_buffer.py` covers basic interpolation | 71 | Test checks exact lookup, interpolation, and clamping. |
| `test_shapes.py` covers basic model output shapes | 72 | Test instantiates model names and checks output shapes. |
| `test_solvers_smoke.py` covers all five solvers at tiny scale | 73 | Test checks finite outputs for every solver family. |
| README exists | 74 | README includes install, quickstart, outputs, configs, runtime notes, tests, and reproducibility notes. |
| Benchmark-specific fidelity diagnostics are implemented | 75 | Evaluator now emits benchmark-specific diagnostics for reaction-diffusion oscillations/thresholds, epidemic peak/attack-rate errors, neural-field pattern errors, wave energy errors, and distributed-memory mean-state errors. |
| Neural-field operator conditioning is production-grade for regular grids | 76 | The solver exports full pairwise matrices in metadata and feeds fixed-count operator-probe static channels that encode kernel and pairwise-delay actions while remaining compatible with resolution transfer. |
| Delayed-wave CFL protection is implemented | 77 | The delayed-wave solver computes a CFL limit, substeps unsafe requested time steps, and records requested/internal step metadata. |
| Distributed-memory structured kernel conditioning is implemented | 78 | The distributed-memory solver emits kernel moment static channels and records the quadrature weights/theta grid in metadata. |
| Sequence baselines reuse the configured spatial backbone family | 79 | ConvLSTM and temporal-transformer baselines now use their sequence core followed by the same configurable spatial readout backbone used by HSNO/current/lag/history models. |
| Correct true k-step rollout targets are implemented | 80 | History windows now include `rollout_targets` with true S_{kΔt} history states sampled from the saved trajectory timeline. |
| Correct supervised rollout loss is implemented | 81 | `rollout_loss` compares autoregressive predictions against the corresponding k-step rollout targets instead of a repeated one-step target. |
| Physical-space one-step/history/rollout metrics are implemented | 82 | Evaluator decodes normalized predictions and targets with the training normalizer before computing relative errors. |
| Robust rollout and oscillatory metrics are implemented | 83 | Evaluator reports per-step rollout errors and computes wave/neural-field amplitude, frequency, and phase metrics from rollout traces. |
| Peak memory and speedup efficiency metrics are implemented | 84 | Evaluator tracks CPU/CUDA peak memory and computes solver speedup from measured trajectory-generation solver timings. |
| Required aggregate plots use real metrics | 85 | Plotting now produces rollout curves, one-step-vs-rollout bars, error-vs-delay, error-vs-resolution, efficiency speedup/memory, and phase/amplitude plots from actual metric columns. |
| `scripts/make_tables.py` produces experiment-suite tables | 86 | The table script writes benchmark/model, regime/model, and benchmark/regime CSV summaries. |
| `scripts/evaluate_one.py` is a standalone evaluator | 87 | The script loads a checkpoint, evaluates it on all concrete regimes, and writes an evaluation CSV without retraining. |
| Expanded regression tests cover rollout/OOD/physical metric foundations | 88 | Tests now cover true k-step rollout targets, held-out/resolution regime construction, and physical-space decode behavior. |
| README/checklist accuracy updated | 89 | Documentation no longer describes the remaining completed partials as placeholders. |
| Recursive one-step application for `step_slices > 1` is implemented | 90 | HSNO supports `append_mode: recursive`, repeatedly predicting one exposed slice and composing exact shift-append updates for multi-slice segments. |
| Semi-implicit time stepping is implemented for local diffusion solvers | 91 | Reaction-diffusion, epidemic, and distributed-memory solvers support IMEX updates with implicit finite-difference diffusion solves. |
| Variable-step model API is implemented | 92 | Models accept raw `delta_t` overrides and replace the normalized `delta_t` conditioning channel through a shared conditioning helper. |
| HSNO full model variant exists | 93 | `hsno_full` builds an HSNO variant configured for recursive shift-append plus rollout and variable-step semiflow losses. |
| Variable-step semiflow consistency loss is implemented | 94 | `variable_semiflow_loss` evaluates `G_r(G_s(h))` against `G_{s+r}(h)` using distinct raw step-size conditioning values. |
| Variable-step semiflow metric is implemented | 95 | The evaluator reports a non-placeholder consistency metric generated from variable-step model calls. |
| Temporal U-Net baseline exists | 96 | `TemporalUNetOperator` performs encoder/decoder convolutions along the history axis at every spatial node and is available through `build_model('temporal_unet')`. |
| Coordinate embeddings are implemented | 97 | `coordinate_channels` generates normalized 1D/2D coordinate channels, and `FutureSliceOperator` can concatenate them via `models.coordinate_embedding`. |
| FiLM/modulation conditioning is implemented | 98 | `FiLM` provides conditioning-generated affine feature modulation for future-slice readouts via `models.film`. |
| Separate model per delay comparison is implemented | 99 | The benchmark runner trains low/high delay-bin HSNO ablations and evaluates each on matching delay-bin subsets. |
| Input history length ablation is implemented | 100 | The benchmark runner expands configured `ablations.history_steps` into concrete HSNO runs with rebuilt datasets. |
| History resolution `M` ablation is implemented | 101 | The benchmark runner expands configured `ablations.history_resolutions` into concrete HSNO runs with rebuilt delay-grid windows. |
| Explicit delay conditioning vs no-delay/separate-delay ablations are implemented | 102 | The runner supports delay-conditioning-off and separate-per-delay ablations in addition to standard delay-conditioned HSNO. |
| Example trajectory prediction-vs-reference plots are implemented | 103 | `save_prediction_example` rolls out a trained model on a held sample and saves predicted-vs-reference observable curves. |
| End-to-end quick-mode test exists | 104 | `tests/test_quick_mode.py` invokes `python run_all_experiments.py --quick` against a temporary output directory. |
| Current-state impossibility toy diagnostic exists | 105 | `tests/test_current_state_impossibility.py` constructs two histories with the same current slice but different delayed-copy futures and verifies current-state-only models produce identical predictions for both. |

## PARTIAL

No remaining items are classified as **PARTIAL**. Items with complete, runnable implementations are listed under **DONE**.

## DNE

No remaining items are classified as **DNE**.
