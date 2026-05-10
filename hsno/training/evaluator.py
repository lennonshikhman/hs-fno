from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from hsno.utils.seed import seed_worker, torch_generator
from .metrics import rel_l2, parameter_count, inference_time, oscillatory_metrics


class Evaluator:
    def __init__(self, model, dataset, device, seed: int = 0):
        self.model = model.to(device)
        self.dataset = dataset
        self.base_dataset = getattr(dataset, "dataset", dataset)
        self.device = device
        self.seed = int(seed)

    def _move(self, b):
        return {k: v.to(self.device) for k, v in b.items()}

    def _decode_history(self, x):
        return self.base_dataset.normalizer.decode(x)

    def _decode_rollout(self, x):
        b, k = x.shape[:2]
        y = x.reshape(b * k, *x.shape[2:])
        y = self.base_dataset.normalizer.decode(y)
        return y.reshape(b, k, *y.shape[1:])

    @staticmethod
    def _mean(values):
        vals = [float(v) for v in values if np.isfinite(v)]
        return float(sum(vals) / max(1, len(vals)))

    def _benchmark_diagnostics(self, benchmark, pred_trace, ref_trace):
        p = np.asarray(pred_trace)
        r = np.asarray(ref_trace)
        out = {}
        if p.size == 0 or r.size == 0:
            return out
        if benchmark == "delayed_reaction_diffusion":
            obs = r.reshape(r.shape[0], -1).mean(1)
            out["rd_oscillation_index"] = float(np.std(np.diff(obs)) / (np.std(obs) + 1e-8)) if len(obs) > 2 else 0.0
            out["rd_threshold_crossing_error"] = float(abs((p > 0.5).mean() - (r > 0.5).mean()))
        elif benchmark == "epidemic_delay":
            po = p.reshape(p.shape[0], -1).mean(1)
            ro = r.reshape(r.shape[0], -1).mean(1)
            out["epidemic_peak_error"] = float(abs(po.max() - ro.max()) / (abs(ro.max()) + 1e-8))
            out["epidemic_attack_rate_error"] = float(abs(po.mean() - ro.mean()) / (abs(ro.mean()) + 1e-8))
        elif benchmark == "nonlocal_neural_field":
            out["neural_field_pattern_rel_l2"] = rel_l2(torch.as_tensor(p[-1]), torch.as_tensor(r[-1]))
        elif benchmark == "delayed_wave":
            disp_p = p[..., 0, :].reshape(p.shape[0], -1) if p.shape[-2] >= 1 else p.reshape(p.shape[0], -1)
            disp_r = r[..., 0, :].reshape(r.shape[0], -1) if r.shape[-2] >= 1 else r.reshape(r.shape[0], -1)
            out["wave_energy_error"] = float(abs(np.mean(disp_p**2) - np.mean(disp_r**2)) / (abs(np.mean(disp_r**2)) + 1e-8))
        elif benchmark == "distributed_memory":
            out["memory_mean_state_error"] = float(abs(p.mean() - r.mean()) / (abs(r.mean()) + 1e-8))
        return out

    def evaluate(self, benchmark, model_name, rollout_steps=3, return_raw: bool = False, context: dict[str, Any] | None = None):
        self.model.eval()
        context = context or {}
        one, hist, roll, semi = [], [], [], []
        step_error_lists = []
        solver_times, delays = [], []
        raw_rows: list[dict[str, Any]] = []
        first_pred_trace, first_ref_trace = None, None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tracemalloc.start()
        eval_start = time.perf_counter()
        loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=torch_generator(self.seed),
            num_workers=0,
        )
        with torch.no_grad():
            for b in loader:
                b = self._move(b)
                pred = self.model(b["history"], b["cond"], b["static"])
                pred_phys = self._decode_history(pred)
                target_phys = self._decode_history(b["target_history"])
                one_err = rel_l2(pred_phys[:, -1], target_phys[:, -1])
                hist_err = rel_l2(pred_phys, target_phys)
                one.append(one_err)
                hist.append(hist_err)
                targets_phys = self._decode_rollout(b["rollout_targets"])
                h = b["history"]
                pred_steps, step_errors = [], []
                kmax = min(int(rollout_steps), targets_phys.shape[1])
                diverged = False
                for k in range(kmax):
                    h = self.model(h, b["cond"], b["static"])
                    h_phys = self._decode_history(h)
                    finite = bool(torch.isfinite(h_phys).all().item())
                    step_err = rel_l2(h_phys[:, -1], targets_phys[:, k, -1]) if finite else float("inf")
                    diverged = diverged or (not finite) or (not np.isfinite(step_err))
                    step_errors.append(step_err)
                    pred_steps.append(h_phys[:, -1].detach().cpu().numpy()[0])
                    solver_step = float(b["solver_time_per_step"].detach().cpu().item())
                    delay = float(b["delay"].detach().cpu().item())
                    raw_rows.append({
                        **context,
                        "model": model_name,
                        "benchmark": benchmark,
                        "trajectory_id": int(b.get("trajectory_id", torch.tensor([-1], device=self.device)).detach().cpu().item()),
                        "window_id": int(b.get("window_id", torch.tensor([-1], device=self.device)).detach().cpu().item()),
                        "rollout_step": int(k + 1),
                        "one_step_rel_l2": float(one_err),
                        "history_rel_l2": float(hist_err),
                        "rollout_rel_l2": float(step_err),
                        "diverged": bool(diverged),
                        "solver_time_per_step": solver_step,
                        "mean_delay": delay,
                        "history_steps": int(b.get("history_steps", torch.tensor([0], device=self.device)).detach().cpu().item()),
                        "history_resolution": int(b.get("history_resolution", torch.tensor([0], device=self.device)).detach().cpu().item()),
                    })
                finite_steps = [e for e in step_errors if np.isfinite(e)]
                roll.append(sum(finite_steps) / max(1, len(finite_steps)) if finite_steps else float("inf"))
                step_error_lists.append(step_errors)
                solver_times.append(float(b["solver_time_per_step"].detach().cpu().item()))
                delays.append(float(b["delay"].detach().cpu().item()))
                cond_raw = b.get("cond_raw")
                if cond_raw is not None:
                    base_dt = cond_raw[:, -1]
                    hs = self.model(b["history"], b["cond"], b["static"], delta_t=base_dt, cond_mean=b.get("cond_mean"), cond_std=b.get("cond_std"))
                    hrs = self.model(hs, b["cond"], b["static"], delta_t=base_dt, cond_mean=b.get("cond_mean"), cond_std=b.get("cond_std"))
                    hsr = self.model(b["history"], b["cond"], b["static"], delta_t=2.0 * base_dt, cond_mean=b.get("cond_mean"), cond_std=b.get("cond_std"))
                    semi_val = rel_l2(self._decode_history(hrs), self._decode_history(hsr))
                    semi.append(semi_val)
                    for rr in raw_rows[-kmax:]:
                        rr["semiflow_error"] = float(semi_val)
                if first_pred_trace is None and pred_steps:
                    first_pred_trace = np.stack(pred_steps, axis=0)
                    first_ref_trace = targets_phys[0, :kmax, -1].detach().cpu().numpy()
        eval_wall_time = time.perf_counter() - eval_start
        _current_mb, peak_mb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        sample = self._move(next(iter(DataLoader(self.dataset, batch_size=1)))) if len(self.dataset) else None
        infer = inference_time(self.model, sample, 3) if sample is not None else 0.0
        solver_per_step = self._mean(solver_times)
        osc = {"amplitude_error": 0.0, "dominant_frequency_error": 0.0, "phase_drift": 0.0}
        diagnostics = {}
        if first_pred_trace is not None and first_ref_trace is not None:
            if benchmark in {"delayed_wave", "nonlocal_neural_field"}:
                osc = oscillatory_metrics(first_pred_trace, first_ref_trace, dt=self.base_dataset.dt_save * self.base_dataset.step_slices)
            diagnostics = self._benchmark_diagnostics(benchmark, first_pred_trace, first_ref_trace)
        cuda_peak = float(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0
        rollout_curve = {}
        if step_error_lists:
            max_len = max(len(x) for x in step_error_lists)
            for k in range(max_len):
                vals = [x[k] for x in step_error_lists if k < len(x)]
                rollout_curve[f"rollout_step_{k+1}_rel_l2"] = self._mean(vals)
        common_metrics = {
            "param_count": parameter_count(self.model),
            "inference_time_per_step": float(infer),
            "output_dimension": int(self.dataset[0]["target_history"].numel()) if len(self.dataset) else 0,
            "peak_memory_mb": max(cuda_peak, float(peak_mb / 1e6)),
            "speedup_vs_solver": float(solver_per_step / infer) if infer > 0 and solver_per_step > 0 else 0.0,
            "eval_wall_time": float(eval_wall_time),
        }
        for rr in raw_rows:
            rr.update(common_metrics)
            rr.setdefault("semiflow_error", self._mean(semi))
        summary = {
            **context,
            "benchmark": benchmark,
            "model": model_name,
            "one_step_rel_l2": self._mean(one),
            "history_rel_l2": self._mean(hist),
            "rollout_rel_l2": self._mean(roll),
            "semiflow_error": self._mean(semi),
            "mean_delay": self._mean(delays),
            **common_metrics,
            **osc,
            **diagnostics,
            **rollout_curve,
        }
        if return_raw:
            return summary, raw_rows
        return summary
