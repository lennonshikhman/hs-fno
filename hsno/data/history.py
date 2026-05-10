"""History-window utilities including the exact HSNO shift-append map."""
from __future__ import annotations
import numpy as np
import torch


def shift_append(history: torch.Tensor, predicted: torch.Tensor, step_slices: int) -> torch.Tensor:
    """Exactly transport known history and append newly predicted future slices.

    history shape is [B, H, C, ...]. predicted may be [B, C, ...] for one slice or
    [B, m, C, ...] for a segment. The output has the same shape as history.
    """
    if history.ndim < 4:
        raise ValueError("history must have shape [B,H,C,...]")
    h = history.shape[1]
    m = int(step_slices)
    if m < 1 or m > h:
        raise ValueError(f"step_slices must be in [1,{h}], got {m}")
    if predicted.ndim == history.ndim - 1:
        predicted = predicted.unsqueeze(1)
    if predicted.ndim != history.ndim:
        raise ValueError("predicted must be [B,C,...] or [B,m,C,...]")
    if predicted.shape[1] != m:
        raise ValueError(f"predicted segment length {predicted.shape[1]} != {m}")
    if predicted.shape[0] != history.shape[0] or predicted.shape[2:] != history.shape[2:]:
        raise ValueError("predicted batch/channel/spatial dimensions must match history")
    return torch.cat([history[:, m:], predicted], dim=1)


def interpolate_fields(fields: np.ndarray, times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Linearly interpolate a trajectory field array at requested times.

    Values outside the available range are clamped to the nearest endpoint,
    matching method-of-steps delay-buffer semantics.
    """
    fields = np.asarray(fields)
    times = np.asarray(times, dtype=np.float64)
    query_times = np.asarray(query_times, dtype=np.float64)
    out = np.empty((len(query_times),) + fields.shape[1:], dtype=fields.dtype)
    for i, t in enumerate(query_times):
        if t <= times[0]:
            out[i] = fields[0]
        elif t >= times[-1]:
            out[i] = fields[-1]
        else:
            j = int(np.searchsorted(times, t))
            t0, t1 = times[j - 1], times[j]
            alpha = (t - t0) / max(t1 - t0, 1e-12)
            out[i] = (1.0 - alpha) * fields[j - 1] + alpha * fields[j]
    return out


def make_history_windows(fields, history_steps: int, step_slices: int = 1):
    """Legacy consecutive-slice history windows for already aligned histories."""
    xs, ys, fut = [], [], []
    for i in range(0, fields.shape[0] - history_steps - step_slices + 1):
        h = fields[i : i + history_steps]
        y = fields[i + step_slices : i + step_slices + history_steps]
        p = fields[i + history_steps : i + history_steps + step_slices]
        xs.append(h)
        ys.append(y)
        fut.append(p)
    return xs, ys, fut


def make_tau_history_windows(
    fields: np.ndarray,
    times: np.ndarray,
    tau: float,
    history_steps: int,
    step_slices: int,
    delta_t: float,
    rollout_steps: int = 1,
):
    """Construct supervised windows on a per-trajectory history grid.

    Each input history is sampled at ``current_time + theta_j`` with
    ``theta_j = -tau + j * tau / (history_steps - 1)``. The target history for
    rollout step ``k`` is sampled after ``k * step_slices * delta_t``. Returning
    all rollout targets makes k-step rollout losses and metrics compare against
    the true history-space solution map instead of a repeated one-step target.
    """
    theta = np.linspace(-float(tau), 0.0, history_steps, dtype=np.float64)
    step_dt = float(step_slices) * float(delta_t)
    start = max(0.0, float(times[0]) + float(tau))
    stop = float(times[-1]) - max(1, int(rollout_steps)) * step_dt
    if stop < start:
        return [], [], [], []
    current_times = np.arange(start, stop + 1e-12, float(delta_t), dtype=np.float64)
    xs, ys, fut, rollouts = [], [], [], []
    future_offsets = float(delta_t) * np.arange(1, step_slices + 1, dtype=np.float64)
    for t in current_times:
        xs.append(interpolate_fields(fields, times, t + theta))
        targets = [interpolate_fields(fields, times, t + k * step_dt + theta) for k in range(1, max(1, int(rollout_steps)) + 1)]
        ys.append(targets[0])
        fut.append(interpolate_fields(fields, times, t + future_offsets))
        rollouts.append(np.stack(targets, axis=0))
    return xs, ys, fut, rollouts
