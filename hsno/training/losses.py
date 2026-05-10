from __future__ import annotations
import torch


def mse(a, b):
    return torch.mean((a - b) ** 2)


def data_loss(pred_hist, target_hist):
    return mse(pred_hist, target_hist)


def slice_loss(pred_hist, target_hist, step_slices=1):
    return mse(pred_hist[:, -step_slices:], target_hist[:, -step_slices:])


def rollout_loss(model, batch, steps=2):
    """Supervised k-step history-space rollout loss against true targets."""
    targets = batch.get("rollout_targets")
    if targets is None:
        targets = batch["target_history"].unsqueeze(1)
    kmax = min(int(steps), targets.shape[1])
    h = batch["history"]
    loss = 0.0
    for k in range(kmax):
        h = model(h, batch.get("cond"), batch.get("static"))
        loss = loss + mse(h, targets[:, k])
    return loss / max(kmax, 1)


def variable_semiflow_loss(model, batch, s_factor=1.0, r_factor=1.0):
    """Variable-step consistency ``G_r(G_s(h))`` versus ``G_{s+r}(h)``.

    The model API accepts raw ``delta_t`` values and replaces the normalized
    delta-t conditioning entry internally. This loss exercises distinct step-size
    conditioning paths rather than reusing a detached one-step placeholder.
    """
    cond_raw = batch.get("cond_raw")
    if cond_raw is None:
        return torch.zeros((), device=batch["history"].device, dtype=batch["history"].dtype)
    base_dt = cond_raw[:, -1]
    mean = batch.get("cond_mean")
    std = batch.get("cond_std")
    hs = model(batch["history"], batch.get("cond"), batch.get("static"), delta_t=base_dt * float(s_factor), cond_mean=mean, cond_std=std)
    hrs = model(hs, batch.get("cond"), batch.get("static"), delta_t=base_dt * float(r_factor), cond_mean=mean, cond_std=std)
    hsr = model(batch["history"], batch.get("cond"), batch.get("static"), delta_t=base_dt * float(s_factor + r_factor), cond_mean=mean, cond_std=std)
    return mse(hrs, hsr)


def semiflow_loss(model, batch):
    return variable_semiflow_loss(model, batch)
