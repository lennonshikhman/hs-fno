from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from hsno.data.datasets import HistoryWindowDataset, generate_trajectories, regime_config
from hsno.data.splits import split_indices
from hsno.models.baselines import build_model
from hsno.utils.naming import canonical_model_name


def device_from_config(cfg):
    return torch.device("cuda" if cfg.get("device", "auto") == "auto" and torch.cuda.is_available() else "cpu")


def _all_indices(path):
    z = np.load(path, allow_pickle=True)
    return list(range(z["fields"].shape[0]))


def prepare_data(cfg, overwrite=False):
    """Prepare train/val/test plus real OOD/resolution datasets."""
    base_path = generate_trajectories(cfg, f"{cfg['output_dir']}/data", overwrite=overwrite, regime="in_distribution")
    n = cfg["data"]["n_train"] + cfg["data"]["n_val"] + cfg["data"]["n_test"]
    sp = split_indices(n, cfg["data"]["n_train"], cfg["data"]["n_val"], cfg.get("seed", 0))
    rollout_steps = cfg["training"].get("rollout_steps", 1)
    drop_delay = cfg["data"].get("drop_delay_conditioning", False)
    train = HistoryWindowDataset(base_path, sp["train"], cfg["data"]["history_steps"], cfg["data"]["step_slices"], rollout_steps=rollout_steps, drop_delay_conditioning=drop_delay)
    val = HistoryWindowDataset(base_path, sp["val"], cfg["data"]["history_steps"], cfg["data"]["step_slices"], train.normalizer, (train.cond_mean, train.cond_std), rollout_steps=rollout_steps, drop_delay_conditioning=drop_delay)
    test = HistoryWindowDataset(base_path, sp["test"], cfg["data"]["history_steps"], cfg["data"]["step_slices"], train.normalizer, (train.cond_mean, train.cond_std), rollout_steps=rollout_steps, drop_delay_conditioning=drop_delay)
    regimes = {"in_distribution": test}
    for regime in ["held_out_delay", "held_out_parameter", "resolution_transfer"]:
        rcfg = regime_config(cfg, regime)
        path = generate_trajectories(rcfg, f"{cfg['output_dir']}/data", overwrite=overwrite, regime=regime)
        regimes[regime] = HistoryWindowDataset(path, _all_indices(path), cfg["data"]["history_steps"], cfg["data"]["step_slices"], train.normalizer, (train.cond_mean, train.cond_std), rollout_steps=rollout_steps, drop_delay_conditioning=drop_delay)
    return train, val, regimes


def make_model(cfg, name, cond_dim=None, static_channels=None):
    name = canonical_model_name(name)
    models_cfg = cfg.get("models", {})
    per_model = models_cfg.get("per_model", {}).get(name, {})
    model_cfg = {**models_cfg, **per_model}
    kwargs = dict(
        history_steps=cfg["data"]["history_steps"],
        channels=cfg["channels"],
        cond_dim=cond_dim if cond_dim is not None else len(cfg["params"]) + 1,
        static_channels=static_channels if static_channels is not None else cfg.get("static_channels", 0),
        step_slices=cfg["data"]["step_slices"],
        backbone=model_cfg.get("backbone", "conv"),
        width=model_cfg.get("width", 32),
        depth=model_cfg.get("depth", 3),
        dim=cfg["data"].get("dim", 1),
        coordinate_embedding=model_cfg.get("coordinate_embedding", False),
        film=model_cfg.get("film", False),
        append_mode=model_cfg.get("append_mode", "segment"),
    )
    if name == "lag_stack":
        kwargs["lags"] = int(model_cfg.get("lags", 3))
    history_space_names = {"hsno", "hsno_full", "hs_fno", "hsno_unet", "hs_transformer", "hs_fno_rollout_semiflow", "hsno_unet_rollout_semiflow"}
    if name not in history_space_names and not name.startswith("hs_fno_"):
        kwargs.pop("append_mode", None)
    return build_model(name, **kwargs)
