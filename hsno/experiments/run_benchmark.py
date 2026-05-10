from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import time
import torch
import pandas as pd
from torch.utils.data import Subset
from hsno.experiments.common import prepare_data, make_model, device_from_config
from hsno.training.trainer import Trainer
from hsno.training.evaluator import Evaluator
from hsno.utils.config import save_config
from hsno.utils.plotting import save_prediction_example
from hsno.utils.naming import canonical_model_name
from hsno.utils.io import write_jsonl, append_jsonl
from hsno.utils.repro import stable_config_hash, write_config_lock, make_run_id


def _update_per_model(cfg, model_name, **updates):
    cfg.setdefault("models", {}).setdefault("per_model", {}).setdefault(model_name, {}).update(updates)


def _configure_run(cfg, model_name: str):
    """Return canonical model name, output label, and run config for a model/ablation."""
    name = canonical_model_name(model_name, warn=model_name != canonical_model_name(model_name))
    rcfg = deepcopy(cfg)
    if name == "hs_fno":
        _update_per_model(rcfg, name, backbone="fno")
    elif name == "hsno_unet":
        _update_per_model(rcfg, name, backbone="unet")
    elif name == "hs_transformer":
        _update_per_model(rcfg, name, backbone="transformer")
    elif name == "hs_fno_no_shift":
        _update_per_model(rcfg, name, backbone="fno")
    elif name == "hs_fno_rollout_semiflow":
        rcfg["training"]["rollout_weight"] = max(float(rcfg["training"].get("rollout_weight", 0.0)), 0.1)
        rcfg["training"]["semiflow_weight"] = max(float(rcfg["training"].get("semiflow_weight", 0.0)), 0.05)
        _update_per_model(rcfg, name, backbone="fno", append_mode="recursive")
    elif name == "hs_fno_coord_conditioning":
        _update_per_model(rcfg, name, backbone="fno", coordinate_embedding=True)
    elif name == "hs_fno_film_conditioning":
        _update_per_model(rcfg, name, backbone="fno", film=True)
    elif name == "hs_fno_no_delay_conditioning":
        rcfg["data"]["drop_delay_conditioning"] = True
        _update_per_model(rcfg, name, backbone="fno")
    elif name.startswith("hs_fno_history_steps_"):
        rcfg["data"]["history_steps"] = int(name.rsplit("_", 1)[-1])
        _update_per_model(rcfg, name, backbone="fno")
    elif name.startswith("hs_fno_history_resolution_"):
        rcfg["data"]["history_steps"] = int(name.rsplit("_", 1)[-1])
        _update_per_model(rcfg, name, backbone="fno")
    elif name == "hs_fno_per_delay_low":
        rcfg["data"]["separate_delay_bin"] = "low"
        _update_per_model(rcfg, name, backbone="fno")
    elif name == "hs_fno_per_delay_high":
        rcfg["data"]["separate_delay_bin"] = "high"
        _update_per_model(rcfg, name, backbone="fno")
    elif name == "hsno_unet_no_shift":
        _update_per_model(rcfg, name, backbone="unet")
    elif name == "hsno_unet_rollout_semiflow":
        rcfg["training"]["rollout_weight"] = max(float(rcfg["training"].get("rollout_weight", 0.0)), 0.1)
        rcfg["training"]["semiflow_weight"] = max(float(rcfg["training"].get("semiflow_weight", 0.0)), 0.05)
        _update_per_model(rcfg, name, backbone="unet", append_mode="recursive")
    return name, name, rcfg


def _model_runs(cfg):
    """Build primary model and HS-FNO ablation run specifications."""
    names = list(cfg["models"].get("selected", [])) + list(cfg.get("ablations", {}).get("enabled", []))
    return [_configure_run(cfg, name) for name in names]


def _dataset_for_delay_bin(dataset, which: str):
    delays = [float(dataset[i]["delay"]) for i in range(len(dataset))]
    if not delays:
        return dataset
    med = sorted(delays)[len(delays)//2]
    idx = [i for i, d in enumerate(delays) if (d <= med if which == "low" else d > med)]
    return Subset(dataset, idx or list(range(len(dataset))))


def _run_is_complete(eval_log: Path, raw_log: Path | None = None) -> bool:
    if not eval_log.exists():
        return False
    if raw_log is not None and not raw_log.exists():
        return False
    try:
        rows = json.loads(eval_log.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    regimes = {row.get("regime") for row in rows if isinstance(row, dict)}
    return {"in_distribution", "held_out_delay", "held_out_parameter", "resolution_transfer"}.issubset(regimes)


def _load_eval_rows(eval_log: Path, seed: int | None = None) -> list[dict]:
    rows = json.loads(eval_log.read_text(encoding="utf-8"))
    if seed is not None:
        for row in rows:
            if isinstance(row, dict):
                row.setdefault("seed", int(seed))
    return rows


def _seeded_label(run_label: str, seed: int | None) -> str:
    return f"{run_label}_seed{int(seed)}" if seed is not None else run_label


def _artifact_path(output_dir, folder: str, benchmark: str, run_label: str, suffix: str):
    return Path(output_dir) / folder / f"{benchmark}_{run_label}{suffix}"




def _model_metadata(run_cfg: dict, model_name: str) -> dict:
    pm = run_cfg.get("models", {}).get("per_model", {}).get(model_name, {})
    return {
        "backbone": pm.get("backbone", run_cfg.get("models", {}).get("backbone", "unknown")),
        "shift_append": bool(pm.get("shift_append", not model_name.endswith("no_shift") and model_name != "history2history")),
        "rollout_loss_on": float(run_cfg.get("training", {}).get("rollout_weight", 0.0)) > 0,
        "semiflow_loss_on": float(run_cfg.get("training", {}).get("semiflow_weight", 0.0)) > 0,
        "rollout_loss_weight": float(run_cfg.get("training", {}).get("rollout_weight", 0.0)),
        "semiflow_loss_weight": float(run_cfg.get("training", {}).get("semiflow_weight", 0.0)),
        "delay_conditioning": not bool(run_cfg.get("data", {}).get("drop_delay_conditioning", False)),
        "history_steps": int(run_cfg.get("data", {}).get("history_steps", 0)),
        "history_resolution": int(run_cfg.get("data", {}).get("history_steps", 0)),
        "train_nx": int(run_cfg.get("data", {}).get("nx", 0)),
        "rollout_horizon": int(run_cfg.get("training", {}).get("rollout_steps", 0)),
    }



def _validation_score_from_history(train_log: Path) -> float | None:
    if not train_log.exists():
        return None
    try:
        hist = json.loads(train_log.read_text(encoding="utf-8"))
    except Exception:
        return None
    vals = [row.get("val_loss") for row in hist if isinstance(row, dict) and row.get("val_loss") is not None]
    return float(min(vals)) if vals else None

def _log_hparams(output_dir: str, benchmark: str, model_name: str, run_cfg: dict, seed: int, selected: dict, validation_score=None) -> None:
    out = Path(output_dir) / "results" / "hparams"
    out.mkdir(parents=True, exist_ok=True)
    search_space = run_cfg.get("hparam_search", {}).get("spaces", {}).get(model_name, {})
    row = {
        "benchmark": benchmark,
        "model": model_name,
        "seed": int(seed),
        "search_space": search_space,
        "number_of_trials": int(run_cfg.get("hparam_search", {}).get("trials", 1)),
        "selected_hyperparameters": selected,
        "validation_metric": "val_loss",
        "validation_score": validation_score,
        "training_budget": run_cfg.get("training", {}),
        "early_stopping_criterion": {"patience": run_cfg.get("training", {}).get("patience"), "min_delta": run_cfg.get("training", {}).get("min_delta", 0.0)},
        "hardware": "cuda" if torch.cuda.is_available() else "cpu",
        "seeds": run_cfg.get("seeds", [seed]),
        "failed": False,
    }
    append_jsonl(out / "hparam_search_log.jsonl", [row])
    selected_path = out / "selected_configs.csv"
    flat = {k: v for k, v in row.items() if k not in {"search_space", "selected_hyperparameters", "training_budget", "early_stopping_criterion", "seeds"}}
    flat["search_space_json"] = json.dumps(search_space, default=str)
    flat["selected_hyperparameters_json"] = json.dumps(selected, default=str)
    flat["training_budget_json"] = json.dumps(run_cfg.get("training", {}), default=str)
    pd.DataFrame([flat]).to_csv(selected_path, mode="a", header=not selected_path.exists(), index=False)

def run_benchmark(cfg, overwrite=False):
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 0))
    logs_dir = Path(cfg["output_dir"]) / "logs"
    save_config(cfg, logs_dir / f"{cfg['benchmark']}_seed{seed}_config.yaml")
    base_train, base_val, base_regimes = prepare_data(cfg, overwrite=overwrite)
    device = device_from_config(cfg)
    rows = []
    cached = {"base": (base_train, base_val, base_regimes)}
    for model_name, run_label, run_cfg in _model_runs(cfg):
        seeded_run_label = _seeded_label(run_label, seed)
        ckpt = _artifact_path(cfg["output_dir"], "checkpoints", cfg["benchmark"], seeded_run_label, ".pt")
        train_log = _artifact_path(cfg["output_dir"], "logs", cfg["benchmark"], seeded_run_label + "_train", ".json")
        eval_log = _artifact_path(cfg["output_dir"], "logs", cfg["benchmark"], seeded_run_label + "_eval", ".json")
        raw_log = _artifact_path(cfg["output_dir"], "logs", cfg["benchmark"], seeded_run_label + "_raw_metrics", ".jsonl")
        if not overwrite and _run_is_complete(eval_log, raw_log):
            print(f"Skipping completed run {cfg['benchmark']} / {run_label} / seed {seed}; using {eval_log}")
            rows.extend(_load_eval_rows(eval_log, seed))
            continue

        key = json.dumps({"history_steps": run_cfg["data"].get("history_steps"), "drop_delay": run_cfg["data"].get("drop_delay_conditioning", False)}, sort_keys=True)
        if key not in cached:
            cached[key] = prepare_data(run_cfg, overwrite=overwrite)
        train, val, regime_datasets = cached[key]
        bin_name = run_cfg["data"].get("separate_delay_bin")
        if bin_name:
            train = _dataset_for_delay_bin(train, bin_name)
            val = _dataset_for_delay_bin(val, bin_name)
        cond_dim = len(train.dataset.cond_keys) if isinstance(train, Subset) else len(train.cond_keys)
        first_sample = train[0]
        sample_static_channels = int(first_sample["static"].shape[0])
        model = make_model(run_cfg, model_name, cond_dim=cond_dim, static_channels=sample_static_channels)

        model_meta = _model_metadata(run_cfg, model_name)
        config_hash = stable_config_hash(run_cfg)
        ckpt_to_load = ckpt if ckpt.exists() else None
        train_wall_time = 0.0
        if ckpt_to_load is not None and ckpt_to_load.exists() and not overwrite:
            try:
                print(f"Skipping training for {cfg['benchmark']} / {run_label} / seed {seed}; loading {ckpt_to_load}")
                state = torch.load(ckpt_to_load, map_location=device)
                model.load_state_dict(state)
            except RuntimeError as exc:
                print(f"Checkpoint {ckpt_to_load} was incompatible for {run_label}; retraining. Error: {exc}")
                start_train = time.perf_counter()
                history = Trainer(model, train, val, run_cfg, device).fit(ckpt)
                train_wall_time = time.perf_counter() - start_train
                train_log.write_text(json.dumps(history, indent=2), encoding="utf-8")
        else:
            start_train = time.perf_counter()
            history = Trainer(model, train, val, run_cfg, device).fit(ckpt)
            train_wall_time = time.perf_counter() - start_train
            train_log.write_text(json.dumps(history, indent=2), encoding="utf-8")

        _log_hparams(cfg["output_dir"], cfg["benchmark"], run_label, run_cfg, seed, run_cfg.get("models", {}).get("per_model", {}).get(model_name, {}), _validation_score_from_history(train_log))

        run_rows = []
        run_raw_rows = []
        for regime, dataset in regime_datasets.items():
            eval_ds = _dataset_for_delay_bin(dataset, bin_name) if bin_name else dataset
            sample_shape = eval_ds[0]["history"].shape if len(eval_ds) else None
            eval_nx = int(sample_shape[-1] if len(sample_shape) == 3 else sample_shape[-2]) if sample_shape is not None else run_cfg["data"]["nx"]
            run_id = make_run_id(cfg["benchmark"], run_label, regime, seed, config_hash)
            lock_path = Path(cfg["output_dir"]) / "results" / "runs" / run_id / "config.lock.json"
            write_config_lock(lock_path, cfg=run_cfg, seed=seed, args={"benchmark": cfg["benchmark"], "model": run_label, "regime": regime}, device=device, run_id=run_id)
            context = {"run_id": run_id, "regime": regime, "seed": seed, "split": regime, "config_path": str(Path("configs") / f"{cfg['benchmark']}.yaml"), "output_dir": cfg["output_dir"], "split_id": f"seed{seed}", "config_hash": config_hash, "train_wall_time": float(train_wall_time), "eval_nx": eval_nx, **model_meta}
            ev, raw = Evaluator(model, eval_ds, device, seed=seed).evaluate(cfg["benchmark"], run_label, run_cfg["training"].get("rollout_steps", 3), return_raw=True, context=context)
            ev["regime"] = regime
            ev["seed"] = seed
            ev["train_history_steps"] = run_cfg["data"]["history_steps"]
            ev["eval_nx"] = eval_nx
            ev.update(context)
            run_rows.append(ev)
            run_raw_rows.extend(raw)
            if regime == "in_distribution":
                save_prediction_example(model, eval_ds, device, Path(cfg["output_dir"]) / "plots", cfg["benchmark"], seeded_run_label)
        eval_log.write_text(json.dumps(run_rows, indent=2), encoding="utf-8")
        write_jsonl(raw_log, run_raw_rows)
        rows.extend(run_rows)
    return rows
