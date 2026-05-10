"""YAML config loading and recursive merging."""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import yaml

def deep_update(base: dict, other: dict) -> dict:
    out = deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict): out[k] = deep_update(out[k], v)
        else: out[k] = deepcopy(v)
    return out

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f) or {}

def load_config(benchmark_path: str | Path, defaults_path: str | Path="configs/defaults.yaml", quick: bool=False) -> dict:
    cfg = deep_update(load_yaml(defaults_path), load_yaml(benchmark_path))
    if quick:
        q = cfg.pop("quick", {})
        cfg = deep_update(cfg, q)
        cfg["quick"] = True
    return cfg

def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: yaml.safe_dump(cfg, f, sort_keys=False)
