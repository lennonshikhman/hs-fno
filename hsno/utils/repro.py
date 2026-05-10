from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import torch


def stable_config_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def package_versions(packages: list[str] | None = None) -> dict[str, str]:
    packages = packages or ["torch", "numpy", "scipy", "pandas", "matplotlib", "yaml", "tqdm"]
    out: dict[str, str] = {}
    for pkg in packages:
        dist = "PyYAML" if pkg == "yaml" else pkg
        try:
            out[pkg] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def runtime_environment(device: str | torch.device | None = None) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "timestamp_unix": time.time(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit_hash(),
        "package_versions": package_versions(),
        "torch_version": torch.__version__,
        "cuda_available": bool(cuda_available),
        "cuda_version": torch.version.cuda,
        "gpu_model": gpu_name,
        "device_name": str(device) if device is not None else ("cuda" if cuda_available else "cpu"),
    }


def write_config_lock(path: str | Path, *, cfg: dict, seed: int, args: dict | None = None, device: str | torch.device | None = None, run_id: str | None = None) -> dict[str, Any]:
    lock = {
        "run_id": run_id,
        "seed": int(seed),
        "config_hash": stable_config_hash(cfg),
        "command_line": sys.argv,
        "command_line_args": args or {},
        "environment": runtime_environment(device),
        "resolved_config": cfg,
        "pid": os.getpid(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2, default=str), encoding="utf-8")
    return lock


def make_run_id(benchmark: str, model: str, regime: str, seed: int, config_hash: str) -> str:
    return f"{benchmark}__{model}__{regime}__seed{int(seed)}__{config_hash[:8]}"
