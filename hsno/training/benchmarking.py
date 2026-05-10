from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .metrics import parameter_count


def benchmark_inference(model, sample: dict[str, torch.Tensor], *, warmup: int = 10, timed: int = 50, precision: str = "float32") -> dict[str, Any]:
    """Reproducible inference benchmark with warmup and repeated timing.

    The caller supplies an already-device-placed sample. CUDA timings are
    synchronized to avoid asynchronous under-reporting.
    """
    device = next(model.parameters()).device
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(int(warmup)):
            _ = model(sample["history"], sample.get("cond"), sample.get("static"))
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        times = []
        for _ in range(int(timed)):
            start = time.perf_counter()
            _ = model(sample["history"], sample.get("cond"), sample.get("static"))
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)
    arr = np.asarray(times, dtype=float)
    return {
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "precision": precision,
        "batch_size": int(sample["history"].shape[0]),
        "warmup_iterations": int(warmup),
        "timed_iterations": int(timed),
        "mean_inference_time": float(arr.mean()) if arr.size else 0.0,
        "median_inference_time": float(np.median(arr)) if arr.size else 0.0,
        "std_inference_time": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1e6) if torch.cuda.is_available() else 0.0,
        "parameter_count": parameter_count(model),
        "output_dimension": int(sample["target_history"].numel()) if "target_history" in sample else 0,
    }
