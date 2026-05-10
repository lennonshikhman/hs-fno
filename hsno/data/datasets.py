from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json, time, numpy as np, torch
from torch.utils.data import Dataset
from .history import make_tau_history_windows
from .normalization import ChannelNormalizer
from hsno.solvers.reaction_diffusion import ReactionDiffusionSolver
from hsno.solvers.epidemic import EpidemicDelaySolver
from hsno.solvers.neural_field import NeuralFieldSolver
from hsno.solvers.delayed_wave import DelayedWaveSolver
from hsno.solvers.distributed_memory import DistributedMemorySolver

SOLVERS = {s.name: s for s in [ReactionDiffusionSolver(), EpidemicDelaySolver(), NeuralFieldSolver(), DelayedWaveSolver(), DistributedMemorySolver()]}
DELAY_KEYS = {"tau", "tau0"}
REGIME_SEEDS = {"in_distribution": 0, "held_out_delay": 101, "held_out_parameter": 202, "resolution_transfer": 303}



def sample_params(rng, ranges, regime: str = "in_distribution"):
    """Sample parameters, optionally outside the training ranges for OOD regimes."""
    params = {}
    for key, value in ranges.items():
        if not (isinstance(value, list) and len(value) == 2):
            params[key] = value
            continue
        lo, hi = float(value[0]), float(value[1])
        width = hi - lo
        if regime == "held_out_delay" and (key in DELAY_KEYS or "tau" in key):
            lo, hi = hi + 0.10 * width, hi + 0.30 * width
        elif regime == "held_out_parameter" and not (key in DELAY_KEYS or "tau" in key):
            lo, hi = hi + 0.10 * width, hi + 0.30 * width
        params[key] = float(rng.uniform(lo, hi))
    return params


def regime_config(cfg: dict, regime: str) -> dict:
    """Return a config copy adjusted for a concrete evaluation/data regime."""
    out = deepcopy(cfg)
    out["regime"] = regime
    if regime in {"held_out_delay", "held_out_parameter", "resolution_transfer"}:
        n = max(1, int(cfg["data"].get("n_test", 1)))
        out["data"]["n_train"] = 0
        out["data"]["n_val"] = 0
        out["data"]["n_test"] = n
    if regime == "resolution_transfer":
        out["data"]["nx"] = max(int(round(cfg["data"]["nx"] * 1.5)), cfg["data"]["nx"] + 2)
        if out["data"].get("dim", 1) == 2:
            out["data"]["ny"] = max(int(round(cfg["data"].get("ny", cfg["data"]["nx"]) * 1.5)), cfg["data"].get("ny", cfg["data"]["nx"]) + 2)
    return out


def _trajectory_count(cfg):
    return int(cfg["data"].get("n_train", 0) + cfg["data"].get("n_val", 0) + cfg["data"].get("n_test", 0))


def _cached_trajectories_are_finite(data_file: Path) -> bool:
    """Return whether an existing trajectory cache can be safely reused."""
    try:
        with np.load(data_file, allow_pickle=True) as z:
            fields = z["fields"]
            static = z["static"] if "static" in z else None
            return bool(np.isfinite(fields).all() and (static is None or np.isfinite(static).all()))
    except Exception:
        return False


def generate_trajectories(cfg, out_dir: str | Path, overwrite=False, regime: str | None = None):
    regime = regime or cfg.get("regime", "in_distribution")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 0))
    regime_suffix = "" if regime == "in_distribution" else f"_{regime}"
    suffix = f"_seed{seed}{regime_suffix}"
    data_file = out / f"{cfg['benchmark']}{suffix}.npz"
    meta_file = out / f"{cfg['benchmark']}{suffix}.json"
    if data_file.exists() and not overwrite:
        if _cached_trajectories_are_finite(data_file):
            return data_file
        print(f"Regenerating {data_file} because it contains non-finite trajectory values.")
    rng = np.random.default_rng(seed + REGIME_SEEDS.get(regime, 0))
    n = _trajectory_count(cfg)
    solver = SOLVERS[cfg["benchmark"]]
    fields, times, params, statics, metas = [], [], [], [], []
    for i in range(n):
        p = sample_params(rng, cfg["params"], regime)
        kw = dict(
            nx=cfg["data"]["nx"],
            ny=cfg["data"].get("ny"),
            dim=cfg["data"].get("dim", 1),
            dt=cfg["data"]["dt"],
            dt_save=cfg["data"]["dt_save"],
            total_time=cfg["data"]["total_time"],
            history_steps=cfg["data"]["history_steps"],
            seed=1000 + seed * 10000 + i + REGIME_SEEDS.get(regime, 0),
        )
        kw.update(cfg.get("solver", {}))
        start = time.perf_counter()
        tr = solver.simulate(p, **kw)
        runtime = time.perf_counter() - start
        tr.metadata = dict(tr.metadata or {})
        tr.metadata["solver_runtime_sec"] = float(runtime)
        tr.metadata["solver_time_per_saved_step_sec"] = float(runtime / max(1, len(tr.times)))
        fields.append(tr.fields)
        times.append(tr.times)
        params.append(p)
        statics.append(tr.static if tr.static is not None else np.zeros_like(tr.fields[0, :1]))
        metas.append(tr.metadata or {})
    np.savez_compressed(
        data_file,
        fields=np.asarray(fields, dtype=np.float32),
        static=np.asarray(statics, dtype=np.float32),
        times=np.asarray(times, dtype=np.float32),
        params_json=json.dumps(params),
        regime=regime,
        dt_save=np.float32(cfg["data"]["dt_save"]),
        metadata_json=json.dumps(metas),
        seed=np.int64(seed),
    )
    meta_file.write_text(json.dumps({"regime": regime, "seed": seed, "params": params, "metadata": metas}, indent=2), encoding="utf-8")
    return data_file


class HistoryWindowDataset(Dataset):
    def __init__(self, npz_path, indices, history_steps, step_slices=1, normalizer=None, cond_stats=None, fit=False, rollout_steps=1, drop_delay_conditioning=False):
        z = np.load(npz_path, allow_pickle=True)
        self.fields = z["fields"]
        self.times = z["times"]
        self.static = z["static"]
        self.params = json.loads(str(z["params_json"]))
        self.indices = list(indices)
        self.history_steps = history_steps
        self.step_slices = step_slices
        self.dt_save = float(z["dt_save"]) if "dt_save" in z else 1.0
        self.rollout_steps = max(1, int(rollout_steps))
        self.drop_delay_conditioning = bool(drop_delay_conditioning)
        self.metadata = json.loads(str(z["metadata_json"])) if "metadata_json" in z else [{} for _ in self.params]
        train_arrays = [self.fields[i].reshape(-1, self.fields.shape[2], *self.fields.shape[3:]) for i in self.indices]
        self.normalizer = normalizer or ChannelNormalizer.fit(train_arrays)
        all_param_keys = sorted(self.params[0].keys())
        param_keys = [k for k in all_param_keys if not (self.drop_delay_conditioning and (k in DELAY_KEYS or "tau" in k))]
        self.cond_keys = param_keys + ["delta_t"]
        cond = np.array([[self.params[i][k] for k in param_keys] + [self.dt_save] for i in self.indices], dtype=np.float32)
        if cond_stats is None:
            self.cond_mean = cond.mean(0)
            self.cond_std = cond.std(0) + 1e-6
        else:
            self.cond_mean, self.cond_std = cond_stats
        self.samples = []
        for ti in self.indices:
            tau = self._history_tau(self.params[ti])
            xs, ys, fut, rollouts = make_tau_history_windows(self.fields[ti], self.times[ti], tau, history_steps, step_slices, self.dt_save, self.rollout_steps)
            c_raw = np.array([self.params[ti][k] for k in param_keys] + [self.dt_save], dtype=np.float32)
            c = (c_raw - self.cond_mean) / self.cond_std
            for wi, (x, y, p, r) in enumerate(zip(xs, ys, fut, rollouts)):
                self.samples.append((x, y, p, r, c, c_raw, self.static[ti], ti, wi))

    @staticmethod
    def _history_tau(params: dict) -> float:
        if "tau" in params:
            return float(params["tau"])
        if "tau0" in params and "alpha" in params:
            return float(params["tau0"] + 0.5 * params["alpha"])
        if "tau0" in params:
            return float(params["tau0"])
        return 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y, p, r, c, c_raw, s, ti, wi = self.samples[i]
        x = self.normalizer.encode(torch.tensor(x, dtype=torch.float32))
        y = self.normalizer.encode(torch.tensor(y, dtype=torch.float32))
        p = self.normalizer.encode(torch.tensor(p, dtype=torch.float32))
        r = self.normalizer.encode(torch.tensor(r, dtype=torch.float32))
        meta = self.metadata[ti] if ti < len(self.metadata) else {}
        solver_step = float(meta.get("solver_time_per_saved_step_sec", 0.0))
        tau = float(self._history_tau(self.params[ti]))
        return {"history": x, "target_history": y, "future": p, "rollout_targets": r, "cond": torch.tensor(c, dtype=torch.float32), "cond_raw": torch.tensor(c_raw, dtype=torch.float32), "cond_mean": torch.tensor(self.cond_mean, dtype=torch.float32), "cond_std": torch.tensor(self.cond_std, dtype=torch.float32), "static": torch.tensor(s, dtype=torch.float32), "solver_time_per_step": torch.tensor(solver_step, dtype=torch.float32), "delay": torch.tensor(tau, dtype=torch.float32), "trajectory_id": torch.tensor(int(ti), dtype=torch.long), "window_id": torch.tensor(int(wi), dtype=torch.long), "history_steps": torch.tensor(int(self.history_steps), dtype=torch.long), "history_resolution": torch.tensor(int(self.history_steps), dtype=torch.long)}
