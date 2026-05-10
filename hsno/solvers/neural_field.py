from __future__ import annotations
import numpy as np
from .base import BaseSolver, Trajectory
from .delay_buffer import DelayBuffer
from .reaction_diffusion import smooth_history


class NeuralFieldSolver(BaseSolver):
    name = "nonlocal_neural_field"

    def simulate(self, params, nx=64, dt=0.005, dt_save=0.05, total_time=1.0, history_steps=6, sigma="tanh", seed=0, **kw):
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 1, nx, endpoint=False)
        dist = np.abs(x[:, None] - x[None, :])
        dist = np.minimum(dist, 1 - dist)
        width = float(params.get("width", 0.08))
        gain = float(params.get("gain", 1.0))
        W = np.exp(-(dist**2) / (2 * width**2))
        W = gain * W / (W.sum(1, keepdims=True) + 1e-9)
        tau_field = float(params["tau0"]) + float(params["alpha"]) * dist
        tau = float(tau_field.max())
        hist = smooth_history(rng, history_steps, 1, nx, 1, None, False, "periodic")
        times = np.linspace(-tau, 0, history_steps)
        u = hist[-1].copy()
        buf = DelayBuffer(times, hist)
        act = np.tanh if sigma == "tanh" else (lambda z: 1 / (1 + np.exp(-z)))
        source_index = np.broadcast_to(np.arange(nx)[None, :], (nx, nx))
        probes = [np.ones(nx, dtype=np.float64)]
        for k in range(1, 4):
            probes.extend([np.sin(2 * np.pi * k * x), np.cos(2 * np.pi * k * x)])
        probe_matrix = np.stack(probes, axis=0)
        operator_channels = []
        channel_names = []
        for i, probe in enumerate(probe_matrix):
            operator_channels.append(W @ probe)
            channel_names.append(f"kernel_probe_{i}")
            operator_channels.append(tau_field @ probe / nx)
            channel_names.append(f"delay_probe_{i}")
        static = np.stack(operator_channels, axis=0).astype(np.float32)
        saved = [h.copy() for h in hist]
        st = list(times.astype(float))
        next_save = dt_save
        for n in range(1, int(np.ceil(total_time / dt)) + 1):
            t = (n - 1) * dt
            delayed = buf.lookup_pairwise_sources(t - tau_field, source_index, channel=0)
            nonlocal_term = (W * act(delayed)).sum(1)[None]
            u = u + dt * (-u + nonlocal_term)
            buf.append(t + dt, u)
            if t + dt + 1e-12 >= next_save:
                saved.append(u.copy())
                st.append(next_save)
                next_save += dt_save
        return Trajectory(
            np.asarray(saved, dtype=np.float32),
            np.asarray(st, dtype=np.float32),
            params,
            static=static,
            metadata={
                "tau_summary": tau,
                "static_channels": channel_names,
                "pairwise_delay": tau_field.astype(float).tolist(),
                "quadrature_kernel": W.astype(float).tolist(),
                "history_times_included": True,
            },
        )
