from __future__ import annotations
import numpy as np
from .base import BaseSolver, Trajectory
from .delay_buffer import DelayBuffer
from .operators import laplacian, is_semi_implicit, semi_implicit_diffusion_step
from .reaction_diffusion import smooth_history


class EpidemicDelaySolver(BaseSolver):
    name = "epidemic_delay"

    def simulate(self, params, nx=64, dt=0.005, dt_save=0.05, total_time=1.0, history_steps=6, boundary="neumann", scheme="finite_difference", seed=0, **kw):
        rng = np.random.default_rng(seed)
        tau = float(params["tau"])
        D = float(params["D"])
        beta = float(params["beta"])
        gamma = float(params["gamma"])
        x = np.linspace(0, 1, nx, endpoint=False)
        S = 1.0 + 0.3 * np.sin(2 * np.pi * x + rng.random()) + 0.15 * np.cos(4 * np.pi * x + rng.random())
        S = np.maximum(S, 0.1).astype(np.float32)[None]
        hist = 0.15 * smooth_history(rng, history_steps, 1, nx, 1, None, True, boundary)
        times = np.linspace(-tau, 0, history_steps)
        u = hist[-1].copy()
        buf = DelayBuffer(times, hist)
        saved = [h.copy() for h in hist]
        st = list(times.astype(float))
        next_save = dt_save
        dx = 1 / nx
        for n in range(1, int(np.ceil(total_time / dt)) + 1):
            t = (n - 1) * dt
            delayed = buf.lookup(t - tau)
            reaction = beta * S * delayed - gamma * u
            if is_semi_implicit(scheme):
                u = semi_implicit_diffusion_step(u + dt * reaction, D, dt, dx, None, boundary)
            else:
                lap = laplacian(u, dx, None, boundary, scheme)
                u = u + dt * (D * lap + reaction)
            u = np.maximum(u, 0)
            buf.append(t + dt, u)
            if t + dt + 1e-12 >= next_save:
                saved.append(u.copy())
                st.append(next_save)
                next_save += dt_save
        return Trajectory(
            np.asarray(saved, dtype=np.float32),
            np.asarray(st, dtype=np.float32),
            params,
            static=S,
            metadata={"static_channel": "S", "history_times_included": True, "method": "method_of_steps_imex" if is_semi_implicit(scheme) else "method_of_steps_explicit", "diffusion_scheme": scheme},
        )
