from __future__ import annotations
import numpy as np
from .base import BaseSolver, Trajectory
from .delay_buffer import DelayBuffer
from .operators import laplacian
from .reaction_diffusion import smooth_history


class DelayedWaveSolver(BaseSolver):
    name = "delayed_wave"

    def simulate(self, params, nx=64, dt=0.005, dt_save=0.05, total_time=1.0, history_steps=6, boundary="periodic", force="tanh", scheme="finite_difference", seed=0, **kw):
        rng = np.random.default_rng(seed)
        tau = float(params["tau"])
        c = float(params["c"])
        alpha = float(params.get("alpha", 1))
        beta = float(params.get("beta", 0.1))
        hist = smooth_history(rng, history_steps, 2, nx, 1, None, False, boundary)
        times = np.linspace(-tau, 0, history_steps)
        z = hist[-1].copy()
        buf = DelayBuffer(times, hist)
        dx = 1 / nx
        cfl_limit = 0.45 * dx / max(abs(c), 1e-8)
        requested_dt = float(dt)
        substeps = max(1, int(np.ceil(requested_dt / cfl_limit)))
        dt = requested_dt / substeps

        def F(q):
            if force == "linear":
                return alpha * q
            if force == "cubic":
                return alpha * q - beta * q**3
            return np.tanh(alpha * q)

        saved = [h.copy() for h in hist]
        st = list(times.astype(float))
        next_save = dt_save
        for n in range(1, int(np.ceil(total_time / dt)) + 1):
            t = (n - 1) * dt
            d = buf.lookup(t - tau)
            u, v = z[0:1], z[1:2]
            un = u + dt * v
            vn = v + dt * (c * c * laplacian(u, dx, None, boundary, scheme) + F(d[0:1]))
            z = np.concatenate([un, vn], 0)
            buf.append(t + dt, z)
            if t + dt + 1e-12 >= next_save:
                saved.append(z.copy())
                st.append(next_save)
                next_save += dt_save
        return Trajectory(
            np.asarray(saved, dtype=np.float32),
            np.asarray(st, dtype=np.float32),
            params,
            metadata={"history_times_included": True, "method": "method_of_steps_explicit_first_order", "diffusion_scheme": scheme, "requested_dt": requested_dt, "internal_dt": dt, "cfl_limit": cfl_limit, "substeps_per_requested_step": substeps},
        )
