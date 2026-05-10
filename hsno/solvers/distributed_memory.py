from __future__ import annotations
import numpy as np
from .base import BaseSolver, Trajectory
from .delay_buffer import DelayBuffer
from .operators import laplacian, is_semi_implicit, semi_implicit_diffusion_step
from .reaction_diffusion import smooth_history


_trapezoid = getattr(np, "trapezoid", np.trapz)


def memory_weights(kind: str, history_steps: int, tau: float):
    theta = np.linspace(-tau, 0, history_steps)
    s = theta + tau
    if kind == "uniform":
        w = np.ones_like(s)
    elif kind == "gamma":
        w = (s + 1e-6) ** 2 * np.exp(-5 * s / max(tau, 1e-6))
    else:
        w = np.exp(3 * theta / max(tau, 1e-6))
    w = np.maximum(w, 0)
    return (w / (_trapezoid(w, theta) + 1e-9)).astype(np.float32), theta


def _explicit_diffusion_substeps(dt: float, nu: float, dx: float, dy: float, dim: int) -> tuple[float, int, float]:
    """Return a stable explicit diffusion step and substep count.

    The distributed-memory benchmark is evaluated at higher resolution in the
    resolution-transfer regime. Keeping the coarse-grid time step there can
    violate the FTCS diffusion CFL condition even when the reaction/memory terms
    are benign. Substepping only the requested solver interval preserves the
    public ``dt``/``dt_save`` cadence while keeping explicit finite differences
    stable.
    """
    if nu <= 0.0:
        return float(dt), 1, float("inf")
    inv_h2_sum = 1.0 / (dx * dx)
    if dim == 2:
        inv_h2_sum += 1.0 / (dy * dy)
    cfl_limit = 0.45 / (float(nu) * inv_h2_sum)
    substeps = max(1, int(np.ceil(float(dt) / cfl_limit)))
    return float(dt) / substeps, substeps, cfl_limit


def _sanitize_bistable_state(u: np.ndarray, max_state: float | None) -> tuple[np.ndarray, bool, bool]:
    """Return a finite symmetrically-clipped state and clipping flags."""
    arr = np.asarray(u, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    nonfinite_clipped = bool(np.any(~finite_mask))
    if max_state is None:
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), nonfinite_clipped, False
    ceiling = float(max_state)
    if ceiling <= 0.0:
        raise ValueError("max_state must be positive or None")
    finite_vals = arr[finite_mask]
    magnitude_clipped = bool(finite_vals.size and np.any(np.abs(finite_vals) > ceiling))
    magnitude_clipped = magnitude_clipped or bool(np.any(np.isinf(arr)))
    finite = np.nan_to_num(arr, nan=0.0, posinf=ceiling, neginf=-ceiling)
    return np.clip(finite, -ceiling, ceiling), nonfinite_clipped, magnitude_clipped


class DistributedMemorySolver(BaseSolver):
    name = "distributed_memory"

    def simulate(
        self,
        params,
        nx=64,
        ny=None,
        dim=1,
        dt=0.005,
        dt_save=0.05,
        total_time=1.0,
        history_steps=6,
        boundary="periodic",
        kernel="exponential",
        scheme="finite_difference",
        seed=0,
        max_state=20.0,
    ):
        rng = np.random.default_rng(seed)
        tau = float(params["tau"])
        nu = float(params["nu"])
        a1 = float(params["a1"])
        a2 = float(params["a2"])
        hist = smooth_history(rng, history_steps, 1, nx, dim, ny, False, boundary).astype(np.float64, copy=False)
        weights, theta = memory_weights(kernel, history_steps, tau)
        weight_prob = weights / (weights.sum() + 1e-9)
        kernel_mean = (weight_prob * theta).sum()
        kernel_moments = np.array(
            [
                float(kernel_mean),
                float(np.sqrt((weight_prob * (theta - kernel_mean) ** 2).sum())),
                float(-(weight_prob * np.log(weight_prob + 1e-9)).sum()),
            ],
            dtype=np.float32,
        )
        static_shape = (nx,) if dim == 1 else (nx, ny)
        static = np.broadcast_to(kernel_moments.reshape(3, *([1] * len(static_shape))), (3, *static_shape)).astype(np.float32)
        u = hist[-1].copy()
        u, nonfinite_now, magnitude_now = _sanitize_bistable_state(u, max_state)
        hist[-1] = u
        buf = DelayBuffer(theta, hist)
        saved = [h.copy() for h in hist]
        st = list(theta.astype(float))
        next_save = dt_save
        dx = 1 / nx
        dy = 1 / (ny or nx)
        requested_dt = float(dt)
        if is_semi_implicit(scheme):
            internal_dt = requested_dt
            substeps = 1
            cfl_limit = float("inf")
        else:
            internal_dt, substeps, cfl_limit = _explicit_diffusion_substeps(requested_dt, nu, dx, dy, dim)
        clipped_nonfinite = bool(nonfinite_now)
        clipped_magnitude = bool(magnitude_now)
        memory_query_offsets = np.linspace(0, tau, history_steps)
        nsteps = int(np.ceil(total_time / internal_dt))
        for n in range(1, nsteps + 1):
            t = (n - 1) * internal_dt
            u, nonfinite_now, magnitude_now = _sanitize_bistable_state(u, max_state)
            clipped_nonfinite = clipped_nonfinite or nonfinite_now
            clipped_magnitude = clipped_magnitude or magnitude_now
            vals = np.stack([buf.lookup(t - tau + s) for s in memory_query_offsets], 0)
            vals, nonfinite_now, magnitude_now = _sanitize_bistable_state(vals, max_state)
            clipped_nonfinite = clipped_nonfinite or nonfinite_now
            clipped_magnitude = clipped_magnitude or magnitude_now
            M = a1 * vals + a2 * vals * vals * vals
            mem = _trapezoid(weights.reshape(-1, 1, *([1] * (u.ndim - 1))) * M, dx=tau / (history_steps - 1), axis=0)
            reaction = u * (1.0 - u * u) + mem
            if is_semi_implicit(scheme):
                u = semi_implicit_diffusion_step(u + internal_dt * reaction, nu, internal_dt, dx, dy, boundary)
            else:
                lap = laplacian(u, dx, dy, boundary, scheme)
                u = u + internal_dt * (nu * lap + reaction)
            u, nonfinite_now, magnitude_now = _sanitize_bistable_state(u, max_state)
            clipped_nonfinite = clipped_nonfinite or nonfinite_now
            clipped_magnitude = clipped_magnitude or magnitude_now
            buf.append(t + internal_dt, u)
            if t + internal_dt + 1e-12 >= next_save:
                saved.append(u.copy())
                st.append(next_save)
                next_save += dt_save
        return Trajectory(
            np.asarray(saved, dtype=np.float32),
            np.asarray(st, dtype=np.float32),
            params,
            static=static,
            metadata={
                "kernel": kernel,
                "kernel_theta": theta.astype(float).tolist(),
                "kernel_weights": weights.astype(float).tolist(),
                "static_channels": [
                    "memory_kernel_mean_theta",
                    "memory_kernel_std_theta",
                    "memory_kernel_entropy",
                ],
                "history_times_included": True,
                "method": "method_of_steps_imex_memory_quadrature"
                if is_semi_implicit(scheme)
                else "method_of_steps_explicit_memory_quadrature",
                "diffusion_scheme": scheme,
                "requested_dt": requested_dt,
                "internal_dt": internal_dt,
                "cfl_limit": cfl_limit,
                "substeps_per_requested_step": substeps,
                "clipped_nonfinite": clipped_nonfinite,
                "clipped_magnitude": clipped_magnitude,
                "max_state": None if max_state is None else float(max_state),
            },
        )
