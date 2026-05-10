from __future__ import annotations
import numpy as np
from .base import BaseSolver, Trajectory
from .delay_buffer import DelayBuffer
from .operators import laplacian, is_semi_implicit, semi_implicit_diffusion_step


def _density_rescale(arr: np.ndarray, floor: float = 0.02, ceiling: float = 1.0) -> np.ndarray:
    arr = arr - arr.min()
    scale = arr.max() + 1e-8
    return (floor + (ceiling - floor) * arr / scale).astype(np.float32)


def _sanitize_density_state(u: np.ndarray, max_state: float | None) -> tuple[np.ndarray, bool, bool]:
    """Return a finite nonnegative density state and clipping flags.

    The delayed logistic reaction-diffusion benchmark is density-like. Numerical
    explicit/IMEX reference solves should therefore never feed negative, NaN, or
    astronomically large values back into the delayed reaction term. This helper
    sanitizes the state before every delayed lookup append and records whether
    lower/upper numerical clipping was required.
    """
    arr = np.asarray(u)
    finite_mask = np.isfinite(arr)
    finite_vals = arr[finite_mask]
    lower_clipped = bool(finite_vals.size and np.any(finite_vals < 0.0))
    nonfinite = bool(np.any(~finite_mask))
    if max_state is None:
        v = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(v, 0.0), lower_clipped or nonfinite, False
    ceiling = float(max_state)
    if ceiling <= 0:
        raise ValueError("max_state must be positive or None")
    upper_clipped = bool(finite_vals.size and np.any(finite_vals > ceiling))
    upper_clipped = upper_clipped or bool(np.any(np.isposinf(arr)))
    lower_clipped = lower_clipped or bool(np.any(np.isneginf(arr)))
    finite = np.nan_to_num(arr, nan=0.0, posinf=ceiling, neginf=0.0)
    return np.clip(finite, 0.0, ceiling), lower_clipped or nonfinite, upper_clipped


def smooth_history(
    rng,
    history_steps,
    channels,
    nx,
    dim=1,
    ny=None,
    nonnegative=True,
    boundary="periodic",
):
    """Generate smooth random histories compatible with common boundaries."""
    xs = np.linspace(0, 1, nx, endpoint=False)
    shape = (history_steps, channels, nx) if dim == 1 else (history_steps, channels, nx, ny)
    arr = np.zeros(shape, dtype=np.float32)
    basis = "fourier" if boundary == "periodic" else ("sine" if boundary == "dirichlet" else "cosine")
    for h in range(history_steps):
        for c in range(channels):
            if dim == 1:
                y = np.zeros(nx, dtype=np.float64)
                for k in range(1, 5):
                    amp = rng.normal(0, 0.25 / k)
                    if basis == "sine":
                        y += amp * np.sin(np.pi * k * xs)
                    elif basis == "cosine":
                        y += amp * np.cos(np.pi * k * xs)
                    else:
                        y += amp * np.sin(2 * np.pi * k * (xs + rng.random()))
                arr[h, c] = y
            else:
                ys = np.linspace(0, 1, ny, endpoint=False)
                X, Y = np.meshgrid(xs, ys, indexing="ij")
                z = np.zeros((nx, ny), dtype=np.float64)
                for k in range(1, 4):
                    amp = rng.normal(0, 0.2 / k)
                    if basis == "sine":
                        z += amp * np.sin(np.pi * k * X) * np.sin(np.pi * k * Y)
                    elif basis == "cosine":
                        z += amp * np.cos(np.pi * k * X) * np.cos(np.pi * k * Y)
                    else:
                        z += amp * np.sin(2 * np.pi * k * (X + rng.random())) * np.cos(2 * np.pi * k * (Y + rng.random()))
                arr[h, c] = z
    if nonnegative:
        arr = _density_rescale(arr)
    return arr.astype(np.float32)


class ReactionDiffusionSolver(BaseSolver):
    name = "delayed_reaction_diffusion"

    def simulate(self, params, nx=64, ny=None, dim=1, dt=0.005, dt_save=0.05, total_time=1.0, history_steps=6, boundary="neumann", scheme="finite_difference", seed=0, max_state=20.0):
        rng = np.random.default_rng(seed)
        tau = float(params["tau"])
        D = float(params["D"])
        r = float(params["r"])
        hist = smooth_history(rng, history_steps, 1, nx, dim, ny, True, boundary)
        times_hist = np.linspace(-tau, 0, history_steps)
        hist = hist.astype(np.float64, copy=False)
        u = hist[-1].copy()
        u, lower_now, upper_now = _sanitize_density_state(u, max_state)
        hist[-1] = u
        buf = DelayBuffer(times_hist, hist)
        saved = [h.copy() for h in hist]
        save_times = list(times_hist.astype(float))
        next_save = dt_save
        clipped = bool(lower_now)
        upper_clipped = bool(upper_now)
        nonfinite_clipped = False
        nsteps = int(np.ceil(total_time / dt))
        dx = 1 / nx
        dy = 1 / (ny or nx)
        for n in range(1, nsteps + 1):
            t = (n - 1) * dt
            u, lower_now, upper_now = _sanitize_density_state(u, max_state)
            clipped = clipped or lower_now
            upper_clipped = upper_clipped or upper_now
            delayed, lower_now, upper_now = _sanitize_density_state(buf.lookup(t - tau), max_state)
            clipped = clipped or lower_now
            upper_clipped = upper_clipped or upper_now
            reaction = r * u * (1.0 - delayed)
            if is_semi_implicit(scheme):
                u = semi_implicit_diffusion_step(u + dt * reaction, D, dt, dx, dy, boundary)
            else:
                lap = laplacian(u, dx, dy, boundary, scheme)
                u = u + dt * (D * lap + reaction)
            if not np.isfinite(u).all():
                nonfinite_clipped = True
            u, lower_now, upper_now = _sanitize_density_state(u, max_state)
            clipped = clipped or lower_now
            upper_clipped = upper_clipped or upper_now
            buf.append(t + dt, u)
            if t + dt + 1e-12 >= next_save:
                saved.append(u.copy())
                save_times.append(next_save)
                next_save += dt_save
        return Trajectory(
            np.asarray(saved, dtype=np.float32),
            np.asarray(save_times, dtype=np.float32),
            params,
            metadata={
                "clipped_nonnegative": clipped,
                "clipped_upper": upper_clipped,
                "clipped_nonfinite": nonfinite_clipped,
                "max_state": None if max_state is None else float(max_state),
                "history_times_included": True,
                "method": "method_of_steps_imex" if is_semi_implicit(scheme) else "method_of_steps_explicit",
                "diffusion_scheme": scheme,
            },
        )
