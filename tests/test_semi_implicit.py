import numpy as np
from hsno.solvers.operators import semi_implicit_diffusion_step


def test_semi_implicit_diffusion_step_is_finite_2d():
    rhs = np.random.default_rng(0).normal(size=(1, 5, 4)).astype(np.float32)
    out = semi_implicit_diffusion_step(rhs, diffusion=0.1, dt=0.05, dx=0.2, dy=0.25, boundary="neumann")
    assert out.shape == rhs.shape
    assert np.isfinite(out).all()
