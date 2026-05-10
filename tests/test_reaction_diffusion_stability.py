import numpy as np

from hsno.solvers.reaction_diffusion import ReactionDiffusionSolver


def test_reaction_diffusion_sanitizes_large_states_without_overflow_warning():
    solver = ReactionDiffusionSolver()
    with np.errstate(over="raise", invalid="raise"):
        tr = solver.simulate(
            {"D": 0.02, "r": 2.5, "tau": 0.2},
            nx=24,
            dt=0.005,
            dt_save=0.05,
            total_time=0.25,
            history_steps=5,
            seed=3,
            max_state=20.0,
        )
    assert np.isfinite(tr.fields).all()
    assert tr.fields.min() >= 0.0
    assert tr.fields.max() <= 20.0 + 1e-5
    assert tr.metadata["max_state"] == 20.0
