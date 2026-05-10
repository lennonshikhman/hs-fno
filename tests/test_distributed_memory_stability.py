import numpy as np

from hsno.solvers.distributed_memory import DistributedMemorySolver


def test_distributed_memory_substeps_resolution_transfer_without_overflow_warning():
    solver = DistributedMemorySolver()
    with np.errstate(over="raise", invalid="raise"):
        tr = solver.simulate(
            {"nu": 0.019, "a1": 0.9, "a2": 0.26, "tau": 0.25},
            nx=96,
            dt=0.005,
            dt_save=0.05,
            total_time=0.25,
            history_steps=6,
            seed=4,
            max_state=20.0,
        )
    assert np.isfinite(tr.fields).all()
    assert np.abs(tr.fields).max() <= 20.0 + 1e-5
    assert tr.metadata["substeps_per_requested_step"] > 1
    assert tr.metadata["internal_dt"] < tr.metadata["requested_dt"]
    assert tr.metadata["max_state"] == 20.0
