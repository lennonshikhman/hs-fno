import numpy as np
from hsno.data.history import make_tau_history_windows


def test_tau_history_windows_return_true_k_step_targets():
    times = np.linspace(-0.2, 0.4, 7, dtype=np.float32)
    fields = times[:, None, None].astype(np.float32)
    xs, ys, fut, roll = make_tau_history_windows(fields, times, tau=0.2, history_steps=3, step_slices=1, delta_t=0.1, rollout_steps=2)
    assert len(xs) >= 1
    assert np.allclose(ys[0], roll[0][0])
    assert np.allclose(roll[0][1, :, 0, 0], [0.0, 0.1, 0.2])
