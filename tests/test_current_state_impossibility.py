import torch

from hsno.models.baselines import build_model


def delayed_copy_target(history: torch.Tensor) -> torch.Tensor:
    """Toy non-Markovian map: the next exposed slice copies the oldest lag."""
    return history[:, 0]


def test_current_state_same_current_different_history_impossibility():
    """Current-state models are structurally unable to separate equal-current histories.

    The two samples have identical instantaneous states u(t, x) and identical
    conditioning/static inputs, but their oldest history slices differ. A delayed
    toy map that copies the oldest lag therefore has two different valid futures.
    Any current-state-only neural operator must produce identical predictions for
    these samples, so it cannot represent this non-Markovian map exactly.
    """
    history = torch.zeros(2, 4, 1, 8)
    history[0, 0] = -1.0
    history[1, 0] = 1.0
    history[:, -1] = 0.25  # same current state for both trajectories
    cond = torch.zeros(2, 2)

    target_future = delayed_copy_target(history)
    assert not torch.allclose(target_future[0], target_future[1])
    assert torch.allclose(history[0, -1], history[1, -1])

    model = build_model(
        "current_state",
        history_steps=4,
        channels=1,
        cond_dim=2,
        static_channels=0,
        step_slices=1,
        width=8,
        depth=1,
        dim=1,
    )
    pred_future = model.predict_future(history, cond, None)[:, 0]

    assert torch.allclose(pred_future[0], pred_future[1], atol=1e-6)
    assert not (torch.allclose(pred_future[0], target_future[0]) and torch.allclose(pred_future[1], target_future[1]))
