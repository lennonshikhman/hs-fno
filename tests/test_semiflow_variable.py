import torch
from hsno.training.losses import variable_semiflow_loss
from hsno.models.baselines import build_model


def test_variable_semiflow_loss_exercises_step_conditioning():
    h = torch.randn(2, 4, 1, 8)
    cond = torch.zeros(2, 3)
    batch = {
        "history": h,
        "cond": cond,
        "static": torch.zeros(2, 0, 8),
        "cond_raw": torch.tensor([[0.1, 0.2, 0.05], [0.2, 0.3, 0.05]]),
        "cond_mean": torch.zeros(2, 3),
        "cond_std": torch.ones(2, 3),
    }
    m = build_model("hsno", history_steps=4, channels=1, cond_dim=3, static_channels=0, step_slices=1, width=4, depth=1, dim=1)
    loss = variable_semiflow_loss(m, batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
