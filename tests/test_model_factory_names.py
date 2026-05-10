import pytest
import torch

from hsno.data.history import shift_append
from hsno.models.baselines import build_model
from hsno.models.hsfno import HistorySpaceFNO
from hsno.utils.naming import canonical_model_name


def _kwargs():
    return dict(history_steps=4, channels=1, cond_dim=3, static_channels=0, step_slices=1, width=8, depth=1, dim=1)


def test_new_history_space_model_names_construct():
    h = torch.randn(2, 4, 1, 16)
    c = torch.randn(2, 3)
    for name in ["hs_fno", "hsno_unet", "hs_transformer", "hs_fno_no_shift", "hs_fno_rollout_semiflow"]:
        model = build_model(name, **_kwargs())
        y = model(h, c, None)
        assert y.shape == h.shape
    assert isinstance(build_model("hs_fno", **_kwargs()), HistorySpaceFNO)


def test_legacy_names_remap_with_warning():
    assert canonical_model_name("ablation_hsno_fno_backbone") == "hs_fno"
    with pytest.warns(DeprecationWarning):
        model = build_model("ablation_hsno_fno_backbone", **_kwargs())
    assert isinstance(model, HistorySpaceFNO)


def test_hs_fno_uses_exact_shift_append():
    model = build_model("hs_fno", **_kwargs())
    h = torch.randn(2, 4, 1, 8)
    c = torch.randn(2, 3)
    future = model.predict_future(h, c, None)
    assert torch.allclose(model(h, c, None), shift_append(h, future, model.step_slices))
