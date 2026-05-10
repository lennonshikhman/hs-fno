from __future__ import annotations
import torch
import torch.nn as nn
from hsno.data.history import shift_append
from .conditioning import broadcast_condition, coordinate_channels, FiLM
from .unet import ConvUNet
from .fno import FNOStyle
from .temporal_transformer import SpatialTransformerBackbone


def make_backbone(kind, in_ch, out_ch, width=32, depth=3, dim=1):
    if kind == "fno":
        return FNOStyle(in_ch, out_ch, width, depth, dim)
    if kind in {"transformer", "spatial_transformer"}:
        return SpatialTransformerBackbone(in_ch, out_ch, width, depth, dim)
    if kind in {"conv", "unet"}:
        return ConvUNet(in_ch, out_ch, width, depth, dim)
    raise ValueError(f"unknown backbone kind: {kind}")


def with_step_condition(cond: torch.Tensor | None, raw_delta_t: float | torch.Tensor | None, cond_mean=None, cond_std=None, delta_t_index: int = -1):
    """Return a conditioning vector with the normalized ``delta_t`` entry replaced."""
    if raw_delta_t is None or cond is None or cond.shape[1] == 0:
        return cond
    out = cond.clone()
    idx = delta_t_index if delta_t_index >= 0 else cond.shape[1] + delta_t_index
    value = torch.as_tensor(raw_delta_t, device=cond.device, dtype=cond.dtype)
    if value.ndim == 0:
        value = value.expand(cond.shape[0])
    mean = 0.0 if cond_mean is None else torch.as_tensor(cond_mean, device=cond.device, dtype=cond.dtype).flatten()[idx]
    std = 1.0 if cond_std is None else torch.as_tensor(cond_std, device=cond.device, dtype=cond.dtype).flatten()[idx]
    out[:, idx] = (value.reshape(-1) - mean) / (std + 1e-12)
    return out


class FutureSliceOperator(nn.Module):
    """Backbone-agnostic predictor P_theta(history, conditioning) for exposed future segment."""

    def __init__(
        self,
        history_steps,
        channels,
        cond_dim=0,
        static_channels=0,
        step_slices=1,
        backbone="conv",
        width=32,
        depth=3,
        dim=1,
        coordinate_embedding=False,
        film=False,
    ):
        super().__init__()
        self.history_steps = history_steps
        self.channels = channels
        self.step_slices = step_slices
        self.dim = dim
        self.cond_dim = cond_dim
        self.static_channels = static_channels
        self.coordinate_embedding = bool(coordinate_embedding)
        coord_ch = dim if self.coordinate_embedding else 0
        in_ch = history_steps * channels + cond_dim + static_channels + coord_ch
        self.net = make_backbone(backbone, in_ch, step_slices * channels, width, depth, dim)
        self.film = FiLM(cond_dim, step_slices * channels) if film else None

    def forward(self, history, cond=None, static=None):
        b, h, c, *sp = history.shape
        x = history.reshape(b, h * c, *sp)
        parts = [x]
        if cond is not None and cond.shape[1] > 0:
            parts.append(broadcast_condition(cond, sp))
        if static is not None and self.static_channels:
            parts.append(static)
        if self.coordinate_embedding:
            parts.append(coordinate_channels(b, sp, history.device, history.dtype))
        y = self.net(torch.cat(parts, 1))
        if self.film is not None:
            y = self.film(y, cond)
        return y.reshape(b, self.step_slices, c, *sp)


class HSNO(nn.Module):
    """History-Space Neural Operator using exact shift-append transport."""

    def __init__(self, *args, append_mode="segment", **kwargs):
        super().__init__()
        self.predictor = FutureSliceOperator(*args, **kwargs)
        self.step_slices = self.predictor.step_slices
        self.append_mode = append_mode
        if append_mode not in {"segment", "recursive"}:
            raise ValueError("append_mode must be 'segment' or 'recursive'")
        self.one_step_predictor = None
        if append_mode == "recursive" and self.step_slices > 1:
            kwargs = dict(kwargs)
            kwargs["step_slices"] = 1
            self.one_step_predictor = FutureSliceOperator(*args, **kwargs)

    def forward(self, history, cond=None, static=None, delta_t=None, cond_mean=None, cond_std=None):
        cond = with_step_condition(cond, delta_t, cond_mean, cond_std)
        if self.append_mode == "recursive" and self.one_step_predictor is not None:
            h = history
            pieces = []
            for _ in range(self.step_slices):
                fut = self.one_step_predictor(h, cond, static)
                pieces.append(fut)
                h = shift_append(h, fut, 1)
            return shift_append(history, torch.cat(pieces, dim=1), self.step_slices)
        future = self.predictor(history, cond, static)
        return shift_append(history, future, self.step_slices)

    def predict_future(self, history, cond=None, static=None, delta_t=None, cond_mean=None, cond_std=None):
        cond = with_step_condition(cond, delta_t, cond_mean, cond_std)
        if self.append_mode == "recursive" and self.one_step_predictor is not None:
            h, pieces = history, []
            for _ in range(self.step_slices):
                fut = self.one_step_predictor(h, cond, static)
                pieces.append(fut)
                h = shift_append(h, fut, 1)
            return torch.cat(pieces, dim=1)
        return self.predictor(history, cond, static)


class HSNOFull(HSNO):
    """Named HSNO-full variant for rollout/semiflow experiments."""

    def __init__(self, *args, append_mode="recursive", **kwargs):
        super().__init__(*args, append_mode=append_mode, **kwargs)

# Generic family alias; HS-FNO is the paper headline model.
HistorySpaceNeuralOperator = HSNO
