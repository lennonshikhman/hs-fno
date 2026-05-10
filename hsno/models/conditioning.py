from __future__ import annotations
import torch
import torch.nn as nn


def broadcast_condition(cond: torch.Tensor, spatial_shape, history_steps: int | None = None) -> torch.Tensor:
    shape = [cond.shape[0]] + ([] if history_steps is None else [history_steps]) + [cond.shape[1]] + list(spatial_shape)
    view = [cond.shape[0]] + ([] if history_steps is None else [1]) + [cond.shape[1]] + [1] * len(spatial_shape)
    return cond.reshape(view).expand(shape)


def coordinate_channels(batch: int, spatial_shape, device=None, dtype=None) -> torch.Tensor:
    """Return normalized coordinate channels broadcast over a batch.

    1D returns ``x``; 2D returns ``x,y`` on ``[0,1]`` with shape ``[B,d,*spatial]``.
    """
    axes = [torch.linspace(0.0, 1.0, int(n), device=device, dtype=dtype) for n in spatial_shape]
    if len(axes) == 1:
        coords = axes[0].reshape(1, 1, -1)
    elif len(axes) == 2:
        x, y = torch.meshgrid(axes[0], axes[1], indexing="ij")
        coords = torch.stack([x, y], dim=0).unsqueeze(0)
    else:
        raise ValueError("coordinate embeddings support 1D or 2D grids")
    return coords.expand(batch, -1, *spatial_shape)


class FiLM(nn.Module):
    """Feature-wise affine modulation generated from conditioning variables."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.channels = int(channels)
        self.affine = nn.Linear(cond_dim, 2 * channels) if cond_dim > 0 else None
        if self.affine is not None:
            nn.init.zeros_(self.affine.weight)
            nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if self.affine is None or cond is None or cond.shape[1] == 0:
            return x
        gamma, beta = self.affine(cond).chunk(2, dim=1)
        shape = [x.shape[0], x.shape[1]] + [1] * (x.ndim - 2)
        return x * (1.0 + gamma.reshape(shape)) + beta.reshape(shape)
