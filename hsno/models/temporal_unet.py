from __future__ import annotations
import torch
import torch.nn as nn


class TemporalUNetOperator(nn.Module):
    """U-Net over the history axis at every spatial node.

    The module preserves spatial variation by reshaping each grid point into an
    independent temporal sequence, applying a 1D encoder/decoder along history,
    and reconstructing future slices at the original grid locations.
    """

    def __init__(self, channels: int, cond_dim: int = 0, step_slices: int = 1, width: int = 32, depth: int = 2):
        super().__init__()
        self.channels = channels
        self.cond_dim = cond_dim
        self.step_slices = step_slices
        self.in_proj = nn.Conv1d(channels + cond_dim, width, 3, padding=1)
        self.downs = nn.ModuleList([nn.Conv1d(width, width, 3, stride=2, padding=1) for _ in range(max(1, depth))])
        self.ups = nn.ModuleList([nn.ConvTranspose1d(width, width, 4, stride=2, padding=1) for _ in range(max(1, depth))])
        self.blocks = nn.ModuleList([nn.Sequential(nn.GELU(), nn.Conv1d(width, width, 3, padding=1), nn.GELU()) for _ in range(2 * max(1, depth) + 1)])
        self.out = nn.Linear(width, step_slices * channels)

    def forward(self, history: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        b, h, c, *sp = history.shape
        n = int(torch.tensor(sp).prod().item())
        x = history.permute(0, *range(3, history.ndim), 2, 1).reshape(b * n, c, h)
        if cond is not None and cond.shape[1] > 0:
            cond_rep = cond[:, None, :, None].expand(b, n, cond.shape[1], h).reshape(b * n, cond.shape[1], h)
            x = torch.cat([x, cond_rep], dim=1)
        x = self.blocks[0](self.in_proj(x))
        skips = []
        for i, down in enumerate(self.downs):
            skips.append(x)
            x = self.blocks[i + 1](down(x))
        for i, up in enumerate(self.ups):
            x = up(x)
            skip = skips[-(i + 1)]
            if x.shape[-1] > skip.shape[-1]:
                x = x[..., : skip.shape[-1]]
            elif x.shape[-1] < skip.shape[-1]:
                x = torch.nn.functional.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = self.blocks[len(self.downs) + 1 + i](x + skip)
        feat = x[..., -1]
        y = self.out(feat).reshape(b, *sp, self.step_slices, c)
        return y.permute(0, len(sp) + 1, len(sp) + 2, *range(1, len(sp) + 1)).contiguous()
