from __future__ import annotations
import torch
import torch.nn as nn


class TemporalTransformerOperator(nn.Module):
    """Transformer-over-history baseline applied independently at each grid point.

    Unlike a global spatial average, this keeps every spatial node as its own
    token sequence over history time. The shared transformer learns temporal
    dependence and returns a spatially varying future field.
    """

    def __init__(self, channels, cond_dim=0, out_steps=1, width=32, nhead=4):
        super().__init__()
        self.channels = channels
        self.cond_dim = cond_dim
        self.out_steps = out_steps
        self.inp = nn.Linear(channels + cond_dim, width)
        enc = nn.TransformerEncoderLayer(width, nhead, batch_first=True, dim_feedforward=2 * width)
        self.tr = nn.TransformerEncoder(enc, 1)
        self.head = nn.Linear(width, out_steps * channels)

    def forward(self, history, cond):
        b, h_steps, channels, *spatial = history.shape
        n_points = 1
        for size in spatial:
            n_points *= size
        # [B,H,C,*S] -> [B*N,H,C]
        x = history.reshape(b, h_steps, channels, n_points).permute(0, 3, 1, 2).reshape(b * n_points, h_steps, channels)
        if cond is not None and cond.numel():
            cond_tokens = cond[:, None, None, :].expand(b, n_points, h_steps, cond.shape[1]).reshape(b * n_points, h_steps, cond.shape[1])
            x = torch.cat([x, cond_tokens], dim=-1)
        y = self.tr(self.inp(x))[:, -1]
        out = self.head(y).reshape(b, n_points, self.out_steps, self.channels)
        out = out.permute(0, 2, 3, 1).reshape(b, self.out_steps, self.channels, *spatial)
        return out


class SpatialTransformerBackbone(nn.Module):
    """Backbone that applies self-attention across spatial grid tokens."""

    def __init__(self, in_channels, out_channels, width=32, depth=2, dim=1, nhead=4):
        super().__init__()
        self.out_channels = out_channels
        self.inp = nn.Linear(in_channels, width)
        enc = nn.TransformerEncoderLayer(width, nhead, batch_first=True, dim_feedforward=2 * width)
        self.encoder = nn.TransformerEncoder(enc, max(1, depth))
        self.head = nn.Linear(width, out_channels)

    def forward(self, x):
        b, channels, *spatial = x.shape
        n_points = 1
        for size in spatial:
            n_points *= size
        tokens = x.reshape(b, channels, n_points).permute(0, 2, 1)
        out = self.head(self.encoder(self.inp(tokens)))
        return out.permute(0, 2, 1).reshape(b, self.out_channels, *spatial)
