from __future__ import annotations
import torch
import torch.nn as nn
from .conditioning import broadcast_condition


class _ConvLSTMCell(nn.Module):
    """Spatial ConvLSTM cell for 1D or 2D regular grids."""

    def __init__(self, in_channels: int, hidden_channels: int, dim: int = 1):
        super().__init__()
        Conv = nn.Conv2d if dim == 2 else nn.Conv1d
        self.hidden_channels = hidden_channels
        self.gates = Conv(in_channels + hidden_channels, 4 * hidden_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None):
        if state is None:
            shape = (x.shape[0], self.hidden_channels, *x.shape[2:])
            h = x.new_zeros(shape)
            c = x.new_zeros(shape)
        else:
            h, c = state
        i, f, g, o = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMOperator(nn.Module):
    """ConvLSTM baseline that preserves spatial structure over the history axis."""

    def __init__(self, channels, cond_dim=0, out_steps=1, width=32, dim=1):
        super().__init__()
        self.out_steps = out_steps
        self.channels = channels
        self.cond_dim = cond_dim
        self.dim = dim
        Conv = nn.Conv2d if dim == 2 else nn.Conv1d
        self.cell = _ConvLSTMCell(channels + cond_dim, width, dim)
        self.head = Conv(width, out_steps * channels, kernel_size=1)

    def forward(self, history, cond):
        b, h_steps, channels, *spatial = history.shape
        state = None
        cond_grid = None
        if cond is not None and cond.numel():
            cond_grid = broadcast_condition(cond, spatial)
        for t in range(h_steps):
            x_t = history[:, t]
            if cond_grid is not None:
                x_t = torch.cat([x_t, cond_grid], dim=1)
            state = self.cell(x_t, state)
        hidden, _ = state
        out = self.head(hidden)
        return out.reshape(b, self.out_steps, self.channels, *spatial)
