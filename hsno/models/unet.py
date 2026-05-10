from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _center_crop_or_pad(x: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """Match the final spatial dimensions by symmetric crop/pad."""
    for axis, target in enumerate(target_shape, start=x.ndim - len(target_shape)):
        size = x.shape[axis]
        if size > target:
            start = (size - target) // 2
            x = x.narrow(axis, start, target)
        elif size < target:
            pad_before = (target - size) // 2
            pad_after = target - size - pad_before
            pads = [0, 0] * len(target_shape)
            rel_axis = axis - (x.ndim - len(target_shape))
            pads[2 * (len(target_shape) - rel_axis - 1)] = pad_before
            pads[2 * (len(target_shape) - rel_axis - 1) + 1] = pad_after
            x = F.pad(x, pads)
    return x


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dim: int):
        super().__init__()
        Conv = nn.Conv2d if dim == 2 else nn.Conv1d
        self.net = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            Conv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ConvUNet(nn.Module):
    """Small but real U-Net backbone with down/up paths and skip connections."""

    def __init__(self, in_channels, out_channels, width=32, depth=3, dim=1):
        super().__init__()
        depth = max(1, int(depth))
        self.dim = dim
        self.pool = nn.MaxPool2d(2) if dim == 2 else nn.MaxPool1d(2)
        self.down = nn.ModuleList()
        channels = []
        ch = in_channels
        for level in range(depth):
            out_ch = width * (2**level)
            self.down.append(_ConvBlock(ch, out_ch, dim))
            channels.append(out_ch)
            ch = out_ch
        self.bottleneck = _ConvBlock(ch, ch * 2, dim)
        ch = ch * 2
        self.up = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        ConvT = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose1d
        for skip_ch in reversed(channels):
            self.up.append(ConvT(ch, skip_ch, kernel_size=2, stride=2))
            self.up_blocks.append(_ConvBlock(skip_ch * 2, skip_ch, dim))
            ch = skip_ch
        Conv = nn.Conv2d if dim == 2 else nn.Conv1d
        self.head = Conv(ch, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        out = x
        for block in self.down:
            out = block(out)
            skips.append(out)
            if min(out.shape[2:]) >= 2:
                out = self.pool(out)
        out = self.bottleneck(out)
        for up, block, skip in zip(self.up, self.up_blocks, reversed(skips)):
            out = up(out)
            out = _center_crop_or_pad(out, tuple(skip.shape[2:]))
            out = block(torch.cat([out, skip], dim=1))
        out = self.head(out)
        return _center_crop_or_pad(out, tuple(x.shape[2:]))
