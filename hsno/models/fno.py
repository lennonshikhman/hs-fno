from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Fourier layer with learned complex weights on low modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x, dim=-1)
        modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(x.shape[0], self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        out_ft[..., :modes] = torch.einsum("bim,iom->bom", x_ft[..., :modes], self.weight[..., :modes])
        return torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)


class SpectralConv2d(nn.Module):
    """2D Fourier layer with learned complex weights on low modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfftn(x, dim=(-2, -1))
        mx = min(self.modes, x_ft.shape[-2])
        my = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(x.shape[0], self.out_channels, x_ft.shape[-2], x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        out_ft[..., :mx, :my] = torch.einsum("bixy,ioxy->boxy", x_ft[..., :mx, :my], self.weight[..., :mx, :my])
        return torch.fft.irfftn(out_ft, s=x.shape[-2:], dim=(-2, -1))


class _FNOBlock(nn.Module):
    def __init__(self, width: int, modes: int, dim: int):
        super().__init__()
        Conv = nn.Conv2d if dim == 2 else nn.Conv1d
        Spectral = SpectralConv2d if dim == 2 else SpectralConv1d
        self.spectral = Spectral(width, width, modes)
        self.local = Conv(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.local(x))


class FNOStyle(nn.Module):
    """Fourier neural-operator backbone for 1D/2D regular grids."""

    def __init__(self, in_channels, out_channels, width=32, depth=3, dim=1, modes=12):
        super().__init__()
        Conv = nn.Conv2d if dim == 2 else nn.Conv1d
        self.lift = Conv(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([_FNOBlock(width, modes, dim) for _ in range(depth)])
        self.proj = nn.Sequential(Conv(width, width, kernel_size=1), nn.GELU(), Conv(width, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(x)
