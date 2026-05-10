from __future__ import annotations
from dataclasses import dataclass
import numpy as np, torch

@dataclass
class ChannelNormalizer:
    mean: np.ndarray
    std: np.ndarray
    @classmethod
    def fit(cls, arrays: list[np.ndarray]) -> "ChannelNormalizer":
        x = np.concatenate([a.reshape(-1, a.shape[-2], *a.shape[-1:]) if False else a for a in arrays], axis=0)
        axes = tuple(i for i in range(x.ndim) if i != 1)
        mean = x.mean(axis=axes); std = x.std(axis=axes) + 1e-6
        return cls(mean.astype(np.float32), std.astype(np.float32))
    def encode(self, x):
        m = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype) if torch.is_tensor(x) else self.mean
        s = torch.as_tensor(self.std, device=x.device, dtype=x.dtype) if torch.is_tensor(x) else self.std
        shape = [1]*(x.ndim); shape[-(x.ndim-2) if False else 1] = -1
        # Supports [N,C,...] and [B,H,C,...]
        if x.ndim >= 4: shape = [1,1,-1] + [1]*(x.ndim-3)
        else: shape = [1,-1] + [1]*(x.ndim-2)
        return (x - m.reshape(shape)) / s.reshape(shape)
    def decode(self, x):
        m = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype) if torch.is_tensor(x) else self.mean
        s = torch.as_tensor(self.std, device=x.device, dtype=x.dtype) if torch.is_tensor(x) else self.std
        shape = [1,1,-1] + [1]*(x.ndim-3) if x.ndim >= 4 else [1,-1]+[1]*(x.ndim-2)
        return x * s.reshape(shape) + m.reshape(shape)
