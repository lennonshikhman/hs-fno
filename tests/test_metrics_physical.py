import numpy as np
import torch
from hsno.data.normalization import ChannelNormalizer


def test_rollout_decode_restores_physical_channel_axis():
    norm = ChannelNormalizer(mean=np.array([10.0], dtype=np.float32), std=np.array([2.0], dtype=np.float32))
    normalized = torch.zeros(2, 3, 4, 1, 5)
    flat = norm.decode(normalized.reshape(6, 4, 1, 5)).reshape(2, 3, 4, 1, 5)
    assert torch.allclose(flat, torch.full_like(flat, 10.0))
