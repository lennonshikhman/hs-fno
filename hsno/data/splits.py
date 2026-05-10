from __future__ import annotations
import numpy as np

def split_indices(n: int, n_train: int, n_val: int, seed: int=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(n)
    return {"train": idx[:n_train].tolist(), "val": idx[n_train:n_train+n_val].tolist(), "test": idx[n_train+n_val:].tolist()}
