from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Trajectory:
    fields: np.ndarray  # [T,C,...]
    times: np.ndarray
    params: dict
    static: np.ndarray | None = None
    metadata: dict | None = None

class BaseSolver:
    name = "base"
    def simulate(self, *args, **kwargs) -> Trajectory: raise NotImplementedError
