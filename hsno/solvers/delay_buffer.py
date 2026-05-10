"""Time-history buffer with sorted linear interpolation for method-of-steps solvers."""
from __future__ import annotations
import bisect
import numpy as np


class DelayBuffer:
    """Store historical PDE states and interpolate delayed values in time.

    The buffer keeps times sorted, which lets method-of-steps solvers seed an
    initial history on ``[-tau, 0]`` and then append computed states at positive
    times. Queries outside the stored interval are clamped to the nearest known
    state; in-range queries use first-order interpolation.
    """

    def __init__(self, times=None, values=None):
        self.times: list[float] = []
        self.values: list[np.ndarray] = []
        if times is not None and values is not None:
            for t, v in sorted(zip(times, values), key=lambda item: item[0]):
                self.append(float(t), np.asarray(v))

    def append(self, t: float, value: np.ndarray) -> None:
        t = float(t)
        v = np.array(value, copy=True)
        if self.times and t < self.times[-1]:
            j = bisect.bisect_left(self.times, t)
            if j < len(self.times) and np.isclose(self.times[j], t):
                self.values[j] = v
            else:
                self.times.insert(j, t)
                self.values.insert(j, v)
            return
        if self.times and np.isclose(self.times[-1], t):
            self.values[-1] = v
        else:
            self.times.append(t)
            self.values.append(v)

    def lookup(self, t: float) -> np.ndarray:
        if not self.times:
            raise ValueError("empty delay buffer")
        if t <= self.times[0]:
            return self.values[0].copy()
        if t >= self.times[-1]:
            return self.values[-1].copy()
        j = bisect.bisect_left(self.times, t)
        t0, t1 = self.times[j - 1], self.times[j]
        a = (t - t0) / max(t1 - t0, 1e-12)
        return (1 - a) * self.values[j - 1] + a * self.values[j]

    def lookup_pairwise_sources(self, query_times: np.ndarray, source_indices: np.ndarray | None = None, channel: int = 0) -> np.ndarray:
        """Vectorized delayed lookup for pairwise source-node delay matrices.

        Args:
            query_times: Matrix with shape [n_target, n_source] giving the
                requested delayed time for each target/source pair.
            source_indices: Optional integer matrix with the same shape as
                ``query_times``. Entry ``[i, j]`` selects which spatial source
                node is sampled for that pair. If omitted, columns are used as
                source indices.
            channel: Field channel to sample from stored values.

        Returns:
            Matrix with the same shape as ``query_times`` containing delayed
            source values. The interpolation is vectorized over all pairs.
        """
        if not self.values:
            raise ValueError("empty delay buffer")
        q = np.asarray(query_times, dtype=np.float64)
        if source_indices is None:
            source_indices = np.broadcast_to(np.arange(q.shape[-1]), q.shape)
        src = np.asarray(source_indices, dtype=np.int64)
        values = np.stack(self.values, axis=0)[:, channel]
        times = np.asarray(self.times, dtype=np.float64)
        flat_q = q.reshape(-1)
        flat_src = src.reshape(-1)
        right = np.searchsorted(times, flat_q, side="left")
        right = np.clip(right, 1, len(times) - 1)
        left = right - 1
        t0 = times[left]
        t1 = times[right]
        alpha = ((flat_q - t0) / np.maximum(t1 - t0, 1e-12)).astype(values.dtype)
        alpha = np.clip(alpha, 0.0, 1.0)
        flat_src = np.clip(flat_src, 0, values.shape[-1] - 1)
        out = (1.0 - alpha) * values[left, flat_src] + alpha * values[right, flat_src]
        out = np.where(flat_q <= times[0], values[0, flat_src], out)
        out = np.where(flat_q >= times[-1], values[-1, flat_src], out)
        return out.reshape(q.shape)

    def lookup_many_sources(self, t_matrix: np.ndarray) -> np.ndarray:
        """Backward-compatible wrapper for column-indexed source lookups."""
        return self.lookup_pairwise_sources(t_matrix)
