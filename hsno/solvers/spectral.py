from __future__ import annotations
import numpy as np


def spectral_laplacian_1d(u: np.ndarray, length: float = 1.0) -> np.ndarray:
    """Fourier pseudospectral periodic 1D Laplacian along the last axis."""
    n = u.shape[-1]
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    return np.fft.ifft(-(k**2) * np.fft.fft(u, axis=-1), axis=-1).real.astype(u.dtype, copy=False)


def spectral_laplacian_2d(u: np.ndarray, length_x: float = 1.0, length_y: float = 1.0) -> np.ndarray:
    """Fourier pseudospectral periodic 2D Laplacian along the last two axes."""
    nx, ny = u.shape[-2], u.shape[-1]
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=length_x / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=length_y / ny)
    multiplier = -(kx[:, None] ** 2 + ky[None, :] ** 2)
    uhat = np.fft.fftn(u, axes=(-2, -1))
    return np.fft.ifftn(multiplier * uhat, axes=(-2, -1)).real.astype(u.dtype, copy=False)
