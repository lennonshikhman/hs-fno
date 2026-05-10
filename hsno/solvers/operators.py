from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .finite_difference import laplacian_1d, laplacian_2d
from .spectral import spectral_laplacian_1d, spectral_laplacian_2d


def laplacian(u: np.ndarray, dx: float, dy: float | None = None, boundary: str = "periodic", scheme: str = "finite_difference") -> np.ndarray:
    """Configurable diffusion operator for solver implementations."""
    dim = 2 if u.ndim >= 3 else 1
    use_spectral = scheme in {"spectral", "fourier", "pseudospectral"} and boundary == "periodic"
    if use_spectral and dim == 1:
        return spectral_laplacian_1d(u, length=1.0)
    if use_spectral and dim == 2:
        return spectral_laplacian_2d(u, length_x=1.0, length_y=1.0)
    return laplacian_1d(u, dx, boundary) if dim == 1 else laplacian_2d(u, dx, dy if dy is not None else dx, boundary)


def is_semi_implicit(scheme: str) -> bool:
    return str(scheme).lower() in {"semi_implicit", "semi-implicit", "implicit_diffusion", "imex"}


def _fd_matrix_1d(n: int, dx: float, boundary: str) -> sparse.csr_matrix:
    inv = 1.0 / (dx * dx)
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    mat = sparse.diags([off, main, off], [-1, 0, 1], shape=(n, n), format="lil")
    if boundary == "periodic":
        mat[0, -1] = 1.0; mat[-1, 0] = 1.0
    elif boundary == "neumann":
        mat[0, 1] = 2.0; mat[-1, -2] = 2.0
    elif boundary == "dirichlet":
        # Boundary values are fixed by the one-sided ghost convention in finite_difference.py.
        pass
    else:
        raise ValueError(f"unknown boundary {boundary}")
    return (inv * mat).tocsr()


def _implicit_matrix(shape: tuple[int, ...], dx: float, dy: float | None, boundary: str, alpha: float) -> sparse.csr_matrix:
    if len(shape) == 1:
        L = _fd_matrix_1d(shape[0], dx, boundary)
    elif len(shape) == 2:
        nx, ny = shape
        Lx = _fd_matrix_1d(nx, dx, boundary)
        Ly = _fd_matrix_1d(ny, dy if dy is not None else dx, boundary)
        L = sparse.kron(sparse.eye(ny, format="csr"), Lx, format="csr") + sparse.kron(Ly, sparse.eye(nx, format="csr"), format="csr")
    else:
        raise ValueError("semi-implicit diffusion supports 1D/2D spatial grids")
    return sparse.eye(L.shape[0], format="csr") - float(alpha) * L


def semi_implicit_diffusion_step(rhs: np.ndarray, diffusion: float, dt: float, dx: float, dy: float | None = None, boundary: str = "periodic") -> np.ndarray:
    """Solve ``(I - dt*diffusion*Δ) u_{n+1} = rhs`` for 1D/2D grids.

    The solve is channelwise and supports periodic, Neumann, and Dirichlet finite-difference
    Laplacians. This gives local solvers an IMEX option: nonlinear/delay terms are explicit,
    diffusion is unconditionally implicit for the configured discrete Laplacian.
    """
    rhs = np.asarray(rhs)
    spatial = rhs.shape[1:]
    A = _implicit_matrix(tuple(spatial), dx, dy, boundary, float(dt) * float(diffusion))
    flat = rhs.reshape(rhs.shape[0], -1)
    solved = np.stack([spsolve(A, flat[c]) for c in range(flat.shape[0])], axis=0)
    return solved.reshape(rhs.shape).astype(rhs.dtype, copy=False)
