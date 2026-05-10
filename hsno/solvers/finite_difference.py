from __future__ import annotations
import numpy as np

def laplacian_1d(u: np.ndarray, dx: float, boundary: str="periodic") -> np.ndarray:
    if boundary == "periodic": return (np.roll(u,1,-1)-2*u+np.roll(u,-1,-1))/dx**2
    out = np.zeros_like(u); out[...,1:-1]=(u[...,:-2]-2*u[...,1:-1]+u[...,2:])/dx**2
    if boundary == "neumann":
        out[...,0]=2*(u[...,1]-u[...,0])/dx**2; out[...,-1]=2*(u[...,-2]-u[...,-1])/dx**2
    elif boundary == "dirichlet":
        out[...,0]=(-2*u[...,0]+u[...,1])/dx**2; out[...,-1]=(u[...,-2]-2*u[...,-1])/dx**2
    else: raise ValueError(boundary)
    return out

def laplacian_2d(u: np.ndarray, dx: float, dy: float, boundary: str="periodic") -> np.ndarray:
    if boundary == "periodic":
        return ((np.roll(u,1,-2)-2*u+np.roll(u,-1,-2))/dx**2 + (np.roll(u,1,-1)-2*u+np.roll(u,-1,-1))/dy**2)
    pad_mode = "edge" if boundary == "neumann" else "constant"
    p = np.pad(u, [(0,0)]*(u.ndim-2)+[(1,1),(1,1)], mode=pad_mode)
    return (p[...,2:,1:-1]-2*u+p[...,:-2,1:-1])/dx**2 + (p[...,1:-1,2:]-2*u+p[...,1:-1,:-2])/dy**2
