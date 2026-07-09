"""Integration helpers shared by problems and adapters."""
from __future__ import annotations
import casadi as ca


def rk4(f, x, u, dt, p=None, steps: int = 1):
    """Fixed-step explicit RK4 of continuous dynamics ``xdot = f(x, u[, p])``.

    Works symbolically (CasADi) or numerically (the same expression graph is
    evaluated either way), so it can be reused by every adapter.
    """
    h = dt / steps
    xk = x
    for _ in range(steps):
        if p is None:
            k1 = f(xk, u)
            k2 = f(xk + h / 2 * k1, u)
            k3 = f(xk + h / 2 * k2, u)
            k4 = f(xk + h * k3, u)
        else:
            k1 = f(xk, u, p)
            k2 = f(xk + h / 2 * k1, u, p)
            k3 = f(xk + h / 2 * k2, u, p)
            k4 = f(xk + h * k3, u, p)
        xk = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return xk


def c2d(Ac, Bc, dt):
    """Exact zero-order-hold discretization of (Ac, Bc)."""
    import numpy as np
    import scipy.linalg as sla
    nx = Ac.shape[0]
    nu = Bc.shape[1]
    M = sla.expm(np.block([[Ac, Bc], [np.zeros((nu, nx + nu))]]) * dt)
    return M[:nx, :nx], M[:nx, nx:nx + nu]


def c2d_affine(Ac, Bc, w, dt):
    """ZOH discretization of an affine system xdot = Ac x + Bc u + w.

    Returns (Ad, Bd, wd) with x+ = Ad x + Bd u + wd.
    """
    import numpy as np
    import scipy.linalg as sla
    nx = Ac.shape[0]
    nu = Bc.shape[1]
    aug = np.zeros((nx + nu + 1, nx + nu + 1))
    aug[:nx, :nx] = Ac
    aug[:nx, nx:nx + nu] = Bc
    aug[:nx, nx + nu] = w
    M = sla.expm(aug * dt)
    return M[:nx, :nx], M[:nx, nx:nx + nu], M[:nx, nx + nu]


def discrete_function(problem, name: str = "fd") -> ca.Function:
    """Return a CasADi Function x_{k+1} = fd(x_k, u_k) for the prediction model."""
    x = ca.SX.sym("x", problem.meta.nx)
    u = ca.SX.sym("u", problem.meta.nu)
    xn = problem.discrete_dynamics(x, u)
    return ca.Function(name, [x, u], [xn], ["x", "u"], ["xn"])
