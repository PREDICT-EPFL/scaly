"""Cart-pole swing-up.

Classic underactuated swing-up: a pole hinged on a cart, single horizontal
force on the cart, with a force limit that makes the upright unreachable by a
single push, so the controller must pump energy.  Angle ``theta`` is measured
from the *upright* (theta=0 unstable, theta=pi hanging).

State  x = [p, theta, dp, dtheta]   Input u = [F]
Goal   regulate to upright x = 0 from the hanging start.

The simulation plant uses a heavier pole than the prediction model (mass
mismatch) and small process noise, so closed-loop scores reflect robustness.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from fastbench.core.problem import Problem, ProblemMeta
from fastbench.core.registry import register_problem

G = 9.81


def _cartpole_rhs(x, u, M, m, l):
    p, th, dp, dth = x[0], x[1], x[2], x[3]
    F = u[0]
    s, c = ca.sin(th), ca.cos(th)
    den = M + m - m * c * c
    ddp = (-m * l * s * dth * dth + m * G * c * s + F) / den
    ddth = (-m * l * c * s * dth * dth + F * c + (M + m) * G * s) / (l * den)
    return ca.vertcat(dp, dth, ddp, ddth)


class CartPoleSwingUp(Problem):
    def __init__(self):
        self.M, self.m, self.l = 1.0, 0.1, 0.8           # model parameters
        self.M_p, self.m_p, self.l_p = 1.0, 0.115, 0.8   # plant (15% heavier pole)
        self.meta = ProblemMeta(
            name="pendulum_swingup", nx=4, nu=1, dt=0.02, N=50, n_sim=150,
            is_lti=False, nonlinear=True, problem_class="swing-up",
            description="Cart-pole energy-pumping swing-up to the upright.",
            sources=[
                "acados pendulum_on_cart example: https://github.com/acados/acados/tree/main/examples/acados_python/pendulum_on_cart",
                "M. Kelly, 'An Introduction to Trajectory Optimization', SIAM Review 59(4), 2017.",
                "OptimTraj cart-pole demo: https://github.com/MatthewPeterKelly/OptimTraj",
            ])
        self.lbu = np.array([-25.0]); self.ubu = np.array([25.0])
        self.lbx = np.array([-2.5, -1e6, -1e6, -1e6])
        self.ubx = np.array([2.5, 1e6, 1e6, 1e6])
        self.Q = np.diag([5.0, 50.0, 0.5, 0.5])
        self.R = np.diag([0.01])
        self.Qf = np.diag([50.0, 500.0, 5.0, 5.0])

    def dynamics_ct(self, x, u):
        return _cartpole_rhs(x, u, self.M, self.m, self.l)

    def stage_cost(self, x, u, xref, uref):
        dx, du = x - xref, u - uref
        return ca.mtimes([dx.T, self.Q, dx]) + ca.mtimes([du.T, self.R, du])

    def terminal_cost(self, x, xref):
        dx = x - xref
        return ca.mtimes([dx.T, self.Qf, dx])

    def reference_traj(self, k):
        return np.zeros(4), np.zeros(1)

    def x0_nominal(self):
        return np.array([0.0, np.pi, 0.0, 0.0])     # hanging down

    def x0_samples(self, n, rng):
        base = self.x0_nominal()
        if n == 1:
            return [base]
        return [base + np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.3, 0.3),
                                 0.0, 0.0]) for _ in range(n)]

    def plant_step(self, x, u, rng=None):
        from fastbench.core.dynamics import rk4
        xx = ca.SX.sym("x", 4); uu = ca.SX.sym("u", 1)
        if not hasattr(self, "_fp"):
            xn = rk4(lambda a, b: _cartpole_rhs(a, b, self.M_p, self.m_p, self.l_p),
                     xx, uu, self.meta.dt)
            self._fp = ca.Function("fp", [xx, uu], [xn])
        xn = np.array(self._fp(x, u)).flatten()
        if rng is not None:
            xn += rng.normal(0, 1e-3, size=4) * np.array([1, 1, 2, 2])
        return xn

    def episode_success(self, X, U):
        xf = X[-1]
        return bool(abs(xf[1]) < 0.15 and abs(xf[3]) < 0.5 and abs(xf[0]) < 0.5)


@register_problem("pendulum_swingup")
def _factory():
    return CartPoleSwingUp()
