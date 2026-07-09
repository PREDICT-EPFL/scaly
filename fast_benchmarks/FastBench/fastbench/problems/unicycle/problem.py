"""Unicycle parking - nonholonomic regulation to a pose.

ẋ = v cosθ, ẏ = v sinθ, θ̇ = ω.  Drive the vehicle from an offset pose to the
origin. Nonholonomic, so the linearization at the target is uncontrollable -
a good stress test for NMPC vs linear MPC.

State x = [X, Y, θ], input u = [v, ω].
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from fastbench.core.problem import Problem, ProblemMeta
from fastbench.core.registry import register_problem


class Unicycle(Problem):
    def __init__(self):
        self.scale_p = 1.05      # plant actuator-gain mismatch
        self.meta = ProblemMeta(
            name="unicycle", nx=3, nu=2, dt=0.1, N=25, n_sim=70,
            is_lti=False, nonlinear=True, problem_class="regulation",
            description="Nonholonomic unicycle parking to the origin pose.",
            sources=["Aguiar & Hespanha, nonholonomic control.",
                     "CommonRoad / mobile-robot MPC benchmarks - https://commonroad.in.tum.de"])
        self.lbu = np.array([-1.0, -2.0]); self.ubu = np.array([1.0, 2.0])
        self.lbx = np.array([-1e6, -1e6, -1e6]); self.ubx = np.array([1e6, 1e6, 1e6])
        self.Q = np.diag([5.0, 5.0, 1.0]); self.R = np.diag([0.1, 0.1])
        self.Qf = np.diag([20.0, 20.0, 4.0])

    def dynamics_ct(self, x, u):
        return ca.vertcat(u[0] * ca.cos(x[2]), u[0] * ca.sin(x[2]), u[1])

    def stage_cost(self, x, u, xref, uref):
        dx, du = x - xref, u - uref
        return ca.mtimes([dx.T, ca.DM(self.Q), dx]) + ca.mtimes([du.T, ca.DM(self.R), du])

    def terminal_cost(self, x, xref):
        dx = x - xref
        return ca.mtimes([dx.T, ca.DM(self.Qf), dx])

    def x0_nominal(self):
        return np.array([2.0, 2.0, 0.0])

    def x0_samples(self, n, rng):
        if n == 1:
            return [self.x0_nominal()]
        return [np.array([rng.uniform(-3, 3), rng.uniform(-3, 3),
                          rng.uniform(-np.pi, np.pi)]) for _ in range(n)]

    def plant_step(self, x, u, rng=None):
        from fastbench.core.dynamics import rk4
        if not hasattr(self, "_fp"):
            xx = ca.SX.sym("x", 3); uu = ca.SX.sym("u", 2)

            def fp(a, b):
                return ca.vertcat(self.scale_p * b[0] * ca.cos(a[2]),
                                  self.scale_p * b[0] * ca.sin(a[2]), b[1])
            self._fp = ca.Function("fp", [xx, uu], [rk4(fp, xx, uu, self.meta.dt)])
        xn = np.array(self._fp(x, u)).flatten()
        if rng is not None:
            xn += rng.normal(0, [5e-3, 5e-3, 2e-3])
        return xn

    def episode_success(self, X, U):
        return bool(np.linalg.norm(X[-1][:2]) < 0.2)


@register_problem("unicycle")
def _factory():
    return Unicycle()
