"""Quadruple-tank process (Johansson 2000) - nonlinear level tracking.

Four interconnected tanks; two pump voltages feed diagonal pairs. The square
-root outflow makes it nonlinear, and the cross-coupling (a known
non-minimum-phase configuration is possible) makes it a classic MIMO process
benchmark. Track level setpoints in the two lower tanks.

State x = [h1, h2, h3, h4] (cm), input u = [v1, v2] (V).
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from fastbench.core.problem import Problem, ProblemMeta
from fastbench.core.registry import register_problem

G = 981.0  # cm/s^2


class QuadrupleTank(Problem):
    def __init__(self):
        self.A = np.array([28.0, 32.0, 28.0, 32.0])     # tank areas cm^2
        self.a = np.array([0.071, 0.057, 0.071, 0.057])  # outlet areas cm^2
        self.a_p = self.a * 1.05                          # plant outlet +5%
        self.k = np.array([3.33, 3.35])                   # pump gains cm^3/Vs
        self.g1, self.g2 = 0.7, 0.6                       # valve splits
        self.meta = ProblemMeta(
            name="quadruple_tank", nx=4, nu=2, dt=2.0, N=25, n_sim=70,
            is_lti=False, nonlinear=True, problem_class="tracking",
            description="Johansson quadruple-tank MIMO level tracking (sqrt nonlinearity).",
            sources=["K.H. Johansson, The quadruple-tank process, IEEE TCST 2000 - https://doi.org/10.1109/87.845876"])
        self.lbu = np.array([0.0, 0.0]); self.ubu = np.array([12.0, 12.0])
        self.lbx = np.zeros(4); self.ubx = np.array([30.0, 30.0, 30.0, 30.0])
        self.Q = np.diag([5.0, 5.0, 0.1, 0.1]); self.R = np.diag([0.05, 0.05])
        self.Qf = np.diag([10.0, 10.0, 0.2, 0.2])
        self.target = np.array([14.0, 14.0, 6.0, 6.0])
        self.uref = np.array([4.0, 4.0])

    def _rhs(self, x, u, a):
        h = [ca.fmax(x[i], 1e-6) for i in range(4)]
        s = [ca.sqrt(2 * G * h[i]) for i in range(4)]
        A, k, g1, g2 = self.A, self.k, self.g1, self.g2
        dh1 = -a[0] / A[0] * s[0] + a[2] / A[0] * s[2] + g1 * k[0] / A[0] * u[0]
        dh2 = -a[1] / A[1] * s[1] + a[3] / A[1] * s[3] + g2 * k[1] / A[1] * u[1]
        dh3 = -a[2] / A[2] * s[2] + (1 - g2) * k[1] / A[2] * u[1]
        dh4 = -a[3] / A[3] * s[3] + (1 - g1) * k[0] / A[3] * u[0]
        return ca.vertcat(dh1, dh2, dh3, dh4)

    def dynamics_ct(self, x, u):
        return self._rhs(x, u, self.a)

    def stage_cost(self, x, u, xref, uref):
        dx, du = x - xref, u - uref
        return ca.mtimes([dx.T, ca.DM(self.Q), dx]) + ca.mtimes([du.T, ca.DM(self.R), du])

    def terminal_cost(self, x, xref):
        dx = x - xref
        return ca.mtimes([dx.T, ca.DM(self.Qf), dx])

    def reference_traj(self, k):
        return self.target.copy(), self.uref.copy()

    def x0_nominal(self):
        return np.array([8.0, 8.0, 3.0, 3.0])

    def x0_samples(self, n, rng):
        if n == 1:
            return [self.x0_nominal()]
        return [np.clip(np.array([rng.uniform(4, 12), rng.uniform(4, 12),
                                  rng.uniform(2, 8), rng.uniform(2, 8)]),
                        0, 30) for _ in range(n)]

    def plant_step(self, x, u, rng=None):
        from fastbench.core.dynamics import rk4
        if not hasattr(self, "_fp"):
            xx = ca.SX.sym("x", 4); uu = ca.SX.sym("u", 2)
            self._fp = ca.Function("fp", [xx, uu],
                                   [rk4(lambda a, b: self._rhs(a, b, self.a_p),
                                        xx, uu, self.meta.dt)])
        xn = np.array(self._fp(x, u)).flatten()
        if rng is not None:
            xn += rng.normal(0, 2e-2, size=4)
        return np.maximum(xn, 0.0)

    def episode_success(self, X, U):
        return bool(np.linalg.norm(X[-1][:2] - self.target[:2]) < 0.5)


@register_problem("quadruple_tank")
def _factory():
    return QuadrupleTank()
