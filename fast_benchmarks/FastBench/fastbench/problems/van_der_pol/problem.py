"""Van der Pol oscillator - classic nonlinear regulation.

ẋ1 = x2,  ẋ2 = mu (1 - x1^2) x2 - x1 + u.  Stabilize the origin against the
limit cycle. A standard small nonlinear OCP test (appears in COPS, ICLOCS2,
CasADi examples).
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from fastbench.core.problem import Problem, ProblemMeta
from fastbench.core.registry import register_problem


class VanDerPol(Problem):
    def __init__(self):
        self.mu = 1.0
        self.mu_p = 1.2          # plant has stronger nonlinearity
        self.meta = ProblemMeta(
            name="van_der_pol", nx=2, nu=1, dt=0.1, N=25, n_sim=60,
            is_lti=False, nonlinear=True, problem_class="regulation",
            description="Van der Pol oscillator stabilized to the origin.",
            sources=["Dolan, More, Munson, COPS benchmark - https://www.mcs.anl.gov/~more/cops/",
                     "CasADi examples - https://github.com/casadi/casadi"])
        self.lbu = np.array([-1.0]); self.ubu = np.array([1.0])
        self.lbx = np.array([-5.0, -5.0]); self.ubx = np.array([5.0, 5.0])
        self.Q = np.diag([1.0, 1.0]); self.R = np.diag([0.1])
        self.Qf = np.diag([5.0, 5.0])

    def _rhs(self, x, u, mu):
        return ca.vertcat(x[1], mu * (1 - x[0] ** 2) * x[1] - x[0] + u[0])

    def dynamics_ct(self, x, u):
        return self._rhs(x, u, self.mu)

    def stage_cost(self, x, u, xref, uref):
        dx, du = x - xref, u - uref
        return ca.mtimes([dx.T, ca.DM(self.Q), dx]) + ca.mtimes([du.T, ca.DM(self.R), du])

    def terminal_cost(self, x, xref):
        dx = x - xref
        return ca.mtimes([dx.T, ca.DM(self.Qf), dx])

    def x0_nominal(self):
        return np.array([2.0, 0.0])

    def x0_samples(self, n, rng):
        if n == 1:
            return [self.x0_nominal()]
        return [np.array([rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5)])
                for _ in range(n)]

    def plant_step(self, x, u, rng=None):
        from fastbench.core.dynamics import rk4
        if not hasattr(self, "_fp"):
            xx = ca.SX.sym("x", 2); uu = ca.SX.sym("u", 1)
            self._fp = ca.Function("fp", [xx, uu],
                                   [rk4(lambda a, b: self._rhs(a, b, self.mu_p),
                                        xx, uu, self.meta.dt)])
        xn = np.array(self._fp(x, u)).flatten()
        if rng is not None:
            xn += rng.normal(0, 1e-3, size=2)
        return xn

    def episode_success(self, X, U):
        return bool(np.linalg.norm(X[-1]) < 0.1)


@register_problem("van_der_pol")
def _factory():
    return VanDerPol()
