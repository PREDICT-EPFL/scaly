"""Exothermic CSTR - nonlinear stabilization of a steady state.

Dimensionless continuous stirred-tank reactor (Uppal-Ray form):

    ẋ1 = -x1 + Da (1-x1) exp(x2)
    ẋ2 = -x2 + B·Da (1-x1) exp(x2) - β (x2 - u)

with x1 conversion, x2 (dimensionless) temperature and u the cooling jacket
input. For these parameters the open loop has a steady state that the
controller must hold against perturbations; the Arrhenius term makes it stiff
and genuinely nonlinear.
"""
from __future__ import annotations

import casadi as ca
import numpy as np
from scipy.optimize import fsolve

from fastbench.core.problem import Problem, ProblemMeta
from fastbench.core.registry import register_problem


class CSTR(Problem):
    def __init__(self):
        self.Da, self.B, self.beta = 0.072, 8.0, 0.3
        self.Da_p = 0.072 * 1.05            # plant reaction-rate mismatch
        self.meta = ProblemMeta(
            name="cstr", nx=2, nu=1, dt=0.2, N=25, n_sim=70,
            is_lti=False, nonlinear=True, problem_class="stabilization",
            description="Exothermic CSTR holding an operating point (stiff, nonlinear).",
            sources=["Uppal, Ray, Poore, CSTR dynamics, Chem. Eng. Sci. 1974.",
                     "Seborg et al., Process Dynamics and Control; do-mpc CSTR example - https://www.do-mpc.com"])
        # operating point (steady state at u=0)
        def ss(z):
            x1, x2 = z
            r = self.Da * (1 - x1) * np.exp(x2)
            return [-x1 + r, -x2 + self.B * r - self.beta * x2]
        self.xs = np.array(fsolve(ss, [0.5, 3.0]))
        self.lbu = np.array([-2.0]); self.ubu = np.array([2.0])
        self.lbx = np.array([0.0, 0.0]); self.ubx = np.array([1.0, 6.0])
        self.Q = np.diag([1.0, 0.5]); self.R = np.diag([0.05])
        self.Qf = np.diag([5.0, 2.5])

    def _rhs(self, x, u, Da):
        r = Da * (1 - x[0]) * ca.exp(x[1])
        return ca.vertcat(-x[0] + r,
                          -x[1] + self.B * r - self.beta * (x[1] - u[0]))

    def dynamics_ct(self, x, u):
        return self._rhs(x, u, self.Da)

    def stage_cost(self, x, u, xref, uref):
        dx, du = x - xref, u - uref
        return ca.mtimes([dx.T, ca.DM(self.Q), dx]) + ca.mtimes([du.T, ca.DM(self.R), du])

    def terminal_cost(self, x, xref):
        dx = x - xref
        return ca.mtimes([dx.T, ca.DM(self.Qf), dx])

    def reference_traj(self, k):
        return self.xs.copy(), np.zeros(1)

    def x0_nominal(self):
        return self.xs + np.array([0.15, -0.5])

    def x0_samples(self, n, rng):
        if n == 1:
            return [self.x0_nominal()]
        return [np.clip(self.xs + rng.uniform(-1, 1, 2) * np.array([0.2, 0.6]),
                        [0.0, 0.0], [1.0, 6.0]) for _ in range(n)]

    def plant_step(self, x, u, rng=None):
        from fastbench.core.dynamics import rk4
        if not hasattr(self, "_fp"):
            xx = ca.SX.sym("x", 2); uu = ca.SX.sym("u", 1)
            self._fp = ca.Function("fp", [xx, uu],
                                   [rk4(lambda a, b: self._rhs(a, b, self.Da_p),
                                        xx, uu, self.meta.dt)])
        xn = np.array(self._fp(x, u)).flatten()
        if rng is not None:
            xn += rng.normal(0, [2e-3, 5e-3])
        return xn

    def episode_success(self, X, U):
        # tolerance covers the steady offset from plant mismatch (no integrator)
        return bool(np.linalg.norm(X[-1] - self.xs) < 0.15)


@register_problem("cstr")
def _factory():
    return CSTR()
