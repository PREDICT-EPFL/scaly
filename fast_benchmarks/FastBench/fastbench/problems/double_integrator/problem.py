"""Double integrator - the minimal LTI/QP regulation benchmark.

State x = [position, velocity], input u = [acceleration]. Position and input
bounds make the QP non-trivial (the unconstrained solution saturates). Ideal
as a fast unit test and a baseline for every QP backend.
"""
from __future__ import annotations

import casadi as ca
import numpy as np
import scipy.linalg as sla

from fastbench.core.dynamics import c2d
from fastbench.core.problem import Problem, ProblemMeta
from fastbench.core.registry import register_problem


class DoubleIntegrator(Problem):
    def __init__(self):
        self.Ac = np.array([[0.0, 1.0], [0.0, 0.0]])
        self.Bc = np.array([[0.0], [1.0]])
        dt = 0.1
        self.meta = ProblemMeta(
            name="double_integrator", nx=2, nu=1, dt=dt, N=20, n_sim=60,
            is_lti=True, convex=True, nonlinear=False, problem_class="regulation",
            description="Position/velocity double integrator with bounds (QP).",
            sources=["Boyd & Vandenberghe, Convex Optimization, 2004 (control examples).",
                     "OSQP examples: https://osqp.org/docs/examples/mpc.html"])
        self.Ad, self.Bd = c2d(self.Ac, self.Bc, dt)
        self.Adp, self.Bdp = c2d(self.Ac, 0.9 * self.Bc, dt)   # 10% actuator loss
        self.lbu = np.array([-1.0]); self.ubu = np.array([1.0])
        self.lbx = np.array([-5.0, -3.0]); self.ubx = np.array([5.0, 3.0])
        self.Q = np.diag([10.0, 1.0]); self.R = np.diag([0.5])
        self.Qf = sla.solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)

    def dynamics_ct(self, x, u):
        return ca.mtimes(ca.DM(self.Ac), x) + ca.mtimes(ca.DM(self.Bc), u)

    def discrete_dynamics(self, x, u):
        return ca.mtimes(ca.DM(self.Ad), x) + ca.mtimes(ca.DM(self.Bd), u)

    def lti_matrices(self):
        return {"A": self.Ad, "B": self.Bd, "Q": self.Q, "R": self.R, "Qf": self.Qf}

    def stage_cost(self, x, u, xref, uref):
        dx, du = x - xref, u - uref
        return ca.mtimes([dx.T, ca.DM(self.Q), dx]) + ca.mtimes([du.T, ca.DM(self.R), du])

    def terminal_cost(self, x, xref):
        dx = x - xref
        return ca.mtimes([dx.T, ca.DM(self.Qf), dx])

    def x0_nominal(self):
        return np.array([3.0, 0.0])

    def x0_samples(self, n, rng):
        if n == 1:
            return [self.x0_nominal()]
        return [np.array([rng.uniform(-2.5, 2.5), rng.uniform(-1.0, 1.0)])
                for _ in range(n)]

    def plant_step(self, x, u, rng=None):
        xn = self.Adp @ np.asarray(x, float) + (self.Bdp @ np.atleast_1d(u)).flatten()
        if rng is not None:
            xn += rng.normal(0, 1e-3, size=2)
        return xn

    def episode_success(self, X, U):
        return bool(np.linalg.norm(X[-1]) < 0.05)


@register_problem("double_integrator")
def _factory():
    return DoubleIntegrator()
