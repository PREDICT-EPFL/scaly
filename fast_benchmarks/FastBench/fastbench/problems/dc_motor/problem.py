"""Armature-controlled DC motor - LTI position tracking (QP).

State x = [angle, speed, current], input u = [voltage]. Setpoint tracking of
the shaft angle under a voltage limit. Classic third-order LTI plant.
"""
from __future__ import annotations

import casadi as ca
import numpy as np
import scipy.linalg as sla

from fastbench.core.dynamics import c2d
from fastbench.core.problem import Problem, ProblemMeta
from fastbench.core.registry import register_problem


def _matrices(J, b, Kt, Kb, R, L):
    Ac = np.array([[0.0, 1.0, 0.0],
                   [0.0, -b / J, Kt / J],
                   [0.0, -Kb / L, -R / L]])
    Bc = np.array([[0.0], [0.0], [1.0 / L]])
    return Ac, Bc


class DCMotor(Problem):
    def __init__(self):
        J, b, Kt, Kb, R, L = 0.01, 0.1, 0.01, 0.01, 1.0, 0.5
        dt = 0.05
        self.Ac, self.Bc = _matrices(J, b, Kt, Kb, R, L)
        self.Acp, self.Bcp = _matrices(J, b, Kt, Kb, 1.1 * R, L)  # +10% resistance
        self.meta = ProblemMeta(
            name="dc_motor", nx=3, nu=1, dt=dt, N=25, n_sim=80,
            is_lti=True, convex=True, nonlinear=False, problem_class="tracking",
            description="Armature-controlled DC motor angle tracking (LTI/QP).",
            sources=["Franklin, Powell, Emami-Naeini, Feedback Control of Dynamic Systems.",
                     "MATLAB DC motor example: https://ctms.engin.umich.edu/CTMS/index.php?example=MotorPosition"])
        self.Ad, self.Bd = c2d(self.Ac, self.Bc, dt)
        self.Adp, self.Bdp = c2d(self.Acp, self.Bcp, dt)
        self.lbu = np.array([-12.0]); self.ubu = np.array([12.0])
        self.lbx = np.array([-1e6, -50.0, -10.0]); self.ubx = np.array([1e6, 50.0, 10.0])
        self.Q = np.diag([50.0, 1.0, 0.1]); self.R = np.diag([0.01])
        self.Qf = sla.solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        self.target = 1.0

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

    def reference_traj(self, k):
        return np.array([self.target, 0.0, 0.0]), np.zeros(1)

    def x0_nominal(self):
        return np.zeros(3)

    def x0_samples(self, n, rng):
        if n == 1:
            return [self.x0_nominal()]
        return [np.array([rng.uniform(-0.5, 0.5), rng.uniform(-2, 2), 0.0])
                for _ in range(n)]

    def plant_step(self, x, u, rng=None):
        xn = self.Adp @ np.asarray(x, float) + (self.Bdp @ np.atleast_1d(u)).flatten()
        if rng is not None:
            xn += rng.normal(0, [1e-4, 1e-3, 1e-3])
        return xn

    def episode_success(self, X, U):
        return bool(abs(X[-1][0] - self.target) < 0.05 and abs(X[-1][1]) < 0.2)


@register_problem("dc_motor")
def _factory():
    return DCMotor()
