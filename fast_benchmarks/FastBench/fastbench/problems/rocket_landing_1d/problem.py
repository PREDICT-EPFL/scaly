"""1-D rocket soft landing - affine-LTI QP.

Vertical descent under gravity with a non-negative thrust (cannot pull down),
a soft-landing target at the ground, and an altitude floor. The constant
gravity term makes the dynamics affine; it is carried through the discrete
model and the QP via the ``c`` term.

State x = [altitude, velocity], input u = [thrust acceleration].
"""
from __future__ import annotations

import casadi as ca
import numpy as np
import scipy.linalg as sla

from fastbench.core.dynamics import c2d_affine
from fastbench.core.problem import Problem, ProblemMeta
from fastbench.core.registry import register_problem

G = 9.81


class RocketLanding1D(Problem):
    def __init__(self):
        self.Ac = np.array([[0.0, 1.0], [0.0, 0.0]])
        self.Bc = np.array([[0.0], [1.0]])
        dt = 0.1
        self.Ad, self.Bd, self.wd = c2d_affine(self.Ac, self.Bc, np.array([0.0, -G]), dt)
        # plant: 3% gravity model error + noise
        _, _, self.wdp = c2d_affine(self.Ac, self.Bc, np.array([0.0, -G * 1.03]), dt)
        self.Adp, self.Bdp = self.Ad, self.Bd
        self.meta = ProblemMeta(
            name="rocket_landing_1d", nx=2, nu=1, dt=dt, N=30, n_sim=60,
            is_lti=True, convex=True, nonlinear=False, problem_class="landing",
            description="Vertical rocket soft landing, thrust>=0, altitude>=0 (affine QP).",
            sources=["Acikmese & Ploen, Convex programming approach to powered descent, JGCD 2007.",
                     "Malyuta et al., Convex Optimization for Trajectory Generation, IEEE CSM 2022 - https://arxiv.org/abs/2106.09125"])
        self.lbu = np.array([0.0]); self.ubu = np.array([2.0 * G])
        self.lbx = np.array([0.0, -20.0]); self.ubx = np.array([200.0, 5.0])
        self.Q = np.diag([5.0, 5.0]); self.R = np.diag([0.1])
        self.Qf = sla.solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)

    def dynamics_ct(self, x, u):
        return ca.vertcat(x[1], u[0] - G)

    def discrete_dynamics(self, x, u):
        return ca.mtimes(ca.DM(self.Ad), x) + ca.mtimes(ca.DM(self.Bd), u) + ca.DM(self.wd)

    def lti_matrices(self):
        return {"A": self.Ad, "B": self.Bd, "Q": self.Q, "R": self.R,
                "Qf": self.Qf, "c": self.wd}

    def stage_cost(self, x, u, xref, uref):
        dx, du = x - xref, u - uref
        return ca.mtimes([dx.T, ca.DM(self.Q), dx]) + ca.mtimes([du.T, ca.DM(self.R), du])

    def terminal_cost(self, x, xref):
        dx = x - xref
        return ca.mtimes([dx.T, ca.DM(self.Qf), dx])

    # glide-slope: descend from H0 to the ground over T_land, then hold
    H0 = 20.0
    T_land = 4.0

    def reference_traj(self, k):
        t = k * self.meta.dt
        if t < self.T_land:
            h = self.H0 * (1.0 - t / self.T_land)
            v = -self.H0 / self.T_land
        else:
            h, v = 0.0, 0.0
        return np.array([h, v]), np.array([G])          # hover-thrust feed-forward

    def x0_nominal(self):
        return np.array([self.H0, -self.H0 / self.T_land])

    def x0_samples(self, n, rng):
        if n == 1:
            return [self.x0_nominal()]
        return [np.array([rng.uniform(15, 25), rng.uniform(-7, -3)]) for _ in range(n)]

    def plant_step(self, x, u, rng=None):
        xn = self.Adp @ np.asarray(x, float) + (self.Bdp @ np.atleast_1d(u)).flatten() + self.wdp
        if rng is not None:
            xn += rng.normal(0, [1e-2, 1e-2])
        return np.maximum(xn, [0.0, -1e9])              # ground floor

    def episode_success(self, X, U):
        return bool(abs(X[-1][0]) < 1.0 and abs(X[-1][1]) < 1.0)


@register_problem("rocket_landing_1d")
def _factory():
    return RocketLanding1D()
