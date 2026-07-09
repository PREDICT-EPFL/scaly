"""Planar (2-D) quadrotor stabilization.

A quadrotor in the vertical plane with two rotor thrusts; nonlinear through the
attitude. Stabilize to hover at the origin from an offset, with a hover-thrust
feed-forward reference and non-negative thrust limits.

State x = [px, pz, phi, vx, vz, omega], input u = [T_left, T_right].
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from fastbench.core.problem import Problem, ProblemMeta

G = 9.81


class PlanarQuadrotor(Problem):
    def __init__(self):
        self.m, self.l, self.I = 0.5, 0.25, 0.0125
        self.m_p = 0.55                      # plant 10% heavier
        self.hover = self.m * G / 2.0
        self.meta = ProblemMeta(
            name="planar_quadrotor", nx=6, nu=2, dt=0.05, N=30, n_sim=100,
            is_lti=False, nonlinear=True, problem_class="stabilization",
            description="Planar quadrotor stabilized to hover at the origin.",
            sources=["Tedrake, Underactuated Robotics - http://underactuated.mit.edu",
                     "Sabatino, Quadrotor control thesis, KTH 2015."])
        self.lbu = np.array([0.0, 0.0]); self.ubu = np.array([2 * self.m * G, 2 * self.m * G])
        self.lbx = np.array([-1e6, -1e6, -1.2, -1e6, -1e6, -1e6])
        self.ubx = np.array([1e6, 1e6, 1.2, 1e6, 1e6, 1e6])
        self.Q = np.diag([8.0, 8.0, 4.0, 1.0, 1.0, 0.5])
        self.R = np.diag([0.05, 0.05])
        self.Qf = np.diag([40.0, 40.0, 20.0, 5.0, 5.0, 2.0])

    def _rhs(self, x, u, m):
        phi, vx, vz, om = x[2], x[3], x[4], x[5]
        tl, tr = u[0], u[1]
        return ca.vertcat(vx, vz, om,
                          -(tl + tr) * ca.sin(phi) / m,
                          (tl + tr) * ca.cos(phi) / m - G,
                          (tr - tl) * self.l / self.I)

    def dynamics_ct(self, x, u):
        return self._rhs(x, u, self.m)

    def stage_cost(self, x, u, xref, uref):
        dx, du = x - xref, u - uref
        return ca.mtimes([dx.T, ca.DM(self.Q), dx]) + ca.mtimes([du.T, ca.DM(self.R), du])

    def terminal_cost(self, x, xref):
        dx = x - xref
        return ca.mtimes([dx.T, ca.DM(self.Qf), dx])

    def reference_traj(self, k):
        return np.zeros(6), np.array([self.hover, self.hover])

    def x0_nominal(self):
        return np.array([-1.0, -0.5, 0.2, 0.0, 0.0, 0.0])

    def x0_samples(self, n, rng):
        if n == 1:
            return [self.x0_nominal()]
        return [np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1, 1),
                          rng.uniform(-0.3, 0.3), 0.0, 0.0, 0.0]) for _ in range(n)]

    def plant_step(self, x, u, rng=None):
        from fastbench.core.dynamics import rk4
        if not hasattr(self, "_fp"):
            xx = ca.SX.sym("x", 6); uu = ca.SX.sym("u", 2)
            self._fp = ca.Function("fp", [xx, uu],
                                   [rk4(lambda a, b: self._rhs(a, b, self.m_p),
                                        xx, uu, self.meta.dt)])
        xn = np.array(self._fp(x, u)).flatten()
        if rng is not None:
            xn += rng.normal(0, 1e-3, size=6) * np.array([1, 1, 1, 2, 2, 2])
        return xn

    def episode_success(self, X, U):
        return bool(np.linalg.norm(X[-1][:3]) < 0.1)


from fastbench.core.registry import register_problem  # noqa: E402


@register_problem("planar_quadrotor")
def _factory():
    return PlanarQuadrotor()
