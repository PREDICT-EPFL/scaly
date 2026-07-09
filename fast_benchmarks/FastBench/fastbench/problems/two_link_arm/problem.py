"""Two-link planar robot arm - fully-actuated nonlinear setpoint tracking.

Rigid-body manipulator dynamics  M(q) q̈ + C(q,q̇) q̇ + g(q) = tau, regulated to
a joint-space setpoint with gravity feed-forward. Fully actuated and stable to
regulate, so it isolates nonlinear-model handling (dense coupling, trig terms)
from the difficulty of underactuation.

State x = [q1, q2, dq1, dq2], input u = [tau1, tau2].
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from fastbench.core.problem import Problem, ProblemMeta
from fastbench.core.registry import register_problem

G = 9.81


class TwoLinkArm(Problem):
    def __init__(self):
        self.m1 = self.m2 = 1.0
        self.l1 = self.l2 = 1.0
        self.lc1 = self.lc2 = 0.5
        self.I1 = self.I2 = 0.083
        self.m2_p = 1.1                      # plant payload mismatch
        self.meta = ProblemMeta(
            name="two_link_arm", nx=4, nu=2, dt=0.05, N=25, n_sim=80,
            is_lti=False, nonlinear=True, problem_class="tracking",
            description="Two-link manipulator joint-space setpoint tracking.",
            sources=["Spong, Hutchinson, Vidyasagar, Robot Modeling and Control, 2006.",
                     "Crocoddyl manipulator examples - https://github.com/loco-3d/crocoddyl"])
        self.lbu = np.array([-20.0, -20.0]); self.ubu = np.array([20.0, 20.0])
        self.lbx = np.array([-1e6, -1e6, -10.0, -10.0])
        self.ubx = np.array([1e6, 1e6, 10.0, 10.0])
        self.Q = np.diag([10.0, 10.0, 0.5, 0.5]); self.R = np.diag([0.05, 0.05])
        self.Qf = np.diag([40.0, 40.0, 2.0, 2.0])
        self.qd = np.array([np.pi / 2, -0.5])      # target joint angles
        self.uref = np.array(self._gravity(self.qd)).flatten()

    def _gravity(self, q):
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1) * G * np.cos(q[0]) \
            + self.m2 * self.lc2 * G * np.cos(q[0] + q[1])
        g2 = self.m2 * self.lc2 * G * np.cos(q[0] + q[1])
        return np.array([g1, g2])

    def _rhs(self, x, u, m2):
        q1, q2, dq1, dq2 = x[0], x[1], x[2], x[3]
        a1 = self.I1 + self.I2 + self.m1 * self.lc1 ** 2 \
            + m2 * (self.l1 ** 2 + self.lc2 ** 2)
        a2 = m2 * self.l1 * self.lc2
        a3 = self.I2 + m2 * self.lc2 ** 2
        M = ca.vertcat(ca.horzcat(a1 + 2 * a2 * ca.cos(q2), a3 + a2 * ca.cos(q2)),
                       ca.horzcat(a3 + a2 * ca.cos(q2), a3))
        C = ca.vertcat(-a2 * ca.sin(q2) * dq2 * (2 * dq1 + dq2),
                       a2 * ca.sin(q2) * dq1 ** 2)
        g = ca.vertcat(
            (self.m1 * self.lc1 + m2 * self.l1) * G * ca.cos(q1)
            + m2 * self.lc2 * G * ca.cos(q1 + q2),
            m2 * self.lc2 * G * ca.cos(q1 + q2))
        qdd = ca.solve(M, ca.vertcat(u[0], u[1]) - C - g)
        return ca.vertcat(dq1, dq2, qdd[0], qdd[1])

    def dynamics_ct(self, x, u):
        return self._rhs(x, u, self.m2)

    def stage_cost(self, x, u, xref, uref):
        dx, du = x - xref, u - uref
        return ca.mtimes([dx.T, ca.DM(self.Q), dx]) + ca.mtimes([du.T, ca.DM(self.R), du])

    def terminal_cost(self, x, xref):
        dx = x - xref
        return ca.mtimes([dx.T, ca.DM(self.Qf), dx])

    def reference_traj(self, k):
        return np.array([self.qd[0], self.qd[1], 0.0, 0.0]), self.uref.copy()

    def x0_nominal(self):
        return np.array([0.0, 0.0, 0.0, 0.0])

    def x0_samples(self, n, rng):
        if n == 1:
            return [self.x0_nominal()]
        return [np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0.0, 0.0])
                for _ in range(n)]

    def plant_step(self, x, u, rng=None):
        from fastbench.core.dynamics import rk4
        if not hasattr(self, "_fp"):
            xx = ca.SX.sym("x", 4); uu = ca.SX.sym("u", 2)
            self._fp = ca.Function("fp", [xx, uu],
                                   [rk4(lambda a, b: self._rhs(a, b, self.m2_p),
                                        xx, uu, self.meta.dt)])
        xn = np.array(self._fp(x, u)).flatten()
        if rng is not None:
            xn += rng.normal(0, 1e-3, size=4)
        return xn

    def episode_success(self, X, U):
        return bool(np.linalg.norm(X[-1][:2] - self.qd) < 0.05)


@register_problem("two_link_arm")
def _factory():
    return TwoLinkArm()
