"""Obstacle-avoidance lane keeping - nonlinear OCP with a path constraint.

A kinematic bicycle keeps a straight lane at a target speed while avoiding a
circular obstacle sitting on the lane. The obstacle is a genuine nonlinear
inequality (not a box), so this problem exercises path-constraint handling and
is only solved by backends that support general inequalities (IPOPT here).

State x = [X, Y, psi, v], input u = [a, delta].
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from fastbench.core.problem import Problem, ProblemMeta
from fastbench.core.registry import register_problem


class VehicleObstacle(Problem):
    has_path_constraints = True

    def __init__(self):
        self.L = 2.7
        self.L_p = 2.9
        self.v0 = 5.0
        self.obs = np.array([15.0, 0.0])   # obstacle centre
        self.r = 1.5                        # obstacle radius
        self.margin = 0.5                   # safety margin
        self.meta = ProblemMeta(
            name="vehicle_obstacle", nx=4, nu=2, dt=0.05, N=40, n_sim=120,
            is_lti=False, nonlinear=True, problem_class="tracking+avoidance",
            description="Lane keeping with circular obstacle avoidance (nonlinear path constraint).",
            sources=["acados race_cars / obstacle examples - https://github.com/acados/acados",
                     "CommonRoad - https://commonroad.in.tum.de"])
        self.lbu = np.array([-3.0, -0.5]); self.ubu = np.array([3.0, 0.5])
        self.lbx = np.array([-1e6, -6.0, -1e6, 0.0]); self.ubx = np.array([1e6, 6.0, 1e6, 8.0])
        self.Q = np.diag([0.0, 0.5, 0.5, 3.0]); self.R = np.diag([0.1, 0.5])
        self.Qf = np.diag([0.0, 1.0, 1.0, 6.0])

    def dynamics_ct(self, x, u):
        psi, v = x[2], x[3]
        return ca.vertcat(v * ca.cos(psi), v * ca.sin(psi),
                          v / self.L * ca.tan(u[1]), u[0])

    def path_constraints(self, x, u):
        d2 = (x[0] - self.obs[0]) ** 2 + (x[1] - self.obs[1]) ** 2
        lh = np.array([(self.r + self.margin) ** 2])
        uh = np.array([1e9])
        return d2, lh, uh

    def stage_cost(self, x, u, xref, uref):
        dx, du = x - xref, u - uref
        return ca.mtimes([dx.T, ca.DM(self.Q), dx]) + ca.mtimes([du.T, ca.DM(self.R), du])

    def terminal_cost(self, x, xref):
        dx = x - xref
        return ca.mtimes([dx.T, ca.DM(self.Qf), dx])

    def reference_traj(self, k):
        return np.array([0.0, 0.0, 0.0, self.v0]), np.zeros(2)

    def x0_nominal(self):
        return np.array([0.0, 0.0, 0.0, self.v0])

    def x0_samples(self, n, rng):
        if n == 1:
            return [self.x0_nominal()]
        return [np.array([0.0, rng.uniform(-0.5, 0.5), 0.0, self.v0 + rng.uniform(-0.5, 0.5)])
                for _ in range(n)]

    def plant_step(self, x, u, rng=None):
        from fastbench.core.dynamics import rk4
        if not hasattr(self, "_fp"):
            xx = ca.SX.sym("x", 4); uu = ca.SX.sym("u", 2)

            def fp(a, b):
                return ca.vertcat(a[3] * ca.cos(a[2]), a[3] * ca.sin(a[2]),
                                  a[3] / self.L_p * ca.tan(b[1]), b[0])
            self._fp = ca.Function("fp", [xx, uu], [rk4(fp, xx, uu, self.meta.dt)])
        xn = np.array(self._fp(x, u)).flatten()
        if rng is not None:
            xn += rng.normal(0, [0.01, 0.01, 0.002, 0.01])
        return xn

    def episode_success(self, X, U):
        d = np.sqrt((X[:, 0] - self.obs[0]) ** 2 + (X[:, 1] - self.obs[1]) ** 2)
        no_crash = bool(np.all(d >= self.r))                  # never hit obstacle
        passed = bool(X[-1, 0] > self.obs[0] + 3.0)            # got past it
        back_on_lane = bool(abs(X[-1, 1]) < 1.0)
        return no_crash and passed and back_on_lane


@register_problem("vehicle_obstacle")
def _factory():
    return VehicleObstacle()
