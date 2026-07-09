"""Kinematic bicycle trajectory tracking.

A kinematic bicycle tracks a time-parameterized sine "lane-change" reference at
a target speed - a compact stand-in for autonomous-driving / racing tracking
MPC.  Nonlinear in heading and speed.

State  x = [X, Y, psi, v]      Input u = [a, delta]
Goal   follow the moving reference (X_ref(t), Y_ref(t), psi_ref(t), v_ref).

The plant uses a longer wheelbase than the model (mismatch) plus measurement
noise.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from fastbench.core.problem import Problem, ProblemMeta


class KinematicVehicle(Problem):
    def __init__(self):
        self.L = 2.7            # model wheelbase [m]
        self.L_p = 2.97         # plant wheelbase (+10%)
        self.v0 = 5.0           # target speed [m/s]
        self.amp = 2.0          # lane amplitude [m]
        self.wave = 2 * np.pi / 30.0   # spatial frequency [1/m]
        self.meta = ProblemMeta(
            name="kinematic_vehicle", nx=4, nu=2, dt=0.05, N=25, n_sim=120,
            is_lti=False, nonlinear=True, problem_class="tracking",
            description="Kinematic bicycle tracking a sine lane-change reference.",
            sources=[
                "Rajamani, 'Vehicle Dynamics and Control', Springer 2012 (kinematic bicycle).",
                "acados race_cars example: https://github.com/acados/acados/tree/main/examples/acados_python/race_cars",
                "CommonRoad motion-planning benchmarks: https://commonroad.in.tum.de",
            ])
        self.lbu = np.array([-3.0, -0.5]); self.ubu = np.array([3.0, 0.5])
        self.lbx = np.array([-1e6, -1e6, -1e6, 0.0])
        self.ubx = np.array([1e6, 1e6, 1e6, 25.0])
        self.Q = np.diag([2.0, 8.0, 4.0, 1.0])
        self.R = np.diag([0.1, 1.0])
        self.Qf = np.diag([4.0, 16.0, 8.0, 2.0])

    def dynamics_ct(self, x, u):
        psi, v = x[2], x[3]
        a, delta = u[0], u[1]
        return ca.vertcat(v * ca.cos(psi), v * ca.sin(psi),
                          v / self.L * ca.tan(delta), a)

    def stage_cost(self, x, u, xref, uref):
        dx, du = x - xref, u - uref
        return ca.mtimes([dx.T, ca.DM(self.Q), dx]) + ca.mtimes([du.T, ca.DM(self.R), du])

    def terminal_cost(self, x, xref):
        dx = x - xref
        return ca.mtimes([dx.T, ca.DM(self.Qf), dx])

    def reference_traj(self, k):
        t = k * self.meta.dt
        X = self.v0 * t
        Y = self.amp * np.sin(self.wave * X)
        slope = self.amp * self.wave * np.cos(self.wave * X)
        psi = np.arctan(slope)
        return np.array([X, Y, psi, self.v0]), np.zeros(2)

    def x0_nominal(self):
        return np.array([0.0, 1.0, 0.0, self.v0])      # start 1 m off the lane

    def x0_samples(self, n, rng):
        if n == 1:
            return [self.x0_nominal()]
        return [np.array([rng.uniform(-1, 1), rng.uniform(-1.5, 1.5),
                          rng.uniform(-0.2, 0.2), self.v0 + rng.uniform(-1, 1)])
                for _ in range(n)]

    def plant_step(self, x, u, rng=None):
        from fastbench.core.dynamics import rk4
        if not hasattr(self, "_fp"):
            xx = ca.SX.sym("x", 4); uu = ca.SX.sym("u", 2)

            def fp(a, b):
                return ca.vertcat(a[3] * ca.cos(a[2]), a[3] * ca.sin(a[2]),
                                  a[3] / self.L_p * ca.tan(b[1]), b[0])
            xn = rk4(fp, xx, uu, self.meta.dt)
            self._fp = ca.Function("fp", [xx, uu], [xn])
        xn = np.array(self._fp(x, u)).flatten()
        if rng is not None:
            xn += rng.normal(0, [0.01, 0.01, 0.002, 0.01])
        return xn

    def episode_success(self, X, U):
        xref_f, _ = self.reference_traj(self.meta.n_sim)
        return bool(np.linalg.norm(X[-1][:2] - xref_f[:2]) < 1.5)


from fastbench.core.registry import register_problem  # noqa: E402


@register_problem("kinematic_vehicle")
def _factory():
    return KinematicVehicle()
