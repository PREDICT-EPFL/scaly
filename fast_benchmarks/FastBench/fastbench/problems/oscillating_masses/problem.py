"""Oscillating masses - a scalable LTI/QP benchmark.

A row of ``nm`` masses connected to each other and to two end walls by springs
and dampers; forces actuate the first and last masses. Lightly damped, so the
open loop oscillates. The state dimension scales as ``2*nm`` (default nm=6 ->
nx=12), giving a sizeable banded QP for stress-testing QP solvers.

State x = [positions(nm), velocities(nm)], input u = [F_first, F_last].
"""
from __future__ import annotations

import casadi as ca
import numpy as np
import scipy.linalg as sla

from fastbench.core.dynamics import c2d
from fastbench.core.problem import Problem, ProblemMeta
from fastbench.core.registry import register_problem


def _matrices(nm, k, m, d):
    K = np.diag(2 * k * np.ones(nm)) - np.diag(k * np.ones(nm - 1), 1) \
        - np.diag(k * np.ones(nm - 1), -1)
    A = np.block([[np.zeros((nm, nm)), np.eye(nm)],
                  [-K / m, -(d / m) * np.eye(nm)]])
    B = np.zeros((2 * nm, 2))
    B[nm, 0] = 1.0 / m          # force on first mass
    B[2 * nm - 1, 1] = 1.0 / m  # force on last mass
    return A, B


class OscillatingMasses(Problem):
    def __init__(self, nm: int = 6):
        self.nm = nm
        self.k, self.m, self.d = 1.0, 1.0, 0.1
        dt = 0.2
        nx, nu = 2 * nm, 2
        self.Ac, self.Bc = _matrices(nm, self.k, self.m, self.d)
        self.Acp, self.Bcp = _matrices(nm, 1.05 * self.k, self.m, self.d)
        self.meta = ProblemMeta(
            name="oscillating_masses", nx=nx, nu=nu, dt=dt, N=30, n_sim=80,
            is_lti=True, convex=True, nonlinear=False, problem_class="regulation",
            description=f"{nm}-mass spring-damper network, regulate to rest (scalable QP).",
            sources=["Stellato et al., OSQP, Math. Prog. Comp. 2020 - https://osqp.org",
                     "Kvasnica et al., Multi-Parametric Toolbox 3 - https://www.mpt3.org"])
        self.Ad, self.Bd = c2d(self.Ac, self.Bc, dt)
        self.Adp, self.Bdp = c2d(self.Acp, self.Bcp, dt)
        self.lbu = np.array([-0.5, -0.5]); self.ubu = np.array([0.5, 0.5])
        self.lbx = np.concatenate([-4 * np.ones(nm), -1e6 * np.ones(nm)])
        self.ubx = np.concatenate([4 * np.ones(nm), 1e6 * np.ones(nm)])
        self.Q = np.diag(np.concatenate([np.ones(nm), 0.1 * np.ones(nm)]))
        self.R = 0.1 * np.eye(2)
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
        e = np.zeros(self.nm); e[0] = 2.0; e[-1] = -1.5
        return np.concatenate([e, np.zeros(self.nm)])

    def x0_samples(self, n, rng):
        if n == 1:
            return [self.x0_nominal()]
        return [np.concatenate([rng.uniform(-2, 2, self.nm),
                                rng.uniform(-0.5, 0.5, self.nm)]) for _ in range(n)]

    def plant_step(self, x, u, rng=None):
        xn = self.Adp @ np.asarray(x, float) + (self.Bdp @ np.atleast_1d(u)).flatten()
        if rng is not None:
            xn += rng.normal(0, 1e-3, size=self.meta.nx)
        return xn

    def episode_success(self, X, U):
        return bool(np.linalg.norm(X[-1]) < 0.3)


@register_problem("oscillating_masses")
def _factory():
    return OscillatingMasses()
