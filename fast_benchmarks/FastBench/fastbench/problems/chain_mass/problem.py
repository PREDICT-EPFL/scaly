"""Chain of masses (linear springs) - an LTI regulation benchmark.

A 1-D chain of ``nm`` point masses connected by linear springs (with light
damping); the left end is anchored to a wall and a horizontal force actuates
the right-most mass.  In deviation coordinates (displacement from the spring
equilibrium) the dynamics are linear and the cost quadratic, so the OCP is a
**convex QP** - the right target for OSQP and PIQP, while remaining a valid
NMPC instance for IPOPT/acados.

State  x = [e_1..e_nm, edot_1..edot_nm]  (deviations)   Input u = [F]
Goal   regulate a perturbed chain back to equilibrium subject to displacement
       and force limits.

This is the linear-spring relative of the nonlinear chain-mass problem used in
the acados benchmarks.
"""
from __future__ import annotations

import casadi as ca
import numpy as np
import scipy.linalg as sla

from fastbench.core.problem import Problem, ProblemMeta
from fastbench.core.registry import register_problem


def _chain_matrices(nm, k, m, c):
    K = np.zeros((nm, nm))
    for i in range(nm):
        K[i, i] += 2 * k if i < nm - 1 else k     # right end has one spring
        if i > 0:
            K[i, i - 1] -= k
        if i < nm - 1:
            K[i, i + 1] -= k
    A = np.block([[np.zeros((nm, nm)), np.eye(nm)],
                  [-K / m, -(c / m) * np.eye(nm)]])
    b = np.zeros((nm, 1)); b[-1, 0] = 1.0
    B = np.block([[np.zeros((nm, 1))], [b / m]])
    return A, B


class ChainMass(Problem):
    def __init__(self, nm: int = 4):
        self.nm = nm
        self.k, self.m, self.c = 10.0, 1.0, 0.4
        self.k_p = 1.05 * self.k                  # plant spring 5% stiffer
        nx, nu = 2 * nm, 1
        dt = 0.05
        self.meta = ProblemMeta(
            name="chain_mass", nx=nx, nu=nu, dt=dt, N=30, n_sim=100,
            is_lti=True, convex=True, nonlinear=False,
            problem_class="regulation",
            description=f"{nm}-mass linear spring chain, regulate to equilibrium (QP).",
            sources=[
                "acados chain-mass benchmark: https://github.com/acados/acados/tree/main/examples/acados_python/chain",
                "Wirsching, Bock, Diehl, 'Fast NMPC of a chain of masses', CCA 2006.",
            ])
        Ac, Bc = _chain_matrices(nm, self.k, self.m, self.c)
        Acp, Bcp = _chain_matrices(nm, self.k_p, self.m, self.c)
        self.Ac, self.Bc = Ac, Bc
        M = sla.expm(np.block([[Ac, Bc], [np.zeros((nu, nx + nu))]]) * dt)
        self.Ad, self.Bd = M[:nx, :nx], M[:nx, nx:]
        Mp = sla.expm(np.block([[Acp, Bcp], [np.zeros((nu, nx + nu))]]) * dt)
        self.Adp, self.Bdp = Mp[:nx, :nx], Mp[:nx, nx:]

        self.lbu = np.array([-10.0]); self.ubu = np.array([10.0])
        self.lbx = np.concatenate([-0.6 * np.ones(nm), -1e6 * np.ones(nm)])
        self.ubx = np.concatenate([0.6 * np.ones(nm), 1e6 * np.ones(nm)])
        self.Q = np.diag(np.concatenate([10.0 * np.ones(nm), 1.0 * np.ones(nm)]))
        self.R = np.diag([0.1])
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

    def reference_traj(self, k):
        return np.zeros(self.meta.nx), np.zeros(self.meta.nu)

    def x0_nominal(self):
        e = np.linspace(0.4, 0.1, self.nm)
        return np.concatenate([e, np.zeros(self.nm)])

    def x0_samples(self, n, rng):
        if n == 1:
            return [self.x0_nominal()]
        out = []
        for _ in range(n):
            e = rng.uniform(-0.35, 0.35, self.nm)
            out.append(np.concatenate([e, rng.uniform(-0.1, 0.1, self.nm)]))
        return out

    def plant_step(self, x, u, rng=None):
        xn = self.Adp @ np.asarray(x, float) + (self.Bdp @ np.atleast_1d(u)).flatten()
        if rng is not None:
            xn += rng.normal(0, 5e-4, size=self.meta.nx)
        return xn

    def episode_success(self, X, U):
        # tolerance accounts for the steady-state offset from plant mismatch
        # (regulator has no integral action); identical across all solvers.
        return bool(np.linalg.norm(X[-1]) < 0.15)


@register_problem("chain_mass")
def _factory():
    return ChainMass()
