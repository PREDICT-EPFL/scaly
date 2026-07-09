"""OSQP adapter for linear-MPC (LTI) problems.

Solves the sparse banded QP from :mod:`fastbench.solvers.qp_mpc` in single
constraint form ``l <= [Aeq; I] z <= u``; only ``q`` and the equality bounds
are updated per step, and OSQP warm-starts internally.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from fastbench.core.registry import register_solver
from fastbench.solvers.base import BuildInfo, SolveStats, SolverAdapter
from fastbench.solvers.qp_mpc import MpcQp


class OsqpAdapter(SolverAdapter):
    name = "osqp"
    requires_lti = True

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self._m = None

    def available(self) -> bool:
        try:
            import osqp  # noqa: F401
            return True
        except Exception:
            return False

    def build(self, problem) -> BuildInfo:
        import osqp
        t0 = time.perf_counter()
        self.problem = problem
        self.qp = MpcQp(problem)
        nz, neq = self.qp.nz, self.qp.neq
        self._A = sp.vstack([self.qp.Aeq, sp.eye(nz, format="csc")], format="csc")
        x0 = problem.x0_nominal()
        xr, ur = self.qp.horizon_refs(0)
        q = self.qp.q(xr, ur)
        beq = self.qp.beq(np.asarray(x0, float))
        l = np.concatenate([beq, self.qp.lb_z])
        u = np.concatenate([beq, self.qp.ub_z])
        self._m = osqp.OSQP()
        self._m.setup(P=self.qp.P.tocsc(), q=q, A=self._A, l=l, u=u,
                      verbose=False, eps_abs=self.eps, eps_rel=self.eps,
                      warm_starting=True)
        self._neq = neq
        return BuildInfo(build_time_s=time.perf_counter() - t0,
                         notes=f"sparse MPC QP, nz={nz}, neq={neq}; OSQP")

    def solve(self, x, k) -> SolveStats:
        xr, ur = self.qp.horizon_refs(k)
        q = self.qp.q(xr, ur)
        beq = self.qp.beq(np.asarray(x, float))
        l = np.concatenate([beq, self.qp.lb_z])
        u = np.concatenate([beq, self.qp.ub_z])
        try:
            self._m.update(q=q, l=l, u=u)
        except Exception:
            self._m.update(q=q)
            self._m.update(l=l, u=u)
        t0 = time.perf_counter()
        res = self._m.solve()
        dt = time.perf_counter() - t0
        info = res.info
        status = str(getattr(info, "status", ""))
        success = ("solved" in status.lower())
        z = np.asarray(res.x, float)
        nx, nu, N = self.qp.nx, self.qp.nu, self.qp.N
        u0 = z[self.qp.iu(0)] if success and np.all(np.isfinite(z)) else None
        cviol = float("nan")
        if u0 is not None:
            cviol = float(np.max(np.abs(self.qp.Aeq @ z - beq), initial=0.0))
        return SolveStats(
            u0=u0, success=success, status=status, solve_time_s=dt,
            iterations=int(getattr(info, "iter", -1)),
            cost=float(getattr(info, "obj_val", float("nan"))),
            kkt_residual=float(getattr(info, "dua_res", float("nan"))),
            constraint_violation=cviol)

    def reset(self):
        pass


@register_solver("osqp")
def _factory():
    return OsqpAdapter()
