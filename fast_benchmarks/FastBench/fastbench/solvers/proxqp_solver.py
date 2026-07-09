"""ProxQP adapter (proxsuite) for linear-MPC (LTI) problems.

Solves the sparse banded MPC QP with equality (dynamics) constraints and box
bounds expressed via ``C = I``; ``g`` and ``b`` are updated each step.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from fastbench.core.registry import register_solver
from fastbench.solvers.base import BuildInfo, SolveStats, SolverAdapter
from fastbench.solvers.qp_mpc import MpcQp


class ProxqpAdapter(SolverAdapter):
    name = "proxqp"
    requires_lti = True

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self._qp = None

    def available(self) -> bool:
        try:
            import proxsuite  # noqa: F401
            return True
        except Exception:
            return False

    def build(self, problem) -> BuildInfo:
        import proxsuite
        from proxsuite import proxqp
        t0 = time.perf_counter()
        self.problem = problem
        self.qp = MpcQp(problem)
        nz, neq = self.qp.nz, self.qp.neq
        self._C = sp.eye(nz, format="csc")
        x0 = np.asarray(problem.x0_nominal(), float)
        xr, ur = self.qp.horizon_refs(0)
        g = self.qp.q(xr, ur)
        b = self.qp.beq(x0)
        self._qp = proxqp.sparse.QP(nz, neq, nz)
        self._qp.settings.eps_abs = self.eps
        self._qp.settings.verbose = False
        self._qp.init(self.qp.P.tocsc(), g, self.qp.Aeq.tocsc(), b,
                      self._C, self.qp.lb_z, self.qp.ub_z)
        self._solved_enum = proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED
        return BuildInfo(build_time_s=time.perf_counter() - t0,
                         notes=f"sparse MPC QP, nz={nz}, neq={neq}; ProxQP")

    def solve(self, x, k) -> SolveStats:
        xr, ur = self.qp.horizon_refs(k)
        g = self.qp.q(xr, ur)
        b = self.qp.beq(np.asarray(x, float))
        self._qp.update(g=g, b=b)
        t0 = time.perf_counter()
        self._qp.solve()
        dt = time.perf_counter() - t0
        res = self._qp.results
        ok = (res.info.status == self._solved_enum)
        z = np.asarray(res.x, float)
        u0 = z[self.qp.iu(0)] if ok and np.all(np.isfinite(z)) else None
        cviol = float(np.max(np.abs(self.qp.Aeq @ z - b), initial=0.0)) if u0 is not None else float("nan")
        return SolveStats(
            u0=u0, success=bool(ok), status=str(res.info.status),
            solve_time_s=dt, iterations=int(res.info.iter),
            cost=float(res.info.objValue),
            kkt_residual=float(getattr(res.info, "pri_res", float("nan"))),
            constraint_violation=cviol)

    def reset(self):
        pass


@register_solver("proxqp")
def _factory():
    return ProxqpAdapter()
