"""PIQP adapter for linear-MPC (LTI) problems.

PIQP solves ``min 0.5 z'Pz + c'z  s.t. Az = b,  Gz <= h,  x_lb <= z <= x_ub``.
We use only the equality (dynamics) block and box bounds; ``c`` and ``b`` are
updated per step.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from fastbench.core.registry import register_solver
from fastbench.solvers.base import BuildInfo, SolveStats, SolverAdapter
from fastbench.solvers.qp_mpc import MpcQp


class PiqpAdapter(SolverAdapter):
    name = "piqp"
    requires_lti = True

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self._s = None

    def available(self) -> bool:
        try:
            import piqp  # noqa: F401
            return True
        except Exception:
            return False

    def build(self, problem) -> BuildInfo:
        import piqp
        t0 = time.perf_counter()
        self.problem = problem
        self.qp = MpcQp(problem)
        nz = self.qp.nz
        x0 = np.asarray(problem.x0_nominal(), float)
        xr, ur = self.qp.horizon_refs(0)
        c = self.qp.q(xr, ur)
        b = self.qp.beq(x0)
        P = self.qp.P.tocsc()
        A = self.qp.Aeq.tocsc()
        self._s = piqp.SparseSolver()
        self._s.settings.verbose = False
        self._s.settings.eps_abs = self.eps
        self._s.settings.eps_rel = self.eps
        # equality = dynamics; box bounds (x_l/x_u) carry input/state limits
        self._s.setup(P, c, A=A, b=b, x_l=self.qp.lb_z, x_u=self.qp.ub_z)
        self._piqp = piqp
        return BuildInfo(build_time_s=time.perf_counter() - t0,
                         notes=f"sparse MPC QP, nz={nz}; PIQP")

    def solve(self, x, k) -> SolveStats:
        xr, ur = self.qp.horizon_refs(k)
        c = self.qp.q(xr, ur)
        b = self.qp.beq(np.asarray(x, float))
        self._s.update(c=c, b=b)
        t0 = time.perf_counter()
        status = self._s.solve()
        dt = time.perf_counter() - t0
        res = self._s.result
        ok = (status == self._piqp.PIQP_SOLVED) if hasattr(self._piqp, "PIQP_SOLVED") \
            else (str(status).lower().endswith("solved"))
        z = np.asarray(res.x, float)
        nx, nu = self.qp.nx, self.qp.nu
        u0 = z[self.qp.iu(0)] if ok and np.all(np.isfinite(z)) else None
        cviol = float(np.max(np.abs(self.qp.Aeq @ z - b), initial=0.0)) if u0 is not None else float("nan")
        return SolveStats(
            u0=u0, success=bool(ok), status=str(status), solve_time_s=dt,
            iterations=int(getattr(res.info, "iter", -1)),
            cost=float(getattr(res.info, "primal_obj", float("nan"))),
            kkt_residual=float(getattr(res.info, "dual_res", float("nan"))),
            constraint_violation=cviol)

    def reset(self):
        pass


@register_solver("piqp")
def _factory():
    return PiqpAdapter()
