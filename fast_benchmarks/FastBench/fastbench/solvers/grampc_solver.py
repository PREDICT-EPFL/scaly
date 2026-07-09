"""GRAMPC adapter (embedded augmented-Lagrangian / gradient NMPC).

GRAMPC is a plain-C real-time NMPC framework targeting (sub)millisecond
sampling.  Its Python binding ``pygrampc`` requires the problem to be supplied
as a ProblemDescription (dynamics, Jacobian-vector products, cost and their
gradients).  These can be generated from the problem's CasADi model, which is
what :meth:`build` does when ``pygrampc`` is installed.

If ``pygrampc`` is not available the adapter reports ``available() == False``
and is skipped by the runner with a clear reason, so the rest of the benchmark
still runs.  The CasADi-to-GRAMPC code generation is the one piece left as a
documented extension point (see ``_build_problem_description``).
"""
from __future__ import annotations

import time

import numpy as np

from fastbench.core.registry import register_solver
from fastbench.solvers.base import BuildInfo, SolveStats, SolverAdapter


class GrampcAdapter(SolverAdapter):
    name = "grampc"

    def __init__(self):
        self._grampc = None

    def available(self) -> bool:
        try:
            import pygrampc  # noqa: F401
            return True
        except Exception:
            return False

    def _build_problem_description(self, problem):
        """Map a CasADi problem onto a pygrampc ProblemDescription.

        Implements f, dfdx_vec, dfdu_vec, l, dldx, dldu, V, dVdx using CasADi
        Functions derived from ``problem``.  Subclass/extend for problems whose
        constraints need GRAMPC's g/h (equality/inequality) interface.
        """
        import casadi as ca
        import pygrampc

        nx, nu = problem.meta.nx, problem.meta.nu
        x = ca.SX.sym("x", nx); u = ca.SX.sym("u", nu)
        xr = ca.SX.sym("xr", nx); ur = ca.SX.sym("ur", nu)
        vlam = ca.SX.sym("lam", nx)
        f = problem.dynamics_ct(x, u)
        l = problem.stage_cost(x, u, xr, ur)
        V = problem.terminal_cost(x, xr)
        F = ca.Function("f", [x, u], [f])
        dfdx = ca.Function("dfdx", [x, u, vlam], [ca.jtimes(f, x, vlam, True)])
        dfdu = ca.Function("dfdu", [x, u, vlam], [ca.jtimes(f, u, vlam, True)])
        L = ca.Function("l", [x, u, xr, ur], [l])
        dldx = ca.Function("dldx", [x, u, xr, ur], [ca.gradient(l, x)])
        dldu = ca.Function("dldu", [x, u, xr, ur], [ca.gradient(l, u)])
        Vt = ca.Function("V", [x, xr], [V])
        dVdx = ca.Function("dVdx", [x, xr], [ca.gradient(V, x)])

        class _Desc(pygrampc.ProblemDescription):
            def __init__(self):
                super().__init__(nx, nu, 0, 0, 0)
                self.ref_x = np.zeros(nx); self.ref_u = np.zeros(nu)

            def ffct(self, out, t, x, u, p):
                out[:] = np.asarray(F(x, u)).flatten()

            def dfdx_vec(self, out, t, x, u, vec, p):
                out[:] = np.asarray(dfdx(x, u, vec)).flatten()

            def dfdu_vec(self, out, t, x, u, vec, p):
                out[:] = np.asarray(dfdu(x, u, vec)).flatten()

            def lfct(self, out, t, x, u, p, xdes, udes):
                out[0] = float(L(x, u, self.ref_x, self.ref_u))

            def dldx(self, out, t, x, u, p, xdes, udes):
                out[:] = np.asarray(dldx(x, u, self.ref_x, self.ref_u)).flatten()

            def dldu(self, out, t, x, u, p, xdes, udes):
                out[:] = np.asarray(dldu(x, u, self.ref_x, self.ref_u)).flatten()

            def Vfct(self, out, T, x, p, xdes):
                out[0] = float(Vt(x, self.ref_x))

            def dVdx(self, out, T, x, p, xdes):
                out[:] = np.asarray(dVdx(x, self.ref_x)).flatten()

        return _Desc()

    def build(self, problem) -> BuildInfo:
        import pygrampc
        t0 = time.perf_counter()
        self.problem = problem
        m = problem.meta
        self._desc = self._build_problem_description(problem)
        self._grampc = pygrampc.Grampc(self._desc)
        self._grampc.set_param("Nhor", m.N + 1)
        self._grampc.set_param("Thor", m.N * m.dt)
        self._grampc.set_param("dt", m.dt)
        lbu, ubu = problem.bounds()[2], problem.bounds()[3]
        self._grampc.set_param("umin", list(lbu))
        self._grampc.set_param("umax", list(ubu))
        return BuildInfo(build_time_s=time.perf_counter() - t0,
                         notes="GRAMPC (augmented Lagrangian + gradient)")

    def solve(self, x, k) -> SolveStats:
        g = self._grampc
        xr, ur = self.problem.reference_traj(k)
        self._desc.ref_x = np.asarray(xr, float)
        self._desc.ref_u = np.asarray(ur, float)
        g.set_param("x0", list(np.asarray(x, float)))
        t0 = time.perf_counter()
        g.run()
        dt = time.perf_counter() - t0
        sol = g.solution
        u0 = np.asarray(sol.unext, float).flatten()
        return SolveStats(
            u0=u0, success=True, status="grampc", solve_time_s=dt,
            iterations=int(getattr(sol, "iter", [-1])[0]) if hasattr(sol, "iter") else -1,
            cost=float(getattr(sol, "J", [float('nan')])[0]) if hasattr(sol, "J") else float("nan"),
            kkt_residual=float("nan"), constraint_violation=float("nan"))

    def reset(self):
        pass


@register_solver("grampc")
def _factory():
    return GrampcAdapter()
