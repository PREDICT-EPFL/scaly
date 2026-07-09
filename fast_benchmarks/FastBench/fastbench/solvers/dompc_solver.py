"""do-mpc adapter (CasADi + IPOPT NMPC wrapper).

do-mpc expresses the cost as ``mterm(x) + sum lterm(x,u) + rterm`` over inputs.
We map the problem's terminal/stage costs onto these terms and feed the
references through time-varying parameters (TVP).  Suited to regulation and
tracking problems.  Import is guarded so the framework loads without do-mpc.
"""
from __future__ import annotations

import time

import numpy as np

from fastbench.core.registry import register_solver
from fastbench.solvers.base import BuildInfo, SolveStats, SolverAdapter


class DoMpcAdapter(SolverAdapter):
    name = "do_mpc"

    def __init__(self):
        self._mpc = None

    def available(self) -> bool:
        try:
            import do_mpc  # noqa: F401
            return True
        except Exception:
            return False

    def build(self, problem) -> BuildInfo:
        import casadi as ca
        import do_mpc

        t0 = time.perf_counter()
        self.problem = problem
        m = problem.meta
        nx, nu, N = m.nx, m.nu, m.N

        model = do_mpc.model.Model("continuous")
        x = model.set_variable("_x", "x", shape=(nx, 1))
        u = model.set_variable("_u", "u", shape=(nu, 1))
        xref = model.set_variable("_tvp", "xref", shape=(nx, 1))
        uref = model.set_variable("_tvp", "uref", shape=(nu, 1))
        model.set_rhs("x", problem.dynamics_ct(x, u))
        # expose costs as expressions of x,u and references
        model.set_expression("lterm", problem.stage_cost(x, u, xref, uref))
        model.set_expression("mterm", problem.terminal_cost(x, xref))
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(n_horizon=N, t_step=m.dt, store_full_solution=False,
                      nlpsol_opts={"ipopt.print_level": 0, "print_time": 0,
                                   "ipopt.sb": "yes"})
        mpc.set_objective(mterm=model.aux["mterm"], lterm=model.aux["lterm"])
        mpc.set_rterm(u=np.zeros(nu))

        lbx, ubx, lbu, ubu = problem.bounds()
        for i in range(nx):
            if lbx[i] > -1e5:
                mpc.bounds["lower", "_x", "x"][i] = lbx[i]
            if ubx[i] < 1e5:
                mpc.bounds["upper", "_x", "x"][i] = ubx[i]
        for i in range(nu):
            mpc.bounds["lower", "_u", "u"][i] = lbu[i]
            mpc.bounds["upper", "_u", "u"][i] = ubu[i]

        tvp_tmpl = mpc.get_tvp_template()
        self._tvp_tmpl = tvp_tmpl

        def tvp_fun(t_now):
            k = int(round(float(t_now) / m.dt))
            for j in range(N + 1):
                xr, ur = problem.reference_traj(k + j)
                tvp_tmpl["_tvp", j, "xref"] = xr
                tvp_tmpl["_tvp", j, "uref"] = ur
            return tvp_tmpl
        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        self._mpc = mpc
        self._nx, self._nu = nx, nu
        self._dt = m.dt
        self._init = False
        return BuildInfo(build_time_s=time.perf_counter() - t0,
                         notes="do-mpc NMPC (IPOPT); TVP references")

    def solve(self, x, k) -> SolveStats:
        mpc = self._mpc
        x = np.asarray(x, float).reshape(-1, 1)
        if not self._init:
            mpc.x0 = x
            mpc.set_initial_guess()
            self._init = True
        t0 = time.perf_counter()
        u0 = mpc.make_step(x)
        dt = time.perf_counter() - t0
        stats = getattr(mpc, "solver_stats", {})
        success = bool(stats.get("success", True))
        return SolveStats(
            u0=np.asarray(u0, float).flatten(), success=success,
            status=str(stats.get("return_status", "")), solve_time_s=dt,
            iterations=int(stats.get("iter_count", -1)),
            cost=float("nan"), kkt_residual=float("nan"),
            constraint_violation=float("nan"))

    def reset(self):
        # re-initialise the guess/history on the next solve without rebuilding
        self._init = False
        if self._mpc is not None:
            try:
                self._mpc.reset_history()
            except Exception:
                pass


@register_solver("do_mpc")
def _factory():
    return DoMpcAdapter()
