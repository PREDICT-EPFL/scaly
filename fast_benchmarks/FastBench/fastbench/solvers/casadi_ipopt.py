"""CasADi + IPOPT adapter (direct multiple shooting).

This is the reference NLP backend: easy to install and, with tight tolerances,
serves as the high-accuracy *ground-truth* controller used to score the
suboptimality of faster solvers.  The NLP is built once with the initial state
and stage references as parameters, then solved repeatedly with warm starts.
"""
from __future__ import annotations

import time

import casadi as ca
import numpy as np

from fastbench.core.registry import register_solver
from fastbench.solvers.base import BuildInfo, SolveStats, SolverAdapter


class CasadiIpoptAdapter(SolverAdapter):
    name = "casadi_ipopt"
    supports_path_constraints = True

    def __init__(self, tol: float = 1e-8, max_iter: int = 200,
                 reference: bool = False):
        self.tol = 1e-10 if reference else tol
        self.max_iter = 1000 if reference else max_iter
        self.reference = reference
        self._solver = None
        self._w0 = None

    def available(self) -> bool:
        try:
            import casadi  # noqa: F401
            return True
        except Exception:
            return False

    # ----------------------------------------------------------------- build
    def build(self, problem) -> BuildInfo:
        t0 = time.perf_counter()
        self.problem = problem
        m = problem.meta
        nx, nu, N = m.nx, m.nu, m.N
        lbx, ubx, lbu, ubu = problem.bounds()

        x = ca.SX.sym("x", nx)
        u = ca.SX.sym("u", nu)
        fd = ca.Function("fd", [x, u], [problem.discrete_dynamics(x, u)])

        # parameters: initial state + per-stage references
        P_x0 = ca.SX.sym("x0", nx)
        P_xref = ca.SX.sym("xref", nx, N + 1)
        P_uref = ca.SX.sym("uref", nu, N)

        Xk = [ca.SX.sym(f"X_{k}", nx) for k in range(N + 1)]
        Uk = [ca.SX.sym(f"U_{k}", nu) for k in range(N)]
        w, lbw, ubw, w0 = [], [], [], []
        g, lbg, ubg = [], [], []

        g.append(Xk[0] - P_x0); lbg += [0] * nx; ubg += [0] * nx
        J = 0
        pc = problem.path_constraints(x, u)
        if pc is not None:
            h_expr, lh, uh = pc
            hfun = ca.Function("h", [x, u], [h_expr])
        for k in range(N):
            w += [Xk[k]]; lbw += list(lbx); ubw += list(ubx); w0 += [0] * nx
            w += [Uk[k]]; lbw += list(lbu); ubw += list(ubu); w0 += [0] * nu
            J += problem.stage_cost(Xk[k], Uk[k], P_xref[:, k], P_uref[:, k])
            g.append(Xk[k + 1] - fd(Xk[k], Uk[k])); lbg += [0] * nx; ubg += [0] * nx
            if pc is not None:
                g.append(hfun(Xk[k], Uk[k])); lbg += list(lh); ubg += list(uh)
        w += [Xk[N]]; lbw += list(lbx); ubw += list(ubx); w0 += [0] * nx
        J += problem.terminal_cost(Xk[N], P_xref[:, N])

        p = ca.veccat(P_x0, ca.vec(P_xref), ca.vec(P_uref))
        nlp = {"x": ca.vertcat(*w), "f": J, "g": ca.vertcat(*g), "p": p}
        opts = {
            "ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes",
            "ipopt.tol": self.tol, "ipopt.max_iter": self.max_iter,
            "ipopt.warm_start_init_point": "yes",
        }
        self._solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        self._lbw, self._ubw = np.array(lbw, float), np.array(ubw, float)
        self._lbg, self._ubg = np.array(lbg, float), np.array(ubg, float)
        self._g_fun = ca.Function("gf", [ca.vertcat(*w), p], [ca.vertcat(*g)])
        self._w0 = np.array(w0, float)
        self._nx, self._nu, self._N = nx, nu, N
        return BuildInfo(build_time_s=time.perf_counter() - t0,
                         notes="direct multiple shooting; IPOPT")

    # ----------------------------------------------------------------- solve
    def solve(self, x, k) -> SolveStats:
        nx, nu, N = self._nx, self._nu, self._N
        xr = np.zeros((nx, N + 1)); ur = np.zeros((nu, N))
        for j in range(N + 1):
            xr[:, j] = self.problem.reference_traj(k + j)[0]
        for j in range(N):
            ur[:, j] = self.problem.reference_traj(k + j)[1]
        p = np.concatenate([np.asarray(x, float), xr.flatten(order="F"),
                            ur.flatten(order="F")])
        t0 = time.perf_counter()
        sol = self._solver(x0=self._w0, lbx=self._lbw, ubx=self._ubw,
                           lbg=self._lbg, ubg=self._ubg, p=p)
        dt = time.perf_counter() - t0
        st = self._solver.stats()
        w_opt = np.array(sol["x"]).flatten()
        self._w0 = w_opt                                   # warm start
        # first input
        u0 = w_opt[nx:nx + nu]
        # constraint violation
        gval = np.array(self._g_fun(w_opt, p)).flatten()
        cviol = float(np.max(np.maximum(self._lbg - gval,
                                        np.maximum(gval - self._ubg, 0.0)), initial=0.0))
        kkt = float("nan")
        try:
            kkt = float(st["iterations"]["inf_du"][-1])
        except Exception:
            pass
        # unpack predicted trajectory
        xpred = np.array([w_opt[i * (nx + nu):i * (nx + nu) + nx] for i in range(N)]
                         + [w_opt[N * (nx + nu):N * (nx + nu) + nx]])
        upred = np.array([w_opt[i * (nx + nu) + nx:i * (nx + nu) + nx + nu] for i in range(N)])
        return SolveStats(
            u0=u0, success=bool(st.get("success", False)),
            status=str(st.get("return_status", "")), solve_time_s=dt,
            iterations=int(st.get("iter_count", -1)), cost=float(sol["f"]),
            kkt_residual=kkt, constraint_violation=cviol,
            x_pred=xpred, u_pred=upred)

    def reset(self):
        if self._w0 is not None:
            self._w0 = np.zeros_like(self._w0)


@register_solver("casadi_ipopt")
def _factory():
    return CasadiIpoptAdapter()


@register_solver("casadi_ipopt_ref")
def _factory_ref():
    return CasadiIpoptAdapter(reference=True)
