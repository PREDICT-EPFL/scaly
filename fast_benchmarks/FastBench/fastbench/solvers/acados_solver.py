"""acados adapter - the embedded/real-time backend.

Builds an OCP from the problem's CasADi model with an EXTERNAL cost (so any
stage/terminal cost is supported) and per-stage parameters carrying the
references.  On build it generates and compiles C code and records codegen
metrics: generation time, compile time, generated source size / LOC, and
compiled-artifact size.  Solves use SQP-RTI by default (the embedded mode).

Requires ``acados_template`` and a built acados (see scripts/install_acados.sh).
The import is guarded so the framework loads without acados installed.
"""
from __future__ import annotations

import os
import time

import numpy as np

from fastbench.core.registry import register_solver
from fastbench.solvers.base import BuildInfo, SolveStats, SolverAdapter


def _dir_stats(path, exts=None):
    """Total bytes and LOC of files under ``path`` (optionally by extension)."""
    nbytes, loc = 0, 0
    for root, _, files in os.walk(path):
        for f in files:
            if exts and not f.endswith(tuple(exts)):
                continue
            fp = os.path.join(root, f)
            try:
                nbytes += os.path.getsize(fp)
                if f.endswith((".c", ".h")):
                    with open(fp, "r", errors="ignore") as fh:
                        loc += sum(1 for _ in fh)
            except OSError:
                pass
    return nbytes, loc


class AcadosAdapter(SolverAdapter):
    name = "acados"

    def __init__(self, nlp_solver: str = "SQP_RTI", rti_iters: int = 1):
        self.nlp_solver = nlp_solver
        self.rti_iters = rti_iters
        self._solver = None

    def available(self) -> bool:
        try:
            import acados_template  # noqa: F401
            return bool(os.environ.get("ACADOS_SOURCE_DIR"))
        except Exception:
            return False

    # ----------------------------------------------------------------- build
    def build(self, problem) -> BuildInfo:
        import casadi as ca
        from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver

        self.problem = problem
        m = problem.meta
        nx, nu, N = m.nx, m.nu, m.N
        lbx, ubx, lbu, ubu = problem.bounds()

        x = ca.SX.sym("x", nx); u = ca.SX.sym("u", nu)
        xref = ca.SX.sym("xref", nx); uref = ca.SX.sym("uref", nu)

        model = AcadosModel()
        model.name = problem.meta.name.replace("-", "_")
        model.x = x
        model.u = u
        model.f_expl_expr = problem.dynamics_ct(x, u)
        xdot = ca.SX.sym("xdot", nx)
        model.xdot = xdot
        model.f_impl_expr = xdot - model.f_expl_expr
        model.p = ca.vertcat(xref, uref)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = N
        ocp.solver_options.tf = N * m.dt

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = problem.stage_cost(x, u, xref, uref)
        ocp.model.cost_expr_ext_cost_e = problem.terminal_cost(x, xref)
        ocp.parameter_values = np.zeros(nx + nu)

        # input box constraints
        ocp.constraints.idxbu = np.arange(nu)
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu
        # state box constraints (skip +-1e6 "free" entries)
        idxbx = [i for i in range(nx) if np.isfinite(lbx[i]) and lbx[i] > -1e5
                 or np.isfinite(ubx[i]) and ubx[i] < 1e5]
        if idxbx:
            ocp.constraints.idxbx = np.array(idxbx)
            ocp.constraints.lbx = lbx[idxbx]
            ocp.constraints.ubx = ubx[idxbx]
        ocp.constraints.x0 = np.asarray(problem.x0_nominal(), float)

        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = self.nlp_solver
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.nlp_solver_max_iter = 50 if self.nlp_solver == "SQP" else 1

        gen_dir = os.path.abspath(f"c_generated_code_{model.name}")
        ocp.code_export_directory = gen_dir
        json_file = f"acados_ocp_{model.name}.json"

        t0 = time.perf_counter()
        self._solver = AcadosOcpSolver(ocp, json_file=json_file, verbose=False)
        build_time = time.perf_counter() - t0

        gen_bytes, gen_loc = _dir_stats(gen_dir, exts=[".c", ".h"])
        comp_bytes, _ = _dir_stats(gen_dir, exts=[".o", ".so", ".a", ".dll", ".dylib"])
        self._nx, self._nu, self._N = nx, nu, N
        return BuildInfo(
            build_time_s=build_time, generation_time_s=build_time,
            compile_time_s=float("nan"),
            generated_code_bytes=gen_bytes, generated_code_loc=gen_loc,
            compiled_bytes=comp_bytes,
            notes=f"{self.nlp_solver}, HPIPM; codegen in {os.path.basename(gen_dir)}")

    # ----------------------------------------------------------------- solve
    def solve(self, x, k) -> SolveStats:
        s = self._solver
        nx, nu, N = self._nx, self._nu, self._N
        x = np.asarray(x, float)
        s.set(0, "lbx", x); s.set(0, "ubx", x)
        for j in range(N):
            xr, ur = self.problem.reference_traj(k + j)
            s.set(j, "p", np.concatenate([xr, ur]))
        xrN, _ = self.problem.reference_traj(k + N)
        s.set(N, "p", np.concatenate([xrN, np.zeros(nu)]))

        t0 = time.perf_counter()
        status = s.solve()
        dt = time.perf_counter() - t0
        u0 = s.get(0, "u")
        try:
            res = np.asarray(s.get_stats("residuals"))
            kkt = float(np.max(res))
        except Exception:
            kkt = float("nan")
        try:
            iters = int(s.get_stats("sqp_iter"))
        except Exception:
            iters = self.rti_iters
        xpred = np.array([s.get(i, "x") for i in range(N + 1)])
        upred = np.array([s.get(i, "u") for i in range(N)])
        return SolveStats(
            u0=np.asarray(u0, float), success=(status == 0),
            status=f"acados_status={status}",
            solve_time_s=float(s.get_stats("time_tot")) if status in (0, 2) else dt,
            iterations=iters, cost=float(s.get_cost()),
            kkt_residual=kkt, constraint_violation=float("nan"),
            x_pred=xpred, u_pred=upred)

    def reset(self):
        pass


@register_solver("acados")
def _factory():
    return AcadosAdapter(nlp_solver="SQP_RTI")


@register_solver("acados_sqp")
def _factory_sqp():
    return AcadosAdapter(nlp_solver="SQP")
