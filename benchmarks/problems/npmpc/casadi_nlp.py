"""CasADi mirror of the neural-process MPC, as a drop-in stand-in for the Alloy solver.

`CasadiNpmpcSolver` presents the same call signature, output keys and statistics as the
`al.SolverFunction` built by `npmpc_nlp`, so one episode loop drives either oracle provider. The
decision-variable layout, the parameter layout, the cost, the equality and inequality rows, the
bounds and the IPOPT options are identical by construction -- both sides read them out of
`ca_npmpc_pieces`, `npmpc_ineq_bounds` and `npmpc_bounds` -- so the only difference is which tool
differentiates and evaluates the oracles. That is the controlled comparison `BENCHMARKS.md` asks
for, and it matters more here than on any other problem in the suite: with 65 decision variables
there is almost no linear algebra for IPOPT to do, so function evaluation is most of the solve.

The reference implementation builds the same problem through `ca.Opti` with per-stage matrices,
which it needs for its own warm-start bookkeeping. Here the variables are one flat vector in
Alloy's order, so the two columns solve a bit-for-bit identical problem.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from alloy.solvers.stats import ALLOY_SOLVER_STATS_VERSION, AlloySolveStatus, SolverStats, SolverStatus
from benchmarks.problems.npmpc import DT, NX, CostWeights, Decoder, ca_npmpc_pieces, n_param

# IPOPT's own termination strings, mapped onto Alloy's solver-agnostic status enum
STATUS_MAP = {
  "Solve_Succeeded": AlloySolveStatus.OK,
  "Solved_To_Acceptable_Level": AlloySolveStatus.ACCEPTABLE,
  "Feasible_Point_Found": AlloySolveStatus.ACCEPTABLE,
  "Maximum_Iterations_Exceeded": AlloySolveStatus.MAX_ITER,
  "Maximum_CpuTime_Exceeded": AlloySolveStatus.MAX_ITER,
  "Infeasible_Problem_Detected": AlloySolveStatus.PRIMAL_INFEASIBLE,
  "Diverging_Iterates": AlloySolveStatus.DUAL_INFEASIBLE,
  "Restoration_Failed": AlloySolveStatus.NUMERICS,
  "Error_In_Step_Computation": AlloySolveStatus.NUMERICS,
  "Search_Direction_Becomes_Too_Small": AlloySolveStatus.NUMERICS,
  "User_Requested_Stop": AlloySolveStatus.USER_STOP,
}
# oracle timings CasADi reports per solve; their sum is the function-evaluation share
FE_TIMERS = ("t_wall_nlp_f", "t_wall_nlp_g", "t_wall_nlp_grad_f", "t_wall_nlp_jac_g", "t_wall_nlp_hess_l")

# CasADi evaluates through its own virtual machine unless told to compile, so this column as it
# currently stands measures an interpreter against generated C. That is a **known unfairness**, not
# a considered choice: `BENCHMARKS.md` section 2.3 defines the CasADi column as JIT-enabled, and
# `race_cars` and `unbumpercars` have the same gap. It is left in place here only so the whole suite
# shares one treatment until the harmonization pass in section 6 of that file fixes all of them
# together -- a suite where one problem JITs and two do not is worse than one where none do, because
# then no two problems' CasADi columns mean the same thing.
#
# All four combinations were measured on the canonical episode first, so the harmonization has
# numbers to start from. Per solve, plus the one-off cost of constructing the solver:
#
#   | expand | jit | build   | total    | FE       | FE share |
#   |--------|-----|---------|----------|----------|----------|
#   | True   | no  |   0.7 s | 17.35 ms | 13.02 ms |      75% |  <- shipped, and the worst of the four
#   | True   | yes | 966.5 s |  7.75 ms |  3.19 ms |      41% |
#   | False  | no  |   0.1 s |  9.50 ms |  4.84 ms |      51% |
#   | False  | yes |  23.9 s |  5.37 ms |  1.46 ms |      27% |  <- fastest, and cheap to build
#
# Two things that table settles. Expanding to scalar SX is a trap: it triples the interpreter's work,
# and once compiled it hands the C compiler some 750 000 lines for a sixteen-minute build only to
# land slower than compiled MX. And with `jit=True, expand=False` the function-evaluation gap against
# Alloy closes from 8.7x to a wash (1.46 against 1.50 ms), which is what the kernel sweeps already
# predicted -- so **no runtime claim may rest on this column as configured**.
JIT_OPTIONS: dict[str, Any] = {
  "jit": True,
  "jit_options": {"flags": ["-O3"], "verbose": False},
  "compiler": "shell",
}
# Matching `race_cars` and `unbumpercars` until the harmonization pass; see the note above.
DEFAULT_EXPAND = True
DEFAULT_JIT = False


def joint_parameter_pieces(
  horizon: int,
  decoder: Decoder = Decoder(),
  *,
  P: np.ndarray | None = None,
  weights: CostWeights = CostWeights(),
  dt: float = DT,
) -> dict[str, Any]:
  """`ca_npmpc_pieces` with the two parameter symbols folded into the single `p` the solvers take.

  The kernels in the sweep read only the decoder tail, so `ca_npmpc_pieces` keeps `xstart` and `pw`
  apart; the solver interfaces want one parameter vector in `n_param` order. Substituting is what
  keeps those two facts from becoming two transcriptions of the same problem.
  """
  import casadi as ca

  pieces = ca_npmpc_pieces(horizon, decoder, ca.MX, P=P, weights=weights, dt=dt)
  p = ca.MX.sym("p", n_param(decoder))
  f, h_eq, g_ineq = ca.substitute(
    [pieces["f"], pieces["h_eq"], pieces["g_ineq"]],
    [pieces["xstart"], pieces["pw"]],
    [p[:NX], p[NX:]],
  )
  return {**pieces, "p": p, "f": f, "h_eq": h_eq, "g_ineq": g_ineq}


def build_casadi_npmpc(config, *, solver: str = "ipopt", P: np.ndarray, jit: bool = DEFAULT_JIT):
  """The CasADi-oracle column for the requested optimizer."""
  pieces = joint_parameter_pieces(config.horizon, config.decoder, P=P, weights=config.weights, dt=config.dt)
  if solver == "ipopt":
    return CasadiNpmpcSolver(config, pieces, jit=jit)
  if solver == "sqp":
    return build_casadi_npmpc_sqp(config, pieces)
  raise ValueError(f"unsupported npmpc CasADi solver {solver!r}")


def build_casadi_npmpc_sqp(config, pieces: dict[str, Any]):
  """`alloy-sqp` driving CasADi-generated oracles, at the same settings as the Alloy column."""
  import casadi as ca

  from alloy_sqp.casadi import build_casadi_external_sqp

  z, p, cost = pieces["z"], pieces["p"], pieces["f"]
  constraints = ca.vertcat(pieces["h_eq"], pieces["g_ineq"])
  lam_f, lam_g = ca.MX.sym("lam_f"), ca.MX.sym("lam_g", int(constraints.shape[0]))
  stem = f"ca_npmpc_sqp_N{config.horizon}"
  return build_casadi_external_sqp(
    name=stem,
    base=ca.Function(f"{stem}_base", [z, p], [cost, constraints]),
    grad=ca.Function(f"{stem}_grad", [z, p], [ca.gradient(cost, z)]),
    jac=ca.Function(f"{stem}_jac", [z, p], [ca.jacobian(constraints, z)]),
    hess=ca.Function(f"{stem}_hess", [z, lam_f, lam_g, p], [ca.hessian(lam_f * cost + ca.dot(lam_g, constraints), z)[0]]),
    n_eq=pieces["n_eq"],
    n_ineq=pieces["n_ineq"],
    x_lb=pieces["x_lb"],
    x_ub=pieces["x_ub"],
    l_ineq=pieces["l_ineq"],
    u_ineq=pieces["u_ineq"],
    # identical SQP settings to the Alloy column in closed_loop.build_solver
    options={"tol": config.sqp_tol, "max_iter": config.sqp_max_iter},
  )


class CasadiNpmpcSolver:
  """`al.SolverFunction`-shaped wrapper around `ca.nlpsol("ipopt", ...)`.

  `jit=True` compiles the oracles instead of interpreting them, which is what a fair baseline needs;
  it is not the default yet, for the reason recorded above `JIT_OPTIONS`.
  """

  def __init__(self, config, pieces: dict[str, Any], *, expand: bool = DEFAULT_EXPAND, jit: bool = DEFAULT_JIT):
    import casadi as ca

    started = time.perf_counter()
    self.n_eq, self.n_ineq = pieces["n_eq"], pieces["n_ineq"]
    self._bounds = {
      "lbx": pieces["x_lb"],
      "ubx": pieces["x_ub"],
      "lbg": np.concatenate([np.zeros(self.n_eq), pieces["l_ineq"]]),
      "ubg": np.concatenate([np.zeros(self.n_eq), pieces["u_ineq"]]),
    }
    self.base = ca.Function(
      f"npmpc_casadi_base_N{config.horizon}",
      [pieces["z"], pieces["p"]],
      [pieces["f"], pieces["h_eq"], pieces["g_ineq"]],
    )
    self.solver = ca.nlpsol(
      f"npmpc_casadi_N{config.horizon}",
      "ipopt",
      {"x": pieces["z"], "p": pieces["p"], "f": pieces["f"], "g": ca.vertcat(pieces["h_eq"], pieces["g_ineq"])},
      {
        "print_time": False,
        "expand": expand,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.tol": config.ipopt_tol,
        "ipopt.max_iter": config.ipopt_max_iter,
        "ipopt.warm_start_init_point": "yes",
        **(JIT_OPTIONS if jit else {}),
      },
    )
    self.build_ms = (time.perf_counter() - started) * 1000.0
    self.last_stats: SolverStats | None = None
    self.last_status: SolverStatus | None = None

  def __call__(self, z0, lam_eq0, lam_ineq0, lam_box0, p) -> dict[str, np.ndarray]:
    started = time.perf_counter()
    solution = self.solver(
      x0=z0,
      p=p,
      lam_g0=np.concatenate([np.asarray(lam_eq0, dtype=np.float64), np.asarray(lam_ineq0, dtype=np.float64)]),
      lam_x0=lam_box0,
      **self._bounds,
    )
    t_total = time.perf_counter() - started
    raw = self.solver.stats()
    t_fe = sum(float(raw.get(name, 0.0)) for name in FE_TIMERS)
    self.last_stats = SolverStats(
      version=ALLOY_SOLVER_STATS_VERSION,
      status=STATUS_MAP.get(str(raw.get("return_status", "")), AlloySolveStatus.ERROR),
      native_status=0,
      iter=int(raw.get("iter_count", 0)),
      obj=float(solution["f"]),
      t_total=t_total,
      t_fe=t_fe,
      t_solver=max(t_total - t_fe, 0.0),
      t_qp=0.0,
      t_globalization=0.0,
      t_glue=0.0,
      n_eval_f=int(raw.get("n_call_nlp_f", 0)),
      n_eval_grad_f=int(raw.get("n_call_nlp_grad_f", 0)),
      n_eval_g=int(raw.get("n_call_nlp_g", 0)),
      n_eval_jac_g=int(raw.get("n_call_nlp_jac_g", 0)),
      n_eval_h=int(raw.get("n_call_nlp_hess_l", 0)),
    )
    self.last_status = self.last_stats.to_solver_status()
    x = np.asarray(solution["x"], dtype=np.float64).reshape(-1)
    f, h_eq, g_ineq = self.base(x, p)
    lam_g = np.asarray(solution["lam_g"], dtype=np.float64).reshape(-1)
    return {
      "x": x,
      "f": np.asarray(f, dtype=np.float64).reshape(()),
      "h_eq": np.asarray(h_eq, dtype=np.float64).reshape(-1),
      "g_ineq": np.asarray(g_ineq, dtype=np.float64).reshape(-1),
      "lam_eq": lam_g[: self.n_eq],
      "lam_ineq": lam_g[self.n_eq :],
      "lam_box": np.asarray(solution["lam_x"], dtype=np.float64).reshape(-1),
    }
