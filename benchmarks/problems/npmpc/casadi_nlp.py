"""CasADi mirror of the neural-process MPC, as a drop-in stand-in for the Scaly solver.

`CasadiNpmpcSolver` presents the same call signature, output keys and statistics as the
`sc.Function` built by `npmpc_nlp`, so one episode loop drives either oracle provider. The
decision-variable layout, the parameter layout, the cost, the equality and inequality rows, the
bounds and the IPOPT options are identical by construction -- both sides read them out of
`ca_npmpc_pieces`, `npmpc_ineq_bounds` and `npmpc_bounds` -- so the only difference is which tool
differentiates and evaluates the oracles. That is the controlled comparison the paper asks
for, and it matters more here than on any other problem in the suite: with 65 decision variables
there is almost no linear algebra for IPOPT to do, so function evaluation is most of the solve.

The reference implementation builds the same problem through `ca.Opti` with per-stage matrices,
which it needs for its own warm-start bookkeeping. Here the variables are one flat vector in
Scaly's order, so the two columns solve a bit-for-bit identical problem.
"""

from __future__ import annotations

from typing import Any

from benchmarks.harness import register_base
from benchmarks.harness.casadi_ipopt import CasadiIpoptSolver
from benchmarks.problems.npmpc import _ca_npmpc_joint_parameter_pieces

DEFAULT_EXPAND = False


def build_casadi_npmpc(config, *, solver: str = "ipopt"):
  """The CasADi-oracle column for the requested optimizer."""
  pieces = _ca_npmpc_joint_parameter_pieces(config.horizon, config.decoder)
  if solver == "ipopt":
    return CasadiNpmpcSolver(config, pieces)
  if solver == "sqp":
    return build_casadi_npmpc_sqp(config, pieces)
  raise ValueError(f"unsupported npmpc CasADi solver {solver!r}")


def build_casadi_npmpc_sqp(config, pieces: dict[str, Any]):
  """`scaly-sqp` driving CasADi-generated oracles, at the same settings as the Scaly column."""
  import casadi as ca

  from scaly_sqp.casadi import build_casadi_external_sqp

  z, p, cost = pieces["z"], pieces["p"], pieces["f"]
  constraints = ca.vertcat(pieces["h_eq"], pieces["g_ineq"])
  lam_f, lam_g = ca.MX.sym("lam_f"), ca.MX.sym("lam_g", int(constraints.shape[0]))
  stem = f"ca_npmpc_sqp_N{config.horizon}"
  base = ca.Function(f"{stem}_base", [z, p], [cost, constraints])
  solver = build_casadi_external_sqp(
    name=stem,
    base=base,
    grad=ca.Function(f"{stem}_grad", [z, p], [ca.gradient(cost, z)]),
    jac=ca.Function(f"{stem}_jac", [z, p], [ca.jacobian(constraints, z)]),
    hess=ca.Function(f"{stem}_hess", [z, p, lam_f, lam_g], [ca.hessian(lam_f * cost + ca.dot(lam_g, constraints), z)[0]]),
    n_eq=pieces["n_eq"],
    n_ineq=pieces["n_ineq"],
    x_lb=pieces["x_lb"],
    x_ub=pieces["x_ub"],
    l_ineq=pieces["l_ineq"],
    u_ineq=pieces["u_ineq"],
    # identical SQP settings to the Scaly column in closed_loop.build_solver
    options={"tol": config.sqp_tol, "max_iter": config.sqp_max_iter},
  )
  register_base(solver.function, base)
  return solver


class CasadiNpmpcSolver(CasadiIpoptSolver):
  """Compiled CasADi oracles and IPOPT solve behind the benchmark solver contract."""

  def __init__(self, config, pieces: dict[str, Any], *, expand: bool = DEFAULT_EXPAND):
    super().__init__(
      f"npmpc_casadi_N{config.horizon}",
      pieces["z"],
      pieces["p"],
      pieces["f"],
      pieces["h_eq"],
      pieces["g_ineq"],
      x_lb=pieces["x_lb"],
      x_ub=pieces["x_ub"],
      l_ineq=pieces["l_ineq"],
      u_ineq=pieces["u_ineq"],
      options={"ipopt.tol": config.ipopt_tol, "ipopt.max_iter": config.ipopt_max_iter},
      expand=expand,
    )
