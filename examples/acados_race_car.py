"""Solve race-car tracking with acados and compare every sequential quadratic programming (SQP) iterate.

The four states are planar position, heading and speed. Throttle and steering drive a
kinematic bicycle model discretized by fourth-order Runge-Kutta. The parameters,
tracking weights, actuator limits and footprint corridor match the race-car
benchmark. A circular reference keeps this example independent of track files.
The formulation eliminates the benchmark's independent terminal controls, whose
minimum penalty is zero.

Requires a C compiler and acados-wheels 0.6.0, which pins acados 5033648. With
access to the private wheel repository, on Linux x86-64 run from this checkout:

    gh release download wheels -R tudoroancea/acados-wheels -p '*manylinux_2_28_x86_64.whl'
    uv run --with ./acados_wheels-0.6.0-py3-none-manylinux_2_28_x86_64.whl examples/acados_race_car.py

Both solvers use acados' SQP with exact Hessians and CasADi corridor constraints.
``build_scaly_solver`` takes Scaly dynamics and cost expression builders. It
generates an acados solver with correctly shaped zero CasADi placeholders, then
replaces the dynamics and initial, path and terminal external costs with Scaly
``casadi=True`` modules before compilation. acados reads the actual sparsity and
workspace requirements from Scaly's external-function queries. The placeholder
sparsity does not need to match. The helper passes ``-Dcasadi_int=int`` through
``ocp.code_gen_options.ext_fun_compile_flags`` to match acados' integer width.

The handwritten CasADi dynamics and costs are used only by the independent
reference solver. Both solvers start with the same states, controls and
multipliers. The script compares all their stored state,
control and multiplier iterates and final costs, and prints the directory
containing both builds under ``examples/generated/acados_race_car/``.
"""

from collections.abc import Callable
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver  # ty: ignore[unresolved-import]
from acados_template.builders import ocp_get_default_cmake_builder  # ty: ignore[unresolved-import]

import scaly as sc
from scaly.codegen import render_c_module
from scaly.function.concrete import ConcreteFunction
from scaly.ir.expr import substitute

NX, NU = 4, 2
PHYSICAL = np.array([1.5706, 0.05, 230.0, 4.950, 297.030, 16.665, 0.6784])
Q = np.array([10.0, 20.0, 50.0, 20.0])
Q_TERMINAL = np.array([1000.0, 1000.0, 500.0, 1000.0])
R = np.array([1e-3, 2.0])


def scaly_dynamics(x: sc.Expr, u: sc.Expr, p: sc.Expr) -> sc.Expr:
  """Advance the bicycle state by one sample using Scaly expressions."""
  wheelbase, dt, mass, c_m0, c_r0, c_r1, c_r2 = [p[4 + i] for i in range(7)]

  def ode(state: sc.Expr) -> sc.Expr:
    beta = 0.5 * u[1]
    v, phi = state[3], state[2]
    vx = v * beta.cos()
    resistance = (c_r0 + c_r1 * vx + c_r2 * vx * vx) * (10.0 * vx).tanh()
    return sc.stack([v * (phi + beta).cos(), v * (phi + beta).sin(), v * beta.sin() / (0.5 * wheelbase), (c_m0 * u[0] - resistance) / mass])

  k1 = ode(x)
  k2 = ode(x + dt / 2 * k1)
  k3 = ode(x + dt / 2 * k2)
  k4 = ode(x + dt * k3)
  return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def casadi_dynamics(x, u, p):
  """Independent CasADi version of the same Runge-Kutta step."""
  wheelbase, dt, mass, c_m0, c_r0, c_r1, c_r2 = [p[4 + i] for i in range(7)]

  def ode(state):
    beta = 0.5 * u[1]
    v, phi = state[3], state[2]
    vx = v * ca.cos(beta)
    resistance = (c_r0 + c_r1 * vx + c_r2 * vx * vx) * ca.tanh(10.0 * vx)
    return ca.vertcat(v * ca.cos(phi + beta), v * ca.sin(phi + beta), v * ca.sin(beta) / (0.5 * wheelbase), (c_m0 * u[0] - resistance) / mass)

  k1 = ode(x)
  k2 = ode(x + dt / 2 * k1)
  k3 = ode(x + dt / 2 * k2)
  k4 = ode(x + dt * k3)
  return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def scaly_cost(x: sc.Expr, u: sc.Expr, p: sc.Expr, stage: str) -> sc.Expr:
  """Control penalty at stage zero, tracking thereafter, terminal tracking only."""
  value = sc.const(0.0)
  if stage != "terminal":
    v_ref = p[3]
    throttle_ref = (10.0 * v_ref).tanh() * (p[8] + p[9] * v_ref + p[10] * v_ref * v_ref) / p[7]
    value = R[0] * (u[0] - throttle_ref) ** 2 + R[1] * u[1] ** 2
  if stage != "initial":
    dx, dy = x[0] - p[0], x[1] - p[1]
    e_lon = p[2].cos() * dx + p[2].sin() * dy
    e_lat = -p[2].sin() * dx + p[2].cos() * dy
    errors = sc.stack([e_lon, e_lat, x[2] - p[2], x[3] - p[3]])
    value = value + (sc.const(Q_TERMINAL if stage == "terminal" else Q) * errors * errors).sum()
  return value


def casadi_cost(x, u, p, stage):
  """Independent CasADi version of the three stage costs."""
  value = 0
  if stage != "terminal":
    v_ref = p[3]
    throttle_ref = ca.tanh(10.0 * v_ref) * (p[8] + p[9] * v_ref + p[10] * v_ref * v_ref) / p[7]
    value = R[0] * (u[0] - throttle_ref) ** 2 + R[1] * u[1] ** 2
  if stage != "initial":
    dx, dy = x[0] - p[0], x[1] - p[1]
    e_lon = ca.cos(p[2]) * dx + ca.sin(p[2]) * dy
    e_lat = -ca.sin(p[2]) * dx + ca.cos(p[2]) * dy
    errors = ca.vertcat(e_lon, e_lat, x[2] - p[2], x[3] - p[3])
    value += ca.dot(ca.DM(Q_TERMINAL if stage == "terminal" else Q), errors * errors)
  return value


def _export(path: Path, inputs: tuple[sc.Expr, ...], outputs: dict[str, sc.Expr | sc.SparseJacobian], replacements: dict[sc.Expr, sc.Expr]) -> None:
  values = [out.values if isinstance(out, sc.SparseJacobian) else out for out in outputs.values()]
  fun = ConcreteFunction.build(
    path.stem,
    sc.group(*(sc.arg(str(expr.name), expr.type) for expr in inputs)),
    inputs,
    sc.group(*(sc.arg(name, value.type) for name, value in zip(outputs, values, strict=True))),
    [substitute(value, replacements) for value in values],
    output_sparsities=[out.sparsity if isinstance(out, sc.SparseJacobian) else None for out in outputs.values()],
  )
  assert path.is_file(), f"acados did not generate {path}"
  path.write_text(render_c_module(fun, casadi=True).body)


def build_scaly_solver(
  ocp: AcadosOcp,
  dynamics: Callable[[sc.Expr, sc.Expr, sc.Expr], sc.Expr],
  cost: Callable[[sc.Expr, sc.Expr, sc.Expr, str], sc.Expr],
) -> AcadosOcpSolver:
  """Build and load an acados solver with Scaly dynamics and external costs.

  The problem uses discrete dynamics, exact Hessians and EXTERNAL costs. It
  declares CasADi SX vectors ``x``, ``u`` and ``p``, with no algebraic variables
  or global parameters ``p_global``.
  ``dynamics(x, u, p)`` returns the next state. ``cost(x, u, p, stage)`` returns
  a scalar for ``initial``, ``path`` or ``terminal``. Terminal ``u`` is empty.
  CasADi zero placeholders supply the shapes for acados generation. Scaly
  modules replace the value and derivative functions, including their sparsity
  and workspace queries, before compilation.
  """
  model = ocp.model
  nx, nu, np_ = model.x.numel(), model.u.numel(), model.p.numel()
  model.disc_dyn_expr = ca.SX.zeros(nx, 1)
  model.cost_expr_ext_cost_0 = model.cost_expr_ext_cost = model.cost_expr_ext_cost_e = ca.SX(0)
  ocp.code_gen_options.ext_fun_compile_flags += " -Dcasadi_int=int"
  directory, name = Path(ocp.code_gen_options.code_export_directory), model.name
  builder = ocp_get_default_cmake_builder()
  AcadosOcpSolver.generate(ocp, cmake_builder=builder, verbose=False)

  ux, x, u, pi, p = sc.sym("ux", nu + nx), sc.sym("x", nx), sc.sym("u", nu), sc.sym("pi", nx), sc.sym("p", np_)
  phi = dynamics(ux[nu:], ux[:nu], p)
  jac = sc.sparse_jacobian(phi, ux)
  transposed = sc.SparseJacobian(sc.SparsityPattern((nu + nx, nx), jac.sparsity.cols, jac.sparsity.rows), jac.values)
  hess = sc.sparse_hessian((phi * pi).sum(), ux)
  outputs = {"phi": phi, "jac_phi_ux": transposed, "hess_lagrangian_ux_ux": hess}
  replacements = {ux: sc.concat([u, x])}
  for suffix, count in (("", 1), ("_jac", 2), ("_jac_hess", 3)):
    inputs = (x, u, pi, p) if count == 3 else (x, u, p)
    _export(directory / f"{name}_model" / f"{name}_dyn_disc_phi_fun{suffix}.c", inputs, dict(list(outputs.items())[:count]), replacements)

  for stage, suffix, stage_nu in (("initial", "_0", nu), ("path", "", nu), ("terminal", "_e", 0)):
    ux, u, z = sc.sym("ux", stage_nu + nx), sc.sym("u", stage_nu), sc.sym("z", 0)
    value = cost(ux[stage_nu:], ux[:stage_nu], p, stage)
    outputs = {
      "cost": value,
      "grad_cost_ux": sc.gradient(value, ux),
      "hess_cost_ux_ux": sc.sparse_hessian(value, ux),
      "hess_cost_z_z": sc.SparseJacobian(sc.SparsityPattern.empty((0, 0)), sc.const(np.empty(0))),
      "hess_cost_z_ux": sc.SparseJacobian(sc.SparsityPattern.empty((0, stage_nu + nx)), sc.const(np.empty(0))),
    }
    for derivative, count in (("", 1), ("_jac", 2), ("_jac_hess", 5)):
      _export(
        directory / f"{name}_cost" / f"{name}_cost_ext_cost{suffix}_fun{derivative}.c",
        (x, u, z, p),
        dict(list(outputs.items())[:count]),
        {ux: sc.concat([u, x])},
      )

  AcadosOcpSolver.build(str(directory), cmake_builder=builder, verbose=False)
  return AcadosOcpSolver(ocp, generate=False, build=False, verbose=False)


def solve_race_car(work: Path, *, horizon: int = 20, scaly: bool = True):
  """Build and solve either version, retaining iterates and generated sources."""
  name = "race_car_scaly" if scaly else "race_car_casadi"
  ocp = AcadosOcp()
  ocp.name = ocp.model.name = name
  model = ocp.model
  model.x, model.u, model.p = ca.SX.sym("x", NX), ca.SX.sym("u", NU), ca.SX.sym("p", 11)
  x, u, p = model.x, model.u, model.p
  if not scaly:
    model.disc_dyn_expr = casadi_dynamics(x, u, p)
    model.cost_expr_ext_cost_0 = casadi_cost(x, u, p, "initial")
    model.cost_expr_ext_cost = casadi_cost(x, u, p, "path")
    model.cost_expr_ext_cost_e = casadi_cost(x, u, p, "terminal")
  ocp.cost.cost_type_0 = ocp.cost.cost_type = ocp.cost.cost_type_e = "EXTERNAL"

  e_lat = -ca.sin(p[2]) * (x[0] - p[0]) + ca.cos(p[2]) * (x[1] - p[1])
  reach = e_lat + 1.4 * ca.sin(x[2] - p[2])
  corridor = ca.vertcat(reach + 0.75 * ca.cos(x[2] - p[2]), reach - 0.75 * ca.cos(x[2] - p[2]))
  model.con_h_expr = model.con_h_expr_e = corridor
  ocp.constraints.lh = ocp.constraints.lh_e = np.full(2, -1.5)
  ocp.constraints.uh = ocp.constraints.uh_e = np.full(2, 1.5)
  ocp.constraints.idxbu, ocp.constraints.lbu, ocp.constraints.ubu = np.arange(NU), np.array([-500.0, -0.5]), np.array([500.0, 0.5])
  ocp.constraints.idxbx = ocp.constraints.idxbx_e = np.array([3])
  ocp.constraints.lbx = ocp.constraints.lbx_e = np.array([0.0])
  ocp.constraints.ubx = ocp.constraints.ubx_e = np.array([5.0])
  ocp.constraints.x0 = np.array([0.0, 0.3, 0.05, 2.5])

  angle = np.arange(horizon + 1) * PHYSICAL[1] * 3.0 / 20.0
  references = np.column_stack([20.0 * np.sin(angle), 20.0 * (1.0 - np.cos(angle)), angle, np.full(horizon + 1, 3.0)])
  ocp.parameter_values = np.r_[references[0], PHYSICAL]
  opts = ocp.solver_options
  opts.N_horizon, opts.tf = horizon, horizon * PHYSICAL[1]
  opts.integrator_type, opts.nlp_solver_type = "DISCRETE", "SQP"
  opts.hessian_approx, opts.globalization = "EXACT", "MERIT_BACKTRACKING"
  opts.regularize_method = "PROJECT"
  opts.nlp_solver_max_iter = 50
  opts.nlp_solver_tol_stat = opts.nlp_solver_tol_eq = opts.nlp_solver_tol_ineq = opts.nlp_solver_tol_comp = 1e-8
  opts.cost_scaling = np.ones(horizon + 1)
  opts.store_iterates = True
  directory = (work / name).resolve()
  ocp.code_gen_options.code_export_directory = str(directory)
  ocp.code_gen_options.json_file = str(directory / "ocp.json")
  solver = build_scaly_solver(ocp, scaly_dynamics, scaly_cost) if scaly else AcadosOcpSolver(ocp, verbose=False)
  for stage in range(horizon + 1):
    solver.set(stage, "p", np.r_[references[stage], PHYSICAL])
    solver.set(stage, "x", ocp.constraints.x0)
    if stage < horizon:
      solver.set(stage, "u", np.array([70.0, 0.05]))
  solver.solve()
  return solver


def compare_iterates(actual: AcadosOcpSolver, reference: AcadosOcpSolver) -> None:
  """Check convergence, every state, control and multiplier iterate, and final cost."""
  assert actual.status == reference.status == 0
  assert actual.get_stats("sqp_iter") == reference.get_stats("sqp_iter")
  for iteration in range(int(actual.get_stats("sqp_iter")) + 1):
    left, right = [solver.get_iterate(iteration).flatten() for solver in (actual, reference)]
    for field in ("x", "u", "pi", "lam"):
      np.testing.assert_allclose(getattr(left, field), getattr(right, field), atol=1e-7, rtol=1e-7, err_msg=f"{field} at SQP iterate {iteration}")
  np.testing.assert_allclose(actual.get_cost(), reference.get_cost(), atol=1e-8, rtol=1e-8)


def main() -> None:
  """Solve both versions and compare all their iterates."""
  work = Path(__file__).resolve().parent / "generated" / "acados_race_car"
  print(f"Generated sources and builds: {work}")
  actual, reference = [solve_race_car(work, scaly=use_scaly) for use_scaly in (True, False)]
  compare_iterates(actual, reference)
  print(f"Both solvers converged in {actual.get_stats('sqp_iter')} SQP iterations, cost {actual.get_cost():.10g}. All iterates agree.")


if __name__ == "__main__":
  main()
