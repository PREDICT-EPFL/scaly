"""IPOPT plugin solver tests (generated C wrapper — the only solve path)."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.codegen.solver import SolverWrapperCtx
from scaly.ir.types import SparsityType
from scaly.solvers.model import ExternalOracle, SolverDescriptor, descriptor_function
from tests.solvers.problem_helpers import build_nlp, solve_nlp


def _wrapper_fixture(rows: tuple[int, ...], cols: tuple[int, ...]) -> sc.Function:
  """Build a descriptor small enough to inspect the IPOPT wrapper without compiling it."""
  base = ExternalOracle("base", "foreign_base_raw", "", (("x", (2,)),), (("f", ()),))
  grad = ExternalOracle("grad", "foreign_grad_raw", "", (("x", (2,)),), (("grad_f", (2,)),))
  hess = ExternalOracle(
    "hess",
    "foreign_hess_raw",
    "",
    (("x", (2,)), ("obj_factor", ())),
    (("hess_lag", (len(rows),)),),
  )
  bounds = ExternalOracle("bounds", "foreign_bounds_raw", "", (), (("x_lb", (2,)), ("x_ub", (2,))))
  descriptor = SolverDescriptor(
    name="ipopt_wrapper_fixture",
    backend="ipopt",
    n=2,
    n_eq=0,
    n_ineq=0,
    input_signature=(("x", (2,)), ("lam:x", (2,)), ("lam_eq", (0,)), ("lam_ineq", (0,))),
    output_signature=(("x", (2,)), ("lam:x", (2,)), ("lam_eq", (0,)), ("lam_ineq", (0,))),
    n_var_blocks=1,
    param_names=(),
    base=base,
    grad=grad,
    hess=hess,
    bounds=bounds,
    jac_sparsity=SparsityType.empty((0, 2)),
    hess_sparsity=SparsityType((2, 2), rows, cols),
  )
  return descriptor_function(descriptor)


def test_ipopt_wrapper_rejects_a_mixed_hessian_triangle() -> None:
  from scaly_ipopt.codegen import render_wrapper

  solver = _wrapper_fixture((0, 1), (1, 0))
  with pytest.raises(ValueError, match="exactly one triangle"):
    render_wrapper(solver, SolverWrapperCtx("ipopt_fixture", "ipopt_fixture_raw", "ipopt_fixture_stats"))


def test_ipopt_wrapper_writes_the_selected_hessian_directly() -> None:
  from scaly_ipopt.codegen import render_wrapper

  solver = _wrapper_fixture((0, 0, 1), (0, 1, 1))
  source = "\n".join(render_wrapper(solver, SolverWrapperCtx("ipopt_fixture", "ipopt_fixture_raw", "ipopt_fixture_stats")))
  assert "h_scratch" not in source
  assert "hess_lower_idx" not in source
  assert "foreign_hess_raw(x, obj_buf, values, ctx->w);" in source
  assert "ipopt_fixture_hess_rows[3] = { 0, 0, 1 }" in source
  assert "ipopt_fixture_hess_cols[3] = { 0, 1, 1 }" in source


@pytest.mark.parametrize(
  ("rows", "cols"),
  [
    ((1, 0), (0, 0)),
    ((0, 1), (0, 1)),
    ((), ()),
  ],
  ids=["lower", "diagonal", "empty"],
)
def test_ipopt_wrapper_accepts_lower_diagonal_and_empty_hessian_patterns(rows: tuple[int, ...], cols: tuple[int, ...]) -> None:
  from scaly_ipopt.codegen import render_wrapper

  solver = _wrapper_fixture(rows, cols)
  source = "\n".join(render_wrapper(solver, SolverWrapperCtx("ipopt_fixture", "ipopt_fixture_raw", "ipopt_fixture_stats")))
  if rows:
    assert "ipopt_fixture_hess_rows" in source
    assert "ipopt_fixture_hess_cols" in source
    assert "foreign_hess_raw(x, obj_buf, values, ctx->w);" in source
  else:
    assert "ipopt_fixture_hess_rows" not in source
    assert "ipopt_fixture_hess_cols" not in source
    assert "foreign_hess_raw" not in source


@pytest.mark.solver("ipopt")
def test_nlp_generated_stats_and_timing_split() -> None:
  x = sc.sym("x", 2)
  f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
  nlp = build_nlp(x=x, f=f, h_eq=sc.stack([x[0] + x[1] - 1.0]), name="nlp_stats_gen")
  out = solve_nlp(nlp, np.array([0.5, 0.5]), np.zeros(1), np.zeros(0), np.zeros(2))
  stats = nlp.solver_stats()
  assert stats is not None
  assert stats.version == sc.SCALY_SOLVER_STATS_VERSION
  assert stats.status == sc.ScalySolveStatus.OK
  assert stats.iter > 0
  assert stats.obj == pytest.approx(float(out["f"]), rel=1e-12, abs=1e-12)
  assert stats.n_eval_f > 0 and stats.n_eval_grad_f > 0 and stats.n_eval_g > 0
  assert stats.n_eval_jac_g > 0 and stats.n_eval_h > 0
  assert all(value >= 0.0 for value in (stats.t_total, stats.t_fe, stats.t_solver, stats.t_qp, stats.t_globalization, stats.t_glue))
  assert stats.t_qp == 0.0 and stats.t_globalization == 0.0
  assert stats.t_total == pytest.approx(stats.t_fe + stats.t_solver + stats.t_qp + stats.t_globalization + stats.t_glue, rel=0.1, abs=1e-12)
  # v3 diagnostics from the intermediate callback; merit penalty and QP
  # iteration count have no IPOPT equivalent and stay zero.
  assert 0.0 <= stats.primal_viol <= 1e-6
  assert stats.step_inf >= 0.0 and 0.0 < stats.alpha <= 1.0
  assert stats.backtracks >= 0
  assert stats.merit_penalty == 0.0 and stats.qp_iter == 0
  assert nlp.solver_stats().to_solver_status() is not None and nlp.solver_stats().to_solver_status().ok
  assert nlp.solver_stats().to_solver_status().iter == stats.iter


@pytest.mark.solver("ipopt")
def test_vendored_ipopt_can_coexist_with_casadi_ipopt() -> None:
  x = sc.sym("x", 1)
  solver = build_nlp(x=x, f=(x[0] - 1.0) ** 2, solver="ipopt", name="ipopt_namespace")
  out = solve_nlp(solver, np.zeros(1), np.zeros(0), np.zeros(0), np.zeros(1))
  assert out["x"][0] == pytest.approx(1.0)

  import casadi as ca

  cx = ca.MX.sym("x")
  casadi_solver = ca.nlpsol("casadi_namespace", "ipopt", {"x": cx, "f": (cx - 2.0) ** 2}, {"ipopt.print_level": 0, "print_time": False})
  casadi_out = casadi_solver(x0=0.0)
  assert float(casadi_out["x"]) == pytest.approx(2.0)


@pytest.mark.solver("ipopt")
def test_nlp_generated_warm_start_reduces_iterations() -> None:
  """Seeding x0 + lam_eq0 + lam_box0 from a previous solve (with
  warm_start_init_point) must converge in fewer iterations than cold."""

  def build(name: str, warm: bool) -> sc.Function:
    x = sc.sym("x", 2)
    f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    options: dict[str, str | int | float] = {"warm_start_init_point": "yes"} if warm else {}
    return build_nlp(x=x, f=f, h_eq=sc.stack([x[0] + x[1] - 1.0]), x_lb=np.full(2, -5.0), x_ub=np.full(2, 5.0), name=name, options=options)

  cold = build("nlp_ws_cold", warm=False)
  warm = build("nlp_ws_warm", warm=True)
  cold_out = solve_nlp(cold, np.array([-1.2, 2.2]), np.zeros(1), np.zeros(0), np.zeros(2))
  assert cold.solver_stats() is not None and cold.solver_stats().status == sc.ScalySolveStatus.OK
  solve_nlp(warm, cold_out["x"], cold_out["lam_eq"], np.zeros(0), cold_out["lam_box"])
  assert warm.solver_stats() is not None and warm.solver_stats().status == sc.ScalySolveStatus.OK
  assert cold.solver_stats().iter >= 1
  assert warm.solver_stats().iter < cold.solver_stats().iter


@pytest.mark.solver("ipopt")
def test_nlp_generated_status_max_iter() -> None:
  x = sc.sym("x", 2)
  f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
  nlp = build_nlp(x=x, f=f, h_eq=sc.stack([x[0] + x[1] - 1.0]), name="nlp_max_iter", options={"max_iter": 1})
  solve_nlp(nlp, np.array([-1.2, 2.2]), np.zeros(1), np.zeros(0), np.zeros(2))
  assert nlp.solver_stats() is not None
  assert nlp.solver_stats().status == sc.ScalySolveStatus.MAX_ITER
  assert nlp.solver_stats().native_status == -1
  assert nlp.solver_stats().iter == 1
  assert nlp.solver_stats().to_solver_status() is not None and not nlp.solver_stats().to_solver_status().ok


@pytest.mark.solver("ipopt")
def test_nlp_generated_rejected_option_reports_error_status() -> None:
  """The generated wrapper checks every AddIpopt*Option return and surfaces
  SCALY_SOLVE_ERROR with Invalid_Option (-12) as the native status and
  defined outputs."""
  x = sc.sym("x", 2)
  nlp = build_nlp(
    x=x, f=(x[0] - 1) ** 2 + (x[1] - 2) ** 2, h_eq=sc.stack([x[0] + x[1] - 1.0]), name="nlp_bad_option", options={"definitely_not_an_ipopt_option": 3}
  )
  x0 = np.array([0.5, -0.5])
  out = solve_nlp(nlp, x0, np.zeros(1), np.zeros(0), np.zeros(2))
  assert nlp.solver_stats() is not None
  assert nlp.solver_stats().status == sc.ScalySolveStatus.ERROR
  assert nlp.solver_stats().native_status == -12
  assert nlp.solver_stats().iter == 0
  np.testing.assert_allclose(out["x"], x0)
  assert nlp.solver_stats().to_solver_status() is not None and not nlp.solver_stats().to_solver_status().ok


@pytest.mark.solver("ipopt")
def test_nlp_equality_constrained_quadratic() -> None:
  """min (x-1)^2 + (y-2)^2  s.t.  x + y == 1.

  Analytic optimum: x* = (0, 1), lam = 2.
  """
  x = sc.sym("x", 2)
  f = (x[0] - 1) ** 2 + (x[1] - 2) ** 2
  h_eq = sc.stack([x[0] + x[1] - 1.0], axis=0)
  nlp = build_nlp(x=x, f=f, h_eq=h_eq)
  out = solve_nlp(nlp, np.array([0.5, 0.5]), np.zeros(1), np.zeros(0), np.zeros(2))
  assert nlp.solver_stats().to_solver_status() is not None and nlp.solver_stats().to_solver_status().ok
  np.testing.assert_allclose(out["x"], [0.0, 1.0], atol=1e-6)
  np.testing.assert_allclose(out["f"], 2.0, atol=1e-6)
  np.testing.assert_allclose(out["lam_eq"], [2.0], atol=1e-6)


@pytest.mark.solver("ipopt")
def test_nlp_box_only_quadratic() -> None:
  """Unconstrained convex objective + box bound that becomes active.

  min (x[0] - 0.5)^2 + (x[1] + 2)^2  s.t.  x[1] >= -1.
  Expected x* = (0.5, -1), lam_box[1] active (>0).
  """
  x = sc.sym("x", 2)
  f = (x[0] - 0.5) ** 2 + (x[1] + 2) ** 2
  nlp = build_nlp(x=x, f=f, x_lb=np.array([-np.inf, -1.0]))
  out = solve_nlp(nlp, np.array([0.0, 0.0]), np.zeros(0), np.zeros(0), np.zeros(2))
  assert nlp.solver_stats().to_solver_status() is not None and nlp.solver_stats().to_solver_status().ok
  np.testing.assert_allclose(out["x"], [0.5, -1.0], atol=1e-6)
  # lam_box = mult_x_U - mult_x_L: negative <=> lower bound active.
  assert out["lam_box"][1] < 0


@pytest.mark.solver("ipopt")
def test_nlp_two_sided_inequality_and_lagrangian_hessian() -> None:
  """min x[0]^2 + 0.5 x[1]^2 + x[0] x[1]   s.t.   0 <= x[0]^2 + x[1] <= 5.

  This exercises:
    - a nonlinear two-sided general inequality,
    - a non-diagonal Lagrangian Hessian (off-diagonal coupling x[0] x[1]),
    - the sparse Jacobian path (g has cross-term 2 x[0]).

  The optimum is the unconstrained minimizer x=(0,0), which lies on the lower
  nonlinear inequality boundary. This keeps the test independent from a second
  IPOPT implementation loaded from CasADi in the same process.
  """
  x = sc.sym("x", 2)
  f = x[0] ** 2 + 0.5 * x[1] ** 2 + x[0] * x[1]
  g = sc.stack([x[0] ** 2 + x[1]], axis=0)
  nlp = build_nlp(x=x, f=f, g_ineq=g, l_ineq=np.array([0.0]), u_ineq=np.array([5.0]))
  x0 = np.array([1.0, -0.5])
  out = solve_nlp(nlp, x0, np.zeros(0), np.zeros(1), np.zeros(2))
  assert nlp.solver_stats().to_solver_status() is not None and nlp.solver_stats().to_solver_status().ok

  np.testing.assert_allclose(out["x"], [0.0, 0.0], atol=2e-4)
  np.testing.assert_allclose(out["g_ineq"], [0.0], atol=2e-4)


@pytest.mark.solver("ipopt")
def test_nlp_mapped_constraints_exact_hessian_matches_unrolled(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  piece_x = sc.sym("piece_x", 2)
  piece = sc.Function._from_exprs("nlp_mapped_constraint_piece", [piece_x], [sc.stack([piece_x[1] - piece_x[0] ** 2])], ["piece_x"], ["h"])
  target = np.array([0.5, 0.25, -0.7, 0.49])

  def build(mapped: bool):
    x = sc.sym("x", 4)
    if mapped:
      h_eq = sc.vmap(piece, 2, [(x, 0, 2)])
    else:
      h_eq = sc.concat([piece(x[2 * it : 2 * (it + 1)]) for it in range(2)])
    return build_nlp(
      x=x,
      f=((x - target) ** 2).sum(),
      h_eq=h_eq,
      name=f"nlp_{'mapped' if mapped else 'unrolled'}_constraint",
    )

  mapped_nlp, unrolled_nlp = build(True), build(False)
  x0 = np.array([0.2, 0.1, -0.3, 0.2])
  mapped_out = solve_nlp(mapped_nlp, x0, np.zeros(2), np.zeros(0), np.zeros(4))
  unrolled_out = solve_nlp(unrolled_nlp, x0, np.zeros(2), np.zeros(0), np.zeros(4))
  assert mapped_nlp.solver_stats().to_solver_status() is not None and mapped_nlp.solver_stats().to_solver_status().ok
  assert unrolled_nlp.solver_stats().to_solver_status() is not None and unrolled_nlp.solver_stats().to_solver_status().ok
  np.testing.assert_allclose(mapped_out["x"], target, atol=2e-6)
  np.testing.assert_allclose(mapped_out["x"], unrolled_out["x"], rtol=1e-7, atol=1e-7)
  np.testing.assert_allclose(mapped_out["f"], unrolled_out["f"], rtol=1e-8, atol=1e-10)
  # The exact-Hessian callback was actually exercised, and the handoff carries correct values:
  # this easy feasible problem could converge identically even with a broken Hessian.
  assert mapped_nlp.solver_stats() is not None and mapped_nlp.solver_stats().n_eval_h > 0

  def hess_dense(mapped: bool, xv: np.ndarray, lam: np.ndarray) -> np.ndarray:
    x = sc.sym("x", 4)
    h_eq = sc.vmap(piece, 2, [(x, 0, 2)]) if mapped else sc.concat([piece(x[2 * it : 2 * (it + 1)]) for it in range(2)])
    base = sc.Function._from_exprs(f"nlp_hess_base_{int(mapped)}", [x], [((x - target) ** 2).sum(), h_eq], ["x"], ["f", "g"])
    shf = sc.sparse_lagrangian_hessian(base, "x")
    sp = shf.output_sparsities[0]
    assert sp is not None
    dense = np.zeros(sp.shape)
    dense[np.asarray(sp.rows), np.asarray(sp.cols)] = np.asarray(shf((xv, (np.array(1.0), lam))))
    return dense

  lam = np.array([0.8, -1.7])
  np.testing.assert_allclose(hess_dense(True, mapped_out["x"], lam), hess_dense(False, mapped_out["x"], lam), rtol=1e-10, atol=1e-12)


@pytest.mark.solver("ipopt")
def test_nlp_with_symbolic_parameter() -> None:
  """Parameter-aware NLP: solve min (x - mu)^2 across different ``mu`` values."""
  x = sc.sym("x", 2)
  mu = sc.sym("mu", 2)
  f = ((x - mu) * (x - mu)).sum()
  nlp = build_nlp(x=x, f=f, p=mu)
  for mu_val in [np.zeros(2), np.array([1.5, -2.3])]:
    out = solve_nlp(nlp, np.array([0.0, 0.0]), np.zeros(0), np.zeros(0), np.zeros(2), mu_val)
    assert nlp.solver_stats().to_solver_status() is not None and nlp.solver_stats().to_solver_status().ok
    np.testing.assert_allclose(out["x"], mu_val, atol=1e-6)


@pytest.mark.solver("ipopt")
def test_nlp_rosenbrock_equality_constrained() -> None:
  """Classic Rosenbrock, equality-constrained.

  min (1 - x)^2 + 100 (y - x^2)^2  s.t.  x + y == 1.
  """
  x_sym = sc.sym("x", 2)
  f = (1 - x_sym[0]) ** 2 + 100 * (x_sym[1] - x_sym[0] ** 2) ** 2
  h_eq = sc.stack([x_sym[0] + x_sym[1] - 1.0], axis=0)
  nlp = build_nlp(x=x_sym, f=f, h_eq=h_eq)
  x0 = np.array([0.5, 0.5])
  out = solve_nlp(nlp, x0, np.zeros(1), np.zeros(0), np.zeros(2))
  assert nlp.solver_stats().to_solver_status() is not None and nlp.solver_stats().to_solver_status().ok

  # Substitute y=1-x. The stationary points solve
  #   400*x^3 + 600*x^2 - 198*x - 202 = 0
  # and the minimizer is the middle real root.
  np.testing.assert_allclose(out["x"], [0.6187956190750259, 0.3812043809249741], atol=1e-5)


@pytest.mark.solver("ipopt")
def test_nested_nlp_in_scaly_function() -> None:
  """NLP solver embedded in a larger Function."""

  @sc.function(sc.L("target", (2,)), output=sc.L("x_proj", ...), name="min_dist_to_unit_circle")
  def proj(target):
    x = sc.sym("x_inner", 2)
    f = (x[0] - target[0]) ** 2 + (x[1] - target[1]) ** 2
    h_eq = sc.stack([x[0] ** 2 + x[1] ** 2 - 1.0], axis=0)
    nlp = build_nlp(x=x, f=f, p=target, h_eq=h_eq)
    out = nlp.symbolic_call((sc.const(np.array([1.0, 0.0])), sc.const(np.zeros(2)), sc.const(np.zeros(1)), sc.const(np.zeros(0)), target))
    return out[0]

  # Projection of (2, 0) onto the unit circle = (1, 0).
  x_proj = proj(np.array([2.0, 0.0]))
  np.testing.assert_allclose(x_proj, [1.0, 0.0], atol=1e-5)
  x_proj = proj(np.array([0.0, 3.0]))
  np.testing.assert_allclose(x_proj, [0.0, 1.0], atol=1e-5)


@pytest.mark.solver("ipopt")
def test_nested_nlp_jit_compiles_through_ipopt() -> None:
  """JIT path for an NLP: projects (target) onto the unit circle."""

  @sc.function(sc.L("target", (2,)), output=sc.L("x_proj", ...), name="proj_circle")
  def proj(target):
    x = sc.sym("x_inner", 2)
    f = (x[0] - target[0]) ** 2 + (x[1] - target[1]) ** 2
    h_eq = sc.stack([x[0] ** 2 + x[1] ** 2 - 1.0], axis=0)
    nlp = build_nlp(x=x, f=f, p=target, h_eq=h_eq)
    out = nlp.symbolic_call((sc.const(np.array([1.0, 0.0])), sc.const(np.zeros(2)), sc.const(np.zeros(1)), sc.const(np.zeros(0)), target))
    return out[0]

  np.testing.assert_allclose(proj(np.array([2.0, 0.0])), [1.0, 0.0], atol=1e-5)
  np.testing.assert_allclose(proj(np.array([0.0, 3.0])), [0.0, 1.0], atol=1e-5)
