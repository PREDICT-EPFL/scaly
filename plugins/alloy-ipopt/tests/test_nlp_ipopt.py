"""IPOPT plugin solver tests (generated C wrapper — the only solve path)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.toolchain import solver_diagnostic, solver_loadable

need_ipopt = pytest.mark.skipif(not solver_loadable("ipopt"), reason=solver_diagnostic("ipopt"))


@need_ipopt
def test_nlp_generated_stats_and_timing_split() -> None:
  x = al.sym("x", 2)
  f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
  nlp = al.nlp(x=x, f=f, h_eq=al.stack([x[0] + x[1] - 1.0]), name="nlp_stats_gen")
  out = nlp(np.array([0.5, 0.5]), np.zeros(1), np.zeros(0), np.zeros(2))
  stats = nlp.last_stats
  assert stats is not None
  assert stats.version == al.ALLOY_SOLVER_STATS_VERSION
  assert stats.status == al.AlloySolveStatus.OK
  assert stats.iter > 0
  assert stats.obj == pytest.approx(float(out["f"]), rel=1e-12, abs=1e-12)
  assert stats.n_eval_f > 0 and stats.n_eval_grad_f > 0 and stats.n_eval_g > 0
  assert stats.n_eval_jac_g > 0 and stats.n_eval_h > 0
  assert all(value >= 0.0 for value in (stats.t_total, stats.t_fe, stats.t_solver, stats.t_glue))
  assert stats.t_total == pytest.approx(stats.t_fe + stats.t_solver + stats.t_glue, rel=0.1, abs=1e-12)
  assert nlp.last_status is not None and nlp.last_status.ok
  assert nlp.last_status.iter == stats.iter


@need_ipopt
def test_nlp_generated_warm_start_reduces_iterations() -> None:
  """Seeding x0 + lam_eq0 + lam_box0 from a previous solve (with
  warm_start_init_point) must converge in fewer iterations than cold."""

  def build(name: str, warm: bool) -> al.SolverFunction:
    x = al.sym("x", 2)
    f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    options: dict[str, str | int | float] = {"warm_start_init_point": "yes"} if warm else {}
    return al.nlp(x=x, f=f, h_eq=al.stack([x[0] + x[1] - 1.0]), x_lb=np.full(2, -5.0), x_ub=np.full(2, 5.0), name=name, options=options)

  cold = build("nlp_ws_cold", warm=False)
  warm = build("nlp_ws_warm", warm=True)
  cold_out = cold(np.array([-1.2, 2.2]), np.zeros(1), np.zeros(0), np.zeros(2))
  assert cold.last_stats is not None and cold.last_stats.status == al.AlloySolveStatus.OK
  warm(cold_out["x"], cold_out["lam_eq"], np.zeros(0), cold_out["lam_box"])
  assert warm.last_stats is not None and warm.last_stats.status == al.AlloySolveStatus.OK
  assert cold.last_stats.iter >= 1
  assert warm.last_stats.iter < cold.last_stats.iter


@need_ipopt
def test_nlp_generated_status_max_iter() -> None:
  x = al.sym("x", 2)
  f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
  nlp = al.nlp(x=x, f=f, h_eq=al.stack([x[0] + x[1] - 1.0]), name="nlp_max_iter", options={"max_iter": 1})
  nlp(np.array([-1.2, 2.2]), np.zeros(1), np.zeros(0), np.zeros(2))
  assert nlp.last_stats is not None
  assert nlp.last_stats.status == al.AlloySolveStatus.MAX_ITER
  assert nlp.last_stats.native_status == -1
  assert nlp.last_stats.iter == 1
  assert nlp.last_status is not None and not nlp.last_status.ok


@need_ipopt
def test_nlp_generated_rejected_option_reports_error_status() -> None:
  """The generated wrapper checks every AddIpopt*Option return and surfaces
  ALLOY_SOLVE_ERROR with Invalid_Option (-12) as the native status and
  defined outputs."""
  x = al.sym("x", 2)
  nlp = al.nlp(
    x=x, f=(x[0] - 1) ** 2 + (x[1] - 2) ** 2, h_eq=al.stack([x[0] + x[1] - 1.0]), name="nlp_bad_option", options={"definitely_not_an_ipopt_option": 3}
  )
  x0 = np.array([0.5, -0.5])
  out = nlp(x0, np.zeros(1), np.zeros(0), np.zeros(2))
  assert nlp.last_stats is not None
  assert nlp.last_stats.status == al.AlloySolveStatus.ERROR
  assert nlp.last_stats.native_status == -12
  assert nlp.last_stats.iter == 0
  np.testing.assert_allclose(out["x"], x0)
  np.testing.assert_allclose(out["f"], 0.0)
  assert nlp.last_status is not None and not nlp.last_status.ok


@need_ipopt
def test_nlp_equality_constrained_quadratic() -> None:
  """min (x-1)^2 + (y-2)^2  s.t.  x + y == 1.

  Analytic optimum: x* = (0, 1), lam = 2.
  """
  x = al.sym("x", 2)
  f = (x[0] - 1) ** 2 + (x[1] - 2) ** 2
  h_eq = al.stack([x[0] + x[1] - 1.0], axis=0)
  nlp = al.nlp(x=x, f=f, h_eq=h_eq)
  out = nlp(np.array([0.5, 0.5]), np.zeros(1), np.zeros(0), np.zeros(2))
  assert nlp.last_status is not None and nlp.last_status.ok
  np.testing.assert_allclose(out["x"], [0.0, 1.0], atol=1e-6)
  np.testing.assert_allclose(out["f"], 2.0, atol=1e-6)
  np.testing.assert_allclose(out["lam_eq"], [2.0], atol=1e-6)


@need_ipopt
def test_nlp_box_only_quadratic() -> None:
  """Unconstrained convex objective + box bound that becomes active.

  min (x[0] - 0.5)^2 + (x[1] + 2)^2  s.t.  x[1] >= -1.
  Expected x* = (0.5, -1), lam_box[1] active (>0).
  """
  x = al.sym("x", 2)
  f = (x[0] - 0.5) ** 2 + (x[1] + 2) ** 2
  nlp = al.nlp(x=x, f=f, x_lb=np.array([-np.inf, -1.0]))
  out = nlp(np.array([0.0, 0.0]), np.zeros(0), np.zeros(0), np.zeros(2))
  assert nlp.last_status is not None and nlp.last_status.ok
  np.testing.assert_allclose(out["x"], [0.5, -1.0], atol=1e-6)
  # lam_box = mult_x_U - mult_x_L: negative <=> lower bound active.
  assert out["lam_box"][1] < 0


@need_ipopt
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
  x = al.sym("x", 2)
  f = x[0] ** 2 + 0.5 * x[1] ** 2 + x[0] * x[1]
  g = al.stack([x[0] ** 2 + x[1]], axis=0)
  nlp = al.nlp(x=x, f=f, g_ineq=g, l_ineq=np.array([0.0]), u_ineq=np.array([5.0]))
  x0 = np.array([1.0, -0.5])
  out = nlp(x0, np.zeros(0), np.zeros(1), np.zeros(2))
  assert nlp.last_status is not None and nlp.last_status.ok

  np.testing.assert_allclose(out["x"], [0.0, 0.0], atol=2e-4)
  np.testing.assert_allclose(out["g_ineq"], [0.0], atol=2e-4)


@need_ipopt
def test_nlp_mapped_constraints_exact_hessian_matches_unrolled(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("ALLOY_STRICT_JVP_MANY", "1")
  piece_x = al.sym("piece_x", 2)
  piece = al.Function("nlp_mapped_constraint_piece", [piece_x], [al.stack([piece_x[1] - piece_x[0] ** 2])], ["piece_x"], ["h"])
  target = np.array([0.5, 0.25, -0.7, 0.49])

  def build(mapped: bool):
    x = al.sym("x", 4)
    if mapped:
      h_eq = al.map_(piece, 2, [(x, 0, 2)])
    else:
      h_eq = al.concat([piece.call([x[2 * it : 2 * (it + 1)]])[0] for it in range(2)])
    return al.nlp(
      x=x,
      f=((x - target) ** 2).sum(),
      h_eq=h_eq,
      name=f"nlp_{'mapped' if mapped else 'unrolled'}_constraint",
    )

  mapped_nlp, unrolled_nlp = build(True), build(False)
  x0 = np.array([0.2, 0.1, -0.3, 0.2])
  mapped_out = mapped_nlp(x0, np.zeros(2), np.zeros(0), np.zeros(4))
  unrolled_out = unrolled_nlp(x0, np.zeros(2), np.zeros(0), np.zeros(4))
  assert mapped_nlp.last_status is not None and mapped_nlp.last_status.ok
  assert unrolled_nlp.last_status is not None and unrolled_nlp.last_status.ok
  np.testing.assert_allclose(mapped_out["x"], target, atol=2e-6)
  np.testing.assert_allclose(mapped_out["x"], unrolled_out["x"], rtol=1e-7, atol=1e-7)
  np.testing.assert_allclose(mapped_out["f"], unrolled_out["f"], rtol=1e-8, atol=1e-10)
  # The exact-Hessian callback was actually exercised, and the handoff carries correct values:
  # this easy feasible problem could converge identically even with a broken Hessian.
  assert mapped_nlp.last_stats is not None and mapped_nlp.last_stats.n_eval_h > 0

  def hess_dense(mapped: bool, xv: np.ndarray, lam: np.ndarray) -> np.ndarray:
    x = al.sym("x", 4)
    h_eq = al.map_(piece, 2, [(x, 0, 2)]) if mapped else al.concat([piece.call([x[2 * it : 2 * (it + 1)]])[0] for it in range(2)])
    base = al.Function(f"nlp_hess_base_{int(mapped)}", [x], [((x - target) ** 2).sum(), h_eq], ["x"], ["f", "g"])
    shf = al.sparse_lagrangian_hessian(base, "x", ["f", "g"])
    sp = shf.output_sparsities[0]
    assert sp is not None
    dense = np.zeros(sp.shape)
    dense[np.asarray(sp.rows), np.asarray(sp.cols)] = np.asarray(shf(xv, np.array(1.0), lam))
    return dense

  lam = np.array([0.8, -1.7])
  np.testing.assert_allclose(hess_dense(True, mapped_out["x"], lam), hess_dense(False, mapped_out["x"], lam), rtol=1e-10, atol=1e-12)


@need_ipopt
def test_nlp_with_symbolic_parameter() -> None:
  """Parameter-aware NLP: solve min (x - mu)^2 across different ``mu`` values."""
  x = al.sym("x", 2)
  mu = al.sym("mu", 2)
  f = ((x - mu) * (x - mu)).sum()
  nlp = al.nlp(x=x, f=f, p=mu)
  for mu_val in [np.zeros(2), np.array([1.5, -2.3])]:
    out = nlp(x0=np.array([0.0, 0.0]), lam_eq0=np.zeros(0), lam_ineq0=np.zeros(0), lam_box0=np.zeros(2), mu=mu_val)
    assert nlp.last_status is not None and nlp.last_status.ok
    np.testing.assert_allclose(out["x"], mu_val, atol=1e-6)


@need_ipopt
def test_nlp_rosenbrock_equality_constrained() -> None:
  """Classic Rosenbrock, equality-constrained.

  min (1 - x)^2 + 100 (y - x^2)^2  s.t.  x + y == 1.
  """
  x_sym = al.sym("x", 2)
  f = (1 - x_sym[0]) ** 2 + 100 * (x_sym[1] - x_sym[0] ** 2) ** 2
  h_eq = al.stack([x_sym[0] + x_sym[1] - 1.0], axis=0)
  nlp = al.nlp(x=x_sym, f=f, h_eq=h_eq)
  x0 = np.array([0.5, 0.5])
  out = nlp(x0, np.zeros(1), np.zeros(0), np.zeros(2))
  assert nlp.last_status is not None and nlp.last_status.ok

  # Substitute y=1-x. The stationary points solve
  #   400*x^3 + 600*x^2 - 198*x - 202 = 0
  # and the minimizer is the middle real root.
  np.testing.assert_allclose(out["x"], [0.6187956190750259, 0.3812043809249741], atol=1e-5)


@need_ipopt
def test_nested_nlp_in_alloy_function() -> None:
  """NLP solver embedded in a larger Function."""

  @al.function("min_dist_to_unit_circle", {"target": (2,)})
  def proj(target):
    x = al.sym("x_inner", 2)
    f = (x[0] - target[0]) ** 2 + (x[1] - target[1]) ** 2
    h_eq = al.stack([x[0] ** 2 + x[1] ** 2 - 1.0], axis=0)
    nlp = al.nlp(x=x, f=f, p=target, h_eq=h_eq)
    out = nlp.call([al.const(np.array([1.0, 0.0])), al.const(np.zeros(1)), al.const(np.zeros(0)), al.const(np.zeros(2)), target])
    return {"x_proj": out[0]}

  # Projection of (2, 0) onto the unit circle = (1, 0).
  x_proj = proj(np.array([2.0, 0.0]))
  np.testing.assert_allclose(x_proj, [1.0, 0.0], atol=1e-5)
  x_proj = proj(np.array([0.0, 3.0]))
  np.testing.assert_allclose(x_proj, [0.0, 1.0], atol=1e-5)


@need_ipopt
def test_nested_nlp_jit_compiles_through_ipopt() -> None:
  """JIT path for an NLP: projects (target) onto the unit circle."""

  @al.function("proj_circle", {"target": (2,)})
  def proj(target):
    x = al.sym("x_inner", 2)
    f = (x[0] - target[0]) ** 2 + (x[1] - target[1]) ** 2
    h_eq = al.stack([x[0] ** 2 + x[1] ** 2 - 1.0], axis=0)
    nlp = al.nlp(x=x, f=f, p=target, h_eq=h_eq)
    out = nlp.call(
      x0=al.const(np.array([1.0, 0.0])),
      lam_eq0=al.const(np.zeros(1)),
      lam_ineq0=al.const(np.zeros(0)),
      lam_box0=al.const(np.zeros(2)),
      target=target,
    )
    return {"x_proj": out[0]}

  np.testing.assert_allclose(proj(np.array([2.0, 0.0])), [1.0, 0.0], atol=1e-5)
  np.testing.assert_allclose(proj(np.array([0.0, 3.0])), [0.0, 1.0], atol=1e-5)
