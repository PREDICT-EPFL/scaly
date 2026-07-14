"""IPOPT plugin solver tests."""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.toolchain import solver_diagnostic, solver_loadable
from alloy_ipopt._ipopt import solve_ipopt

need_ipopt = pytest.mark.skipif(not solver_loadable("ipopt"), reason=solver_diagnostic("ipopt"))


@need_ipopt
def test_solve_ipopt_warm_start_and_stats() -> None:
  def solve(x0: np.ndarray, previous=None):
    return solve_ipopt(
      n=2,
      m=1,
      x0=x0,
      x_L=np.full(2, -5.0),
      x_U=np.full(2, 5.0),
      g_L=np.zeros(1),
      g_U=np.zeros(1),
      jac_rows=np.array([0, 0], dtype=np.int32),
      jac_cols=np.array([0, 1], dtype=np.int32),
      hess_rows=np.array([0, 1, 1], dtype=np.int32),
      hess_cols=np.array([0, 0, 1], dtype=np.int32),
      eval_f=lambda x: (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2,
      eval_grad_f=lambda x: np.array([2.0 * (x[0] - 1.0) - 400.0 * x[0] * (x[1] - x[0] ** 2), 200.0 * (x[1] - x[0] ** 2)]),
      eval_g=lambda x: np.array([x[0] + x[1] - 1.0]),
      eval_jac_g=lambda _x: np.ones(2),
      eval_h=lambda x, obj_factor, _lam: obj_factor * np.array([2.0 - 400.0 * x[1] + 1200.0 * x[0] ** 2, -400.0 * x[0], 200.0]),
      options={"print_level": 0, "sb": "yes", **({"warm_start_init_point": "yes"} if previous is not None else {})},
      lam_g0=None if previous is None else previous.mult_g,
      z_L0=None if previous is None else previous.mult_x_L,
      z_U0=None if previous is None else previous.mult_x_U,
    )

  cold = solve(np.array([-1.2, 2.2]))
  warm = solve(cold.x, cold)
  assert cold.status in (0, 1, 6) and warm.status in (0, 1, 6)
  assert cold.iters >= 1 and warm.iters >= 1 and warm.iters <= cold.iters
  assert all(count > 0 for count in cold.eval_counts.values())
  assert all(count > 0 for count in warm.eval_counts.values())


@need_ipopt
def test_nlp_solver_status_stats() -> None:
  x = al.sym("x", 2)
  nlp = al.nlp(x=x, f=(x[0] - 1) ** 2 + (x[1] - 2) ** 2, h_eq=al.stack([x[0] + x[1] - 1.0]), backend="python")
  nlp(np.array([2.0, -1.0]), np.zeros(1), np.zeros(0))
  assert nlp.last_status is not None and nlp.last_status.ok
  assert nlp.last_status.iter > 0
  assert nlp.last_status.stats is not None
  assert nlp.last_status.stats["iters"] == nlp.last_status.iter
  assert all(nlp.last_status.stats[name] > 0 for name in ("eval_f", "eval_grad_f", "eval_g", "eval_jac_g", "eval_h"))


@need_ipopt
def test_nlp_equality_constrained_quadratic() -> None:
  """min (x-1)^2 + (y-2)^2  s.t.  x + y == 1.

  Analytic optimum: x* = (0, 1), lam = 2.
  """
  x = al.sym("x", 2)
  f = (x[0] - 1) ** 2 + (x[1] - 2) ** 2
  h_eq = al.stack([x[0] + x[1] - 1.0], axis=0)
  nlp = al.nlp(x=x, f=f, h_eq=h_eq)
  out = nlp(np.array([0.5, 0.5]), np.zeros(1), np.zeros(0))
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
  out = nlp(np.array([0.0, 0.0]), np.zeros(0), np.zeros(0))
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
  out = nlp(x0, np.zeros(0), np.zeros(1))
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
      backend="python",
    )

  mapped_nlp, unrolled_nlp = build(True), build(False)
  x0 = np.array([0.2, 0.1, -0.3, 0.2])
  mapped_out = mapped_nlp(x0, np.zeros(2), np.zeros(0))
  unrolled_out = unrolled_nlp(x0, np.zeros(2), np.zeros(0))
  assert mapped_nlp.last_status is not None and mapped_nlp.last_status.ok
  assert unrolled_nlp.last_status is not None and unrolled_nlp.last_status.ok
  np.testing.assert_allclose(mapped_out["x"], target, atol=2e-6)
  np.testing.assert_allclose(mapped_out["x"], unrolled_out["x"], rtol=1e-7, atol=1e-7)
  np.testing.assert_allclose(mapped_out["f"], unrolled_out["f"], rtol=1e-8, atol=1e-10)
  # The exact-Hessian callback was actually exercised, and the handoff carries correct values:
  # this easy feasible problem could converge identically even with a broken Hessian.
  assert mapped_nlp.last_status.stats is not None and mapped_nlp.last_status.stats["eval_h"] > 0

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
    out = nlp(x0=np.array([0.0, 0.0]), lam_eq0=np.zeros(0), lam_ineq0=np.zeros(0), mu=mu_val)
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
  out = nlp(x0, np.zeros(1), np.zeros(0))
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
    out = nlp.call([al.const(np.array([1.0, 0.0])), al.const(np.zeros(1)), al.const(np.zeros(0)), target])
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
      target=target,
    )
    return {"x_proj": out[0]}

  np.testing.assert_allclose(proj(np.array([2.0, 0.0])), [1.0, 0.0], atol=1e-5)
  np.testing.assert_allclose(proj(np.array([0.0, 3.0])), [0.0, 1.0], atol=1e-5)
