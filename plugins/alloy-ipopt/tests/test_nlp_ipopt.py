"""IPOPT plugin solver tests."""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.toolchain import solver_diagnostic, solver_loadable

need_ipopt = pytest.mark.skipif(not solver_loadable("ipopt"), reason=solver_diagnostic("ipopt"))


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
