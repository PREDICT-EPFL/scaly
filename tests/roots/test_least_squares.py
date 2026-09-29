"""``GaussNewton`` and ``LevenbergMarquardt`` against SciPy's ``least_squares``, and the fitted
parameters' derivative in the data through the stationarity condition, against finite differences."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize

import scaly as sc

T = np.linspace(0.0, 4.0, 25)
TRUE = np.array([2.5, 1.3, 0.4])


def model_np(z: np.ndarray, t: np.ndarray = T) -> np.ndarray:
  return z[0] * np.exp(-z[1] * t) + z[2]


DATA = model_np(TRUE) + 0.02 * np.random.default_rng(3).standard_normal(T.size)


@sc.roots.least_squares(vars=sc.L("z", 3), params=sc.L("y", T.size), name="decay_fit")
def DECAY(z: sc.Expr, y: sc.Expr) -> sc.Expr:
  return z[0] * (-z[1] * sc.const(T)).exp() + z[2] - y


def reference(y: np.ndarray = DATA) -> np.ndarray:
  return optimize.least_squares(lambda z: model_np(z) - y, np.array([1.0, 1.0, 0.0]), method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15).x


@pytest.mark.parametrize("method", ["gauss_newton", "levenberg_marquardt"])
def test_a_fit_matches_scipys_least_squares(method: str) -> None:
  z, info = sc.roots.solver(DECAY, method, name=f"decay_{method}")(np.array([1.0, 1.0, 0.0]), DATA)
  np.testing.assert_allclose(z, reference(), rtol=1e-9)
  assert sc.Status(int(info.status)) == sc.Status.OK and float(info.residual) < 1e-9


def test_levenberg_marquardt_is_the_default_and_starts_where_gauss_newton_cannot() -> None:
  @sc.roots.least_squares(vars=sc.L("x", 2), name="rosenbrock")
  def rosenbrock(x: sc.Expr) -> sc.Expr:
    return sc.stack([10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]])

  assert isinstance(sc.roots.REGISTRY.auto(rosenbrock), sc.roots.LevenbergMarquardt)
  x, info = sc.roots.solver(rosenbrock, name="rosenbrock_lm")(np.array([-1.2, 1.0]))
  np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-10)
  assert sc.Status(int(info.status)) == sc.Status.OK

  # Every point of x0 x1 = 1 fits: J has rank one everywhere, J^T J is singular, and Gauss-Newton's
  # Cholesky fails where Levenberg-Marquardt's damping carries it to the curve.
  @sc.roots.least_squares(vars=sc.L("x", 2), name="degenerate")
  def degenerate(x: sc.Expr) -> sc.Expr:
    return sc.stack([x[0] * x[1] - 1.0, 2.0 * x[0] * x[1] - 2.0])

  _, gn = sc.roots.solver(degenerate, "gauss_newton", name="degenerate_gn")(np.array([1.0, 0.0]))
  x, lm = sc.roots.solver(degenerate, "levenberg_marquardt", name="degenerate_lm")(np.array([1.0, 0.0]))
  assert sc.Status(int(gn.status)) == sc.Status.NUMERICS
  assert sc.Status(int(lm.status)) == sc.Status.OK
  np.testing.assert_allclose(x[0] * x[1], 1.0, atol=1e-10)


def test_the_fit_is_differentiable_in_the_data() -> None:
  # At the fit, J^T r = 0; with r = model(z) - y, dz/dy = H^{-1} J^T for the full Hessian H of
  # 1/2 |r|^2, which Gauss-Newton's J^T J approximates. (Finite differences of SciPy's fits are
  # limited by how far each converges.)
  y = sc.sym("y", T.size)
  z, _ = sc.roots.solver(DECAY, name="decay_diff")(sc.const(np.array([1.0, 1.0, 0.0])), y)
  jac = sc.Function.from_exprs("decay_fit_jac", [y], [sc.jacobian(z, y)], ["y"], ["J"])(DATA)
  a, b, c = reference()
  e = np.exp(-b * T)
  j = np.stack([e, -a * T * e, np.ones_like(T)], axis=1)
  r = a * e + c - DATA
  hess = j.T @ j
  hess[0, 1] += (r * -T * e).sum()
  hess[1, 0] += (r * -T * e).sum()
  hess[1, 1] += (r * a * T * T * e).sum()
  np.testing.assert_allclose(jac, np.linalg.solve(hess, j.T), atol=1e-8)
  assert np.abs(jac - np.linalg.solve(j.T @ j, j.T)).max() > 1e-3  # the second-order term matters here


def test_levenberg_marquardt_rejects_a_step_that_raises_the_cost() -> None:
  # r = atan(x) from x = 3: the Gauss-Newton step lands at -9.5, where the cost is higher, and
  # taking such steps diverges. Levenberg-Marquardt refuses them and raises its damping instead.
  @sc.roots.least_squares(vars=sc.L("x", 1), name="arctan")
  def arctan(x: sc.Expr) -> sc.Expr:
    return x.atan()

  # Gauss-Newton overflows to x = inf, where the gradient vanishes and a step test relative to |x|
  # passes: a solution that is not finite is never OK.
  x_gn, gn = sc.roots.solver(arctan, "gauss_newton", name="arctan_gn")(np.array([3.0]))
  x, lm = sc.roots.solver(arctan, "levenberg_marquardt", name="arctan_lm")(np.array([3.0]))
  assert not np.isfinite(x_gn[0]) and sc.Status(int(gn.status)) == sc.Status.NUMERICS
  assert sc.Status(int(lm.status)) == sc.Status.OK and abs(float(x[0])) < 1e-10
  # Its first trial step is that same leap to -9.5: refused, it leaves x where it was.
  x1, _ = sc.roots.solver(arctan, sc.roots.LevenbergMarquardt(max_iter=1), name="arctan_lm_once")(np.array([3.0]))
  assert float(x1[0]) == 3.0
