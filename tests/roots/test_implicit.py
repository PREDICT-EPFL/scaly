"""``custom_root``: the implicit-function derivative of a root found by a loop written by hand, at
the point the loop reached, never through its iterations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

import scaly as sc


def residual(z: sc.Expr, params: Sequence[sc.Expr]) -> sc.Expr:
  (p,) = params
  return z * z * z + z - p


@sc.function
def fixed_point_step(z: sc.Expr, p: sc.Expr) -> sc.Expr:
  return p / (1.0 + z * z)


@sc.function
def always(z: sc.Expr, p: sc.Expr) -> sc.Expr:
  return sc.greater(p.abs().sum() + 1.0, 0.0)


def crude_root(p: sc.Expr, iterations: int) -> sc.Expr:
  """``z^3 + z = p`` by fixed-point steps ``z <- p / (1 + z^2)`` in a ``while_loop``: not converged
  after a few."""
  z, _ = sc.while_loop(always, fixed_point_step, p * 0.0, max_iter=iterations, params=(p,))
  return z


P = np.array([0.3, -0.5, 0.8])


@pytest.mark.parametrize("iterations", [2, 60])
def test_the_rule_holds_at_the_point_found_whatever_the_iteration(iterations: int) -> None:
  root = sc.roots.custom_root(residual, sc.sym("z", 3), [sc.sym("p", 3)], name=f"cubic_root_{iterations}", names=["p"])
  p = sc.sym("p", 3)
  zstar = crude_root(p, iterations)
  z = root((p, zstar))
  fn = sc.Function.from_exprs(
    f"cubic_root_derivs_{iterations}",
    [p],
    [z, sc.jacobian(z, p), sc.gradient(z.sum(), p), sc.hessian((z * z).sum(), p)],
    ["p"],
    ["z", "J", "g", "H"],
  )
  value, jac, grad, hess = fn(P)
  slope = 1.0 / (3.0 * value**2 + 1.0)  # dz/dp for z^3 + z = p, at the z the loop reached
  np.testing.assert_allclose(value, sc.Function.from_exprs(f"crude_{iterations}", [p], [zstar], ["p"], ["z"])(P))
  np.testing.assert_allclose(jac, np.diag(slope), rtol=1e-13)
  np.testing.assert_allclose(grad, slope, rtol=1e-13)
  # d/dp (z^2) = 2 z z', and z'' = -6 z z'^3 by differentiating 1 / (3 z^2 + 1): the rules' second level.
  np.testing.assert_allclose(hess, np.diag(2.0 * slope**2 + 2.0 * value * (-6.0 * value * slope**3)), rtol=1e-12)
  if iterations == 60:
    np.testing.assert_allclose(value**3 + value, P, atol=1e-14)


def test_its_arguments_are_checked() -> None:
  with pytest.raises(ValueError, match="vector of unknowns"):
    sc.roots.custom_root(residual, sc.sym("z", (2, 2)), [sc.sym("p", 3)], name="bad_shape")
  with pytest.raises(ValueError, match="2 names for 1 parameters"):
    sc.roots.custom_root(residual, sc.sym("z", 3), [sc.sym("p", 3)], name="bad_names", names=["p", "q"])
