"""``root`` and ``least_squares``: what they accept, what they refuse, and the solver Function's
signature over trees of unknowns and parameters."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc


def test_a_tree_of_unknowns_and_parameters_round_trips() -> None:
  @sc.roots.root(vars=sc.G(sc.L("a", 2), sc.L("b", (2, 2))), params=sc.G(sc.L("s", ()), sc.L("t", 2)), name="tree")
  def tree(z: tuple[sc.Expr, sc.Expr], p: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr]:
    a, b = z
    s, t = p
    return a - s * t, b @ b - sc.const(np.eye(2)) * (1.0 + a[0] * a[0])

  assert (tree.n, tree.m) == (6, 6)
  solve = sc.roots.solver(tree, name="tree_newton")
  assert solve.input_names == ("a", "b", "s", "t")
  assert solve.output_names == ("a", "b", "info:status", "info:iter", "info:residual")
  (a, b), info = solve((np.ones(2), np.eye(2) * 1.2), (np.array(2.0), np.array([0.5, -1.0])))
  np.testing.assert_allclose(a, [1.0, -2.0], atol=1e-12)
  np.testing.assert_allclose(b @ b, 2.0 * np.eye(2), atol=1e-10)
  assert sc.Status(int(info.status)) == sc.Status.OK


def test_without_parameters_the_solver_takes_the_warm_start_alone() -> None:
  @sc.roots.root(vars=sc.L("x", ()), name="golden")
  def golden(x: sc.Expr) -> sc.Expr:
    return x * x - x - 1.0

  x, _ = sc.roots.solver(golden, name="golden_newton")(np.array(2.0))
  np.testing.assert_allclose(x, (1 + 5**0.5) / 2, rtol=1e-12)


def test_what_a_problem_refuses() -> None:
  with pytest.raises(ValueError, match="3 equations in 2 unknowns"):
    sc.roots.root(vars=sc.L("z", 2), name="tall")(lambda z: sc.concat([z, z[:1]]))
  with pytest.raises(ValueError, match="1 residuals in 2 unknowns"):
    sc.roots.least_squares(vars=sc.L("z", 2), name="wide")(lambda z: z.sum().reshape((1,)))
  with pytest.raises(ValueError, match=r"reads symbols it does not declare: \['q'\]"):
    sc.roots.root(vars=sc.L("z", 1), name="undeclared")(lambda z: z - sc.sym("q", 1))
  with pytest.raises(ValueError, match="bounds must not depend on the unknowns"):
    sc.roots.root(vars=sc.L("z", ()), name="moving")(lambda z: sc.roots.RootSpec(z, lb=z - 1.0, ub=sc.const(1.0)))
  with pytest.raises(TypeError, match="least squares takes no bounds"):
    sc.roots.least_squares(vars=sc.L("z", ()), name="bounded_fit")(lambda z: sc.roots.RootSpec(z, lb=sc.const(0.0)))
  fit = sc.roots.least_squares(vars=sc.L("z", ()), name="fit")(lambda z: sc.stack([z - 1.0, z + 1.0]))
  with pytest.raises(TypeError, match="roots.newton solves Root, not LeastSquares"):
    sc.roots.solver(fit, sc.roots.Newton())
