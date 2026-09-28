"""A product or quotient with a constant: its tangent leaves out the constant's zero term, so a
linear map stays provably affine behind any depth of Function calls, and PIQP accepts it."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import scaly as sc
from scaly.ad import jacobian_sparsity, jvp, jvp_many
from scaly.passes.expr import simplify_cse_fixpoint

C = np.array([2.0, -3.0, 0.5])
M = np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 4.0], [3.0, 0.0, 1.0]])

LINEAR: dict[str, tuple[Callable[[sc.Expr], sc.Expr], np.ndarray]] = {
  "scalar_times": (lambda u: 2.0 * u, 2.0 * np.eye(3)),
  "times_const": (lambda u: u * sc.const(C), np.diag(C)),
  "over_const": (lambda u: u / sc.const(C), np.diag(1.0 / C)),
  "matmul": (lambda u: sc.const(M) @ u, M),
}
"""Linear maps of ``u`` in R^3, each with its Jacobian."""


def _wrap(inner: sc.Function, name: str) -> sc.Function:
  @sc.function(3, output="y", name=name)
  def wrapper(u):
    return inner(u)

  return wrapper


def _nested(case: str, depth: int) -> sc.Function:
  """The linear map ``case`` called through ``depth`` levels of Functions."""
  fn = sc.function(3, output="y", name=f"zero_tangent_{case}_0")(LINEAR[case][0])
  for level in range(1, depth):
    fn = _wrap(fn, f"zero_tangent_{case}_{level}")
  return fn


@pytest.mark.parametrize("case", LINEAR)
def test_the_tangent_of_a_linear_map_reads_only_its_seed(case: str) -> None:
  # Unsimplified, as a call's derivative helper keeps its body: a ``0 * u`` term would read ``u``.
  u, seed = sc.sym("u", 3), sc.sym("du", 3)
  y = LINEAR[case][0](u)
  assert not jacobian_sparsity(jvp(y, u, seed), u).rows
  assert not jacobian_sparsity(jvp_many(y, u, sc.const(np.eye(3))), u).rows


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("case", LINEAR)
def test_a_linear_map_behind_calls_has_a_constant_jacobian(case: str, depth: int) -> None:
  fn = _nested(case, depth)
  u = sc.sym("u", 3)
  derivative = simplify_cse_fixpoint(sc.jacobian(fn(u), u))  # as the QP proof forms it
  assert not jacobian_sparsity(derivative, u).rows
  np.testing.assert_allclose(sc.jacobian(fn, "y", "u")(np.array([0.3, -1.2, 2.0])), LINEAR[case][1], rtol=1e-12)


@sc.function(1, output="y", name="zero_tangent_inner")
def _inner(u):
  return 2.0 * u


@sc.function(1, output="y", name="zero_tangent_outer")
def _outer(u):
  return _inner(u)


@sc.problem(vars=sc.L("u", 1), name="zero_tangent_two_levels")
def _two_levels(u):
  return sc.ProblemSpec(minimize=(u * u).sum(), eq=(_outer(u) - 1.0,))


@pytest.mark.solver("piqp")
def test_piqp_accepts_a_constraint_scaled_behind_two_calls() -> None:
  solve = sc.solver(_two_levels, "piqp", options={"eps_abs": 1e-10, "eps_rel": 1e-10})
  u, *_ = solve.numerical_call(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(0), ())
  np.testing.assert_allclose(u, [0.5], atol=1e-8)  # 2 u = 1
