"""Gauss, Radau and Lobatto nodes and the Lagrange integrals the collocation tableaus are built from."""

from __future__ import annotations

import numpy as np
import pytest

from numpy.polynomial import legendre

from scaly.integrators.polynomial import (
  differentiation_matrix,
  gauss_nodes,
  interpolation_matrix,
  lagrange_integrals,
  lgl,
  lobatto_nodes,
  radau_nodes,
)


def _exact_to(nodes: np.ndarray) -> int:
  """The highest degree the interpolatory quadrature on ``nodes`` integrates exactly over [0, 1]."""
  weights = lagrange_integrals(nodes, np.ones(1))[0]
  degree = 0
  while abs(weights @ nodes**degree - 1.0 / (degree + 1)) < 1e-13:
    degree += 1
  return degree - 1


@pytest.mark.parametrize("s", range(1, 9))
def test_gauss_and_radau_nodes_reach_their_quadrature_degree(s: int) -> None:
  gauss, radau = gauss_nodes(s), radau_nodes(s)
  assert _exact_to(gauss) == 2 * s - 1 and _exact_to(radau) == 2 * s - 2
  assert radau[-1] == 1.0 and np.all(np.diff(radau) > 0) and 0 < radau[0]
  np.testing.assert_allclose(gauss, (np.polynomial.legendre.leggauss(s)[0] + 1) / 2, atol=1e-15)


@pytest.mark.parametrize("s", range(2, 9))
def test_lobatto_nodes_include_both_ends(s: int) -> None:
  nodes = lobatto_nodes(s)
  assert nodes[0] == 0.0 and nodes[-1] == 1.0 and np.all(np.diff(nodes) > 0)
  assert _exact_to(nodes) == 2 * s - 3
  np.testing.assert_allclose(nodes, 1 - nodes[::-1], atol=1e-15)  # symmetric about 1/2


def test_lagrange_integrals_integrate_the_monomials() -> None:
  nodes, upper = radau_nodes(4), np.array([0.2, 0.7, 1.0])
  table = lagrange_integrals(nodes, upper)
  for k in range(4):  # degree up to s - 1 is interpolated exactly
    np.testing.assert_allclose(table @ nodes**k, upper ** (k + 1) / (k + 1), rtol=1e-14)
  with pytest.raises(ValueError, match="at least 2"):
    lobatto_nodes(1)


@pytest.mark.parametrize("nodes_of", [gauss_nodes, radau_nodes, lobatto_nodes])
@pytest.mark.parametrize("s", [2, 5, 9])
def test_differentiation_and_interpolation_are_exact_on_polynomials(nodes_of, s: int) -> None:
  nodes = np.concatenate([[0.0], nodes_of(s)]) if nodes_of is not lobatto_nodes else nodes_of(s)
  d = differentiation_matrix(nodes)
  np.testing.assert_allclose(d.sum(axis=1), 0.0, atol=1e-12)
  at = np.array([0.0, 0.123, 0.5, 1.0, nodes[1]])
  table = interpolation_matrix(nodes, at)
  for k in range(nodes.size):
    np.testing.assert_allclose(d @ nodes**k, k * nodes ** max(k - 1, 0) if k else np.zeros(nodes.size), atol=1e-10)
    np.testing.assert_allclose(table @ nodes**k, at**k, atol=1e-12)
  assert np.array_equal(table[-1], np.eye(nodes.size)[1])  # a node reads its own value exactly


@pytest.mark.parametrize("n", [1, 2, 3, 6, 12, 30])
def test_the_lgl_rule_its_nodes_weights_and_derivatives(n: int) -> None:
  x, w, d = lgl(n)
  basis = np.zeros(n + 1)
  basis[-1] = 1.0
  inner = legendre.legroots(legendre.legder(basis)) if n > 1 else np.zeros(0)
  np.testing.assert_allclose(x, np.concatenate([[-1.0], np.sort(inner), [1.0]]), atol=2e-15)
  np.testing.assert_array_equal(x, -x[::-1])
  np.testing.assert_array_equal(w, w[::-1])
  exact = [np.polynomial.polynomial.polyval(x, np.eye(2 * n)[k]) @ w - (2 / (k + 1) if k % 2 == 0 else 0.0) for k in range(2 * n)]
  assert np.abs(exact).max() < 1e-14  # every degree up to 2n - 1
  np.testing.assert_allclose(w @ legendre.legval(x, basis) ** 2, 2.0 / n, rtol=1e-13)  # not P_n^2, whose integral is 2 / (2n + 1)
  coefficients = np.random.default_rng(n).standard_normal(n + 1)  # a polynomial of degree n
  np.testing.assert_allclose(
    d @ np.polynomial.polynomial.polyval(x, coefficients),
    np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coefficients)),
    atol=1e-11 * n * n,
  )
