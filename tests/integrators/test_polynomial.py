"""Gauss, Radau and Lobatto nodes and the Lagrange integrals the collocation tableaus are built from."""

from __future__ import annotations

import numpy as np
import pytest

from scaly.integrators.polynomial import gauss_nodes, lagrange_integrals, lobatto_nodes, radau_nodes


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
