"""Gauss, Radau and Lobatto nodes on ``[0, 1]`` and integrals of the Lagrange basis over them."""

from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre

__all__ = ["gauss_nodes", "lagrange_integrals", "lobatto_nodes", "radau_nodes"]


def _polished(coefficients: np.ndarray, roots: np.ndarray) -> np.ndarray:
  """Roots of a Legendre series on ``[-1, 1]`` refined by Newton's method: the companion matrix's
  eigenvalues are accurate to a few units in 1e-15, and two steps take them to rounding."""
  deriv = legendre.legder(coefficients)
  for _ in range(3):
    roots = roots - legendre.legval(roots, coefficients) / legendre.legval(roots, deriv)
  return np.sort(roots)


def gauss_nodes(s: int) -> np.ndarray:
  """The ``s`` Gauss-Legendre nodes on ``(0, 1)``: the roots of the shifted Legendre polynomial ``P_s``."""
  return (legendre.leggauss(s)[0] + 1.0) / 2.0


def radau_nodes(s: int) -> np.ndarray:
  """The ``s`` right Radau nodes on ``(0, 1]``, the last one 1: the roots of ``P_s - P_{s-1}``."""
  coefficients = np.zeros(s + 1)
  coefficients[s], coefficients[s - 1] = 1.0, -1.0
  x = _polished(coefficients, legendre.legroots(coefficients).real)
  x[-1] = 1.0
  return (x + 1.0) / 2.0


def lobatto_nodes(s: int) -> np.ndarray:
  """The ``s >= 2`` Lobatto nodes on ``[0, 1]``: both ends and the roots of ``P'_{s-1}``."""
  if s < 2:
    raise ValueError(f"Lobatto nodes need at least 2 points, got {s}")
  basis = np.zeros(s)
  basis[s - 1] = 1.0
  inner = legendre.legder(basis)
  x = _polished(inner, legendre.legroots(inner).real) if s > 2 else np.zeros(0)
  return (np.concatenate([[-1.0], x, [1.0]]) + 1.0) / 2.0


def lagrange_integrals(nodes: np.ndarray, upper: np.ndarray) -> np.ndarray:
  """``out[i, j] = integral from 0 to upper[i] of l_j``, ``l_j`` the Lagrange polynomial of ``nodes``
  that is 1 at node ``j`` and 0 at the others; by a Gauss rule exact for its degree."""
  nodes, upper = np.asarray(nodes, dtype=np.float64), np.asarray(upper, dtype=np.float64)
  s = nodes.size
  x, w = legendre.leggauss(s + 1)
  out = np.zeros((upper.size, s))
  for i, top in enumerate(upper):
    t, weights = top * (x + 1.0) / 2.0, top * w / 2.0
    for j in range(s):
      others = np.delete(nodes, j)
      basis = np.prod((t[:, None] - others) / (nodes[j] - others), axis=1)
      out[i, j] = weights @ basis
  return out
