"""``linalg.banded``: tridiagonal and cyclic tridiagonal solves against dense NumPy solves, for vector
and matrix right-hand sides, with their derivative in the right-hand side."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.linalg.banded import solve_cyclic_tridiagonal, solve_tridiagonal

RNG = np.random.default_rng(11)


def _bands(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  lower, upper = RNG.uniform(-1.0, 1.0, n), RNG.uniform(-1.0, 1.0, n)
  return lower, 2.5 + RNG.uniform(0.0, 1.0, n), upper  # diagonally dominant: no zero pivot


def _dense(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, cyclic: bool) -> np.ndarray:
  n = diag.size
  a = np.diag(diag) + np.diag(lower[1:], -1) + np.diag(upper[:-1], 1)
  if cyclic:
    a[0, n - 1] += lower[0]
    a[n - 1, 0] += upper[-1]
  return a


@pytest.mark.parametrize("cyclic", [False, True], ids=["tridiagonal", "cyclic"])
@pytest.mark.parametrize("n", [3, 8, 41])
@pytest.mark.parametrize("trailing", [(), (3,)], ids=["vector", "matrix"])
def test_the_solve_matches_a_dense_one(cyclic: bool, n: int, trailing: tuple[int, ...]) -> None:
  lower, diag, upper = _bands(n)
  solve = solve_cyclic_tridiagonal if cyclic else solve_tridiagonal
  b = sc.sym("b", (n, *trailing))
  x = solve(lower, diag, upper, b)
  weights = RNG.normal(size=(n, *trailing))
  cost = (x * sc.const(weights)).sum()
  fn = sc.Function.from_exprs(f"banded_{int(cyclic)}_{n}_{len(trailing)}", [b], [x, sc.gradient(cost, b)], ["b"], ["x", "g"])
  bv = RNG.normal(size=(n, *trailing))
  got, grad = fn._flat_numerical_call(bv)
  a = _dense(lower, diag, upper, cyclic)
  np.testing.assert_allclose(got, np.linalg.solve(a, bv), rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(grad, np.linalg.solve(a.T, weights), rtol=1e-12, atol=1e-12)  # the cotangent is a transposed solve


def test_a_band_of_the_wrong_length_or_rows_are_refused() -> None:
  lower, diag, upper = _bands(4)
  with pytest.raises(ValueError, match="one entry per row"):
    solve_tridiagonal(lower[:3], diag, upper, sc.sym("b", 4))
  with pytest.raises(ValueError, match="needs 4 rows"):
    solve_tridiagonal(lower, diag, upper, sc.sym("b", 5))
