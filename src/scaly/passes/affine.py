"""Recovers the affine structure of a concrete integer index array, so a gather or scatter needs no table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AffineIndexMap:
  """``idx[k] == residual[k % len(residual)] + sum(c * coord(k) for c, coord in ...)``.

  ``dims`` are the recovered range lengths, outermost first, and ``coeffs`` their coefficients;
  the ``i``-th coordinate of ``k`` is ``(k // prod(dims[i+1:]) // len(residual)) % dims[i]``.
  ``residual`` is whatever was not affine, of length one in the fully affine case.
  """

  dims: tuple[int, ...]
  coeffs: tuple[int, ...]
  residual: np.ndarray

  def evaluate(self, length: int) -> np.ndarray:
    """The array this map stands for. The inverse of ``affine_index_map``, for tests and asserts."""
    out = np.tile(self.residual, length // len(self.residual))
    stride = len(self.residual)
    for dim, coeff in zip(reversed(self.dims), reversed(self.coeffs)):
      out += coeff * (np.arange(length, dtype=np.int64) // stride % dim)
      stride *= dim
    return out


def _divisors(n: int) -> list[int]:
  small = [d for d in range(1, int(n**0.5) + 1) if n % d == 0]
  return sorted({*small, *(n // d for d in small)})


def affine_index_map(indices: np.ndarray) -> AffineIndexMap:
  """Factor ``indices`` into ranges whose contribution is affine, plus a residual table.

  Greedy, outermost first: pick the smallest inner block length ``m`` dividing the remaining length
  whose reshaped rows differ by one constant offset, record that range, and recurse into the first
  row. Peeling the smallest block that works leaves the smallest residual. A trailing coefficient of
  zero (a repeated block) is kept, because the caller drops the term.
  """
  rest = np.asarray(indices, dtype=np.int64).reshape(-1)
  dims: list[int] = []
  coeffs: list[int] = []
  while len(rest) > 1:
    for m in _divisors(len(rest))[:-1]:
      rows = rest.reshape(-1, m)
      delta = int(rows[1, 0]) - int(rows[0, 0])
      if np.array_equal(rows, rows[0] + delta * np.arange(len(rows), dtype=np.int64)[:, None]):
        dims.append(len(rows))
        coeffs.append(delta)
        rest = rows[0]
        break
    else:
      break
  return AffineIndexMap(tuple(dims), tuple(coeffs), rest)
