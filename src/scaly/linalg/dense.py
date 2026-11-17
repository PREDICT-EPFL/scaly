"""Dense triangular solves, Cholesky and the positive definite solve, as library Functions over LOOP."""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

from ..function.api import custom_derivative, function
from ..function.model import Function
from ..function.sugar import broadcast, scan, vmap
from ..function.tree import arg, group
from ..ir.expr import Expr, as_expr, equal, gather, scatter, where, zeros_like
from ..ir.types import TensorType, dtypes

_INDEX = TensorType((), dtypes.int64)


def _index(value: Any) -> Expr:
  return Expr.const(np.asarray(value, dtype=np.int64), dtype=dtypes.int64)


def _masked(G: Expr, mask: np.ndarray) -> Expr:
  return G * Expr.const(mask.astype(G.type.dtype.numpy()), dtype=G.type.dtype)


def _lower_cotangent(G: Expr) -> Expr:
  """The cotangent of a stored lower triangle standing for a symmetric matrix with full cotangent ``G``.

  An off-diagonal stored entry stands for both mirrored entries, so it collects both contributions.
  """
  n = G.shape[0]
  return _masked(G + G.T, np.tril(np.ones((n, n)), -1)) + _masked(G, np.eye(n))


def _name(lower: bool, trans: bool, unit: bool) -> str:
  return f"solve_triangular_{'l' if lower else 'u'}{'t' if trans else 'n'}{'u' if unit else 'n'}"


@cache
def _substitution(lower: bool, trans: bool, unit: bool) -> Function:
  """``X`` solving ``op(T) X = B`` by substitution, ``op(T)`` being ``T`` or its transpose, read on one triangle."""
  forward = lower != trans
  name = _name(lower, trans, unit)

  def entry(n: int, i: Expr, k: Expr) -> Expr:
    return k * n + i if trans else i * n + k

  @function(arg("r"), arg("t", _INDEX), arg("T"), arg("X"), arg("i", _INDEX), name=f"{name}_update")
  def update(r: Expr, t: Expr, T: Expr, X: Expr, i: Expr) -> Expr:
    n = T.shape[0]
    k = t if forward else i + 1 + t
    return r - gather(T, entry(n, i, k)) * gather(X, k, axis=0)

  @function(arg("X"), arg("i", _INDEX), arg("T"), arg("B"), name=f"{name}_row")
  def row(X: Expr, i: Expr, T: Expr, B: Expr) -> Expr:
    n = T.shape[0]
    r, _ = scan(update, gather(B, i, axis=0), (broadcast(T), broadcast(X), broadcast(i)), length=n, trip=i if forward else n - 1 - i, index=True)
    return scatter(X, i, r if unit else r / gather(T, i * n + i), axis=0, combine="set")

  @function(arg("T"), arg("B"), outputs=arg("X"), name=name)
  def solve(T: Expr, B: Expr) -> Expr:
    X, _ = scan(row, zeros_like(B), (broadcast(T), broadcast(B)), length=T.shape[0], index=True, reverse=not forward, in_place=(0,))
    return X

  return solve


@cache
def _triangular(lower: bool, trans: bool, unit: bool) -> Function:
  """The substitution with reverse rules: a solve with the transposed matrix and an outer product.

  The rule calls the plain transposed substitution, since a transposed solve with rules would call
  this one back while its rules are bound.
  """
  name = _name(lower, trans, unit)

  @function(arg("T"), arg("B"), outputs=group(arg("X"), group(arg("T_"), arg("X_"))), name=f"{name}_fwd")
  def fwd(T: Expr, B: Expr) -> tuple[Expr, tuple[Expr, Expr]]:
    X = _substitution(lower, trans, unit)(T, B)
    return X, (T, X)

  @function(group(arg("T"), arg("X")), arg("X_bar"), name=f"{name}_bwd")
  def bwd(residuals: tuple[Expr, Expr], X_bar: Expr) -> tuple[Expr, Expr]:
    T, X = residuals
    B_bar = _substitution(lower, not trans, unit)(T, X_bar)
    G = -(X @ B_bar.T) if trans else -(B_bar @ X.T)
    n = T.shape[0]
    return _masked(G, np.tril(np.ones((n, n)), -int(unit)) if lower else np.triu(np.ones((n, n)), int(unit))), B_bar

  return custom_derivative(_substitution(lower, trans, unit), fwd=fwd, bwd=bwd)


@function(arg("s", ()), arg("k", _INDEX), arg("L"), arg("i", _INDEX), arg("j", _INDEX), name="cholesky_dot")
def _cholesky_dot(s: Expr, k: Expr, L: Expr, i: Expr, j: Expr) -> Expr:
  n = L.shape[0]
  return s + gather(L, i * n + k) * gather(L, j * n + k)


@function(arg("t", _INDEX), arg("L"), arg("A"), arg("j", _INDEX), name="cholesky_entry")
def _cholesky_entry(t: Expr, L: Expr, A: Expr, j: Expr) -> tuple[Expr]:
  n = L.shape[0]
  i = j + t
  s, _ = scan(_cholesky_dot, Expr.const(0.0), (broadcast(L), broadcast(i), broadcast(j)), length=n, trip=j, index=True)
  return (gather(A, i * n + j) - s,)


@function(arg("L"), arg("status", _INDEX), arg("j", _INDEX), arg("A"), name="cholesky_column")
def _cholesky_column(L: Expr, status: Expr, j: Expr, A: Expr) -> tuple[Expr, Expr]:
  n = L.shape[0]
  _, (c,) = scan(_cholesky_entry, (), (broadcast(L), broadcast(A), broadcast(j)), length=n, trip=n - j, index=True)
  pivot, diagonal = c[0], gather(A, j * n + j)
  unusable = ~(pivot > float(np.finfo(A.type.dtype.numpy()).eps) * diagonal.abs())
  d = pivot.sqrt()
  rows = _index(np.arange(n))
  column = where(rows > j, gather(c, rows - j, mode="fill") / d, where(equal(rows, j), d, Expr.const(0.0)))
  return scatter(L, rows * n + j, column, combine="set"), where(unusable & equal(status, 0), j + 1, status)


def _cholesky_parts(A: Expr) -> tuple[Expr, Expr]:
  n = A.shape[0]
  (L, status), _ = scan(_cholesky_column, (zeros_like(A), _index(0)), (broadcast(A),), length=n, index=True, in_place=(0,))
  return L, status


_FACTOR_OUTPUTS = group(arg("L"), arg("status", _INDEX))


@function(arg("A"), outputs=group(_FACTOR_OUTPUTS, arg("L_")), name="cholesky_fwd")
def _cholesky_fwd(A: Expr) -> tuple[tuple[Expr, Expr], Expr]:
  L, status = _cholesky_parts(A)
  return (L, status), L


@function(arg("L"), group(arg("L_bar"), arg("status_bar", _INDEX)), name="cholesky_bwd")
def _cholesky_bwd(L: Expr, weights: tuple[Expr, Expr]) -> Expr:
  L_bar = weights[0]
  n = L.shape[0]
  inner = L.T @ L_bar
  P = _masked(inner, np.tril(np.ones((n, n)), -1)) + _masked(inner, 0.5 * np.eye(n))
  below = _triangular(True, True, False)
  S = below(L, below(L, P).T).T
  return _lower_cotangent(S)


_cholesky = custom_derivative(function(arg("A"), outputs=_FACTOR_OUTPUTS, name="cholesky")(_cholesky_parts), fwd=_cholesky_fwd, bwd=_cholesky_bwd)


def _positive_parts(K: Expr, B: Expr) -> tuple[Expr, Expr, Expr]:
  L, status = _cholesky(K)
  X = _triangular(True, True, False)(L, _triangular(True, False, False)(L, B))
  return X, status, L


_SOLVE_OUTPUTS = group(arg("X"), arg("status", _INDEX))


@function(arg("K"), arg("B"), outputs=group(_SOLVE_OUTPUTS, group(arg("L"), arg("X_"))), name="solve_pos_fwd")
def _positive_fwd(K: Expr, B: Expr) -> tuple[tuple[Expr, Expr], tuple[Expr, Expr]]:
  X, status, L = _positive_parts(K, B)
  return (X, status), (L, X)


@function(group(arg("L"), arg("X")), group(arg("X_bar"), arg("status_bar", _INDEX)), name="solve_pos_bwd")
def _positive_bwd(residuals: tuple[Expr, Expr], weights: tuple[Expr, Expr]) -> tuple[Expr, Expr]:
  (L, X), X_bar = residuals, weights[0]
  B_bar = _triangular(True, True, False)(L, _triangular(True, False, False)(L, X_bar))
  return _lower_cotangent(-(B_bar @ X.T)), B_bar


@function(arg("K"), arg("B"), outputs=_SOLVE_OUTPUTS, name="solve_pos")
def _positive_body(K: Expr, B: Expr) -> tuple[Expr, Expr]:
  X, status, _ = _positive_parts(K, B)
  return X, status


_positive = custom_derivative(_positive_body, fwd=_positive_fwd, bwd=_positive_bwd)


def _square(name: str, A: Expr) -> tuple[int, ...]:
  if A.type.pattern is not None:
    raise TypeError(f"{name} takes a dense matrix, got a sparse one")
  if len(A.shape) < 2 or A.shape[-1] != A.shape[-2]:
    raise ValueError(f"{name} needs a square matrix, or a stack of them, got shape {A.shape}")
  return A.shape[:-2]


def _right_hand_side(name: str, A: Expr, b: Any) -> tuple[Expr, bool]:
  b = as_expr(b)
  batch, n = A.shape[:-2], A.shape[-1]
  if b.shape[: len(batch)] != batch or b.shape[len(batch) :] not in ((n,), (n, b.shape[-1])):
    raise ValueError(f"{name}: the right-hand side has shape {b.shape}, expected {(*batch, n)} or {(*batch, n, 'k')}")
  vector = len(b.shape) == len(batch) + 1
  return (b.reshape((*b.shape, 1)) if vector else b), vector


def _mapped(fn: Function[Any, Any, Any, Any], batch: tuple[int, ...], *args: Expr) -> Any:
  """Call ``fn`` on each matrix of a stack whose leading axes are ``batch``."""
  if not batch:
    return fn(*args)
  count = int(np.prod(batch))
  results = vmap(fn, count)(*(a.reshape((count, *a.shape[len(batch) :])) for a in args))
  single = not isinstance(results, tuple)
  results = tuple(r.reshape((*batch, *r.shape[1:])) for r in ((results,) if single else results))
  return results[0] if single else results


def solve_triangular(T: Any, b: Any, *, lower: bool = True, trans: bool = False, unit_diagonal: bool = False) -> Expr:
  """Solve ``T x = b``, or ``T^T x = b`` with ``trans``, for a triangular ``T``.

  ``T`` is square, or a stack of square matrices over leading axes. Only the triangle that
  ``lower`` names is read, and its diagonal only when ``unit_diagonal`` is false. With
  ``unit_diagonal`` the diagonal is taken as ones. ``b`` is one right-hand side, shaped
  ``(*batch, n)``, or several as the columns of ``(*batch, n, k)``, and ``x`` has its shape.
  A zero diagonal entry gives infinities or NaN in ``x``.

  The result is differentiable in ``T`` and ``b``. The reverse derivative is one solve with the
  transposed matrix, and the forward derivative follows the substitution.
  """
  T = as_expr(T)
  batch = _square("solve_triangular", T)
  B, vector = _right_hand_side("solve_triangular", T, b)
  X = _mapped(_triangular(lower, trans, unit_diagonal), batch, T, B)
  return X.reshape(X.shape[:-1]) if vector else X


def cholesky(A: Any) -> tuple[Expr, Expr]:
  """Factor a symmetric positive definite ``A`` as ``L L^T``, returning ``(L, status)``.

  Only the lower triangle of ``A`` is read, and an entry below the diagonal stands for itself
  and its mirror. ``A`` is square, or a stack of square matrices over leading axes. ``L`` is lower
  triangular with zeros above the diagonal.

  ``status`` is an int64 per matrix: 0 when every pivot is usable, otherwise ``j + 1`` for the
  first unusable pivot ``j``, as LAPACK numbers them. A pivot is unusable when it is not
  greater than machine epsilon times the diagonal entry of ``A`` it came from, which includes
  zero, negative and NaN pivots. The entries of ``L`` from an unusable pivot on are meaningless.

  ``L`` is differentiable in ``A``. The derivative of an entry below the diagonal is the sum of
  the derivatives with respect to the two mirrored entries it stands for.
  """
  A = as_expr(A)
  return _mapped(_cholesky, _square("cholesky", A), A)
