"""Dense triangular solves, the Cholesky, LDL^T and LU factorizations and their solves, as library Functions over LOOP."""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

from ..function.api import custom_derivative, function
from ..function.model import Function
from ..function.sugar import broadcast, scan, vmap
from ..function.tree import arg, group
from ..ir.expr import Expr, argmax, as_expr, equal, gather, minimum, scatter, stack, where, zeros_like
from ..ir.types import TensorType, dtypes

_INDEX = TensorType((), dtypes.int64)


def _index(value: Any) -> Expr:
  return Expr.const(np.asarray(value, dtype=np.int64), dtype=dtypes.int64)


def _masked(G: Expr, mask: np.ndarray) -> Expr:
  return G * Expr.const(mask)


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
    X, _ = scan(row, zeros_like(B), (broadcast(T), broadcast(B)), length=T.shape[0], index=True, reverse=not forward)
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
  unusable = ~(pivot > float(np.finfo(np.float64).eps) * diagonal.abs())
  d = pivot.sqrt()
  rows = _index(np.arange(n))
  column = where(rows > j, gather(c, rows - j, mode="fill") / d, where(equal(rows, j), d, Expr.const(0.0)))
  return scatter(L, rows * n + j, column, combine="set"), where(unusable & equal(status, 0), j + 1, status)


def _cholesky_parts(A: Expr) -> tuple[Expr, Expr]:
  n = A.shape[0]
  (L, status), _ = scan(_cholesky_column, (zeros_like(A), _index(0)), (broadcast(A),), length=n, index=True)
  return L, status


_FACTOR_OUTPUTS = group(arg("L"), arg("status", _INDEX))


@function(arg("A"), outputs=group(_FACTOR_OUTPUTS, arg("L_")), name="cholesky_fwd")
def _cholesky_fwd(A: Expr) -> tuple[tuple[Expr, Expr], Expr]:
  L, status = _cholesky_parts(A)
  return (L, status), L


def _phi(M: Expr) -> Expr:
  """The lower triangle of ``M`` with its diagonal halved."""
  n = M.shape[0]
  return _masked(M, np.tril(np.ones((n, n)), -1)) + _masked(M, 0.5 * np.eye(n))


@function(arg("A"), arg("dA"), outputs=_FACTOR_OUTPUTS, name="cholesky_jvp")
def _cholesky_jvp(A: Expr, dA: Expr) -> tuple[Expr, Expr]:
  """``dL = L phi(L^-1 dA L^-T)``, with ``dA`` the symmetric matrix its lower triangle stands for.

  A forward rule receives no residuals, so it factors ``A`` again rather than differentiate the loops.
  """
  n = A.shape[0]
  L, _ = _cholesky_parts(A)
  low = _masked(dA, np.tril(np.ones((n, n)), -1))
  lower = _triangular(True, False, False)
  M = lower(L, lower(L, low + low.T + _masked(dA, np.eye(n))).T).T
  return L @ _phi(M), _index(0)


@function(arg("L"), group(arg("L_bar"), arg("status_bar", _INDEX)), name="cholesky_bwd")
def _cholesky_bwd(L: Expr, weights: tuple[Expr, Expr]) -> Expr:
  L_bar = weights[0]
  below = _triangular(True, True, False)
  S = below(L, below(L, _phi(L.T @ L_bar)).T).T
  return _lower_cotangent(S)


_cholesky = custom_derivative(
  function(arg("A"), outputs=_FACTOR_OUTPUTS, name="cholesky")(_cholesky_parts), jvp=_cholesky_jvp, fwd=_cholesky_fwd, bwd=_cholesky_bwd
)


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


def _unusable(pivot: Expr, diagonal: Expr) -> Expr:
  """A pivot of either sign not greater in magnitude than eps times its diagonal entry, or infinite, or NaN."""
  return ~(pivot.abs() > float(np.finfo(np.float64).eps) * diagonal.abs()) | ~pivot.isfinite()


def _strict_lower(n: int) -> np.ndarray:
  return np.tril(np.ones((n, n)), -1)


def _diagonal(M: Expr) -> Expr:
  """The diagonal of each matrix of ``M``, a vector over its last axis."""
  return (M * Expr.const(np.eye(M.shape[-1]))).sum(-1)


@function(arg("s", ()), arg("k", _INDEX), arg("LD"), arg("v"), arg("i", _INDEX), name="ldl_dot")
def _ldl_dot(s: Expr, k: Expr, LD: Expr, v: Expr, i: Expr) -> Expr:
  return s + gather(LD, i * LD.shape[0] + k) * gather(v, k)


@function(arg("t", _INDEX), arg("LD"), arg("v"), arg("A"), arg("j", _INDEX), name="ldl_entry")
def _ldl_entry(t: Expr, LD: Expr, v: Expr, A: Expr, j: Expr) -> tuple[Expr]:
  n = LD.shape[0]
  i = j + t
  s, _ = scan(_ldl_dot, Expr.const(0.0), (broadcast(LD), broadcast(v), broadcast(i)), length=n, trip=j, index=True)
  return (gather(A, i * n + j) - s,)


@function(arg("LD"), arg("status", _INDEX), arg("j", _INDEX), arg("A"), name="ldl_column")
def _ldl_column(LD: Expr, status: Expr, j: Expr, A: Expr) -> tuple[Expr, Expr]:
  """Column ``j`` of ``L`` below the diagonal and the pivot ``d_j`` on it, with ``v = d * L[j, :]`` shared by every row's dot product."""
  n = LD.shape[0]
  rows = _index(np.arange(n))
  v = gather(LD, rows * (n + 1)) * gather(LD, j * n + rows)
  _, (c,) = scan(_ldl_entry, (), (broadcast(LD), broadcast(v), broadcast(A), broadcast(j)), length=n, trip=n - j, index=True)
  pivot = c[0]
  column = where(rows > j, gather(c, rows - j, mode="fill") / pivot, where(equal(rows, j), pivot, Expr.const(0.0)))
  status = where(_unusable(pivot, gather(A, j * n + j)) & equal(status, 0), j + 1, status)
  return scatter(LD, rows * n + j, column, combine="set"), status


def _ldl_parts(A: Expr) -> tuple[Expr, Expr]:
  """``L`` below the diagonal and ``d`` on it, packed in one matrix so that a forward derivative runs one rule."""
  (LD, status), _ = scan(_ldl_column, (zeros_like(A), _index(0)), (broadcast(A),), length=A.shape[0], index=True)
  return LD, status


def _symmetric_tangent(dA: Expr) -> Expr:
  """The symmetric matrix that the lower triangle of ``dA`` stands for."""
  n = dA.shape[0]
  low = _masked(dA, _strict_lower(n))
  return low + low.T + _masked(dA, np.eye(n))


_LDL_OUTPUTS = group(arg("LD"), arg("status", _INDEX))


@function(arg("A"), arg("dA"), outputs=_LDL_OUTPUTS, name="ldl_jvp")
def _ldl_jvp(A: Expr, dA: Expr) -> tuple[Expr, Expr]:
  """With ``F = L^-1 dA L^-T``: ``dd = diag(F)`` and ``dL = L X``, ``X`` the strict lower triangle of ``F`` over ``d`` by column.

  A forward rule receives no residuals, so it factors ``A`` again rather than differentiate the loops.
  """
  n = A.shape[0]
  LD, _ = _ldl_parts(A)
  unit = _triangular(True, False, True)
  F = unit(LD, unit(LD, _symmetric_tangent(dA)).T).T
  L = _masked(LD, _strict_lower(n)) + Expr.const(np.eye(n))
  return L @ (_masked(F, _strict_lower(n)) / _diagonal(LD).reshape((1, n))) + _masked(F, np.eye(n)), _index(0)


_ldl = custom_derivative(function(arg("A"), outputs=_LDL_OUTPUTS, name="ldl")(_ldl_parts), jvp=_ldl_jvp)


@function(arg("s", ()), arg("m", _INDEX), arg("M"), arg("c"), arg("i", _INDEX), name="lu_dot")
def _lu_dot(s: Expr, m: Expr, M: Expr, c: Expr, i: Expr) -> Expr:
  return s + gather(M, i * M.shape[0] + m) * gather(c, m)


@function(arg("c"), arg("i", _INDEX), arg("M"), arg("k", _INDEX), name="lu_row")
def _lu_row(c: Expr, i: Expr, M: Expr, k: Expr) -> Expr:
  """Entry ``i`` of column ``k``: minus the dot product of row ``i`` of ``L`` with the entries of the column above it."""
  s, _ = scan(_lu_dot, Expr.const(0.0), (broadcast(M), broadcast(c), broadcast(i)), length=M.shape[0], trip=minimum(i, k), index=True)
  return scatter(c, i, gather(c, i) - s, combine="set")


@cache
def _lu(n: int) -> Function:
  """``P A = L U`` with partial pivoting for order ``n``, whose permutation is an int64 vector of that length."""
  perm_type = TensorType((n,), dtypes.int64)
  outputs = group(arg("LU"), arg("piv", perm_type), arg("perm", perm_type), arg("status", _INDEX))

  @function(arg("M"), arg("perm", perm_type), arg("status", _INDEX), arg("k", _INDEX), arg("A"), name="lu_column")
  def column(M: Expr, perm: Expr, status: Expr, k: Expr, A: Expr) -> tuple[Expr, Expr, Expr, Expr]:
    """Column ``k`` of the left-looking factorization: the column, the pivot of largest magnitude from row ``k`` on, the row swap."""
    rows = _index(np.arange(n))
    c, _ = scan(_lu_row, gather(M, rows * n + k), (broadcast(M), broadcast(k)), length=n, index=True)
    p = argmax(where(rows >= k, c.abs(), Expr.const(-1.0)))
    pivot, swapped = gather(c, p), gather(c, k)
    c = where(equal(rows, k), pivot, where(equal(rows, p), swapped, c))
    perm = where(equal(rows, k), gather(perm, p), where(equal(rows, p), gather(perm, k), perm))
    status = where(_unusable(pivot, gather(A, gather(perm, k) * n + k)) & equal(status, 0), k + 1, status)
    swap = stack([k, p])
    M = scatter(M, swap, gather(M, swap[::-1], axis=0), axis=0, combine="set")
    return scatter(M, rows * n + k, where(rows > k, c / pivot, c), combine="set"), perm, status, p

  def parts(A: Expr) -> tuple[Expr, Expr, Expr, Expr]:
    (M, perm, status), (piv,) = scan(column, (A, _index(np.arange(n)), _index(0)), (broadcast(A),), length=n, index=True)
    return M, piv, perm, status

  @function(arg("A"), arg("dA"), outputs=outputs, name="lu_jvp")
  def jvp(A: Expr, dA: Expr) -> tuple[Expr, Expr, Expr, Expr]:
    """With ``G = L^-1 P dA`` and ``S = tril(G U^-1, -1)``: ``dL = L S`` and ``dU = triu(G - S U)``, packed as the factor is.

    ``dU`` is ``triu(G U^-1) U`` without forming ``G U^-1``, which overflows when a pivot is tiny.

    A forward rule receives no residuals, so it factors ``A`` again rather than differentiate the loops.
    """
    M, piv, perm, _ = parts(A)
    strict = _strict_lower(n)
    G = _triangular(True, False, True)(M, gather(dA, perm, axis=0))
    S = _masked(_triangular(False, True, False)(M, G.T).T, strict)
    L, U = _masked(M, strict) + Expr.const(np.eye(n)), _masked(M, 1.0 - strict)
    return _masked(L @ S, strict) + _masked(G - S @ U, 1.0 - strict), zeros_like(piv), zeros_like(perm), _index(0)

  return custom_derivative(function(arg("A"), outputs=outputs, name="lu")(parts), jvp=jvp)


def _ldl_solve(LD: Expr, B: Expr) -> Expr:
  unit = _triangular(True, False, True)
  return _triangular(True, True, True)(LD, unit(LD, B) / _diagonal(LD).reshape((LD.shape[0], 1)))


def _symmetric_parts(K: Expr, B: Expr) -> tuple[Expr, Expr, Expr]:
  LD, status = _ldl(K)
  return _ldl_solve(LD, B), status, LD


@function(arg("K"), arg("B"), outputs=group(_SOLVE_OUTPUTS, group(arg("LD"), arg("X_"))), name="solve_sym_fwd")
def _symmetric_fwd(K: Expr, B: Expr) -> tuple[tuple[Expr, Expr], tuple[Expr, Expr]]:
  X, status, LD = _symmetric_parts(K, B)
  return (X, status), (LD, X)


@function(group(arg("LD"), arg("X")), group(arg("X_bar"), arg("status_bar", _INDEX)), name="solve_sym_bwd")
def _symmetric_bwd(residuals: tuple[Expr, Expr], weights: tuple[Expr, Expr]) -> tuple[Expr, Expr]:
  (LD, X), X_bar = residuals, weights[0]
  B_bar = _ldl_solve(LD, X_bar)
  return _lower_cotangent(-(B_bar @ X.T)), B_bar


@function(arg("K"), arg("B"), outputs=_SOLVE_OUTPUTS, name="solve_sym")
def _symmetric_body(K: Expr, B: Expr) -> tuple[Expr, Expr]:
  X, status, _ = _symmetric_parts(K, B)
  return X, status


_symmetric = custom_derivative(_symmetric_body, fwd=_symmetric_fwd, bwd=_symmetric_bwd)


@cache
def _general(n: int) -> Function:
  """The solve with a general matrix of order ``n``, through ``_lu(n)``."""
  perm_type = TensorType((n,), dtypes.int64)

  def parts(K: Expr, B: Expr) -> tuple[Expr, Expr, Expr, Expr]:
    M, _, perm, status = _lu(n)(K)
    X = _triangular(False, False, False)(M, _triangular(True, False, True)(M, gather(B, perm, axis=0)))
    return X, status, M, perm

  @function(arg("K"), arg("B"), outputs=group(_SOLVE_OUTPUTS, group(arg("LU"), arg("perm", perm_type), arg("X_"))), name="solve_gen_fwd")
  def fwd(K: Expr, B: Expr) -> tuple[tuple[Expr, Expr], tuple[Expr, Expr, Expr]]:
    X, status, M, perm = parts(K, B)
    return (X, status), (M, perm, X)

  @function(group(arg("LU"), arg("perm", perm_type), arg("X")), group(arg("X_bar"), arg("status_bar", _INDEX)), name="solve_gen_bwd")
  def bwd(residuals: tuple[Expr, Expr, Expr], weights: tuple[Expr, Expr]) -> tuple[Expr, Expr]:
    """``B_bar = K^-T X_bar = P^T L^-T U^-T X_bar`` and ``K_bar = -B_bar X^T``."""
    (M, perm, X), X_bar = residuals, weights[0]
    V = _triangular(True, True, True)(M, _triangular(False, True, False)(M, X_bar))
    B_bar = scatter(zeros_like(V), perm, V, axis=0, combine="set")
    return -(B_bar @ X.T), B_bar

  @function(arg("K"), arg("B"), outputs=_SOLVE_OUTPUTS, name="solve_gen")
  def body(K: Expr, B: Expr) -> tuple[Expr, Expr]:
    X, status, _, _ = parts(K, B)
    return X, status

  return custom_derivative(body, fwd=fwd, bwd=bwd)


def _dense(name: str, what: str, x: Expr) -> None:
  if x.type.pattern is not None:
    raise TypeError(f"{name} takes a dense {what}, got a sparse one")
  if x.type.dtype != dtypes.float64:
    raise TypeError(f"{name} takes a float64 {what}, got {x.type.dtype}")


def _square(name: str, A: Expr) -> tuple[int, ...]:
  _dense(name, "matrix", A)
  if len(A.shape) < 2 or A.shape[-1] != A.shape[-2]:
    raise ValueError(f"{name} needs a square matrix, or a stack of them, got shape {A.shape}")
  return A.shape[:-2]


def _right_hand_side(name: str, A: Expr, b: Any) -> tuple[Expr, bool]:
  b = as_expr(b)
  _dense(name, "right-hand side", b)
  batch, n = A.shape[:-2], A.shape[-1]
  if b.shape[: len(batch)] != batch or b.shape[len(batch) : len(batch) + 1] != (n,) or len(b.shape) > len(batch) + 2:
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


def solve_triangular(T: Any, b: Any, *, lower: bool, trans: bool = False, unit_diagonal: bool = False) -> Expr:
  """Solve ``T x = b``, or ``T^T x = b`` with ``trans``, for a triangular ``T``.

  ``T`` is a float64 square matrix, or a stack of them over leading axes. ``lower`` has no
  default, since SciPy's default reads the upper triangle. Only the triangle that ``lower``
  names is read, and its diagonal only when ``unit_diagonal`` is false. With ``unit_diagonal``
  the diagonal is taken as ones. ``b`` is one right-hand side, shaped
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
  and its mirror. ``A`` is a float64 square matrix, or a stack of them over leading axes. ``L`` is lower
  triangular with zeros above the diagonal.

  ``status`` holds one int64 per matrix. It is 0 when every pivot is usable, and otherwise
  ``j + 1`` for the first unusable pivot ``j`` counted from zero, so pivots are numbered from one
  as LAPACK numbers them. A pivot is unusable when it is not
  greater than machine epsilon times the diagonal entry of ``A`` it came from, which includes
  zero, negative and NaN pivots. The entries of ``L`` from an unusable pivot on are meaningless.

  ``L`` is differentiable in ``A``. The derivative of an entry below the diagonal is the sum of
  the derivatives with respect to the two mirrored entries it stands for. The reverse derivative
  reuses ``L``, and the forward derivative factors ``A`` once more.
  """
  A = as_expr(A)
  return _mapped(_cholesky, _square("cholesky", A), A)


def ldl(A: Any) -> tuple[Expr, Expr, Expr]:
  """Factor a symmetric quasi-definite ``A`` as ``L D L^T`` without pivoting, returning ``(L, d, status)``.

  Only the lower triangle of ``A`` is read, as in `cholesky`. ``A`` is a float64 square matrix, or
  a stack of them over leading axes. ``L`` is unit lower triangular, with ones on the diagonal and
  zeros above it, and ``d`` holds the diagonal of ``D``. Without pivoting the factorization exists
  for quasi-definite matrices, such as ``[[P, B^T], [B, -R]]`` with ``P`` and ``R`` positive definite,
  and its pivots may have either sign. Another symmetric matrix can give small pivots that pass
  the status check and factors with no accurate digits.

  ``status`` follows `cholesky`, with a pivot unusable when its magnitude is not greater than
  machine epsilon times the magnitude of the diagonal entry of ``A`` it came from, or when it is
  infinite.

  The factors are meant to be read, not differentiated: differentiate `solve` instead. They have
  a forward rule, which factors ``A`` once more for every direction, and no reverse rule, so a
  gradient through them differentiates the loops of the factorization.
  """
  A = as_expr(A)
  LD, status = _mapped(_ldl, _square("ldl", A), A)
  n = A.shape[-1]
  return _masked(LD, _strict_lower(n)) + Expr.const(np.eye(n)), _diagonal(LD), status


def lu_factor(A: Any) -> tuple[Expr, Expr, Expr]:
  """Factor a general ``A`` as ``P A = L U`` with partial pivoting, returning ``(lu, piv, status)``.

  ``lu`` and ``piv`` are those of ``scipy.linalg.lu_factor``: ``lu`` holds ``U`` on and above the
  diagonal and the entries of the unit lower triangular ``L`` below it, and ``piv`` is an int64
  vector saying that row ``k`` was swapped with row ``piv[k]`` at step ``k``, counted from zero.
  Each step takes the pivot of largest magnitude in its column, the first one on a tie. ``A`` is
  a float64 square matrix, or a stack of them over leading axes.

  ``status`` follows `cholesky`, with a pivot unusable when its magnitude is not greater than
  machine epsilon times the magnitude of the entry of ``A`` it came from, or when it is infinite.
  A status of 0 does not show that ``A`` is nonsingular.

  The factors are meant to be read, not differentiated: differentiate `solve` instead. They have
  a forward rule, which factors ``A`` once more for every direction, and no reverse rule, so a
  gradient through them differentiates the loops of the factorization.
  """
  A = as_expr(A)
  batch = _square("lu_factor", A)
  LU, piv, _, status = _mapped(_lu(A.shape[-1]), batch, A)
  return LU, piv, status
