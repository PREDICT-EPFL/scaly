"""Sparse Cholesky and LDL^T and their solves, as library Functions over loops that read the tables of ``symbolic.ldl``."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from ..function.api import custom_derivative, function
from ..function.model import Function
from ..function.sugar import broadcast, scan
from ..function.tree import arg, group
from ..ir.expr import Expr, concat, equal, gather, pack, scatter, values, where
from ..ir.types import CanonicalPattern, TensorType, dtypes
from . import symbolic
from .dense import _eps, _like, _status, _unusable
from .symbolic import LDLSymbolic, Ordering

_INDEX = TensorType((), dtypes.int64)
_STATUS = TensorType((), dtypes.int32)

# The tables the loops read, as two groups: those of the matrix and those of its factor.
_A_TABLES = ("a_ptr", "a_rows", "a_source", "diagonal")
_L_TABLES = ("l_ptr", "l_rows", "r_ptr", "r_cols", "r_pos")


@dataclass(frozen=True, eq=False)
class _Tables:
  """``symbolic.ldl``'s tables as the loops read them, with the rows of ``L`` beside its columns.

  ``r_ptr``, ``r_cols`` and ``r_pos`` list row ``i`` of ``L`` below the diagonal: the columns ``k``
  it has an entry in, ascending, and the position of that entry in ``l_rows``. ``diagonal[j]`` is
  the position of the stored diagonal entry of column ``j`` of the permuted matrix among the
  matrix's values, or -1 when it is not stored, which a ``fill`` gather reads as zero.
  """

  analysis: LDLSymbolic
  arrays: dict[str, np.ndarray]

  def exprs(self, names: tuple[str, ...]) -> tuple[Expr, ...]:
    return tuple(Expr.const(self.arrays[name], dtype=dtypes.int64) for name in names)


# By digest and weakly held, so the tables live only as long as a Function that reads them.
_BY_DIGEST: weakref.WeakValueDictionary[str, _Tables] = weakref.WeakValueDictionary()


def _tables(pattern: CanonicalPattern, ordering: Ordering) -> _Tables:
  analysis = symbolic.ldl(pattern, ordering)
  if (cached := _BY_DIGEST.get(analysis.digest)) is not None:
    return cached
  n = analysis.n
  l_ptr, l_rows = np.asarray(analysis.l_ptr, dtype=np.int64), np.asarray(analysis.l_rows, dtype=np.int64)
  l_cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(l_ptr))
  by_row = np.lexsort((l_cols, l_rows))
  a_ptr, a_rows, a_source = (np.asarray(t, dtype=np.int64) for t in (analysis.a_ptr, analysis.a_rows, analysis.a_source))
  a_cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(a_ptr))
  diagonal = np.full(n, -1, dtype=np.int64)
  stored = a_rows == a_cols
  diagonal[a_cols[stored]] = a_source[stored]
  arrays = {
    "a_ptr": a_ptr,
    "a_rows": a_rows,
    "a_source": a_source,
    "l_ptr": l_ptr,
    "l_rows": l_rows,
    "r_ptr": np.concatenate([[0], np.cumsum(np.bincount(l_rows, minlength=n))]).astype(np.int64),
    "r_cols": l_cols[by_row],
    "r_pos": by_row.astype(np.int64),
    "diagonal": diagonal,
  }
  return _BY_DIGEST.setdefault(analysis.digest, _Tables(analysis, arrays))


def _group(names: tuple[str, ...]) -> Any:
  return group(*(arg(name, dtype=...) for name in names))


def _whole(tables: tuple[Expr, ...]) -> tuple[Any, ...]:
  """A group of tables that every step of a loop reads whole."""
  return tuple(broadcast(table) for table in tables)


def _ragged(body: Function, carry: Any, ptr: Expr, j: Expr, length: int, *xs: Any, reverse: bool = False) -> Any:
  """Run ``body`` over the entries ``ptr[j] <= p < ptr[j + 1]`` of a ragged table, ``body`` taking ``start = ptr[j]`` last."""
  start, stop = gather(ptr, j), gather(ptr, j + 1)
  out, _ = scan(body, carry, (*xs, broadcast(start)), length=length, trip=stop - start, index=True, reverse=reverse)
  return out


# The factorization, left-looking: column j of the permuted matrix goes into a dense work column w,
# each column k with an entry in row j of L subtracts its contribution from row j down, and w then
# holds the pivot and column j of L, whose entries are cleared from w for the next column.
#
# Every carry of these loops, and of the substitutions below, is updated in place: the inner loops
# thread the work column through the column step, which lowering proves link by link. The loops do
# not demand it with ``in_place``, which a derivative storing their trajectories would contradict;
# tests/linalg/test_sparse.py checks it in the generated C instead.


@function(
  arg("w", dtype=...),
  arg("t", _INDEX),
  arg("A", dtype=...),
  arg("a_rows", dtype=...),
  arg("a_source", dtype=...),
  arg("start", _INDEX),
  name="sparse_factor_load",
)
def _load(w: Expr, t: Expr, A: Expr, a_rows: Expr, a_source: Expr, start: Expr) -> Expr:
  """``w[a_rows[p]] = A[a_source[p]]`` for entry ``p = start + t`` of the permuted lower triangle."""
  p = start + t
  return scatter(w, gather(a_rows, p), gather(A, gather(a_source, p)), combine="set")


@function(
  arg("w", dtype=...),
  arg("t", _INDEX),
  arg("Lx", dtype=...),
  arg("l_rows", dtype=...),
  arg("scale", dtype=...),
  arg("start", _INDEX),
  name="sparse_factor_axpy",
)
def _axpy(w: Expr, t: Expr, Lx: Expr, l_rows: Expr, scale: Expr, start: Expr) -> Expr:
  """``w[l_rows[p]] += Lx[p] * scale`` for entry ``p = start + t`` of a column of ``L``."""
  p = start + t
  return scatter(w, gather(l_rows, p), gather(Lx, p) * scale, combine="add")


@cache
def _update(kind: str) -> Function:
  """The contribution of the ``t``-th column ``k`` with an entry in row ``j`` of ``L``, subtracted from the work column from row ``j`` down."""

  @function(
    arg("w", dtype=...),
    arg("t", _INDEX),
    arg("Lx", dtype=...),
    arg("d", dtype=...),
    _group(_L_TABLES),
    arg("start", _INDEX),
    name=f"sparse_{kind}_update",
  )
  def update(w: Expr, t: Expr, Lx: Expr, d: Expr, l_tables: tuple[Expr, ...], start: Expr) -> Expr:
    l_ptr, l_rows, _, r_cols, r_pos = l_tables
    q = start + t
    k, pos = gather(r_cols, q), gather(r_pos, q)
    entry = gather(Lx, pos)
    scale = -(gather(d, k) * entry) if kind == "ldl" else -entry
    stop = gather(l_ptr, k + 1)
    w, _ = scan(_axpy, w, (broadcast(Lx), broadcast(l_rows), broadcast(scale), broadcast(pos)), length=w.shape[0], trip=stop - pos, index=True)
    return w

  return update


@function(
  arg("Lx", dtype=...),
  arg("w", dtype=...),
  arg("t", _INDEX),
  arg("l_rows", dtype=...),
  arg("scale", dtype=...),
  arg("start", _INDEX),
  name="sparse_factor_store",
)
def _store(Lx: Expr, w: Expr, t: Expr, l_rows: Expr, pivot: Expr, start: Expr) -> tuple[Expr, Expr]:
  """``Lx[p] = w[i] / pivot`` and ``w[i] = 0`` for entry ``p = start + t`` of the column, at row ``i = l_rows[p]``.

  A division rather than a product with ``1 / pivot``, whose reciprocal overflows for a subnormal pivot.
  """
  p = start + t
  i = gather(l_rows, p)
  return scatter(Lx, p, gather(w, i) / pivot, combine="set"), scatter(w, i, _like(w, 0.0), combine="set")


@cache
def _column(kind: str) -> Function:
  """Column ``j``: the pivot ``d[j]``, or the diagonal of ``L`` for Cholesky, the entries of ``L`` below it, and the status."""

  @function(
    arg("Lx", dtype=...),
    arg("d", dtype=...),
    arg("w", dtype=...),
    arg("status", _STATUS),
    arg("j", _INDEX),
    arg("A", dtype=...),
    _group(_A_TABLES),
    _group(_L_TABLES),
    name=f"sparse_{kind}_column",
  )
  def column(
    Lx: Expr, d: Expr, w: Expr, status: Expr, j: Expr, A: Expr, a_tables: tuple[Expr, ...], l_tables: tuple[Expr, ...]
  ) -> tuple[Expr, Expr, Expr, Expr]:
    a_ptr, a_rows, a_source, diagonal_source = a_tables
    l_ptr, l_rows, r_ptr, _, _ = l_tables
    n = w.shape[0]
    w = _ragged(_load, w, a_ptr, j, n, broadcast(A), broadcast(a_rows), broadcast(a_source))
    w = _ragged(_update(kind), w, r_ptr, j, n, broadcast(Lx), broadcast(d), _whole(l_tables))
    pivot = gather(w, j)
    diagonal = gather(A, gather(diagonal_source, j), mode="fill")
    if kind == "ldl":
      unusable, value = _unusable(pivot, diagonal), pivot
    else:
      unusable, value = ~(pivot > _eps(A) * diagonal.abs()), pivot.sqrt()
    Lx, w = _ragged(_store, (Lx, w), l_ptr, j, n, broadcast(l_rows), broadcast(value))
    status = where(unusable & equal(status, 0), _status(j + 1), status)
    return Lx, scatter(d, j, value, combine="set"), w, status

  return column


def _factor_parts(kind: str, A: Expr, tables: _Tables) -> tuple[Expr, Expr, Expr]:
  """``Lx``, ``d`` and the status of the matrix whose stored values, in canonical order, are ``A``."""
  analysis = tables.analysis
  zeros = lambda size: Expr.const(np.zeros(size, dtype=A.type.dtype.numpy()), dtype=A.type.dtype)  # noqa: E731
  init = (zeros(analysis.nnz_l), zeros(analysis.n), zeros(analysis.n), _status(0))
  (Lx, d, _, status), _ = scan(
    _column(kind),
    init,
    (broadcast(A), _whole(tables.exprs(_A_TABLES)), _whole(tables.exprs(_L_TABLES))),
    length=analysis.n,
    index=True,
  )
  return Lx, d, status


@cache
def _factor_layout(analysis: LDLSymbolic) -> tuple[CanonicalPattern, np.ndarray]:
  """The pattern of ``L`` with its diagonal, and for each stored entry its position in ``[Lx | diagonal]``."""
  n = analysis.n
  rows = np.concatenate([np.asarray(analysis.l_rows, dtype=np.int64), np.arange(n, dtype=np.int64)])
  cols = np.concatenate([np.repeat(np.arange(n, dtype=np.int64), np.diff(analysis.l_ptr)), np.arange(n, dtype=np.int64)])
  order = np.lexsort((cols, rows))
  return CanonicalPattern((n, n), tuple(rows[order].tolist()), tuple(cols[order].tolist())), order.astype(np.int64)


def _factor_value(Lx: Expr, diagonal: Expr, analysis: LDLSymbolic) -> Expr:
  pattern, order = _factor_layout(analysis)
  return pack(gather(concat([Lx, diagonal]), Expr.const(order, dtype=dtypes.int64)), pattern)


# The substitutions with the factor's columns: Y holds one right-hand side per column, in the
# reordered numbering.


@function(
  arg("Y", dtype=...),
  arg("t", _INDEX),
  arg("Lx", dtype=...),
  arg("l_rows", dtype=...),
  arg("y", dtype=...),
  arg("start", _INDEX),
  name="sparse_solve_axpy",
)
def _solve_axpy(Y: Expr, t: Expr, Lx: Expr, l_rows: Expr, y: Expr, start: Expr) -> Expr:
  """``Y[l_rows[p]] -= Lx[p] * y`` for entry ``p = start + t`` of a column of ``L``."""
  p = start + t
  return scatter(Y, gather(l_rows, p), -(gather(Lx, p) * y), axis=0, combine="add")


@function(
  arg("s", dtype=...),
  arg("t", _INDEX),
  arg("Lx", dtype=...),
  arg("l_rows", dtype=...),
  arg("Y", dtype=...),
  arg("start", _INDEX),
  name="sparse_solve_dot",
)
def _solve_dot(s: Expr, t: Expr, Lx: Expr, l_rows: Expr, Y: Expr, start: Expr) -> Expr:
  """``s + Lx[p] * Y[l_rows[p]]`` for entry ``p = start + t`` of a column of ``L``."""
  p = start + t
  return s + gather(Lx, p) * gather(Y, gather(l_rows, p), axis=0)


@cache
def _lower_sweep(kind: str) -> Function:
  """Column ``j`` of ``L^{-1}``: with Cholesky's diagonal, ``Y[j] /= d[j]`` first; then the rows below ``j`` lose ``L[i, j] Y[j]``."""

  @function(
    arg("Y", dtype=...),
    arg("j", _INDEX),
    arg("Lx", dtype=...),
    arg("d", dtype=...),
    arg("l_ptr", dtype=...),
    arg("l_rows", dtype=...),
    name=f"sparse_{kind}_forward",
  )
  def forward(Y: Expr, j: Expr, Lx: Expr, d: Expr, l_ptr: Expr, l_rows: Expr) -> Expr:
    y = gather(Y, j, axis=0)
    if kind == "cholesky":
      y = y / gather(d, j)
      Y = scatter(Y, j, y, axis=0, combine="set")
    return _ragged(_solve_axpy, Y, l_ptr, j, Y.shape[0], broadcast(Lx), broadcast(l_rows), broadcast(y))

  return forward


@cache
def _upper_sweep(kind: str) -> Function:
  """Row ``j`` of ``L^{-T}``, from the last: ``Y[j]`` loses the dot product of column ``j`` of ``L`` with the rows below, then with Cholesky's diagonal ``Y[j] /= d[j]``."""

  @function(
    arg("Y", dtype=...),
    arg("j", _INDEX),
    arg("Lx", dtype=...),
    arg("d", dtype=...),
    arg("l_ptr", dtype=...),
    arg("l_rows", dtype=...),
    name=f"sparse_{kind}_backward",
  )
  def backward(Y: Expr, j: Expr, Lx: Expr, d: Expr, l_ptr: Expr, l_rows: Expr) -> Expr:
    zero = Expr.const(np.zeros(Y.shape[1:], dtype=Y.type.dtype.numpy()), dtype=Y.type.dtype)
    s = _ragged(_solve_dot, zero, l_ptr, j, Y.shape[0], broadcast(Lx), broadcast(l_rows), broadcast(Y))
    y = gather(Y, j, axis=0) - s
    return scatter(Y, j, y / gather(d, j) if kind == "cholesky" else y, axis=0, combine="set")

  return backward


def _substitute(kind: str, Lx: Expr, d: Expr, B: Expr, tables: _Tables) -> Expr:
  """``K^{-1} B`` from the factor ``(Lx, d)`` of ``P K P^T``, for ``B`` in the original numbering."""
  analysis = tables.analysis
  n = analysis.n
  l_ptr, l_rows = (Expr.const(tables.arrays[name], dtype=dtypes.int64) for name in ("l_ptr", "l_rows"))
  Y = gather(B, Expr.const(np.asarray(analysis.perm, dtype=np.int64), dtype=dtypes.int64), axis=0)
  sweep = (broadcast(Lx), broadcast(d), broadcast(l_ptr), broadcast(l_rows))
  Y, _ = scan(_lower_sweep(kind), Y, sweep, length=n, index=True)
  if kind == "ldl":
    Y = Y / d.reshape((n, *(1,) * (len(B.shape) - 1)))
  Y, _ = scan(_upper_sweep(kind), Y, sweep, length=n, index=True, reverse=True)
  return gather(Y, Expr.const(np.asarray(analysis.iperm, dtype=np.int64), dtype=dtypes.int64), axis=0)


# The matrix entries a factorization reads: of a pair stored in both triangles, the lower one.


def _read_coordinates(pattern: CanonicalPattern, tables: _Tables) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  source = tables.arrays["a_source"]
  return source, np.asarray(pattern.rows, dtype=np.int64)[source], np.asarray(pattern.cols, dtype=np.int64)[source]


def _symmetric_product(dK: Expr, X: Expr, pattern: CanonicalPattern, tables: _Tables) -> Expr:
  """``S X`` for the symmetric ``S`` that the read entries of the values ``dK`` stand for."""
  source, rows, cols = _read_coordinates(pattern, tables)
  off = rows != cols
  read = np.concatenate([source, source[off]])
  into, at = np.concatenate([rows, cols[off]]), np.concatenate([cols, rows[off]])
  index = lambda a: Expr.const(a, dtype=dtypes.int64)  # noqa: E731
  weights = gather(dK, index(read)).reshape((read.size, *(1,) * (len(X.shape) - 1)))
  zero = Expr.const(np.zeros(X.shape, dtype=X.type.dtype.numpy()), dtype=X.type.dtype)
  return scatter(zero, index(into), weights * gather(X, index(at), axis=0), axis=0, combine="add")


def _sampled_cotangent(B_bar: Expr, X: Expr, size: int, pattern: CanonicalPattern, tables: _Tables) -> Expr:
  """The values of ``-B_bar X^T`` on the read entries, both mirrored products on each off-diagonal one, and zero elsewhere."""
  source, rows, cols = _read_coordinates(pattern, tables)
  index = lambda a: Expr.const(a, dtype=dtypes.int64)  # noqa: E731
  axes = tuple(range(1, len(X.shape)))
  product = lambda r, c: (
    (gather(B_bar, index(r), axis=0) * gather(X, index(c), axis=0)).sum(axes) if axes else gather(B_bar, index(r)) * gather(X, index(c))
  )  # noqa: E731
  mirror = Expr.const((rows != cols).astype(X.type.dtype.numpy()), dtype=X.type.dtype)
  entries = -(product(rows, cols) + mirror * product(cols, rows))
  zero = Expr.const(np.zeros(size, dtype=X.type.dtype.numpy()), dtype=X.type.dtype)
  return scatter(zero, index(source), entries, combine="set")


# The library Functions, one per factorization and ordering, each a template over the pattern and
# the dtype of the matrix. Sparse values cross their boundaries in canonical, row-major order.


def _type_pattern(type_: TensorType) -> CanonicalPattern:
  """The pattern of a sparse type, which a sparse declaration guarantees."""
  assert type_.pattern is not None
  return type_.pattern


def _pattern(A: Expr) -> CanonicalPattern:
  return _type_pattern(A.type)


def _matrix(name: str) -> Any:
  return arg(name, dtype=..., pattern=..., layout="csr")


@cache
def _factorization(kind: str, ordering: Ordering, matrix: TensorType) -> Function:
  """The factorization of one matrix type, whose factor outputs are declared nondifferentiable.

  The outputs' types depend on the analysis of the pattern, so the Function is built for each
  type rather than as a template over its pattern.
  """
  tables = _tables(_type_pattern(matrix), ordering)
  analysis = tables.analysis
  n, dtype = analysis.n, matrix.dtype
  factor_type = TensorType((n, n), dtype, diff=False, pattern=_factor_layout(analysis)[0])
  outputs = (
    arg("L", factor_type, layout="csr"),
    *((arg("d", TensorType((n,), dtype, diff=False)),) if kind == "ldl" else ()),
    arg("status", _STATUS),
  )
  token = f"{'f32_' if dtype == dtypes.float32 else ''}p{_type_pattern(matrix).digest[:8]}"

  @function(arg("A", matrix, layout="csr"), outputs=group(*outputs), name=f"sparse_{kind}_{ordering}_{token}")
  def factor(A: Expr) -> tuple[Expr, ...]:
    Lx, d, status = _factor_parts(kind, values(A), tables)
    if kind == "cholesky":
      return _factor_value(Lx, d, analysis), status
    return _factor_value(Lx, _like(d, np.ones(n)), analysis), d, status

  return factor


@cache
def _solve(kind: str, ordering: Ordering) -> Function:
  """``(X, status)`` for ``K X = B``, with the implicit rules ``dX = K^{-1} (dB - dK X)``, ``B_bar = K^{-1} X_bar``, ``K_bar = -B_bar X^T``."""
  assume = "pos" if kind == "cholesky" else "sym"
  name = f"solve_{assume}_sparse_{ordering}"
  solve_outputs = group(arg("X", dtype=...), arg("status", _STATUS))

  def parts(K: Expr, B: Expr) -> tuple[Expr, Expr, Expr, Expr, _Tables]:
    tables = _tables(_pattern(K), ordering)
    Lx, d, status = _factor_parts(kind, values(K), tables)
    return _substitute(kind, Lx, d, B, tables), status, Lx, d, tables

  @function(_matrix("K"), arg("B", dtype=...), outputs=solve_outputs, name=name)
  def body(K: Expr, B: Expr) -> tuple[Expr, Expr]:
    X, status, *_ = parts(K, B)
    return X, status

  residuals = group(_matrix("K_"), arg("Lx", dtype=...), arg("d", dtype=...), arg("X_", dtype=...))

  @function(_matrix("K"), arg("B", dtype=...), outputs=group(solve_outputs, residuals), name=f"{name}_fwd")
  def fwd(K: Expr, B: Expr) -> tuple[tuple[Expr, Expr], tuple[Expr, Expr, Expr, Expr]]:
    X, status, Lx, d, _ = parts(K, B)
    return (X, status), (K, Lx, d, X)

  @function(residuals, group(arg("X_bar", dtype=...), arg("status_bar", _STATUS)), name=f"{name}_bwd")
  def bwd(saved: tuple[Expr, Expr, Expr, Expr], weights: tuple[Expr, Expr]) -> tuple[Expr, Expr]:
    (K, Lx, d, X), X_bar = saved, weights[0]
    tables = _tables(_pattern(K), ordering)
    B_bar = _substitute(kind, Lx, d, X_bar, tables)
    return pack(_sampled_cotangent(B_bar, X, _pattern(K).nnz, _pattern(K), tables), _pattern(K)), B_bar

  @function(residuals, group(_matrix("dK"), arg("dB", dtype=...)), outputs=solve_outputs, name=f"{name}_tangent")
  def tangent(saved: tuple[Expr, Expr, Expr, Expr], directions: tuple[Expr, Expr]) -> tuple[Expr, Expr]:
    """``dX = K^{-1} (dB - dK X)``, solved with the factorization and the solution of the call."""
    (K, Lx, d, X), (dK, dB) = saved, directions
    tables = _tables(_pattern(K), ordering)
    return _substitute(kind, Lx, d, dB - _symmetric_product(values(dK), X, _pattern(K), tables), tables), _status(0)

  return custom_derivative(body, fwd=fwd, tangent=tangent, bwd=bwd)


def _sparse_matrix(name: str, A: Expr) -> None:
  if A.type.dtype not in (dtypes.float32, dtypes.float64):
    raise TypeError(f"{name} takes a float32 or float64 matrix, got {A.type.dtype}")
  if len(A.shape) != 2:
    raise ValueError(f"{name} takes one sparse matrix, not a stack of them, got shape {A.shape}")
  if A.shape[0] != A.shape[1]:
    raise ValueError(f"{name} needs a square matrix, got shape {A.shape}")


def _declared(A: Expr) -> TensorType:
  """The type a factorization declares for its matrix: ``A``'s shape, dtype and pattern."""
  return TensorType(A.shape, A.type.dtype, pattern=A.type.pattern)


def cholesky(A: Expr, ordering: Ordering) -> tuple[Expr, Expr]:
  _sparse_matrix("cholesky", A)
  return _factorization("cholesky", ordering, _declared(A))(A)


def ldl(A: Expr, ordering: Ordering) -> tuple[Expr, Expr, Expr]:
  _sparse_matrix("ldl", A)
  return _factorization("ldl", ordering, _declared(A))(A)


def solve(K: Expr, B: Expr, assume: str, ordering: Ordering) -> tuple[Expr, Expr]:
  _sparse_matrix("solve", K)
  return _solve("cholesky" if assume == "pos" else "ldl", ordering)(K, B)
