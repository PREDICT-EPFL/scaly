"""``SparseMatrix``: a matrix whose pattern is fixed when the graph is built and whose values are an ``Expr``.

The pattern is compressed sparse column (CSC) with sorted, distinct row indices, held in NumPy;
the values are one ``(nnz,)`` expression in that order. Every operation works out the result's
pattern at build time and expresses its values with the ordinary static-index expression ops
(``gather``, ``segment_sum``, ``scatter``), so derivatives, sparsity analysis and code generation
see nothing new: a ``SparseMatrix`` is a library value, not an IR type. Across a ``Function``
boundary it travels as its compact values, with ``sparsity`` as the pattern metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import EllipsisType
from typing import Any, cast

import numpy as np
from scipy import sparse

from ..ad.sparse import SparseJacobian
from ..function.tree import SymbolicValue, Tree
from ..ir.expr import Expr, ExprOp, as_expr, concat, gather, scatter, segment_sum
from ..ir.types import SparsityType, TensorType


def _pattern_arrays(shape: tuple[int, int], rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Sort coordinates into CSC order: ``(indptr, indices, order)`` with ``order`` the permutation of
  the given coordinates that produces it. The coordinates must be distinct."""
  m, n = shape
  rows, cols = np.asarray(rows, dtype=np.int64).reshape(-1), np.asarray(cols, dtype=np.int64).reshape(-1)
  if rows.size and (rows.min() < 0 or rows.max() >= m or cols.min() < 0 or cols.max() >= n):
    raise ValueError(f"sparse coordinates out of bounds for shape {shape}")
  order = np.lexsort((rows, cols))
  keys = cols[order] * m + rows[order]
  if keys.size and np.any(keys[1:] == keys[:-1]):
    raise ValueError("sparse coordinates must be distinct")
  indptr = np.zeros(n + 1, dtype=np.int64)
  np.add.at(indptr, cols + 1, 1)
  return np.cumsum(indptr), rows[order], order


def _zero(dtype: Any) -> Expr:
  return Expr.const(np.zeros(1), dtype=dtype)


def _pick(values: Expr, positions: np.ndarray) -> Expr:
  """``values[positions]``, where a position equal to ``values.size`` reads zero."""
  positions = np.asarray(positions, dtype=np.int64)
  if positions.size and positions.max() >= values.size:
    values = concat([values, _zero(values.type.dtype)])
  return gather(values, positions)


@dataclass(frozen=True, eq=False)
class SparseMatrix(SymbolicValue):
  """An ``m x n`` matrix with a static CSC pattern and an ``Expr`` of its ``nnz`` values.

  ``indptr`` (``n + 1``) and ``indices`` (``nnz``, sorted within each column) are NumPy arrays and
  never change at run time; ``values`` is any float expression of shape ``(nnz,)``: an input, a
  computed quantity, a sparse Jacobian's values. Operations build new patterns at generation time
  and new values from static gathers and segment sums, so everything is differentiable through the
  values and generates straight-line or looped code with tables.
  """

  shape: tuple[int, int]
  indptr: np.ndarray
  indices: np.ndarray
  values: Expr

  # Expressions and NumPy arrays defer to this class's reflected operators (``x @ A``, ``2.0 * A``).
  __array_ufunc__ = None

  def __post_init__(self) -> None:
    m, n = (int(d) for d in self.shape)
    object.__setattr__(self, "shape", (m, n))
    indptr = np.asarray(self.indptr, dtype=np.int64).reshape(-1)
    indices = np.asarray(self.indices, dtype=np.int64).reshape(-1)
    object.__setattr__(self, "indptr", indptr)
    object.__setattr__(self, "indices", indices)
    values = as_expr(self.values)
    object.__setattr__(self, "values", values)
    if indptr.size != n + 1 or indptr[0] != 0 or indptr[-1] != indices.size or np.any(np.diff(indptr) < 0):
      raise ValueError(f"indptr must have {n + 1} nondecreasing entries from 0 to nnz")
    if indices.size and (indices.min() < 0 or indices.max() >= m):
      raise ValueError(f"row indices out of bounds for {m} rows")
    cols = np.repeat(np.arange(n), np.diff(indptr))
    keys = cols * max(m, 1) + indices
    if keys.size and np.any(np.diff(keys) <= 0):
      raise ValueError("row indices must be sorted and distinct within each column")
    if values.shape != (indices.size,):
      raise ValueError(f"values must have shape ({indices.size},), got {values.shape}")

  # --- construction ---------------------------------------------------------------------------

  @staticmethod
  def from_coo(rows: Any, cols: Any, values: Any, shape: tuple[int, int]) -> SparseMatrix:
    """From coordinate triplets; repeated coordinates are summed."""
    values = as_expr(values)
    values = values.reshape((values.size,))
    rows, cols = np.asarray(rows, dtype=np.int64).reshape(-1), np.asarray(cols, dtype=np.int64).reshape(-1)
    if not rows.size == cols.size == values.size:
      raise ValueError(f"from_coo needs as many rows, cols and values, got {rows.size}, {cols.size}, {values.size}")
    m, n = shape
    if rows.size and (rows.min() < 0 or rows.max() >= m or cols.min() < 0 or cols.max() >= n):
      raise ValueError(f"sparse coordinates out of bounds for shape {shape}")
    keys = cols * m + rows
    unique, inverse = np.unique(keys, return_inverse=True)
    indptr, indices, _ = _pattern_arrays(shape, unique % max(m, 1), unique // max(m, 1))
    if unique.size == keys.size:
      order = np.argsort(inverse, kind="stable")
      return SparseMatrix(shape, indptr, indices, gather(values, order) if np.any(order != np.arange(order.size)) else values)
    return SparseMatrix(shape, indptr, indices, segment_sum(values, inverse, unique.size))

  @staticmethod
  def from_pattern(pattern: SparsityType | np.ndarray | sparse.sparray | sparse.spmatrix, values: Any) -> SparseMatrix:
    """From a pattern and values in that pattern's own order: coordinate order for a
    ``SparsityType``, row-major nonzero order for a boolean mask, stored order for a SciPy matrix."""
    rows, cols, shape = _coordinates(pattern)
    values = as_expr(values)
    if values.shape != (rows.size,):
      raise ValueError(f"values for a pattern with {rows.size} entries must have shape ({rows.size},), got {values.shape}")
    indptr, indices, order = _pattern_arrays(shape, rows, cols)
    reordered = values if np.array_equal(order, np.arange(order.size)) else gather(values, order)
    return SparseMatrix(shape, indptr, indices, reordered)

  @staticmethod
  def symbol(name: str, pattern: SparsityType | np.ndarray | sparse.sparray | sparse.spmatrix) -> SparseMatrix:
    """A matrix whose values are a new input symbol ``name`` of shape ``(nnz,)``, in CSC order. As
    a ``Function`` input it is just that vector."""
    rows, cols, shape = _coordinates(pattern)
    indptr, indices, _ = _pattern_arrays(shape, rows, cols)
    return SparseMatrix(shape, indptr, indices, Expr.sym(name, (indices.size,)))

  @staticmethod
  def from_scipy(a: sparse.sparray | sparse.spmatrix) -> SparseMatrix:
    """A constant matrix with SciPy's pattern (explicit zeros included) and values."""
    csc = sparse.csc_array(a, dtype=np.float64)
    csc.sum_duplicates()
    csc.sort_indices()
    return SparseMatrix(csc.shape, csc.indptr, csc.indices, Expr.const(np.asarray(csc.data, dtype=np.float64)))

  @staticmethod
  def from_dense(x: Any, pattern: SparsityType | np.ndarray | sparse.sparray | sparse.spmatrix | None = None) -> SparseMatrix:
    """The entries of a dense ``(m, n)`` expression on ``pattern``. Without one, a constant keeps
    its nonzeros and anything else keeps every entry."""
    x = as_expr(x)
    if len(x.shape) != 2:
      raise ValueError(f"from_dense needs a rank-2 expression, got shape {x.shape}")
    if pattern is None:
      mask = np.asarray(x.value) != 0 if x.op == ExprOp.CONST and x.value is not None else np.ones(x.shape, dtype=bool)
      pattern = mask
    rows, cols, shape = _coordinates(pattern)
    if shape != x.shape:
      raise ValueError(f"pattern shape {shape} does not match the expression's {x.shape}")
    indptr, indices, _ = _pattern_arrays(shape, rows, cols)
    flat = (indices * shape[1] + np.repeat(np.arange(shape[1]), np.diff(indptr))).astype(np.int64)
    return SparseMatrix(shape, indptr, indices, gather(x.reshape((x.size,)), flat))

  @staticmethod
  def from_sparse_jacobian(sj: SparseJacobian) -> SparseMatrix:
    """The result of ``sparse_jacobian``/``sparse_hessian`` as a matrix."""
    return SparseMatrix.from_pattern(sj.sparsity, sj.values)

  @staticmethod
  def diag(v: Any) -> SparseMatrix:
    """A square diagonal matrix with ``v`` on its diagonal."""
    v = as_expr(v)
    n = v.size
    return SparseMatrix((n, n), np.arange(n + 1), np.arange(n), v.reshape((n,)))

  @staticmethod
  def identity(n: int) -> SparseMatrix:
    return SparseMatrix.diag(Expr.const(np.ones(int(n))))

  @staticmethod
  def zeros(shape: tuple[int, int]) -> SparseMatrix:
    """An all-zero matrix with no stored entries."""
    return SparseMatrix(shape, np.zeros(shape[1] + 1, dtype=np.int64), np.zeros(0, dtype=np.int64), Expr.const(np.zeros(0)))

  @staticmethod
  def block(blocks: Sequence[Sequence[SparseMatrix | Expr | np.ndarray | sparse.sparray | sparse.spmatrix | None]]) -> SparseMatrix:
    """Assemble a block matrix, such as a KKT matrix. Each entry is a ``SparseMatrix``, a dense rank-2
    block (as ``from_dense``: a constant stores its nonzeros, an expression every entry) or ``None``
    for a zero block. Every block row needs a block that fixes its height and every block column one
    that fixes its width. A zero block stores nothing, diagonal included: ``SparseLDL`` needs every
    diagonal entry stored, which ``add_diagonal`` gives (as zeros if need be)."""
    grid = [[_as_sparse(b) for b in row] for row in blocks]
    if not grid or len({len(row) for row in grid}) != 1:
      raise ValueError("block needs a non-empty rectangular grid of blocks")
    heights = [_block_size([b.shape[0] for b in row if b is not None], f"block row {i}") for i, row in enumerate(grid)]
    columns = [[row[j] for row in grid] for j in range(len(grid[0]))]
    widths = [_block_size([b.shape[1] for b in col if b is not None], f"block column {j}") for j, col in enumerate(columns)]
    row_off, col_off = np.cumsum([0, *heights]), np.cumsum([0, *widths])
    rows, cols, parts = [], [], []
    for i, row in enumerate(grid):
      for j, b in enumerate(row):
        if b is None or not b.nnz:
          continue
        r, c = b.coordinates()
        rows.append(r + row_off[i])
        cols.append(c + col_off[j])
        parts.append(b.values)
    shape = (int(row_off[-1]), int(col_off[-1]))
    if not parts:
      return SparseMatrix.zeros(shape)
    return SparseMatrix.from_pattern(
      SparsityType(shape, tuple(int(x) for x in np.concatenate(rows)), tuple(int(x) for x in np.concatenate(cols))),
      concat(parts) if len(parts) > 1 else parts[0],
    )

  # --- inspection -----------------------------------------------------------------------------

  @property
  def nnz(self) -> int:
    return int(self.indices.size)

  def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, cols)`` of the stored entries, in the values' order."""
    return self.indices.copy(), np.repeat(np.arange(self.shape[1]), np.diff(self.indptr))

  @property
  def sparsity(self) -> SparsityType:
    """The pattern as a ``SparsityType`` whose coordinate order is the values' order, which is what a
    ``Function`` output's sparsity metadata needs."""
    rows, cols = self.coordinates()
    return SparsityType(self.shape, tuple(int(x) for x in rows), tuple(int(x) for x in cols))

  def _keys(self) -> np.ndarray:
    rows, cols = self.coordinates()
    return cols * self.shape[0] + rows

  def position(self, row: int, col: int) -> int | None:
    """Where entry ``(row, col)`` sits in ``values``, or None when it is not stored."""
    lo, hi = self.indptr[col], self.indptr[col + 1]
    k = lo + int(np.searchsorted(self.indices[lo:hi], row))
    return int(k) if k < hi and self.indices[k] == row else None

  def to_dense(self) -> Expr:
    """The dense ``(m, n)`` expression."""
    rows, cols = self.coordinates()
    return scatter(self.values, rows * self.shape[1] + cols, self.shape)

  def to_scipy(self, values: np.ndarray) -> sparse.csc_array:
    """A SciPy matrix with this pattern and numeric ``values``: for checking results."""
    return sparse.csc_array((np.asarray(values, dtype=np.float64).reshape(-1), self.indices, self.indptr), shape=self.shape)

  def with_values(self, values: Any) -> SparseMatrix:
    """The same pattern with other values, in the same order."""
    return SparseMatrix(self.shape, self.indptr, self.indices, as_expr(values))

  def diagonal(self) -> Expr:
    """The main diagonal as a dense vector; entries off the pattern are zero."""
    picks = [self.position(i, i) for i in range(min(self.shape))]
    return _pick(self.values, np.array([self.nnz if p is None else p for p in picks], dtype=np.int64))

  # --- structure-changing operations ----------------------------------------------------------

  @property
  def T(self) -> SparseMatrix:
    rows, cols = self.coordinates()
    indptr, indices, order = _pattern_arrays((self.shape[1], self.shape[0]), cols, rows)
    return SparseMatrix((self.shape[1], self.shape[0]), indptr, indices, gather(self.values, order) if self.nnz else self.values)

  def transpose(self) -> SparseMatrix:
    return self.T

  def select(self, keep: np.ndarray) -> SparseMatrix:
    """Only the stored entries where the boolean ``keep`` (one per entry, in values' order) holds."""
    keep = np.asarray(keep, dtype=bool).reshape(-1)
    if keep.size != self.nnz:
      raise ValueError(f"select needs one flag per stored entry ({self.nnz}), got {keep.size}")
    rows, cols = self.coordinates()
    indptr = np.concatenate([[0], np.cumsum(np.bincount(cols[keep], minlength=self.shape[1]))])
    return SparseMatrix(self.shape, indptr, rows[keep], gather(self.values, np.flatnonzero(keep)))

  def tril(self, k: int = 0) -> SparseMatrix:
    rows, cols = self.coordinates()
    return self.select(rows - cols >= -k)

  def triu(self, k: int = 0) -> SparseMatrix:
    rows, cols = self.coordinates()
    return self.select(cols - rows >= k)

  def with_pattern(self, other: SparseMatrix) -> SparseMatrix:
    """These values placed on a larger pattern that contains this one, zero elsewhere; also the way
    to make two matrices share one pattern before a loop."""
    if other.shape != self.shape:
      raise ValueError(f"shapes differ: {self.shape} vs {other.shape}")
    mine, theirs = self._keys(), other._keys()
    at = np.searchsorted(mine, theirs)
    found = (at < mine.size) & (mine[np.minimum(at, max(mine.size - 1, 0))] == theirs) if mine.size else np.zeros(theirs.size, dtype=bool)
    if int(found.sum()) != self.nnz:
      raise ValueError("the target pattern does not contain every stored entry")
    return SparseMatrix(self.shape, other.indptr, other.indices, _pick(self.values, np.where(found, at, self.nnz)))

  # --- arithmetic -----------------------------------------------------------------------------

  def __neg__(self) -> SparseMatrix:
    return self.with_values(-self.values)

  def __add__(self, other: Any) -> SparseMatrix:
    _refuse_scalar(other, "add")
    other = _as_sparse(other)
    if not isinstance(other, SparseMatrix):
      return NotImplemented
    if other.shape != self.shape:
      raise ValueError(f"cannot add {self.shape} and {other.shape}")
    a, b = self._keys(), other._keys()
    union = np.union1d(a, b)
    m = self.shape[0]
    shape = self.shape
    indptr = np.concatenate([[0], np.cumsum(np.bincount(union // max(m, 1), minlength=shape[1]))])
    indices = union % max(m, 1)
    values = _union_part(self.values, a, union) + _union_part(other.values, b, union)
    return SparseMatrix(shape, indptr, indices, values)

  __radd__ = __add__

  def __sub__(self, other: Any) -> SparseMatrix:
    _refuse_scalar(other, "subtract")
    other = _as_sparse(other)
    if not isinstance(other, SparseMatrix):
      return NotImplemented
    return self + (-other)

  def __rsub__(self, other: Any) -> SparseMatrix:
    return (-self) + other

  def __mul__(self, other: Any) -> SparseMatrix:
    """By a scalar, or elementwise by another ``SparseMatrix`` (the pattern intersection)."""
    if isinstance(other, SparseMatrix):
      if other.shape != self.shape:
        raise ValueError(f"cannot multiply {self.shape} and {other.shape} elementwise")
      a, b = self._keys(), other._keys()
      both, ia, ib = np.intersect1d(a, b, assume_unique=True, return_indices=True)
      m = self.shape[0]
      indptr = np.concatenate([[0], np.cumsum(np.bincount(both // max(m, 1), minlength=self.shape[1]))])
      return SparseMatrix(self.shape, indptr, both % max(m, 1), gather(self.values, ia) * gather(other.values, ib))
    s = as_expr(other)
    if s.size != 1:
      raise ValueError(f"a SparseMatrix scales by a scalar, got shape {s.shape}; use scale_rows or scale_cols")
    return self.with_values(self.values * s.reshape(()))

  __rmul__ = __mul__

  def __truediv__(self, other: Any) -> SparseMatrix:
    s = as_expr(other)
    if s.size != 1:
      raise ValueError(f"a SparseMatrix divides by a scalar, got shape {s.shape}")
    return self.with_values(self.values / s.reshape(()))

  def scale_rows(self, d: Any) -> SparseMatrix:
    """``diag(d) @ A``."""
    d = as_expr(d).reshape((self.shape[0],))
    return self.with_values(self.values * gather(d, self.indices))

  def scale_cols(self, d: Any) -> SparseMatrix:
    """``A @ diag(d)``."""
    d = as_expr(d).reshape((self.shape[1],))
    return self.with_values(self.values * gather(d, self.coordinates()[1]))

  def add_diagonal(self, d: Any) -> SparseMatrix:
    """``A + diag(d)`` for a square ``A``; ``d`` a vector or a scalar."""
    if self.shape[0] != self.shape[1]:
      raise ValueError(f"add_diagonal needs a square matrix, got {self.shape}")
    d = as_expr(d)
    n = self.shape[0]
    if d.size == 1 and n != 1:
      d = d.reshape(()) * Expr.const(np.ones(n))
    return self + SparseMatrix.diag(d)

  def __matmul__(self, other: Any) -> Any:
    if isinstance(other, SparseMatrix):
      return _spgemm(self, other)
    x = as_expr(other)
    if not x.shape or x.shape[0] != self.shape[1] or len(x.shape) > 2:
      raise ValueError(f"cannot multiply {self.shape} by {x.shape}")
    return _spmm(self, x)

  def __rmatmul__(self, other: Any) -> Any:
    x = as_expr(other)
    if len(x.shape) == 1:
      return self.T @ x
    return (self.T @ x.T).T

  def matvec(self, x: Any) -> Expr:
    return self @ x


class S(Tree[SparseMatrix, sparse.sparray]):
  """Declare one named sparse matrix: its pattern is part of the ``Function``'s signature.

  ``pattern`` is anything ``SparseMatrix.symbol`` takes (a ``SparsityType``, a boolean mask, a SciPy
  matrix) or a ``SparseMatrix`` whose pattern to copy; an output declared with ``...`` takes the
  pattern the body returns. The body receives a ``SparseMatrix``. A symbolic call must pass a
  ``SparseMatrix`` with exactly this pattern and an evaluation a SciPy sparse matrix with exactly
  this pattern, explicit zeros included; anything else is refused. An output comes back as a
  ``SparseMatrix`` from a symbolic call and a ``scipy.sparse.csc_array`` from an evaluation.

  Only the ``(nnz,)`` values, in CSC order, cross the generated C signature, exactly as for
  ``SparseMatrix.symbol``; the pattern is metadata fixed when the graph is built. An output's
  pattern is also the output's sparsity metadata in the generated header.
  """

  pattern: SparsityType | None

  def __init__(self, name: str, pattern: Any, /) -> None:
    if not isinstance(name, str) or not name:
      raise ValueError("S needs a non-empty name")
    self.names = (name,)
    if pattern is Ellipsis:
      self.pattern, self.decls = None, (Ellipsis,)
      return
    if isinstance(pattern, SparseMatrix):
      shape, indptr, indices = pattern.shape, pattern.indptr, pattern.indices
    else:
      rows, cols, shape = _coordinates(pattern)
      indptr, indices, _ = _pattern_arrays(shape, rows, cols)
    self._set_pattern(shape, indptr, indices)
    self.decls = (TensorType((self._indices.size,)),)

  def _set_pattern(self, shape: tuple[int, int], indptr: np.ndarray, indices: np.ndarray) -> None:
    self._shape = (int(shape[0]), int(shape[1]))
    self._indptr = np.asarray(indptr, dtype=np.int64)
    self._indices = np.asarray(indices, dtype=np.int64)
    self._scipy_index = np.int32 if max(self._indices.size, *self._shape) < np.iinfo(np.int32).max else np.int64
    cols = np.repeat(np.arange(self._shape[1]), np.diff(self._indptr))
    self.pattern = SparsityType(self._shape, tuple(int(r) for r in self._indices), tuple(int(c) for c in cols))

  def _copy(self, name: str, decl: TensorType | EllipsisType) -> S:
    out = S.__new__(S)
    out.names, out.decls, out.pattern = (name,), (decl,), self.pattern
    if self.pattern is not None:
      out._shape, out._indptr, out._indices, out._scipy_index = self._shape, self._indptr, self._indices, self._scipy_index
    return out

  def _require_pattern(self) -> None:
    if self.pattern is None:
      raise TypeError(f"sparse leaf {self.names[0]!r} has an inferred pattern; resolve it by tracing first")

  def _matrix(self, values: Expr) -> SparseMatrix:
    self._require_pattern()
    return SparseMatrix(self._shape, self._indptr, self._indices, values)

  def _mismatch(self, shape: tuple[int, ...], indptr: np.ndarray, indices: np.ndarray) -> str | None:
    """Why a pattern differs from the declared one, or None when it is the same."""
    if tuple(shape) != self._shape:
      return f"shape {tuple(shape)}, expected {self._shape}"
    if np.array_equal(indptr, self._indptr) and np.array_equal(indices, self._indices):
      return None
    for j in range(self._shape[1]):
      got, want = indices[indptr[j] : indptr[j + 1]], self._indices[self._indptr[j] : self._indptr[j + 1]]
      if not np.array_equal(got, want):
        return f"{indices.size} stored entries, expected {self._indices.size}; column {j} stores rows {got.tolist()}, expected {want.tolist()}"
    raise AssertionError("patterns differ but no column does")

  @property
  def sparsities(self) -> tuple[SparsityType | None, ...]:
    return (self.pattern,)

  def symbols(self, *, diff: bool | None = None) -> SparseMatrix:
    type_ = self.types[0]
    if diff is not None:
      type_ = TensorType(type_.shape, type_.dtype, type_.sparsity, diff)
    return self._matrix(Expr(ExprOp.INPUT, type=type_, name=self.names[0]))

  def relabel(self, prefix: str) -> S:
    return self._copy(prefix + self.names[0], self.decls[0])

  def with_types(self, types: tuple[TensorType, ...]) -> S:
    if len(types) != 1:
      raise ValueError(f"S expects one resolved type, got {len(types)}")
    if self.pattern is not None and types[0].shape != (self._indices.size,):
      raise ValueError(f"sparse leaf {self.names[0]!r} stores {self._indices.size} values, got type {types[0]}")
    return self._copy(self.names[0], types[0])

  def infer(self, value: SparseMatrix) -> S:
    if self.pattern is not None:
      return self
    out = self._copy(self.names[0], self.decls[0])
    out._set_pattern(value.shape, value.indptr, value.indices)
    return out

  def flatten_symbolic(self, value: SparseMatrix, what: str, *, allow_scalar: bool = False) -> tuple[Expr, ...]:
    if not isinstance(value, SparseMatrix):
      raise ValueError(f"{what}: expected a SparseMatrix for {self.names[0]!r}, got {type(value).__name__}")
    if self.pattern is not None and (why := self._mismatch(value.shape, value.indptr, value.indices)) is not None:
      raise ValueError(f"{what}: {self.names[0]!r} has a different sparsity pattern than declared: {why}")
    return (value.values,)

  def flatten_numerical(self, value: sparse.sparray, what: str) -> tuple[np.ndarray, ...]:
    if not isinstance(value, (sparse.sparray, sparse.spmatrix)):
      raise ValueError(f"{what}: expected a SciPy sparse matrix for {self.names[0]!r}, got {type(value).__name__}")
    # Float64 CSC, the form an evaluation returns, is used as it is: a conversion costs more than the
    # call it feeds for small matrices. Canonicalizing below copies first.
    is_csc = isinstance(value, (sparse.csc_array, sparse.csc_matrix)) and value.dtype == np.float64
    csc = cast(sparse.csc_array, value) if is_csc else sparse.csc_array(value, dtype=np.float64)
    if not csc.has_canonical_format:
      csc = csc.copy()
      csc.sum_duplicates()
    if (why := self._mismatch(csc.shape, csc.indptr, csc.indices)) is not None:
      raise ValueError(f"{what}: {self.names[0]!r} has a different sparsity pattern than declared: {why}")
    return (np.require(csc.data, dtype=np.float64, requirements="C"),)

  def unflatten(self, values: tuple[Any, ...]) -> Any:
    if len(values) != 1:
      raise ValueError(f"S expects one flat value, got {len(values)}")
    (value,) = values
    if isinstance(value, Expr):
      return self._matrix(value)
    self._require_pattern()
    # Fresh index arrays per result: a caller may edit a returned matrix's structure in place.
    index = self._scipy_index
    return sparse.csc_array(
      (np.asarray(value, dtype=np.float64).reshape(-1), self._indices.astype(index), self._indptr.astype(index)), shape=self._shape
    )


def _refuse_scalar(other: Any, what: str) -> None:
  if other is None or isinstance(other, (SparseMatrix, sparse.sparray, sparse.spmatrix)):
    return
  if (other.shape if isinstance(other, Expr) else np.shape(other)) == ():
    raise TypeError(f"cannot {what} a scalar and a sparse matrix: it would store every entry; use add_diagonal, or to_dense() first")


def _as_sparse(b: Any) -> SparseMatrix | None:
  if b is None or isinstance(b, SparseMatrix):
    return b
  if isinstance(b, (sparse.sparray, sparse.spmatrix)):
    return SparseMatrix.from_scipy(b)
  e = as_expr(b)
  if len(e.shape) != 2:
    raise ValueError(f"a dense block must be rank-2, got shape {e.shape}")
  return SparseMatrix.from_dense(e)


def _block_size(sizes: list[int], where: str) -> int:
  if not sizes:
    raise ValueError(f"{where} has no block that fixes its size")
  if len(set(sizes)) != 1:
    raise ValueError(f"{where} has blocks of different sizes {sorted(set(sizes))}")
  return sizes[0]


def _coordinates(pattern: Any) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
  if isinstance(pattern, SparsityType):
    return np.asarray(pattern.rows, dtype=np.int64), np.asarray(pattern.cols, dtype=np.int64), pattern.shape
  if isinstance(pattern, SparseMatrix):
    rows, cols = pattern.coordinates()
    return rows, cols, pattern.shape
  if isinstance(pattern, (sparse.sparray, sparse.spmatrix)):
    coo = sparse.coo_array(pattern)
    return np.asarray(coo.row, dtype=np.int64), np.asarray(coo.col, dtype=np.int64), (int(coo.shape[0]), int(coo.shape[1]))
  mask = np.asarray(pattern, dtype=bool)
  if mask.ndim != 2:
    raise ValueError(f"a pattern mask must be rank-2, got shape {mask.shape}")
  rows, cols = np.nonzero(mask)
  return rows.astype(np.int64), cols.astype(np.int64), (int(mask.shape[0]), int(mask.shape[1]))


def _union_part(values: Expr, keys: np.ndarray, union: np.ndarray) -> Expr:
  """``values`` spread over the union pattern, zero where this operand has no entry."""
  if keys.size == union.size:
    return values
  at = np.searchsorted(keys, union)
  hit = (at < keys.size) & (keys[np.minimum(at, max(keys.size - 1, 0))] == union) if keys.size else np.zeros(union.size, dtype=bool)
  return _pick(values, np.where(hit, at, keys.size))


def _spmm(a: SparseMatrix, x: Expr) -> Expr:
  """``A @ x`` for a dense vector or matrix: one product per stored entry, summed into its row."""
  m, n = a.shape
  rows, cols = a.coordinates()
  if len(x.shape) == 1:
    if not a.nnz:
      return Expr.const(np.zeros(m))
    return segment_sum(a.values * gather(x, cols), rows, m)
  k = x.shape[1]
  if not a.nnz:
    return Expr.const(np.zeros((m, k)))
  lanes = np.arange(k)
  picked = gather(x.reshape((n * k,)), (cols[:, None] * k + lanes[None, :]).reshape(-1)).reshape((a.nnz, k))
  products = (a.values.reshape((a.nnz, 1)) * picked).reshape((a.nnz * k,))
  return segment_sum(products, (rows[:, None] * k + lanes[None, :]).reshape(-1), m * k).reshape((m, k))


def _spgemm(a: SparseMatrix, b: SparseMatrix) -> SparseMatrix:
  """``A @ B`` with the product pattern worked out at build time: every pair of an entry ``(i, k)``
  of ``A`` and an entry ``(k, j)`` of ``B`` contributes to ``(i, j)``."""
  if a.shape[1] != b.shape[0]:
    raise ValueError(f"cannot multiply {a.shape} by {b.shape}")
  m, n = a.shape[0], b.shape[1]
  b_rows, b_cols = b.coordinates()
  counts = np.diff(a.indptr)[b_rows]  # entries of A's column k for each entry (k, j) of B
  pb = np.repeat(np.arange(b.nnz), counts)
  starts = np.repeat(a.indptr[b_rows], counts)
  offsets = np.arange(pb.size) - np.repeat(np.cumsum(counts) - counts, counts)
  pa = starts + offsets
  if not pa.size:
    return SparseMatrix.zeros((m, n))
  keys = b_cols[pb] * m + a.indices[pa]
  unique, inverse = np.unique(keys, return_inverse=True)
  indptr = np.concatenate([[0], np.cumsum(np.bincount(unique // m, minlength=n))])
  products = gather(a.values, pa) * gather(b.values, pb)
  values = products if unique.size == keys.size and np.array_equal(inverse, np.arange(keys.size)) else segment_sum(products, inverse, unique.size)
  return SparseMatrix((m, n), indptr, unique % m, values)
