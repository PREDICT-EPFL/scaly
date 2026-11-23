"""The entry points ``solve``, ``cholesky`` and ``ldl``: one spelling each, the factorization chosen by the matrix's type and ``assume``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import scipy.sparse

from ..function.model import Function
from ..ir.expr import Expr, _operands, as_expr, promote_dtype
from . import dense, sparse
from .dense import _general, _mapped, _positive, _symmetric, _system
from .symbolic import ORDERINGS, Ordering

_DENSE: dict[str, Callable[[int], Function]] = {"pos": lambda n: _positive, "sym": lambda n: _symmetric, "gen": _general}


def _ordering(name: str, A: Expr, ordering: Ordering | None) -> Ordering | None:
  """The ordering of a sparse ``A``, ``"amd"`` by default; a dense ``A`` takes none."""
  if A.type.pattern is None:
    if ordering is not None:
      raise TypeError(f"{name}: ordering applies to a sparse matrix, and this one is dense")
    return None
  if ordering is None:
    return "amd"
  if ordering not in ORDERINGS:
    raise ValueError(f"{name}: ordering must be one of {ORDERINGS}, got {ordering!r}")
  return ordering


def solve(K: Any, b: Any, *, assume: str, ordering: Ordering | None = None) -> tuple[Expr, Expr]:
  """Solve ``K x = b``, returning ``(x, status)``.

  ``assume`` states the class of ``K`` and has no default, since NumPy's and SciPy's solves take
  a general matrix. ``"pos"`` means symmetric positive definite, factored by
  `cholesky`. ``"sym"`` means symmetric quasi-definite, factored by `ldl` without pivoting.
  ``"gen"`` means general, factored by `lu_factor` with partial pivoting. The rules of each
  factorization for reading the lower triangle and for ``status`` apply here. ``K`` is
  a float32 or float64 square matrix, or a stack of them over leading axes. ``b`` is shaped ``(*batch, n)`` or
  ``(*batch, n, k)``, and ``x`` has its shape. ``b`` has the dtype of ``K``, which a Python number or
  list takes, and so has ``x``.

  A sparse ``K`` takes ``"pos"`` or ``"sym"``, and ``ordering`` as `cholesky` and `ldl` do. ``b``
  and ``x`` stay dense and in the original numbering, since the solve applies the ordering
  itself, and the status counts pivots in the reordered numbering.

  ``x`` is differentiable in ``K`` and ``b``. The reverse derivative reuses the factorization
  of the solve: ``b_bar = K^{-T} x_bar`` and ``K_bar = -b_bar x^T``, with the two mirrored
  entries of each lower-triangle entry summed when only the lower triangle is read. Forward
  derivatives are ``dx = K^{-1} (db - dK x)``, solved with the same factorization, so a
  solution and its Jacobian factor ``K`` once. The derivative is meaningless when ``status`` is
  nonzero.
  """
  # A SciPy matrix becomes a sparse expression first, so that a Python ``b`` takes its dtype.
  K, b = _operands(as_expr(K) if scipy.sparse.issparse(K) else K, b)
  ordering = _ordering("solve", K, ordering)
  if ordering is not None:
    if assume not in ("pos", "sym"):
      raise ValueError(f"solve: a sparse matrix takes assume 'pos' or 'sym', got {assume!r}")
    return sparse.solve(K, _sparse_right_hand_side(K, b), assume, ordering)
  if assume not in _DENSE:
    raise ValueError(f"solve: assume must be one of {sorted(_DENSE)}, got {assume!r}")
  K, B, batch, vector = _system("solve", K, b)
  X, status = _mapped(_DENSE[assume](K.shape[-1]), batch, K, B)
  return (X.reshape(X.shape[:-1]) if vector else X), status


def _sparse_right_hand_side(K: Expr, b: Expr) -> Expr:
  if b.type.pattern is not None:
    raise TypeError("solve takes a dense right-hand side, got a sparse one")
  try:
    promote_dtype(K, b)
  except TypeError as error:
    raise TypeError(f"solve: {error}") from None
  n = K.shape[-1]
  if b.shape[:1] != (n,) or len(b.shape) > 2:
    raise ValueError(f"solve: the right-hand side has shape {b.shape}, expected {(n,)} or {(n, 'k')}")
  return b


def cholesky(A: Any, *, ordering: Ordering | None = None) -> tuple[Expr, Expr]:
  """Factor a symmetric positive definite ``A`` as ``L L^T``, returning ``(L, status)``.

  Only the lower triangle of ``A`` is read, and an entry below the diagonal stands for itself
  and its mirror. ``A`` is a float32 or float64 square matrix, or a stack of them over leading axes. ``L`` is lower
  triangular with zeros above the diagonal, and has the dtype of ``A``.

  ``status`` holds one int32 per matrix. It is 0 when every pivot is usable, and otherwise
  ``j + 1`` for the first unusable pivot ``j`` counted from zero, so pivots are numbered from one
  as LAPACK numbers them. A pivot is unusable when it is not
  greater than the machine epsilon of ``A``'s dtype times the diagonal entry of ``A`` it came from, which includes
  zero, negative and NaN pivots. The entries of ``L`` from an unusable pivot on are meaningless.

  A sparse ``A`` is one matrix, whose pattern may store the lower triangle, the upper one or
  both: an entry stored in one triangle stands for its mirror, and of a pair stored in both the
  lower entry is read. ``ordering`` reorders it symmetrically before factoring, as
  ``P A P^T = L L^T``, to limit fill: ``"amd"``, approximate minimum degree and the default,
  ``"rcm"``, reverse Cuthill-McKee, or ``"natural"``. ``L`` is then sparse, with the structural pattern of the
  factor of ``P A P^T`` and its diagonal, in the reordered numbering, and the status counts pivots
  in that numbering. ``ordering`` with a dense ``A`` raises ``TypeError``.

  ``L`` is differentiable in ``A``. The derivative of an entry below the diagonal is the sum of
  the derivatives with respect to the two mirrored entries it stands for. The reverse derivative
  and the forward derivative of a dense factor reuse ``L``. A sparse
  ``L`` is meant to be read, not differentiated, and is declared nondifferentiable: differentiate
  `solve` instead.
  """
  A = as_expr(A)
  ordering = _ordering("cholesky", A, ordering)
  return dense.cholesky(A) if ordering is None else sparse.cholesky(A, ordering)


def ldl(A: Any, *, ordering: Ordering | None = None) -> tuple[Expr, Expr, Expr]:
  """Factor a symmetric quasi-definite ``A`` as ``L D L^T`` without pivoting, returning ``(L, d, status)``.

  Only the lower triangle of ``A`` is read, as in `cholesky`. ``A`` is a float32 or float64 square matrix, or
  a stack of them over leading axes. ``L`` is unit lower triangular, with ones on the diagonal and
  zeros above it, and ``d`` holds the diagonal of ``D``. Without pivoting the factorization exists
  for quasi-definite matrices, such as ``[[P, B^T], [B, -R]]`` with ``P`` and ``R`` positive definite,
  and its pivots may have either sign. Another symmetric matrix can give small pivots that pass
  the status check and factors with no accurate digits.

  ``status`` follows `cholesky`, with a pivot unusable when its magnitude is not greater than
  machine epsilon times the magnitude of the diagonal entry of ``A`` it came from, or when it is
  infinite.

  A sparse ``A`` follows `cholesky`: ``ordering`` gives ``P A P^T = L D L^T``, ``L`` is sparse with
  the pattern of the factor and ones on its diagonal, and ``d`` is a dense vector, both in the
  reordered numbering.

  The factors are meant to be read, not differentiated: differentiate `solve` instead. Dense
  factors have a forward rule, which reads the factors of the call, and no reverse rule, so a
  gradient through them differentiates the loops of the factorization. Sparse factors
  are declared nondifferentiable.
  """
  A = as_expr(A)
  ordering = _ordering("ldl", A, ordering)
  return dense.ldl(A) if ordering is None else sparse.ldl(A, ordering)
