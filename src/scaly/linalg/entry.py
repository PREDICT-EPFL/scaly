"""The ``solve`` entry point: one spelling, the factorization chosen by the matrix's type and ``assume``."""

from __future__ import annotations

from typing import Any

from ..function.model import Function
from ..ir.expr import Expr, as_expr
from .dense import _mapped, _positive, _right_hand_side, _square

_DENSE: dict[str, Function] = {"pos": _positive}


def solve(K: Any, b: Any, *, assume: str) -> tuple[Expr, Expr]:
  """Solve ``K x = b``, returning ``(x, status)``.

  ``assume`` states the class of ``K`` and has no default, since NumPy's and SciPy's solves take
  a general matrix. ``"pos"`` means symmetric positive definite, factored by
  `cholesky`, whose rules for reading the lower triangle and for ``status`` apply here. ``K`` is
  a float64 square matrix, or a stack of them over leading axes. ``b`` is shaped ``(*batch, n)`` or
  ``(*batch, n, k)``, and ``x`` has its shape.

  ``x`` is differentiable in ``K`` and ``b``. The reverse derivative reuses the factorization
  of the solve: ``b_bar = K^{-1} x_bar`` and ``K_bar = -b_bar x^T``, with the two mirrored
  entries of each lower-triangle entry summed. Forward derivatives use the forward rule of
  `cholesky` and follow the substitutions. The derivative is meaningless when ``status`` is nonzero.
  """
  K = as_expr(K)
  if assume not in _DENSE:
    raise ValueError(f"solve: assume must be one of {sorted(_DENSE)}, got {assume!r}")
  batch = _square("solve", K)
  B, vector = _right_hand_side("solve", K, b)
  X, status = _mapped(_DENSE[assume], batch, K, B)
  return (X.reshape(X.shape[:-1]) if vector else X), status
