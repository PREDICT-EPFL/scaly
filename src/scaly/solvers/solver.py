"""Select a solver plugin for a typed backend-free problem and wrap the result in ``Solver``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, overload

import numpy as np

from ..function.model import Function
from ..ir.expr import Expr
from .nlp import build_nlp
from .qp import build_qp
from .problem import Problem
from .registry import get_backend, require_backend
from .stats import SolverStats

type SolverFunction[SV, NV, SP, NP] = Function[
  tuple[SV, SV, Expr, Expr, SP],
  tuple[NV, NV, np.ndarray, np.ndarray, NP],
  tuple[SV, SV, Expr, Expr],
  tuple[NV, NV, np.ndarray, np.ndarray],
]
"""The plain ``Function`` behind a ``Solver``: inputs ``(vars_init, lam_box0, lam_eq0, lam_ineq0, params)``,
outputs ``(vars, lam_box, lam_eq, lam_ineq)``."""


def _zeros(tree: Any, symbolic: bool) -> Any:
  leaves = tuple(np.zeros(shape) for shape in tree.shapes)
  return tree.unflatten(tuple(Expr.const(leaf) for leaf in leaves) if symbolic else leaves)


@dataclass(frozen=True, slots=True)
class Solver[SV, NV, SP, NP]:
  """A compiled solver called with its parameters. The initial point and multipliers default to zero.

  ``function`` is the plain ``Function`` with the full five-group signature, for code generation
  and symbolic composition. Its four outputs are its first four
  inputs, so passing a previous result as ``warm`` warm-starts the next call.
  """

  function: SolverFunction[SV, NV, SP, NP]

  @overload
  def __call__(
    self, params: NP, /, *, x0: NV | None = None, warm: tuple[NV, NV, np.ndarray, np.ndarray] | None = None
  ) -> tuple[NV, NV, np.ndarray, np.ndarray]: ...

  @overload
  def __call__(self, params: SP, /, *, x0: SV | None = None, warm: tuple[SV, SV, Expr, Expr] | None = None) -> tuple[SV, SV, Expr, Expr]: ...

  def __call__(self, params: Any, /, *, x0: Any = None, warm: Any = None) -> Any:
    """Solve for ``params`` from ``warm``, or from ``x0`` and zero multipliers, or from zero."""
    if x0 is not None and warm is not None:
      raise TypeError("pass either x0 or warm, not both")
    parts = cast(Any, self.function.instantiate().input_tree).parts
    symbolic = parts[4].is_symbolic(params)
    if warm is None:
      warm = (_zeros(parts[0], symbolic) if x0 is None else x0, *(_zeros(part, symbolic) for part in parts[1:4]))
    init = tuple(warm)
    return self.function(*init, params)

  def stats(self) -> SolverStats:
    """The statistics of the latest numerical call."""
    return self.function.solver_stats()


def solver[SV, NV, SP, NP](
  problem: Problem[SV, NV, SP, NP],
  backend: str,
  /,
  *,
  name: str | None = None,
  options: dict[str, Any] | None = None,
) -> Solver[SV, NV, SP, NP]:
  """Build a typed ``Solver`` for one backend-free problem.

  Options are checked at construction and supplied to each numerical call.
  Solvers differing only in options share the generated module and its cache entry.
  """
  selected = get_backend(backend)
  solver_name = name or f"{problem.name}_{backend}"
  if selected.kind == "nlp":
    nlp_backend = require_backend(backend, "nlp")
    return Solver(build_nlp(problem, nlp_backend, name=solver_name, options=options))
  if selected.kind == "qp":
    return Solver(build_qp(problem, require_backend(backend, "qp"), name=solver_name, options=options))
  raise ValueError(f"solver plugin {backend!r} has unknown kind {selected.kind!r}")
