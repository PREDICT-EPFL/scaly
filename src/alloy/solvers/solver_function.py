"""Opaque solver wrapper around an Alloy ``Function``.

A :class:`SolverFunction` is a real :class:`alloy.Function` whose outputs are
``ExprOp.SOLVER_CALL`` expression nodes. That means it can be:

- called directly from Python (``solver_function(...)`` returns a dict of
  numpy arrays, like before);
- nested inside a larger ``Function`` graph via the inherited
  :meth:`Function.call` (the call returns an :class:`Expr` per output);
- rendered to C by the standard code generator, which lowers a
  ``SOLVER_CALL`` to a solver-plugin-rendered wrapper over the vendored
  solver's C API.

The expression-graph side of the solver is opaque: ``SOLVER_CALL`` nodes are
marked non-differentiable, and their attrs carry a :class:`SolverDescriptor`
holding every Python-level Function (oracle, gradient, Jacobian, Hessian,
bounds) plus solver options. Two ``SOLVER_CALL`` nodes that point to the same
descriptor and the same argument list describe the same solve — the Program IR
lowerer and generated C solver wrapper deduplicate on that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..ir.expr import Expr, ExprOp
from ..function import Function
from ..function.tree import flat_tree
from ..ir.types import SparsityType, TensorType
from .stats import SolverStats, SolverStatus


@dataclass(frozen=True, slots=True)
class ExternalOracle:
  """A C-ABI oracle supplied by a plugin consumer instead of an Alloy graph.

  ``source`` must define ``raw_symbol`` with the same flat-buffer convention
  as generated Alloy kernels: one ``const double*`` per input, one ``double*``
  per output, and a trailing ``double*`` workspace argument. ``workspace_size``
  declares the number of doubles available through that final argument.
  """

  name: str
  raw_symbol: str
  source: str
  input_signature: tuple[tuple[str, tuple[int, ...]], ...]
  output_signature: tuple[tuple[str, tuple[int, ...]], ...]
  workspace_size: int = 0

  def __post_init__(self) -> None:
    if self.workspace_size < 0:
      raise ValueError(f"external oracle workspace_size must be nonnegative, got {self.workspace_size}")


@dataclass(frozen=True)
class SolverDescriptor:
  """Everything a solver plugin's generated C wrapper needs to drive a solve.

  Stored as a single attr on every ``ExprOp.SOLVER_CALL`` node so that nodes for
  different outputs of the same solve share one identity. Frozen + identity
  hash (via ``id``) so it can live inside ``Expr.attrs`` without surprising
  structural equality.
  """

  name: str
  backend: str  # solver plugin name (an ``alloy.solvers`` entry point, e.g. "piqp", "ipopt")
  n: int
  n_eq: int
  n_ineq: int
  # call-time input signature, in call order
  input_signature: tuple[tuple[str, tuple[int, ...]], ...]
  # output names + shapes, in solve-result order
  output_signature: tuple[tuple[str, tuple[int, ...]], ...]
  # param names (subset of input_signature beyond x0/lam_eq0/lam_ineq0)
  param_names: tuple[str, ...]
  # Functions
  oracle: Function | None = None  # QP only
  base: Function | ExternalOracle | None = None  # NLP only
  grad: Function | ExternalOracle | None = None
  jac: Function | ExternalOracle | None = None
  hess: Function | ExternalOracle | None = None
  bounds: Function | ExternalOracle | None = None
  # Sparsity (NLP)
  jac_sparsity: SparsityType | None = None
  hess_sparsity: SparsityType | None = None
  # Sparse QP (PIQP sparse interface): structural CSC patterns of P (upper
  # triangle), A_eq, G_ineq, baked into the generated wrapper as static
  # tables; the oracle emits compact CSC-ordered value buffers. None => dense.
  sparse: bool = False
  P_sparsity: SparsityType | None = None
  A_sparsity: SparsityType | None = None
  G_sparsity: SparsityType | None = None
  # Solver-specific options
  options: tuple[tuple[str, Any], ...] = ()
  # Oracle output naming (QP); the order in which the oracle's outputs encode
  # the QP data buffers.
  oracle_output_names: tuple[str, ...] = ()
  # Hash key used as a stable identifier (set in __post_init__)
  _key: int = field(default=0, hash=False, compare=False, repr=False)
  # Mutable per-descriptor runtime state (e.g. PIQP workspace handle). Frozen
  # is fine — we mutate the dict's contents, not the binding itself.
  runtime: dict[str, Any] = field(default_factory=dict, hash=False, compare=False, repr=False)

  def __post_init__(self) -> None:
    object.__setattr__(self, "_key", id(self))

  def __hash__(self) -> int:
    return self._key

  def __eq__(self, other: object) -> bool:
    return self is other

  def structural_key(self) -> tuple[Any, ...]:
    """Stable per-instance key used by ``Expr.structural_key`` for SOLVER_CALL nodes."""
    return ("SolverDescriptor", self._key, self.name, self.backend, self.n, self.n_eq, self.n_ineq)


class SolverFunction(Function):
  """Function whose body is one ``ExprOp.SOLVER_CALL`` per output.

  Built by :func:`alloy.qp` and :func:`alloy.nlp`; not constructed directly.
  Inherits all of ``Function``'s call-time behavior (JIT-as-default,
  ``.call(...)`` for nested embedding). The descriptor
  side-table is shared by every SOLVER_CALL Expr produced for this solver.
  """

  def __init__(self, descriptor: SolverDescriptor) -> None:
    self.descriptor = descriptor
    input_exprs = [Expr.sym(n, s if s else (), diff=False) for n, s in descriptor.input_signature]
    input_names = [n for n, _ in descriptor.input_signature]
    args = tuple(input_exprs)
    output_exprs = [
      Expr(
        ExprOp.SOLVER_CALL,
        args,
        TensorType(shape, diff=False),
        attrs={"solver": descriptor, "output": i, "output_name": name},
      )
      for i, (name, shape) in enumerate(descriptor.output_signature)
    ]
    output_names = [n for n, _ in descriptor.output_signature]
    self._init_graph(
      descriptor.name,
      input_exprs,
      output_exprs,
      flat_tree(tuple(input_names), tuple(expr.type for expr in input_exprs)),
      flat_tree(tuple(output_names), tuple(expr.type for expr in output_exprs)),
    )
    self.last_status: SolverStatus | None = None
    self.last_stats: SolverStats | None = None

  def __repr__(self) -> str:
    return f"SolverFunction({self.name!r}, {self.input_names}->{self.output_names})"

  def call(self, args: Sequence[Any] | None = None, /, **kwargs: Any) -> tuple[Expr, ...]:  # ty: ignore[invalid-method-override]
    """Embed this solver as a node in a larger expression graph.

    Either positional (list of args in input order) or keyword, matching
    ``__call__``'s resolution. Returns one ``Expr`` per solver output.
    """
    if args is not None and kwargs:
      raise TypeError("pass positional inputs or keyword inputs, not both")
    if kwargs:
      missing = [n for n in self.input_names if n not in kwargs]
      extra = [n for n in kwargs if n not in self.input_names]
      if missing or extra:
        parts: list[str] = []
        if missing:
          parts.append(f"missing keyword inputs: {missing}")
        if extra:
          parts.append(f"unexpected keyword inputs: {extra}")
        raise TypeError(", ".join(parts))
      ordered = [kwargs[n] for n in self.input_names]
    else:
      ordered = list(args) if args is not None else []
    return Function.call(self, ordered)

  def __call__(self, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:  # ty: ignore[invalid-method-override]
    """Run the solver and return a name->array dict.

    This intentionally departs from ``Function.__call__``'s convention
    (returning a tuple/single array) because the solve has many semantically
    distinct outputs (primal, multipliers, status). Callers that just want
    the array list can use :meth:`eval_list` on the inherited interface.

    The solve runs through the generated C wrapper — the only solve path.
    """
    ordered = self._resolve_inputs(args, kwargs)
    coerced = coerce_solver_inputs(self.descriptor, ordered)
    compiled = self._compile()
    outs = compiled.run(coerced)
    self.last_stats = compiled.solver_stats(self.name)
    self.last_status = self.last_stats.to_solver_status()
    return dict(zip(self.output_names, outs, strict=True))


def _coerce_input(name: str, shape: tuple[int, ...], val: Any) -> np.ndarray:
  arr = np.asarray(val, dtype=np.float64)
  if shape and arr.shape != shape:
    try:
      arr = arr.reshape(shape)
    except ValueError as exc:
      raise ValueError(f"input {name!r}: cannot reshape {arr.shape} to {shape}") from exc
  return np.ascontiguousarray(arr)


def coerce_solver_inputs(descriptor: SolverDescriptor, args: Sequence[Any]) -> list[np.ndarray]:
  """Validate / reshape an argument list against the descriptor's signature."""
  if len(args) != len(descriptor.input_signature):
    raise TypeError(f"expected {len(descriptor.input_signature)} inputs, got {len(args)}")
  return [_coerce_input(n, s, a) for (n, s), a in zip(descriptor.input_signature, args, strict=True)]
