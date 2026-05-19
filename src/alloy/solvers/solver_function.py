"""Opaque solver wrapper around an Alloy ``Function``.

A :class:`SolverFunction` is a real :class:`alloy.Function` whose outputs are
``Ops.SOLVER_CALL`` expression nodes. That means it can be:

- called directly from Python (``solver_function(...)`` returns a dict of
  numpy arrays, like before);
- nested inside a larger ``Function`` graph via the inherited
  :meth:`Function.call` (the call returns an :class:`Expr` per output);
- rendered to C by the standard code generator, which knows how to lower a
  ``SOLVER_CALL`` to a vendored PIQP/IPOPT invocation.

The expression-graph side of the solver is opaque: ``SOLVER_CALL`` nodes are
marked non-differentiable, and their attrs carry a :class:`SolverDescriptor`
holding every Python-level Function (oracle, gradient, Jacobian, Hessian,
bounds) plus solver options. Two ``SOLVER_CALL`` nodes that point to the same
descriptor and the same argument list describe the same solve — the tape
interpreter and the C renderer both deduplicate on that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..expr import Expr
from ..function import Function
from ..ops import Ops
from ..types import SparsityType, TensorType


@dataclass(frozen=True, slots=True)
class SolverStatus:
  code: int
  name: str
  iter: int = 0

  @property
  def ok(self) -> bool:
    # PIQP solved == 1, IPOPT solved == 0 / 1 / 6.
    return self.code in (0, 1, 6)


@dataclass(frozen=True)
class SolverDescriptor:
  """Everything needed to drive PIQP or IPOPT from either Python or generated C.

  Stored as a single attr on every ``Ops.SOLVER_CALL`` node so that nodes for
  different outputs of the same solve share one identity. Frozen + identity
  hash (via ``id``) so it can live inside ``Expr.attrs`` without surprising
  structural equality.
  """

  name: str
  backend: str  # "piqp" | "ipopt"
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
  base: Function | None = None  # NLP only
  grad: Function | None = None
  jac: Function | None = None
  hess: Function | None = None
  bounds: Function | None = None
  # Sparsity (NLP)
  jac_sparsity: SparsityType | None = None
  hess_sparsity: SparsityType | None = None
  hess_lower_mask: tuple[bool, ...] = ()
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
  """Function whose body is one ``Ops.SOLVER_CALL`` per output.

  Built by :func:`alloy.qp` and :func:`alloy.nlp`; not constructed directly.
  Inherits all of ``Function``'s call-time behavior (JIT-as-default, tape
  interpreter fallback, ``.call(...)`` for nested embedding). The descriptor
  side-table is shared by every SOLVER_CALL Expr produced for this solver.
  """

  def __init__(self, descriptor: SolverDescriptor) -> None:
    self.descriptor = descriptor
    input_exprs = [Expr.sym(n, s if s else (), diff=False) for n, s in descriptor.input_signature]
    input_names = [n for n, _ in descriptor.input_signature]
    args = tuple(input_exprs)
    output_exprs = [
      Expr(
        Ops.SOLVER_CALL,
        args,
        TensorType(shape, diff=False),
        attrs={"solver": descriptor, "output": i, "output_name": name},
      )
      for i, (name, shape) in enumerate(descriptor.output_signature)
    ]
    output_names = [n for n, _ in descriptor.output_signature]
    super().__init__(descriptor.name, input_exprs, output_exprs, input_names, output_names)
    self.last_status: SolverStatus | None = None

  def __repr__(self) -> str:
    return f"SolverFunction({self.name!r}, {self.input_names}->{self.output_names})"

  def __call__(self, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:  # ty: ignore[invalid-method-override]
    """Run the solver and return a name->array dict.

    This intentionally departs from ``Function.__call__``'s convention
    (returning a tuple/single array) because the solve has many semantically
    distinct outputs (primal, multipliers, status). Callers that just want
    the array list can use :meth:`eval_list` on the inherited interface.

    Top-level calls go through ``run_solver_backend`` directly so
    ``last_status`` is populated. Nested (graph-embedded) SOLVER_CALL nodes
    go through the tape interpreter / generated C, which drops the status.
    """
    ordered = self._resolve_inputs(args, kwargs)
    coerced = coerce_solver_inputs(self.descriptor, ordered)
    outs, status = run_solver_backend(self.descriptor, coerced)
    self.last_status = status
    return dict(zip(self.output_names, outs, strict=True))


def run_solver_backend(descriptor: SolverDescriptor, inputs: Sequence[np.ndarray]) -> tuple[list[np.ndarray], SolverStatus]:
  """Single entry point that runs PIQP or IPOPT from a SOLVER_CALL.

  Used by the tape interpreter and by :class:`SolverFunction.__call__` (which
  goes through the tape interpreter for SOLVER_CALL nodes since the JIT path
  for this op is not implemented yet). Inputs are positional in the same
  order as ``descriptor.input_signature``.
  """
  # Local imports avoid a runtime cycle: nlp.py imports SolverFunction.
  if descriptor.backend == "piqp":
    from .qp import _qp_backend

    return _qp_backend(descriptor, inputs)
  if descriptor.backend == "ipopt":
    from .nlp import _nlp_backend

    return _nlp_backend(descriptor, inputs)
  raise ValueError(f"unknown solver backend {descriptor.backend!r}")


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


