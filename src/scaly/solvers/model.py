"""Solver descriptors and their opaque plain-ConcreteFunction expression graphs."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any

from ..ir.expr import Expr, ExprOp
from ..function.concrete import ConcreteFunction
from ..function.model import Function
from ..function.tree import Tree
from ..function.tree import flat_tree, flat_parameters
from ..ir.types import SparsityPattern, TensorType


class CSolverOption(ctypes.Structure):
  """One call-time solver option; a null name terminates the option array."""

  _fields_ = [("name", ctypes.c_char_p), ("kind", ctypes.c_int), ("integer", ctypes.c_int64), ("number", ctypes.c_double), ("text", ctypes.c_char_p)]


def solver_options(options: dict[str, Any]) -> ctypes.Array[CSolverOption]:
  """Pack validated options for the generated solver entry point."""
  values = []
  for name, value in options.items():
    if isinstance(value, (bool, int)):
      values.append(CSolverOption(name.encode(), 0, int(value), 0.0, None))
    elif isinstance(value, float):
      values.append(CSolverOption(name.encode(), 1, 0, value, None))
    else:
      values.append(CSolverOption(name.encode(), 2, 0, 0.0, value.encode()))
  return (CSolverOption * (len(values) + 1))(*values)


@dataclass(frozen=True, slots=True)
class ExternalOracle:
  """A C-ABI oracle supplied by a plugin consumer instead of an Scaly graph.

  ``source`` must define ``raw_symbol`` with the same flat-buffer convention
  as generated Scaly kernels: one ``const double*`` per input, one ``double*``
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
  backend: str  # solver plugin name (an ``scaly.solvers`` entry point, e.g. "piqp", "ipopt")
  n: int
  n_eq: int
  n_ineq: int
  # call-time input signature, in call order
  input_signature: tuple[tuple[str, tuple[int, ...]], ...]
  # output names + shapes, in solve-result order
  output_signature: tuple[tuple[str, tuple[int, ...]], ...]
  # param names (ordered parameter leaves after the fixed warm-start groups)
  param_names: tuple[str, ...]
  # Number of variable leaves at each end of the typed solver signature.
  n_var_blocks: int
  # Functions
  oracle: ConcreteFunction | None = None  # QP only
  base: ConcreteFunction | ExternalOracle | None = None  # NLP only
  grad: ConcreteFunction | ExternalOracle | None = None
  jac: ConcreteFunction | ExternalOracle | None = None
  hess: ConcreteFunction | ExternalOracle | None = None
  bounds: ConcreteFunction | ExternalOracle | None = None
  # Sparsity (NLP)
  jac_sparsity: SparsityPattern | None = None
  hess_sparsity: SparsityPattern | None = None
  # Sparse QP (PIQP sparse interface): structural CSC patterns of P (upper
  # triangle), A_eq, G_ineq, baked into the generated wrapper as static
  # tables; the oracle emits compact CSC-ordered value buffers. None => dense.
  P_sparsity: SparsityPattern | None = None
  A_sparsity: SparsityPattern | None = None
  G_sparsity: SparsityPattern | None = None
  # Oracle output naming (QP); the order in which the oracle's outputs encode
  # the QP data buffers.
  oracle_output_names: tuple[str, ...] = ()
  # Hash key used as a stable identifier (set in __post_init__)
  _key: int = field(default=0, hash=False, compare=False, repr=False)
  runtime_options: ctypes.Array[CSolverOption] = field(default_factory=lambda: (CSolverOption * 1)(), hash=False, compare=False, repr=False)

  def __post_init__(self) -> None:
    object.__setattr__(self, "_key", id(self))

  def __hash__(self) -> int:
    return self._key

  def __eq__(self, other: object) -> bool:
    return self is other

  def structural_key(self) -> tuple[Any, ...]:
    """Stable per-instance key used by ``Expr.structural_key`` for SOLVER_CALL nodes."""
    return ("SolverDescriptor", self._key, self.name, self.backend, self.n, self.n_eq, self.n_ineq)


def descriptor_function(
  descriptor: SolverDescriptor,
  input_tree: Tree[Any, Any] | None = None,
  output_tree: Tree[Any, Any] | None = None,
) -> Function[Any, Any, Any, Any]:
  """Build a Function whose concrete graph has opaque outputs sharing ``descriptor``."""
  input_exprs = tuple(Expr.sym(name, shape if shape else (), diff=False) for name, shape in descriptor.input_signature)
  args = tuple(input_exprs)
  output_exprs = tuple(
    Expr(
      ExprOp.SOLVER_CALL,
      args,
      TensorType(shape, diff=False),
      attrs={"solver": descriptor, "output": i, "output_name": name},
    )
    for i, (name, shape) in enumerate(descriptor.output_signature)
  )
  inputs = input_tree or flat_parameters(tuple(name for name, _ in descriptor.input_signature), tuple(expr.type for expr in input_exprs))
  outputs = output_tree or flat_tree(tuple(name for name, _ in descriptor.output_signature), tuple(expr.type for expr in output_exprs))
  function = ConcreteFunction._from_exprs(
    descriptor.name,
    input_exprs,
    output_exprs,
    tuple(name for name, _ in descriptor.input_signature),
    tuple(name for name, _ in descriptor.output_signature),
  )._with_trees(inputs, outputs)
  function.descriptor = descriptor
  return Function._from_instance(function)
