"""Solver descriptors and the Functions that call their external bridges."""

from __future__ import annotations

import ctypes
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, TYPE_CHECKING

from ..ir.expr import Expr
from ..ir.layout import DenseLayout
from ..function.concrete import ConcreteFunction
from ..function.extern import CLibrary, extern
from ..function.model import Function
from ..function.tree import Tree, arg
from ..function.tree import flat_tree, flat_parameters
from ..ir.types import SparsityPattern, TensorType
from .stages import StageLayout, detect_stage_layout


if TYPE_CHECKING:
  from ..codegen.toolchain import BuildRecipe


@dataclass(frozen=True)
class _SolverLibrary(CLibrary):
  """A native plugin declaration whose bridge is rendered with the module recipe."""

  function: ConcreteFunction = field(kw_only=True)
  recipe: BuildRecipe | None = None


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
class StageOracles:
  """Mapped dense derivatives in recorded stage order, with row-major stage blocks.

  ``grad`` returns the stacked objective gradient. ``jac`` returns the dynamics,
  initial, path and terminal Jacobians, each in local ``[x, u]`` order.
  ``hess`` takes the objective scale and stacked constraint multipliers after the
  ordinary inputs. It returns stage and terminal Lagrangian blocks, each a dense
  Hessian followed by its gradient row. A stage excludes the incoming transition's
  linear ``-xnext`` term. Box bounds are handled by the plugin.
  """

  grad: ConcreteFunction
  jac: ConcreteFunction
  hess: ConcreteFunction


@dataclass(frozen=True, eq=False)
class SolverDescriptor:
  """Everything a solver plugin's generated C wrapper needs to drive a solve.

  The solver Function retains this record for runtime options and statistics.
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
  _stage_layout: StageLayout | None = None
  stage_oracles: StageOracles | None = None
  # Sparse QP (PIQP sparse interface): structural CSC patterns of P (upper
  # triangle), A_eq, G_ineq, baked into the generated wrapper as static
  # tables; the oracle emits compact CSC-ordered value buffers. None => dense.
  P_sparsity: SparsityPattern | None = None
  A_sparsity: SparsityPattern | None = None
  G_sparsity: SparsityPattern | None = None
  # Oracle output naming (QP); the order in which the oracle's outputs encode
  # the QP data buffers.
  oracle_output_names: tuple[str, ...] = ()
  compile_options: tuple[tuple[str, Any], ...] = ()
  runtime_options: ctypes.Array[CSolverOption] = field(default_factory=lambda: (CSolverOption * 1)(), hash=False, compare=False, repr=False)

  @cached_property
  def stage_layout(self) -> StageLayout:
    """The supplied stage layout, or a layout detected and checked on first access."""
    if self._stage_layout is not None:
      return self._stage_layout
    if self.jac_sparsity is None or self.hess_sparsity is None:
      raise ValueError("stage detection requires NLP Jacobian and Lagrangian Hessian sparsity")
    return detect_stage_layout(self.jac_sparsity, self.hess_sparsity, self.n_eq)


def descriptor_function(
  descriptor: SolverDescriptor,
  input_tree: Tree[Any, Any] | None = None,
  output_tree: Tree[Any, Any] | None = None,
) -> Function[Any, Any, Any, Any]:
  """Build a Function calling the plugin's concrete extern bridge."""
  from .registry import get_backend

  input_exprs = tuple(Expr.sym(name, shape) for name, shape in descriptor.input_signature)
  output_exprs = tuple(Expr.const(np.zeros(shape)) for _, shape in descriptor.output_signature)
  inputs = input_tree or flat_parameters(tuple(name for name, _ in descriptor.input_signature), tuple(expr.type for expr in input_exprs))
  outputs = output_tree or flat_tree(tuple(name for name, _ in descriptor.output_signature), tuple(expr.type for expr in output_exprs))
  if any(isinstance(layout, DenseLayout) for layout in (*inputs.layouts, *outputs.layouts)):
    raise ValueError("solvers require row-major dense leaves; use an ordinary Function wrapper for a column-major interface")
  inputs = inputs.with_types(tuple(TensorType(type_.shape, type_.dtype) for type_ in inputs.types))
  outputs = outputs.with_types(tuple(TensorType(type_.shape, type_.dtype) for type_ in outputs.types))
  prototype = ConcreteFunction.build(descriptor.name, inputs, input_exprs, outputs, output_exprs, descriptor=descriptor)
  from .paths import solver_paths
  from ..utils.names import c_ident

  backend = get_backend(descriptor.backend)
  paths = solver_paths()
  load = paths.loads.get(backend.name)
  directories = (str(Path(load).parent),) if load and Path(load).is_absolute() else (() if load else (str(backend.lib_dir()),))
  library = _SolverLibrary(
    c_ident(descriptor.name),
    function=prototype,
    include_dirs=tuple(str(path) for path in paths.include_dirs if (path / backend.header).is_file()) or (str(backend.include_dir()),),
    library_dirs=directories,
    link=tuple(flag[2:] for flag in backend.link_flags if flag.startswith("-l")),
    compile_flags=tuple(flag for flag in backend.link_flags if not flag.startswith("-l")),
  )
  call = extern(
    library,
    f"{c_ident(descriptor.name)}_solve",
    inputs=tuple(arg(name, value.type) for name, value in zip(inputs.names, input_exprs)),
    outputs=outputs,
  )
  return Function._from_instance(prototype._replace(outputs=call._call(input_exprs)))
