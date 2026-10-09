"""Concrete expression graphs, numerical calls, and the dependency-light DerivSpec base for named expression-dialect graphs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, Mapping, Sequence, cast, overload

import numpy as np

from ..ir.expr import Expr, ExprOp, check_prints_reach, linear_combination, recording_prints, topo
from ..ir.expr_spec import verify_expr
from ..ir.match import _apply_lowering
from ..ir.types import DeviceSpec, Lowering, SparsityPattern, TensorType, dtypes
from .tree import Tree, flat_tree, flat_parameters, inferred_outputs

if TYPE_CHECKING:
  from ..solvers.stats import SolverStats


def _jit():
  """The one sanctioned frontend->backend seam: calling a ``ConcreteFunction`` JIT-compiles it.

  Deferred so the frontend does not import the backend at module scope (see the import-layer table
  in ``docs/how_it_works/architecture.md``); every other use of the backend from here goes through it.
  """
  from ..codegen import jit

  return jit


@dataclass(frozen=True, slots=True)
class DerivSpec:
  """Base class for typed requests passed to ConcreteFunction.factory."""

  kind: ClassVar[str]
  of: str
  wrt: str

  @property
  def output_name(self) -> str:
    return f"{self.kind}_{self.of}_{self.wrt}"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, SparsityPattern | None, int | None]:
    raise NotImplementedError

  def _in(self, inputs: Mapping[str, Expr], name: str) -> Expr:
    if name not in inputs:
      raise ValueError(f"unknown factory input {name!r} in output {self}")
    return inputs[name]

  def _out(self, outputs: Mapping[str, Expr], name: str) -> Expr:
    if name not in outputs:
      raise ValueError(f"unknown factory output {name!r} in output {self}")
    return outputs[name]


Role = Literal["forward", "adjoint"]
"""What built a derived helper Function: a forward-mode or an adjoint call rule."""


@dataclass
class _Memo:
  """State computed from a ConcreteFunction and cached beside it, never part of its definition."""

  compiled: Any = None
  derivatives: dict[Any, ConcreteFunction] = field(default_factory=dict)
  maps: dict[int, ConcreteFunction] = field(default_factory=dict)
  helpers: dict[Any, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False, repr=False)
class ConcreteFunction[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
  """A named expression graph: named inputs, named outputs, and the computation between them.

  ``ConcreteFunction`` is the unit of composition, differentiation and compilation. Calling ``fn(inputs)``
  with ``Expr`` leaves puts one call node in a larger graph. Lowering may still inline a small
  callee when it expands a procedure into scalar code. ``fn.factory(...)`` derives a new
  ``ConcreteFunction`` carrying the requested derivatives. Calling ``fn(inputs)`` with array leaves lowers
  it, renders C, compiles and caches a shared library, and calls it through the pointer entry that
  every generated function shares.

  Each call takes one argument per parameter in the input tree. A group remains a tuple-valued
  parameter. Empty calls evaluate numerically; ``symbolic_call()`` embeds a zero-input graph.
  A single-output result is a bare leaf.

  Input and output names matter beyond display. Derivatives are requested by name, and the
  generated C symbols are built from them.

  A ConcreteFunction is immutable. Use ``@scaly.function(...)`` to build one from a Python body,
  or ``ConcreteFunction.build`` from expressions already in hand.
  """

  name: str
  input_tree: Tree[SymbolicInputs, NumericalInputs]
  inputs: tuple[Expr, ...]
  output_tree: Tree[SymbolicOutputs, NumericalOutputs]
  outputs: tuple[Expr, ...]
  output_sparsities: tuple[SparsityPattern | None, ...]
  output_coloring_widths: tuple[int | None, ...]
  device: DeviceSpec
  descriptor: Any = None
  role: Role | None = None
  _memo: _Memo = field(init=False, repr=False, default_factory=_Memo)

  @classmethod
  def build(
    cls,
    name: str,
    input_tree: Tree[Any, Any],
    inputs: Sequence[Expr],
    output_tree: Tree[Any, Any],
    outputs: Sequence[Expr],
    *,
    output_sparsities: Sequence[SparsityPattern | None] | None = None,
    output_coloring_widths: Sequence[int | None] | None = None,
    device: DeviceSpec | str | None = None,
    descriptor: Any = None,
    role: Role | None = None,
  ) -> ConcreteFunction[Any, Any, Any, Any]:
    """Build a function from its trees and the expressions at their leaves.

    ``inputs`` are the ``INPUT`` expressions of ``input_tree``'s leaves and ``outputs`` the
    expressions of ``output_tree``'s leaves, both in flat leaf order. Every input the outputs
    depend on must be among ``inputs``. Sparse outputs carry the pattern of their compact values.
    """
    return cls(
      name,
      input_tree,
      tuple(inputs),
      output_tree,
      tuple(outputs),
      tuple(output_sparsities) if output_sparsities is not None else (None,) * len(outputs),
      tuple(output_coloring_widths) if output_coloring_widths is not None else (None,) * len(outputs),
      DeviceSpec.parse(device),
      descriptor,
      role,
    )

  @classmethod
  def _trace(
    cls,
    name: str,
    fn: Callable[..., SymbolicOutputs],
    inputs: Tree[SymbolicInputs, NumericalInputs],
    outputs: Tree[SymbolicOutputs, NumericalOutputs] | None,
    output_name: str | None = None,
  ) -> ConcreteFunction[SymbolicInputs, NumericalInputs, SymbolicOutputs, NumericalOutputs]:
    symbolic_inputs = inputs.symbols()
    input_exprs = inputs.flatten_symbolic(symbolic_inputs, f"{name} inputs")
    with recording_prints() as prints:
      symbolic_outputs = fn(*cast(tuple[Any, ...], symbolic_inputs))
    if outputs is None:
      outputs = inferred_outputs(symbolic_outputs, output_name or name)
    try:
      output_exprs = outputs.flatten_symbolic(symbolic_outputs, f"{name} outputs")
    except ValueError as exc:
      raise TypeError(str(exc)) from exc
    check_prints_reach(prints, output_exprs, f"function {name!r}")
    output_types = outputs.resolved(tuple(expr.type for expr in output_exprs))
    return cls.build(name, inputs, input_exprs, outputs.with_types(output_types), output_exprs)

  @classmethod
  def _from_exprs(
    cls,
    name: str,
    inputs: Sequence[Expr],
    outputs: Sequence[Expr],
    input_names: Sequence[str],
    output_names: Sequence[str],
    output_sparsities: Sequence[SparsityPattern | None] | None = None,
    output_coloring_widths: Sequence[int | None] | None = None,
    role: Role | None = None,
  ) -> ConcreteFunction[Any, Any, Any, Any]:
    input_tree = flat_parameters(tuple(input_names), tuple(expr.type for expr in inputs))
    output_tree = flat_tree(tuple(output_names), tuple(expr.type for expr in outputs))
    return cls.build(
      name, input_tree, inputs, output_tree, outputs, output_sparsities=output_sparsities, output_coloring_widths=output_coloring_widths, role=role
    )

  def __post_init__(self) -> None:
    name = self.name
    if not all(isinstance(part, tuple) for part in (self.inputs, self.outputs, self.output_sparsities, self.output_coloring_widths)):
      raise TypeError(f"function {name!r}: construct ConcreteFunction through ConcreteFunction.build")
    non_inputs = [expr.name or f"%{expr.id}" for expr in self.inputs if expr.op != ExprOp.INPUT]
    if non_inputs:
      raise ValueError(f"function {name!r} declares non-input expressions as inputs: {non_inputs}")
    verify_expr((*self.inputs, *self.outputs))
    for expr in (*self.inputs, *self.outputs):
      if expr.type.dtype not in dtypes.all():
        raise ValueError(f"function {name!r} cannot lower dtype {expr.type.dtype} (input/output '{expr.name or '<?>'}').")

    def layout(types: Sequence[TensorType]) -> list[tuple[tuple[int, ...], Any]]:
      return [(type_.shape, type_.dtype) for type_ in types]

    if layout(self.input_tree.types) != layout([expr.type for expr in self.inputs]):
      raise ValueError(f"function {name!r}: input tree {self.input_tree.names} does not match the input expressions")
    if layout(self.output_tree.types) != layout([expr.type for expr in self.outputs]):
      raise ValueError(f"function {name!r}: output tree {self.output_tree.names} does not match the output expressions")
    if len(self.output_sparsities) != len(self.outputs):
      raise ValueError(f"expected {len(self.outputs)} output sparsities, got {len(self.output_sparsities)}")
    if len(self.output_coloring_widths) != len(self.outputs):
      raise ValueError(f"expected {len(self.outputs)} output coloring widths, got {len(self.output_coloring_widths)}")
    for output_name, out, sparsity in zip(self.output_names, self.outputs, self.output_sparsities, strict=True):
      if sparsity is not None and out.size != sparsity.nnz:
        raise ValueError(
          f"sparse output metadata for {output_name!r} has {sparsity.nnz} nonzeros, but output shape {out.shape} has {out.size} entries"
        )
    if len(set(self.input_names)) != len(self.input_names):
      raise ValueError(f"duplicate input names in {self.input_names}")
    if len(set(self.output_names)) != len(self.output_names):
      raise ValueError(f"duplicate output names in {self.output_names}")
    declared = {e.id for e in self.inputs}
    missing = [e.name or f"%{e.id}" for e in self.nodes if e.op == ExprOp.INPUT and e.id not in declared]
    if missing:
      raise ValueError(f"function {name!r} has undeclared symbolic inputs: {missing}")

  def __repr__(self) -> str:
    return f"ConcreteFunction({self.name!r}, {self.input_names}->{self.output_names})"

  @property
  def input_names(self) -> tuple[str, ...]:
    """The input leaf names in C-signature order."""
    return self.input_tree.names

  @property
  def output_names(self) -> tuple[str, ...]:
    """The output leaf names in C-signature order."""
    return self.output_tree.names

  def _replace(self, **changes: Any) -> ConcreteFunction[Any, Any, Any, Any]:
    """The one copy path: a validated copy with ``changes`` applied and no compiled or derived state."""
    return replace(self, **changes)

  def with_device(self, device: DeviceSpec | str) -> ConcreteFunction[Any, Any, Any, Any]:
    """Return a copy of this ConcreteFunction placed on ``device``."""
    return self._replace(device=DeviceSpec.parse(device))

  def _with_trees(self, input_tree: Tree[Any, Any], output_tree: Tree[Any, Any]) -> ConcreteFunction[Any, Any, Any, Any]:
    return self._replace(input_tree=input_tree, output_tree=output_tree)

  def input_map(self) -> dict[str, Expr]:
    """The symbolic input leaves keyed by their declared names, in declaration order."""
    return dict(zip(self.input_names, self.inputs, strict=True))

  def output_map(self) -> dict[str, Expr]:
    """The output expressions keyed by their declared names, in declaration order."""
    return dict(zip(self.output_names, self.outputs, strict=True))

  def output_sparsity_map(self) -> dict[str, SparsityPattern | None]:
    return dict(zip(self.output_names, self.output_sparsities, strict=True))

  @cached_property
  def nodes(self) -> tuple[Expr, ...]:
    """Every expression node the outputs reach, each after its operands."""
    return tuple(topo(self.outputs))

  @property
  def input_shapes(self) -> tuple[tuple[int, ...], ...]:
    """The input leaf shapes in C-signature order."""
    return tuple(expr.shape for expr in self.inputs)

  @property
  def output_shapes(self) -> tuple[tuple[int, ...], ...]:
    """The output leaf shapes in C-signature order."""
    return tuple(expr.shape for expr in self.outputs)

  @overload
  def __call__[*Ns](self: ConcreteFunction[SymbolicInputs, tuple[*Ns], SymbolicOutputs, NumericalOutputs], *args: *Ns) -> NumericalOutputs: ...

  @overload
  def __call__[*Ss](self: ConcreteFunction[tuple[*Ss], NumericalInputs, SymbolicOutputs, NumericalOutputs], *args: *Ss) -> SymbolicOutputs: ...

  def __call__(self, *args: Any) -> SymbolicOutputs | NumericalOutputs:
    """Call one argument per parameter, dispatching on symbolic or numerical leaves.

    An empty call evaluates numerically. Use ``symbolic_call()`` to embed a zero-input call.
    """
    inputs = cast(SymbolicInputs | NumericalInputs, args)
    if self.input_tree.is_symbolic(inputs):
      return self._symbolic(inputs)
    if self.input_tree.is_numerical(inputs):
      return self._numerical(inputs)
    raise TypeError(f"{self.name}: inputs mix Expr and numerical leaves; wrap numerical constants in scaly.const for a symbolic call")

  def symbolic_call[*Ss](self: ConcreteFunction[tuple[*Ss], NumericalInputs, SymbolicOutputs, NumericalOutputs], *args: *Ss) -> SymbolicOutputs:
    """Embed a call using one symbolic argument per declared parameter."""
    return self._symbolic(args)

  def _symbolic(self, args: SymbolicInputs) -> SymbolicOutputs:
    actuals = self.input_tree.flatten_symbolic(args, f"{self.name}.symbolic_call")
    actual_diff = any(arg.type.diff for arg in actuals)
    outputs = tuple(
      Expr(ExprOp.CALL, actuals, TensorType(out.shape, out.type.dtype, diff=out.type.diff and actual_diff), attrs={"callee": self, "output": i})
      for i, out in enumerate(self.outputs)
    )
    return cast(SymbolicOutputs, self.output_tree.unflatten(outputs))

  def numerical_call[*Ns](self: ConcreteFunction[SymbolicInputs, tuple[*Ns], SymbolicOutputs, NumericalOutputs], *args: *Ns) -> NumericalOutputs:
    """Compile and evaluate one numerical argument per declared parameter."""
    return self._numerical(args)

  def _numerical(self, args: NumericalInputs) -> NumericalOutputs:
    actuals = self.input_tree.flatten_numerical(args, f"{self.name}.numerical_call")
    return cast(NumericalOutputs, self.output_tree.unflatten(self._flat_numerical_call(*actuals)))

  def compile(self) -> None:
    """Compile ahead of the first numerical call, or reuse the cached library."""
    self._compile()

  def _compile(self) -> Any:
    """Lazily JIT-compile this function and cache the handle."""
    if self._memo.compiled is None:
      self._memo.compiled = _jit().CompiledFunction(self)
    return self._memo.compiled

  def _flat_numerical_call(self, *args: Any) -> tuple[np.ndarray, ...]:
    """Evaluate from flat leaves: lazily compile and run through the universal ABI.

    The leaf-level seam under ``numerical_call``. Nothing outside ``function/`` should reach for
    it; a caller holding flat leaves has ``input_tree.unflatten`` to build the declared tree.
    """
    return tuple(self._compile().run(list(args)))

  def recompile(self) -> None:
    """Drop the cached compiled handle and remove the on-disk cache entry for this function."""
    jit = _jit()
    self._memo.compiled = None
    jit.invalidate_cache(self)

  def solver_stats(self, name: str | None = None) -> SolverStats:
    """Return the latest stats for a solver reached by this compiled function."""
    jit = _jit()
    if self._memo.compiled is None:
      raise jit.JitError(f"function {self.name!r} has not been compiled or run")
    return self._memo.compiled.solver_stats(name)

  @cached_property
  def _effective_lowering(self) -> Lowering:
    """The hint that selects this ConcreteFunction's procedure: ``block`` or ``opaque`` anywhere wins, then ``scalar``, else ``auto``."""
    hints = {n.lowering for n in (*self.inputs, *self.nodes)}
    return "block" if hints & {"block", "opaque"} else "scalar" if "scalar" in hints else "auto"

  def _inherit_lowering(self, derived: Expr, lowering: Lowering | None = None) -> Expr:
    """Apply this ConcreteFunction's effective lowering policy to a derived expression."""
    policy = self._effective_lowering if lowering is None else lowering
    return _apply_lowering(derived, policy)

  def _with_outputs(self, outputs: Sequence[Expr]) -> ConcreteFunction[Any, Any, Any, Any]:
    return self._replace(outputs=tuple(outputs))

  def factory(
    self, name: str, inputs: Sequence[str], outputs: Sequence[str | DerivSpec], aux: Mapping[str, Sequence[str]] | None = None
  ) -> ConcreteFunction:
    """Build a new ``ConcreteFunction`` whose outputs mix this function's outputs and derivatives of them.

    Args:
      name: the new function's name, which also names its generated C symbols.
      inputs: the input names, in order. Besides the declared inputs, ``"fwd:<input>"`` names the
        seed of a forward derivative and ``"lam:<output>"`` the weight of an adjoint or of an
        ``aux`` combination.
      outputs: output names and ``DerivSpec`` requests such as ``Grad("cost", "x")``. A request's
        output is named ``{kind}_{of}_{wrt}``.
      aux: extra outputs usable by name in ``outputs`` and in requests. Each maps a new name to a
        list of output names and stands for their sum weighted by the matching ``"lam:<output>"``
        inputs, as in a Lagrangian.

    Returns:
      A ``ConcreteFunction`` with the selected inputs and the requested outputs.

    Raises:
      ValueError: if a name in ``inputs``, ``outputs`` or ``aux`` is unknown, or an ``aux`` name
        shadows an output.
    """
    in_expr = self.input_map()
    out_expr = self.output_map()
    aux = aux or {}

    duals: dict[str, Expr] = {}
    for out_name, out in out_expr.items():
      duals[out_name] = Expr(ExprOp.INPUT, type=out.type, name=f"lam:{out_name}")
    seeds: dict[str, Expr] = {}
    for in_name, inp in in_expr.items():
      seeds[in_name] = Expr(ExprOp.INPUT, type=inp.type, name=f"fwd:{in_name}")

    all_inputs = dict(in_expr)
    all_inputs.update({f"lam:{k}": v for k, v in duals.items()})
    all_inputs.update({f"fwd:{k}": v for k, v in seeds.items()})
    all_outputs = dict(out_expr)
    for aux_name, names in aux.items():
      if aux_name in out_expr:
        raise ValueError(f"factory aux output {aux_name!r} shadows an existing output")
      unknown = [n for n in names if n not in out_expr]
      if unknown:
        raise ValueError(f"unknown factory aux outputs for {aux_name!r}: {unknown}")
      all_outputs[aux_name] = linear_combination(out_expr, duals, list(names))

    unknown_inputs = [s for s in inputs if s not in all_inputs]
    if unknown_inputs:
      raise ValueError(f"unknown factory inputs: {unknown_inputs}")
    ret_inputs = [all_inputs[s] for s in inputs]
    ret_outputs: list[Expr] = []
    ret_output_names: list[str] = []
    ret_sparsities: list[SparsityPattern | None] = []
    ret_coloring_widths: list[int | None] = []
    for spec in outputs:
      if isinstance(spec, str):
        if spec not in all_outputs:
          raise ValueError(f"unknown factory output {spec!r}")
        output, sparsity, output_name = all_outputs[spec], None, spec
        coloring_width = None
      else:
        output, sparsity, coloring_width = spec.build(all_inputs, all_outputs)
        output_name = spec.output_name
      ret_outputs.append(output)
      ret_output_names.append(output_name)
      ret_sparsities.append(sparsity)
      ret_coloring_widths.append(coloring_width)
    return ConcreteFunction._from_exprs(
      name, ret_inputs, ret_outputs, inputs, ret_output_names, ret_sparsities, output_coloring_widths=ret_coloring_widths
    )
