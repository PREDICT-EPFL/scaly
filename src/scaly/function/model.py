"""Function and the dependency-light DerivSpec base for named expression-dialect graphs."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping, Sequence, cast, overload

import numpy as np

from ..ir.expr import Expr, ExprOp, as_expr, linear_combination, topo
from ..ir.match import _apply_lowering
from ..ir.types import DeviceSpec, Lowering, SparsityType, TensorType, backend_supports
from .tree import Tree, _G, flat_tree, is_symbolic_call, param_list

if TYPE_CHECKING:
  from ..solvers.stats import SolverStats


_JIT: Any = None


def _jit():
  """The one sanctioned frontend->backend seam: calling a ``Function`` JIT-compiles it.

  Deferred so the frontend does not import the backend at module scope (see the import-layer table
  in ``docs/how_it_works/architecture.md``); every other use of the backend from here goes through it.
  The module is kept after the first call: the import statement costs a third of a trivial call.
  """
  global _JIT
  if _JIT is None:
    from ..codegen import jit

    _JIT = jit
  return _JIT


@dataclass(frozen=True, slots=True)
class DerivSpec:
  """Base class for typed requests passed to Function.factory."""

  kind: ClassVar[str]
  of: str
  wrt: str

  @property
  def output_name(self) -> str:
    return f"{self.kind}_{self.of}_{self.wrt}"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, SparsityType | None, int | None]:
    raise NotImplementedError

  def _in(self, inputs: Mapping[str, Expr], name: str) -> Expr:
    if name not in inputs:
      raise ValueError(f"unknown factory input {name!r} in output {self}")
    return inputs[name]

  def _out(self, outputs: Mapping[str, Expr], name: str) -> Expr:
    if name not in outputs:
      raise ValueError(f"unknown factory output {name!r} in output {self}")
    return outputs[name]


class ConcreteFunction[**PS, **PN, SO, NO]:
  """A named expression graph: named inputs, named outputs, and the computation between them.

  ``Function`` is the unit of three things at once. **Composition** — ``fn(inputs)`` with ``Expr``
  leaves puts a first-class call node in a larger graph. Lowering may inline small pure callees
  when it scalarizes a procedure. **Differentiation** — ``fn.factory(...)``
  derives a new ``Function`` carrying the requested derivatives. **Compilation** — ``fn(inputs)``
  with array leaves lowers it, renders C, compiles and caches a shared library, and dispatches
  through the universal ABI.

  ``__call__`` takes one argument per parameter, each shaped as that parameter's declared tree, and
  dispatches on the leaves to ``symbolic_call`` or ``numerical_call``; call those directly when the
  distinction is the point. A one-leaf tree is the bare value on both sides — see ``scaly.L`` — so
  a single-output result must not be destructured. The type parameters are the symbolic and the
  numerical parameter lists and the symbolic and numerical output trees.

  Names are load-bearing: input and output names are how derivatives are requested and what the
  generated C symbols are built from.

  Use ``@scaly.function(...)`` to build one from a Python body.
  """

  descriptor: Any
  input_tree: _G
  output_tree: Tree[SO, NO]

  def __init__(
    self,
    name: str,
    fn: Callable[PS, SO],
    inputs: _G,
    outputs: Tree[SO, NO],
    *,
    device: DeviceSpec | str | None = None,
  ) -> None:
    """Trace ``fn`` over ``inputs``, a parameter list from ``tree.params``: the body gets one argument per slot."""
    symbolic_inputs = inputs.symbols()
    input_exprs = inputs.flatten_symbolic(symbolic_inputs, f"{name} inputs")
    symbolic_outputs = cast(Callable[..., Any], fn)(*symbolic_inputs)
    try:
      output_exprs = outputs.flatten_symbolic(symbolic_outputs, f"{name} outputs")
    except ValueError as exc:
      raise TypeError(str(exc)) from exc
    output_types = outputs.resolved(tuple(expr.type for expr in output_exprs))
    output_tree = outputs.infer(symbolic_outputs).with_types(output_types)
    self._init_graph(name, input_exprs, output_exprs, inputs, output_tree, output_tree.sparsities, device=device)
    try:
      self._signature = inspect.signature(fn)
    except (TypeError, ValueError):
      pass

  @classmethod
  def _from_exprs(
    cls,
    name: str,
    inputs: Sequence[Expr],
    outputs: Sequence[Expr],
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    output_sparsities: Sequence[SparsityType | None] | None = None,
    device: DeviceSpec | str | None = None,
    output_coloring_widths: Sequence[int | None] | None = None,
  ) -> ConcreteFunction[Any, Any, Any, Any]:
    inputs = tuple(inputs)
    outputs = tuple(outputs)
    raw_input_names = tuple(input_names) if input_names is not None else tuple(expr.name for expr in inputs)
    if any(name is None for name in raw_input_names):
      raise ValueError("all inputs must have names")
    resolved_input_names = cast(tuple[str, ...], raw_input_names)
    resolved_output_names = tuple(output_names) if output_names is not None else tuple(expr.name or f"out{i}" for i, expr in enumerate(outputs))
    if len(resolved_input_names) != len(inputs):
      raise ValueError(f"expected {len(inputs)} input names, got {len(resolved_input_names)}")
    if len(resolved_output_names) != len(outputs):
      raise ValueError(f"expected {len(outputs)} output names, got {len(resolved_output_names)}")
    instance = cls.__new__(cls)
    instance._init_graph(
      name,
      inputs,
      outputs,
      param_list(flat_tree(resolved_input_names, tuple(expr.type for expr in inputs))) if inputs else param_list(),
      flat_tree(resolved_output_names, tuple(expr.type for expr in outputs)),
      output_sparsities,
      device,
      output_coloring_widths,
    )
    return instance

  def _init_graph(
    self,
    name: str,
    inputs: Sequence[Expr],
    outputs: Sequence[Expr],
    input_tree: _G,
    output_tree: Tree[Any, Any],
    output_sparsities: Sequence[SparsityType | None] | None = None,
    device: DeviceSpec | str | None = None,
    output_coloring_widths: Sequence[int | None] | None = None,
  ) -> None:
    self.name = name
    # The body's signature binds keyword arguments; a Function with no body takes positional ones only.
    self._signature: inspect.Signature | None = None
    self.inputs = tuple(inputs)
    self.outputs = tuple(outputs)
    self.input_tree = input_tree
    self.output_tree = output_tree
    self.device: DeviceSpec = DeviceSpec.parse(device)
    for expr in (*self.inputs, *self.outputs):
      if not backend_supports(self.device, expr.type.dtype):
        raise ValueError(
          f"function {name!r} placed on {self.device} cannot lower dtype {expr.type.dtype} (input/output '{expr.name or '<?>'}'). "
          f"Use a different device or cast to a supported dtype."
        )
    self.input_names = input_tree.names
    self.output_names = output_tree.names
    self.output_sparsities = tuple(output_sparsities) if output_sparsities is not None else (None,) * len(self.outputs)
    self.output_coloring_widths = tuple(output_coloring_widths) if output_coloring_widths is not None else (None,) * len(self.outputs)
    if len(self.input_names) != len(self.inputs):
      raise ValueError(f"expected {len(self.inputs)} input names, got {len(self.input_names)}")
    if len(self.output_names) != len(self.outputs):
      raise ValueError(f"expected {len(self.outputs)} output names, got {len(self.output_names)}")
    if len(self.output_sparsities) != len(self.outputs):
      raise ValueError(f"expected {len(self.outputs)} output sparsities, got {len(self.output_sparsities)}")
    if len(self.output_coloring_widths) != len(self.outputs):
      raise ValueError(f"expected {len(self.outputs)} output coloring widths, got {len(self.output_coloring_widths)}")
    for name, out, sparsity in zip(self.output_names, self.outputs, self.output_sparsities, strict=True):
      if sparsity is not None and out.size != sparsity.nnz:
        raise ValueError(f"sparse output metadata for {name!r} has {sparsity.nnz} nonzeros, but output shape {out.shape} has {out.size} entries")
    if len(set(self.input_names)) != len(self.input_names):
      raise ValueError(f"duplicate input names in {self.input_names}")
    if len(set(self.output_names)) != len(self.output_names):
      raise ValueError(f"duplicate output names in {self.output_names}")
    declared = {e.id for e in self.inputs}
    missing = [e.name or f"%{e.id}" for e in topo(self.outputs) if e.op == ExprOp.INPUT and e.id not in declared]
    if missing:
      raise ValueError(f"function {self.name!r} has undeclared symbolic inputs: {missing}")
    self._compiled: Any = None
    # Derivative rules that replace differentiating the body; set by ``sc.custom_derivative``.
    self.custom_jvp: ConcreteFunction | None = None
    self.custom_vjp: ConcreteFunction | None = None
    # ``(output index, input index) -> pattern``, set by ``sc.custom_derivative(sparsity=...)``.
    self.custom_sparsity: Any = None

  def __repr__(self) -> str:
    suffix = f" device={self.device}" if self.device.kind != "host" else ""
    return f"Function({self.name!r}, {self.input_names}->{self.output_names}{suffix})"

  def with_device(self, device: DeviceSpec | str) -> "ConcreteFunction":
    """Return a copy of this Function placed on ``device``.

    This is a placement policy hint (see roadmap Phase 1 / Phase 9). Today
    only ``host`` actually lowers; non-host devices are accepted and tracked
    so debug output and verifier diagnostics can see them, but compilation
    only succeeds for placements with a registered backend.
    """
    instance = type(self).__new__(type(self))
    instance._init_graph(
      self.name,
      self.inputs,
      self.outputs,
      self.input_tree,
      self.output_tree,
      self.output_sparsities,
      device,
      self.output_coloring_widths,
    )
    instance._signature = self._signature
    return instance

  def _with_trees(self, input_tree: _G, output_tree: Tree[Any, Any]) -> ConcreteFunction[Any, Any, Any, Any]:
    """Replace the declared trees of a graph built from expressions; ``input_tree`` is a parameter list."""
    if input_tree.names != self.input_names or input_tree.types != tuple(expr.type for expr in self.inputs):
      raise ValueError("replacement input tree does not match the Function graph")
    if output_tree.names != self.output_names or output_tree.types != tuple(expr.type for expr in self.outputs):
      raise ValueError("replacement output tree does not match the Function graph")
    self.input_tree = input_tree
    self.output_tree = output_tree
    return self

  def input_map(self) -> dict[str, Expr]:
    return dict(zip(self.input_names, self.inputs, strict=True))

  def output_map(self) -> dict[str, Expr]:
    return dict(zip(self.output_names, self.outputs, strict=True))

  def output_sparsity_map(self) -> dict[str, SparsityType | None]:
    return dict(zip(self.output_names, self.output_sparsities, strict=True))

  @property
  def input_shapes(self) -> tuple[tuple[int, ...], ...]:
    """The input leaf shapes in C-signature order."""
    return tuple(expr.shape for expr in self.inputs)

  @property
  def output_shapes(self) -> tuple[tuple[int, ...], ...]:
    """The output leaf shapes in C-signature order."""
    return tuple(expr.shape for expr in self.outputs)

  # The numerical overload comes first on purpose: a Function built from bare expressions has
  # ``Any`` parameter lists, both overloads then match, and the first one wins. Evaluation is the
  # reading that untyped code wants, and a typed symbolic call still resolves exactly because
  # ``Expr`` is not assignable to the numerical leaf type.
  @overload
  def __call__(self, *args: PN.args, **kwargs: PN.kwargs) -> NO: ...

  @overload
  def __call__(self, *args: PS.args, **kwargs: PS.kwargs) -> SO: ...

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Call with one argument per parameter, dispatching on the leaves.

    All-``Expr`` leaves take the symbolic path and build a call node; anything else takes the
    numerical path and runs the compiled artifact, as does a call with no arguments (inside a traced
    body, write ``f.symbolic_call()``). A call mixing the two is an error: wrap the numerical leaves
    in ``scaly.const`` to make a symbolic call explicit. Keywords bind by the body's parameter names.
    """
    if kwargs:
      args = self._bind(args, kwargs)
    if is_symbolic_call(args, self.name):
      return self.symbolic_call(*args)
    return self.numerical_call(*args)

  def symbolic_call(self, *args: PS.args, **kwargs: PS.kwargs) -> SO:
    """Embed a call node; each argument has its parameter's declared symbolic structure."""
    values = self._bind(args, kwargs) if kwargs else args
    slots = self.input_tree.parts
    if len(values) != len(slots):
      raise self._arity_error(len(values))
    what = f"{self.name}.symbolic_call"
    actuals = [expr for slot, arg in zip(slots, values) for expr in slot.flatten_symbolic(arg, what)]
    return cast(SO, self.output_tree.unflatten(self._flat_symbolic_call(actuals)))

  def numerical_call(self, *args: PN.args, **kwargs: PN.kwargs) -> NO:
    """Compile and evaluate; each argument has its parameter's declared numerical structure."""
    values = self._bind(args, kwargs) if kwargs else args
    slots = self.input_tree.parts
    if len(values) != len(slots):
      raise self._arity_error(len(values))
    what = f"{self.name}.numerical_call"
    actuals = [array for slot, arg in zip(slots, values) for array in slot.flatten_numerical(arg, what)]
    return cast(NO, self.output_tree.unflatten(self._flat_numerical_call(*actuals)))

  def _bind(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    if self._signature is None:
      raise TypeError(f"{self.name}() takes positional arguments only, got {', '.join(kwargs)}")
    return self._signature.bind(*args, **kwargs).args

  def _arity_error(self, got: int) -> TypeError:
    if self._signature is not None:
      labels = list(self._signature.parameters)
    else:
      labels = [slot.names[0] if slot.size == 1 else f"({', '.join(slot.names)})" for slot in self.input_tree.parts]
    return TypeError(f"{self.name}() takes {len(labels)} argument{'' if len(labels) == 1 else 's'} ({', '.join(labels)}), got {got}")

  def _compile(self) -> Any:
    """Lazily JIT-compile this function and cache the handle."""
    if self._compiled is None:
      self._compiled = _jit().CompiledFunction(self)
    return self._compiled

  def _flat_numerical_call(self, *args: Any) -> tuple[np.ndarray, ...]:
    """Evaluate from flat leaves: lazily compile and run through the universal ABI.

    The leaf-level seam under ``numerical_call``. Nothing outside ``function/`` should reach for
    it; a caller holding flat leaves has ``input_tree.unflatten`` to build the declared tree.
    """
    jit = _jit()
    if self.device.kind != "host":
      raise jit.JitError(f"function {self.name!r} placed on {self.device}, but only host lowering is implemented.")
    return tuple(self._compile().run(list(args)))

  def recompile(self) -> None:
    """Drop the cached compiled handle and remove the on-disk cache entry for this function."""
    jit = _jit()
    self._compiled = None
    jit.invalidate_cache(self)

  def solver_stats(self, name: str | None = None) -> SolverStats:
    """Return the latest stats for a solver reached by this compiled function."""
    jit = _jit()
    if self._compiled is None:
      raise jit.JitError(f"function {self.name!r} has not been compiled or run")
    return self._compiled.solver_stats(name)

  def _effective_lowering(self) -> Lowering:
    """The hint that selects this Function's procedure: ``block`` or ``opaque`` anywhere wins, then ``scalar``, else ``auto``."""
    hints = {n.lowering for n in (*self.inputs, *topo(self.outputs))}
    return "block" if hints & {"block", "opaque"} else "scalar" if "scalar" in hints else "auto"

  def _inherit_lowering(self, derived: Expr, lowering: Lowering | None = None) -> Expr:
    """Apply this Function's effective lowering policy to a derived expression."""
    policy = self._effective_lowering() if lowering is None else lowering
    return _apply_lowering(derived, policy)

  def _with_outputs(self, outputs: Sequence[Expr]) -> ConcreteFunction[Any, Any, Any, Any]:
    """Return a private graph copy with replacement outputs and preserved Function metadata."""
    instance = type(self).__new__(type(self))
    instance._init_graph(
      self.name,
      self.inputs,
      outputs,
      self.input_tree,
      self.output_tree,
      self.output_sparsities,
      self.device,
      self.output_coloring_widths,
    )
    instance._signature = self._signature
    return instance

  def _flat_symbolic_call(self, args: Sequence[Any], /) -> tuple[Expr, ...]:
    """Build a call node from flat leaves. Raw values are coerced with ``as_expr``.

    The leaf-level seam under ``symbolic_call``. Differentiation is the one consumer outside
    ``function/``: it synthesizes callees from flat expression lists and calls them with the
    same list, including the zero-input case that ``__call__`` cannot route symbolically.
    """
    actuals = tuple(as_expr(arg) for arg in args)
    if len(actuals) != len(self.inputs):
      raise ValueError(f"expected {len(self.inputs)} call arguments, got {len(actuals)}")
    for name, expected, actual in zip(self.input_names, self.inputs, actuals, strict=True):
      if expected.shape != actual.shape:
        raise ValueError(f"call argument {name!r} has shape {actual.shape}, expected {expected.shape}")
    actual_diff = any(arg.type.diff for arg in actuals)
    return tuple(
      Expr(
        ExprOp.CALL,
        actuals,
        TensorType(out.shape, out.type.dtype, out.type.sparsity, diff=out.type.diff and actual_diff),
        attrs={"callee": self, "output": i},
      )
      for i, out in enumerate(self.outputs)
    )

  def factory(
    self, name: str, inputs: Sequence[str], outputs: Sequence[str | DerivSpec], aux: Mapping[str, Sequence[str]] | None = None
  ) -> ConcreteFunction:
    in_expr = self.input_map()
    out_expr = self.output_map()
    aux = aux or {}

    duals: dict[str, Expr] = {}
    for out_name, out in out_expr.items():
      duals[out_name] = Expr.sym(f"lam:{out_name}", out.shape)
    seeds: dict[str, Expr] = {}
    for in_name, inp in in_expr.items():
      seeds[in_name] = Expr.sym(f"fwd:{in_name}", inp.shape)

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
    ret_sparsities: list[SparsityType | None] = []
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


# The temporary alias while the tree migrates: the template takes the name `Function` in P2.
Function = ConcreteFunction
