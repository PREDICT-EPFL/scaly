"""Function (a body instantiated per argument signature), ConcreteFunction (one named graph), and DerivSpec."""

from __future__ import annotations

import hashlib
import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping, Sequence, cast, overload

import numpy as np

from ..ir.expr import Expr, ExprOp, as_expr, linear_combination, topo
from ..ir.match import _apply_lowering
from ..ir.target import Target, get_target
from ..ir.types import DeviceSpec, Lowering, SparsityType, TensorType, as_shape, backend_supports, dtypes
from .tree import Hole, LeafDecl, SymbolicValue, Tree, _G, _leaves, flat_tree, inferred_tree, is_symbolic_call, param_list, skeleton

if TYPE_CHECKING:
  from .extern import ExternCallee


_log = logging.getLogger(__name__)
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


class NotConcrete(TypeError):
  """A Function with shape holes was used where one concrete instance is needed; ``instantiate`` it first."""


# The graph attributes of an instance. On a template they are its one instance's, so asking a template
# with holes for one raises NotConcrete, which names the fix, instead of an AttributeError.
_GRAPH_ATTRIBUTES = frozenset(
  {
    "input_tree",
    "output_tree",
    "input_names",
    "output_names",
    "inputs",
    "outputs",
    "device",
    "output_sparsities",
    "output_coloring_widths",
    "extern",
    "custom_jvp",
    "custom_vjp",
    "custom_sparsity",
    "_compiled",
    "_compiled_for",
    "_compile",
    "_flat_numerical_call",
    "_flat_symbolic_call",
    "_with_trees",
    "_with_outputs",
    "_effective_lowering",
    "_inherit_lowering",
  }
)


class Function[**PS, **PN, SO, NO]:
  """A Python body over declared parameters, traced into one concrete instance per argument signature.

  ``@scaly.function`` builds one. Where every parameter's shape is declared, the body is traced at
  the decorator and the result is that instance, a ``ConcreteFunction``: ``is_concrete`` holds and
  ``concrete`` is the function itself. Where a declaration leaves holes (``sc.L()``, ``(n, None)``),
  each call binds them to its arguments' shapes, traces the body once per distinct binding, and
  caches the instance under a name that spells the bound shapes (``f__3x4``), which is also its C
  symbol. ``instances`` holds what has been built; ``instantiate`` builds one ahead of time.

  A call dispatches as a ``ConcreteFunction``'s does: ``Expr`` arguments build a call node, arrays
  run the compiled instance. The graph attributes (``input_names``, ``inputs``, ...) belong to an
  instance, so on a template with holes they raise ``NotConcrete``. The type parameters are the
  symbolic and numerical parameter lists and the symbolic and numerical output trees.
  """

  name: str
  input_tree: _G
  output_tree: Tree[SO, NO]
  input_names: tuple[str, ...]
  output_names: tuple[str, ...]
  inputs: tuple[Expr, ...]
  outputs: tuple[Expr, ...]
  device: DeviceSpec
  output_sparsities: tuple[SparsityType | None, ...]
  output_coloring_widths: tuple[int | None, ...]
  extern: ExternCallee | None
  _signature: inspect.Signature | None

  def __init__(
    self, name: str, fn: Callable[PS, SO], slots: _G | None, output: Tree[SO, NO] | None, *, device: DeviceSpec | str | None = None
  ) -> None:
    """A template over ``slots``, a named parameter list with holes, or with ``None`` every argument's
    structure, shape and dtype bound at the call; an ``output`` of ``None`` is read off the trace."""
    self.name = name
    self._fn = fn
    self._slots = slots
    self._output = output
    self._device = device
    self._signature = inspect.signature(fn)
    self._cache: dict[Any, ConcreteFunction[PS, PN, SO, NO]] = {}
    self._instances: dict[str, ConcreteFunction[PS, PN, SO, NO]] = {}
    # The same instances by the arguments' flat shapes and dtypes, which decide the binding when the
    # arguments bind at all: a hit skips building the bound declaration, and the instance's own call
    # still checks structure and fixed shapes.
    self._by_arguments: dict[tuple[Any, ...], ConcreteFunction[PS, PN, SO, NO]] = {}
    # A derived template's instance is ``transform`` of its source's; see ``lift``.
    self._source: Function[Any, Any, Any, Any] | None = None
    self._transform: Callable[[ConcreteFunction[Any, Any, Any, Any]], ConcreteFunction[PS, PN, SO, NO]] | None = None
    self._extra: tuple[str, ...] = ()

  def lift(
    self, transform: Callable[[ConcreteFunction[Any, Any, Any, Any]], ConcreteFunction[Any, Any, Any, Any]], name: str, extra: tuple[str, ...] = ()
  ) -> Function[Any, Any, Any, Any]:
    """A template whose instance for a call is ``transform`` of this one's instance for the call's
    leading arguments. ``extra`` names the parameters the transform appends (a seed, multipliers):
    this template's instance determines their shapes, so they need no holes of their own."""
    lifted: Function[Any, Any, Any, Any] = Function(name, self._fn, self._slots, None, device=self._device)
    lifted._source, lifted._transform, lifted._extra = self, transform, extra
    names = [*cast(inspect.Signature, self._signature).parameters, *extra]
    lifted._signature = inspect.Signature([inspect.Parameter(n, inspect.Parameter.POSITIONAL_ONLY) for n in names])
    return lifted

  def __repr__(self) -> str:
    if self._source is not None:
      return f"Function({self.name!r}, derived from {self._source.name!r}, instances={list(self._instances)})"
    if self._slots is None:
      slots = ", ".join(inspect.signature(self._fn).parameters)
    else:
      slots = ", ".join(f"{name}: {decl.shape if isinstance(decl, TensorType) else decl}" for name, decl in zip(self._slots.names, self._slots.decls))
    outputs = "inferred" if self._output is None else self._output.names
    return f"Function({self.name!r}, ({slots}) -> {outputs}, instances={list(self._instances)})"

  def __getattr__(self, name: str) -> Any:
    # Reached only for an attribute the object lacks: on a template, a graph attribute is its instance's.
    if name in _GRAPH_ATTRIBUTES and not self.is_concrete:
      return getattr(self.concrete, name)
    raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

  @property
  def is_concrete(self) -> bool:
    """Whether every parameter's shape is declared, so that this Function is its one instance."""
    return False

  @property
  def concrete(self) -> ConcreteFunction[PS, PN, SO, NO]:
    """The one instance of a fully declared Function; a template with holes raises ``NotConcrete``."""
    if self._source is not None:
      holes = f"the shapes of {self._source.name}'s parameters"
    elif self._slots is None:
      holes = "every parameter's structure and shape"
    else:
      holes = ", ".join(f"{name}: {decl}" for name, decl in zip(self._slots.names, self._slots.decls) if isinstance(decl, Hole))
    raise NotConcrete(
      f"{self.name} leaves {holes} to its calls; build an instance with {self.name}.instantiate(...), "
      "one shape, TensorType or example per parameter, or declare the shapes in @sc.function"
    )

  @property
  def instances(self) -> dict[str, ConcreteFunction[PS, PN, SO, NO]]:
    """The instances built so far, by name; each one's name is its C symbol."""
    return dict(self._instances)

  def instantiate(self, *specs: Any, **kwargs: Any) -> ConcreteFunction[PS, PN, SO, NO]:
    """Bind the holes ahead of time and return the instance, cached as a call's would be.

    One declaration per parameter, as in the decorator: a shape (``3``, ``(n, m)``), a
    ``TensorType``, a tree without holes, or an example ``ndarray``, ``Expr`` or ``SparseMatrix``.
    A tuple of ints is a shape; any other tuple is a group's parts.
    """
    values = self._bind(specs, kwargs) if kwargs else specs
    return self._instance(tuple(_example(spec) for spec in values), f"{self.name}.instantiate")

  # The numerical overload comes first on purpose: a Function built from bare expressions has
  # ``Any`` parameter lists, both overloads then match, and the first one wins. Evaluation is the
  # reading that untyped code wants, and a typed symbolic call still resolves exactly because
  # ``Expr`` is not assignable to the numerical leaf type.
  @overload
  def __call__(self, *args: PN.args, **kwargs: PN.kwargs) -> NO: ...

  @overload
  def __call__(self, *args: PS.args, **kwargs: PS.kwargs) -> SO: ...

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Call the instance the arguments bind; see ``ConcreteFunction.__call__``."""
    values = self._bind(args, kwargs) if kwargs else args
    return self._instance(values, self.name)(*values)

  def symbolic_call(self, *args: PS.args, **kwargs: PS.kwargs) -> SO:
    """Embed a call node to the instance the arguments bind, building it if needed."""
    values: Any = self._bind(args, kwargs) if kwargs else args
    return self._instance(values, f"{self.name}.symbolic_call").symbolic_call(*values)

  def numerical_call(self, *args: PN.args, **kwargs: PN.kwargs) -> NO:
    """Evaluate the instance the arguments bind, building and compiling it if needed."""
    values: Any = self._bind(args, kwargs) if kwargs else args
    return self._instance(values, f"{self.name}.numerical_call").numerical_call(*values)

  def _instance(self, args: tuple[Any, ...], what: str) -> ConcreteFunction[PS, PN, SO, NO]:
    """The instance for these arguments: the holes bound to their shapes, traced on first use."""
    arguments = _argument_key(args)
    if arguments is not None and (hit := self._by_arguments.get(arguments)) is not None:
      return hit
    if self._source is not None:
      return self._derived_instance(args, arguments, what)
    if self._slots is None:
      # Bare: the arguments declare themselves, nesting included, so the nesting is part of the key.
      names = list(inspect.signature(self._fn).parameters)
      if len(args) != len(names):
        raise self._arity_error(len(args))
      bound = param_list(*(inferred_tree(arg, param, what, argument=True) for arg, param in zip(args, names, strict=True)))
      structure = skeleton(args) if any(isinstance(arg, tuple) for arg in args) else None
      decls: Sequence[LeafDecl] = (Hole(),) * bound.size
    else:
      if len(args) != len(self._slots.parts):
        raise self._arity_error(len(args))
      bound = param_list(*(slot.bind(arg, what) for slot, arg in zip(self._slots.parts, args, strict=True)))
      structure, decls = None, self._slots.decls
    key = (bound.types, bound.sparsities, structure)
    instance = self._cache.get(key)
    if instance is None:
      tokens = instance_tokens(decls, bound.types, bound.sparsities, structure)
      name = f"{self.name}__{tokens}"
      if name in self._instances:
        raise RuntimeError(f"{self.name}: two argument signatures would share the instance name {name!r}")
      _log.debug("instantiating %s", name)
      instance = ConcreteFunction(name, self._fn, bound, self._output, device=self._device, output_name=self.name)
      instance.tokens = tokens
      self._cache[key] = self._instances[name] = instance
    if arguments is not None:
      self._by_arguments[arguments] = instance
    return instance

  def _derived_instance(self, args: tuple[Any, ...], arguments: tuple[Any, ...] | None, what: str) -> ConcreteFunction[PS, PN, SO, NO]:
    assert self._source is not None and self._transform is not None
    if len(args) != len(cast(inspect.Signature, self._signature).parameters):
      raise self._arity_error(len(args))
    source = self._source._instance(args[: len(args) - len(self._extra)], what)
    instance = self._cache.get(source)
    if instance is None:
      instance = self._transform(source)
      if instance.name in self._instances:
        raise RuntimeError(f"{self.name}: two argument signatures would share the instance name {instance.name!r}")
      instance.tokens = source.tokens
      self._cache[source] = self._instances[instance.name] = instance
    if arguments is not None:
      self._by_arguments[arguments] = instance
    return instance

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

  @property
  def input_shapes(self) -> tuple[tuple[int, ...], ...]:
    """The input leaf shapes in C-signature order."""
    return self.concrete.input_shapes

  @property
  def output_shapes(self) -> tuple[tuple[int, ...], ...]:
    """The output leaf shapes in C-signature order."""
    return self.concrete.output_shapes

  def input_map(self) -> dict[str, Expr]:
    return self.concrete.input_map()

  def output_map(self) -> dict[str, Expr]:
    return self.concrete.output_map()

  def output_sparsity_map(self) -> dict[str, SparsityType | None]:
    return self.concrete.output_sparsity_map()

  def factory(
    self, name: str, inputs: Sequence[str], outputs: Sequence[str | DerivSpec], aux: Mapping[str, Sequence[str]] | None = None
  ) -> ConcreteFunction:
    """Derive a Function carrying the requested outputs and derivatives; see ``ConcreteFunction.factory``."""
    return self.concrete.factory(name, inputs, outputs, aux)

  def with_device(self, device: DeviceSpec | str) -> Function[PS, PN, SO, NO]:
    """The same template, its instances placed on ``device``."""
    if self._source is not None and self._transform is not None:
      return self._source.with_device(device).lift(self._transform, self.name, self._extra)
    return Function(self.name, self._fn, self._slots, self._output, device=device)

  def recompile(self) -> None:
    """Drop every instance's compiled handle and on-disk cache entry."""
    for instance in self._instances.values():
      instance.recompile()

  def callee_state(self, name: str | None = None) -> Any:
    """The state an extern callee of the one instance exposes; see ``ConcreteFunction.callee_state``."""
    return self.concrete.callee_state(name)

  @staticmethod
  def from_exprs(
    name: str,
    inputs: Sequence[Expr],
    outputs: Sequence[Expr],
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    output_sparsities: Sequence[SparsityType | None] | None = None,
    device: DeviceSpec | str | None = None,
    output_coloring_widths: Sequence[int | None] | None = None,
  ) -> ConcreteFunction[Any, Any, Any, Any]:
    """A ConcreteFunction over expressions already built: no inputs take no argument, one input one leaf, more one group."""
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
    instance = ConcreteFunction.__new__(ConcreteFunction)
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


def instance_tokens(
  decls: Sequence[LeafDecl],
  types: Sequence[TensorType],
  sparsities: Sequence[SparsityType | None] = (),
  structure: Any = None,
) -> str:
  """What a template's instance adds to the template's name, after ``__``: one token per hole, its
  bound dimensions joined by ``x`` (only the ``None`` ones of a partial shape), ``s`` for a scalar,
  and the dtype's name when the hole left the dtype open and it is not ``float64``; a sparse leaf's
  token is ``p`` and a digest of its pattern. A bare template called with tuples appends ``t`` and a
  digest of the nesting. The tokens hold no ``_``, so ``f__3_4`` cannot be read two ways."""
  tokens = []
  for decl, type_, sparsity in zip(decls, types, sparsities or (None,) * len(types), strict=True):
    if not isinstance(decl, Hole):
      continue
    if sparsity is not None:
      tokens.append("p" + _digest(np.asarray(sparsity.shape), np.asarray(sparsity.rows), np.asarray(sparsity.cols))[:8])
      continue
    dims = type_.shape if decl.dims is None else tuple(n for d, n in zip(decl.dims, type_.shape) if d is None)
    token = "x".join(map(str, dims)) if dims else "s"
    tokens.append(token + (type_.dtype.name if decl.dtype is None and type_.dtype != dtypes.float64 else ""))
  if structure is not None:
    tokens.append("t" + hashlib.sha256(repr(structure).encode()).hexdigest()[:6])
  return "_".join(tokens)


def _digest(*arrays: np.ndarray) -> str:
  """A hex digest of integer arrays, independent of their dtype and of Python's hash seed."""
  sha = hashlib.sha256()
  for array in arrays:
    sha.update(np.ascontiguousarray(array, dtype=np.int64).tobytes() + b"|")
  return sha.hexdigest()


def _argument_key(args: tuple[Any, ...]) -> tuple[Any, ...] | None:
  """The nesting, flat shapes and ``Expr`` dtypes of a call's arguments, or ``None`` when a leaf is a
  ``SymbolicValue``, whose pattern may matter too, or a list, which may hold expressions."""
  key: list[Any] = [skeleton(args)]
  for leaf in _leaves(args):
    if isinstance(leaf, Expr):
      key.append((leaf.shape, leaf.type.dtype))
    elif isinstance(leaf, (SymbolicValue, list)):
      return None
    else:
      key.append(np.shape(leaf))
  return tuple(key)


def _example(spec: Any) -> Any:
  """A value standing for one parameter's declaration in ``instantiate``: arrays and symbolic values as
  they are, a shape or ``TensorType`` as a symbol of it, a tree as its symbols, another tuple as parts."""
  if isinstance(spec, (Expr, SymbolicValue, np.ndarray)):
    return spec
  if isinstance(spec, Tree):
    return spec.named("_").symbols()
  if isinstance(spec, TensorType):
    return Expr.sym("_", spec.shape, dtype=spec.dtype)
  shape_like = (int, np.integer)
  if isinstance(spec, shape_like) or (isinstance(spec, tuple) and all(isinstance(d, shape_like) for d in spec)):
    return Expr.sym("_", as_shape(tuple(int(d) for d in spec) if isinstance(spec, tuple) else int(spec)))
  if isinstance(spec, tuple):
    return tuple(_example(item) for item in spec)
  raise TypeError(f"instantiate takes a shape, a TensorType, a tree or an example array per parameter, got {spec!r}")


class ConcreteFunction[**PS, **PN, SO, NO](Function[PS, PN, SO, NO]):
  """A named expression graph: named inputs, named outputs, and the computation between them.

  The one instance of a fully declared ``Function``, or one of a template's. It is the unit of three
  things at once. **Composition** — ``fn(inputs)`` with ``Expr``
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

  # On a template's instance: what it adds to the template's name, which its derivatives reuse.
  tokens: str

  def __init__(  # no super().__init__(): an instance keeps no template state
    self,
    name: str,
    fn: Callable[PS, SO],
    inputs: _G,
    outputs: Tree[SO, NO] | None = None,
    *,
    device: DeviceSpec | str | None = None,
    output_name: str | None = None,
  ) -> None:
    """Trace ``fn`` over ``inputs``, a parameter list from ``tree.param_list``: the body gets one
    argument per slot. Without ``outputs`` the output tree is read off the traced value, its leaves
    named after ``output_name`` (default ``name``): ``y``, or ``y_0``, ``y_1``, ... for a tuple."""
    symbolic_inputs = inputs.symbols()
    input_exprs = inputs.flatten_symbolic(symbolic_inputs, f"{name} inputs")
    symbolic_outputs = cast(Callable[..., Any], fn)(*symbolic_inputs)
    if outputs is None:
      outputs = cast(Tree[SO, NO], inferred_tree(symbolic_outputs, output_name or name, f"{name} outputs"))
    outputs = outputs.infer(symbolic_outputs)
    try:
      output_exprs = outputs.flatten_symbolic(symbolic_outputs, f"{name} outputs")
    except ValueError as exc:
      raise TypeError(str(exc)) from exc
    output_types = outputs.resolved(tuple(expr.type for expr in output_exprs))
    output_tree = outputs.with_types(output_types)
    self._init_graph(name, input_exprs, output_exprs, inputs, output_tree, output_tree.sparsities, device=device)
    try:
      self._signature = inspect.signature(fn)
    except (TypeError, ValueError):
      pass

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
    self._compiled_for: Target | None = None  # the target ``_compiled`` was rendered for
    # The C body of a Function the compiler does not generate (``function/extern.py``).
    self.extern: ExternCallee | None = None
    # Derivative rules that replace differentiating the body; set by ``sc.custom_derivative``.
    self.custom_jvp: ConcreteFunction | None = None
    self.custom_vjp: ConcreteFunction | None = None
    # ``(output index, input index) -> pattern``, set by ``sc.custom_derivative(sparsity=...)``.
    self.custom_sparsity: Any = None

  def __repr__(self) -> str:
    suffix = f" device={self.device}" if self.device.kind != "host" else ""
    return f"ConcreteFunction({self.name!r}, {self.input_names}->{self.output_names}{suffix})"

  @property
  def is_concrete(self) -> bool:
    return True

  @property
  def concrete(self) -> ConcreteFunction[PS, PN, SO, NO]:
    return self

  @property
  def instances(self) -> dict[str, ConcreteFunction[PS, PN, SO, NO]]:
    return {self.name: self}

  def instantiate(self, *specs: Any, **kwargs: Any) -> ConcreteFunction[PS, PN, SO, NO]:
    """Check the declarations against this Function's and return it; see ``Function.instantiate``."""
    values = self._bind(specs, kwargs) if kwargs else specs
    if len(values) != len(self.input_tree.parts):
      raise self._arity_error(len(values))
    for slot, spec in zip(self.input_tree.parts, values, strict=True):
      slot.bind(_example(spec), f"{self.name}.instantiate")
    return self

  def with_device(self, device: DeviceSpec | str) -> ConcreteFunction[PS, PN, SO, NO]:
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

  def _compile(self) -> Any:
    """Lazily JIT-compile this function for the target in force (``sc.target``) and cache the
    handle; a call under another target renders and compiles again, or reuses that one's library."""
    target = get_target()
    if self._compiled is None or (self._compiled_for is not target and self._compiled_for != target):
      self._compiled = _jit().CompiledFunction(self)
      self._compiled_for = target
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
    self._compiled_for = None
    jit.invalidate_cache(self)

  def callee_state(self, name: str | None = None) -> Any:
    """The state the extern callee ``name`` (``function/extern.py``) exposed after the latest call,
    decoded by the callee; ``name`` may be left out when this Function reaches one. A solver's is its
    statistics, which ``sc.opt.solver_stats`` reads."""
    jit = _jit()
    if self._compiled is None:
      raise jit.JitError(f"function {self.name!r} has not been compiled or run")
    return self._compiled.callee_state(name)

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
      if expected.type.dtype != actual.type.dtype:
        raise ValueError(f"call argument {name!r} has dtype {actual.type.dtype.name}, expected {expected.type.dtype.name}")
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
    return ConcreteFunction.from_exprs(
      name, ret_inputs, ret_outputs, inputs, ret_output_names, ret_sparsities, output_coloring_widths=ret_coloring_widths
    )
