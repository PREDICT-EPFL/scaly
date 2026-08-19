"""``Function``: a named expression-dialect graph, and the unit of composition and compilation.

Also holds ``Port``, the ``DerivSpec`` request base (its concrete kinds are ``function/factory.py``,
where the AD they dispatch to is reachable), and ``_jit`` — the one sanctioned upward seam, since
calling a ``Function`` compiles it. See ``docs/how_it_works/architecture.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Sequence, cast

import numpy as np

from ..ir.expr import Expr, ExprOp, as_expr, linear_combination, topo
from ..ir.types import DeviceSpec, SparsityType, TensorType, backend_supports

if TYPE_CHECKING:
  from ..solvers.stats import SolverStats


def _jit():
  """The one sanctioned frontend->backend seam: calling a ``Function`` JIT-compiles it.

  Deferred so the frontend does not import the backend at module scope (see the layering table
  in ``docs/how_it_works/architecture.md``); every other use of the backend from here goes through it.
  """
  from ..codegen import jit

  return jit


@dataclass(frozen=True, slots=True)
class DerivSpec:
  """One typed output request for ``Function.factory``.

  ``of`` names an output of the source function (or an ``aux`` output), ``wrt`` one of its inputs.
  The concrete kinds — ``jac``, ``grad``, ``hess``, ``spjac``, ``sphess``, ``fwd``, ``adj`` — are in
  ``function/factory.py``, where the AD they dispatch to is reachable; ``factory`` only needs the
  request shape, so the base sits next to it.
  """

  kind: ClassVar[str]
  of: str
  wrt: str

  @property
  def output_name(self) -> str:
    return f"{self.kind}_{self.of}_{self.wrt}"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, SparsityType | None]:
    raise NotImplementedError

  def _in(self, inputs: Mapping[str, Expr], name: str) -> Expr:
    if name not in inputs:
      raise ValueError(f"unknown factory input {name!r} in output {self}")
    return inputs[name]

  def _out(self, outputs: Mapping[str, Expr], name: str) -> Expr:
    if name not in outputs:
      raise ValueError(f"unknown factory output {name!r} in output {self}")
    return outputs[name]


@dataclass(frozen=True, slots=True)
class Port:
  name: str
  expr: Expr
  sparsity: SparsityType | None = None

  @property
  def shape(self) -> tuple[int, ...]:
    return self.expr.shape


class Function:
  """A named expression graph: named inputs, named outputs, and the computation between them.

  ``Function`` is the unit of three things at once. **Composition** — ``fn.call(args)`` puts a
  first-class call node in a larger graph, and the callee survives into the generated C as a real
  C function rather than being inlined. **Differentiation** — ``fn.factory(...)`` derives a new
  ``Function`` carrying the requested derivatives. **Compilation** — calling one lowers it,
  renders C, compiles and caches a shared library, and dispatches through the universal ABI.

  Names are load-bearing: input and output names are how derivatives are requested and what the
  generated C symbols are built from.

  Use ``@alloy.function(...)`` to build one from a Python body; construct it directly when you
  already have the expressions.
  """

  def __init__(
    self,
    name: str,
    inputs: Sequence[Expr],
    outputs: Sequence[Expr],
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    output_sparsities: Sequence[SparsityType | None] | None = None,
    device: DeviceSpec | str | None = None,
  ):
    self.name = name
    self.inputs = tuple(inputs)
    self.outputs = tuple(outputs)
    self.device: DeviceSpec = DeviceSpec.parse(device)
    for expr in (*self.inputs, *self.outputs):
      if not backend_supports(self.device, expr.type.dtype):
        raise ValueError(
          f"function {name!r} placed on {self.device} cannot lower dtype {expr.type.dtype} (input/output '{expr.name or '<?>'}'). "
          f"Use a different device or cast to a supported dtype."
        )
    raw_input_names = tuple(input_names) if input_names is not None else tuple(i.name for i in inputs)
    if any(n is None for n in raw_input_names):
      raise ValueError("all inputs must have names")
    self.input_names: tuple[str, ...] = cast("tuple[str, ...]", raw_input_names)
    self.output_names: tuple[str, ...] = (
      tuple(output_names) if output_names is not None else tuple(o.name or f"out{i}" for i, o in enumerate(outputs))
    )
    self.output_sparsities = tuple(output_sparsities) if output_sparsities is not None else (None,) * len(self.outputs)
    if len(self.input_names) != len(self.inputs):
      raise ValueError(f"expected {len(self.inputs)} input names, got {len(self.input_names)}")
    if len(self.output_names) != len(self.outputs):
      raise ValueError(f"expected {len(self.outputs)} output names, got {len(self.output_names)}")
    if len(self.output_sparsities) != len(self.outputs):
      raise ValueError(f"expected {len(self.outputs)} output sparsities, got {len(self.output_sparsities)}")
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

  def __repr__(self) -> str:
    suffix = f" device={self.device}" if self.device.kind != "host" else ""
    return f"Function({self.name!r}, {self.input_names}->{self.output_names}{suffix})"

  def with_device(self, device: DeviceSpec | str) -> "Function":
    """Return a copy of this Function placed on ``device``.

    This is a placement policy hint (see roadmap Phase 1 / Phase 9). Today
    only ``host`` actually lowers; non-host devices are accepted and tracked
    so debug output and verifier diagnostics can see them, but compilation
    only succeeds for placements with a registered backend.
    """
    return Function(
      self.name,
      self.inputs,
      self.outputs,
      self.input_names,
      self.output_names,
      self.output_sparsities,
      device=device,
    )

  def input_map(self) -> dict[str, Expr]:
    return dict(zip(self.input_names, self.inputs, strict=True))

  def output_map(self) -> dict[str, Expr]:
    return dict(zip(self.output_names, self.outputs, strict=True))

  def output_sparsity_map(self) -> dict[str, SparsityType | None]:
    return dict(zip(self.output_names, self.output_sparsities, strict=True))

  def _resolve_inputs(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[Any, ...]:
    if args and kwargs:
      raise TypeError("pass positional inputs or keyword inputs, not both")
    if kwargs:
      expected = set(self.input_names)
      missing = [n for n in self.input_names if n not in kwargs]
      extra = [n for n in kwargs if n not in expected]
      if missing or extra:
        parts = []
        if missing:
          parts.append(f"missing keyword inputs: {missing}")
        if extra:
          parts.append(f"unexpected keyword inputs: {extra}")
        raise TypeError(", ".join(parts))
      return tuple(kwargs[name] for name in self.input_names)
    if len(args) != len(self.inputs):
      raise TypeError(f"expected {len(self.inputs)} inputs, got {len(args)}")
    return args

  def _compile(self) -> Any:
    """Lazily JIT-compile this function and cache the handle."""
    if self._compiled is None:
      self._compiled = _jit().CompiledFunction(self)
    return self._compiled

  def eval_list(self, *args: Any, **kwargs: Any) -> list[np.ndarray]:
    """Default dispatch: lazily compile and run through the universal ABI."""
    jit = _jit()
    if self.device.kind != "host":
      raise jit.JitError(f"function {self.name!r} placed on {self.device}, but only host lowering is implemented.")
    ordered = self._resolve_inputs(args, kwargs)
    return self._compile().run(list(ordered))

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

  def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray | tuple[np.ndarray, ...]:
    outs = self.eval_list(*args, **kwargs)
    return outs[0] if len(outs) == 1 else tuple(outs)

  def call(self, args: Sequence[Any]) -> tuple[Expr, ...]:
    actuals = tuple(as_expr(arg) for arg in args)
    if len(actuals) != len(self.inputs):
      raise ValueError(f"expected {len(self.inputs)} call arguments")
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

  def factory(self, name: str, inputs: Sequence[str], outputs: Sequence[str | DerivSpec], aux: Mapping[str, Sequence[str]] | None = None) -> Function:
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
    for spec in outputs:
      if isinstance(spec, str):
        if spec not in all_outputs:
          raise ValueError(f"unknown factory output {spec!r}")
        output, sparsity, output_name = all_outputs[spec], None, spec
      else:
        output, sparsity = spec.build(all_inputs, all_outputs)
        output_name = spec.output_name
      ret_outputs.append(output)
      ret_output_names.append(output_name)
      ret_sparsities.append(sparsity)
    return Function(name, ret_inputs, ret_outputs, inputs, ret_output_names, ret_sparsities)
