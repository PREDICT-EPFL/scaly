from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import numpy as np

from .ad import gradient, hessian, jacobian, jvp, linear_combination, vjp
from .expr import Expr, as_expr, topo
from .ops import Ops
from .rewrite import simplify
from .sparsity import sparse_hessian, sparse_jacobian
from .tape import Tape, linearize
from .types import DeviceSpec, SparsityType, TensorType, backend_supports


@dataclass(frozen=True, slots=True)
class Port:
  name: str
  expr: Expr
  sparsity: SparsityType | None = None

  @property
  def shape(self) -> tuple[int, ...]:
    return self.expr.shape


class Function:
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
    missing = [e.name or f"%{e.id}" for e in topo(self.outputs) if e.op == Ops.INPUT and e.id not in declared]
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

  def eval_interpreter(self, *args: Any, **kwargs: Any) -> list[np.ndarray]:
    """Evaluate ``self`` through the Python tape interpreter (reference path)."""
    ordered = self._resolve_inputs(args, kwargs)
    env = dict(zip(self.input_names, ordered, strict=True))
    return self.tape().evaluate(env)

  def eval_list(self, *args: Any, **kwargs: Any) -> list[np.ndarray]:
    """Default dispatch: lazily compile and run via the universal ABI, falling back to the interpreter."""
    from .jit import CompiledFunction, JitError, JitUnavailable, jit_disabled, jit_required

    if jit_disabled():
      return self.eval_interpreter(*args, **kwargs)
    if self.device.kind == "metal":
      from .metal_runtime import MetalCompiledFunction, metal_available

      if not metal_available():
        raise JitError(
          f"function {self.name!r} placed on {self.device} but Metal runtime is unavailable on this machine. "
          f"Install the Xcode Metal Toolchain via `xcodebuild -downloadComponent MetalToolchain`."
        )
      ordered = self._resolve_inputs(args, kwargs)
      compiled_metal = self._compiled
      if compiled_metal is None:
        compiled_metal = MetalCompiledFunction(self)
        self._compiled = compiled_metal
      return compiled_metal.run(list(ordered))
    if self.device.kind != "host":
      raise JitError(
        f"function {self.name!r} placed on {self.device}, but only host lowering is implemented. "
        f"Use ALLOY_DISABLE_JIT=1 to fall back to the interpreter, or call .with_device('host') for now."
      )
    ordered = self._resolve_inputs(args, kwargs)
    compiled: CompiledFunction | None = self._compiled
    if compiled is None:
      try:
        compiled = CompiledFunction(self)
      except JitUnavailable:
        if jit_required():
          raise
        return self.eval_interpreter(*args, **kwargs)
      except JitError:
        raise
      self._compiled = compiled
    return compiled.run(list(ordered))

  def recompile(self) -> None:
    """Drop the cached compiled handle and remove the on-disk cache entry for this function."""
    from .jit import invalidate_cache

    self._compiled = None
    invalidate_cache(self)

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
        Ops.CALL,
        actuals,
        TensorType(out.shape, out.type.dtype, out.type.sparsity, diff=out.type.diff and actual_diff),
        attrs={"callee": self, "output": i},
      )
      for i, out in enumerate(self.outputs)
    )

  def tape(self) -> Tape:
    return linearize(self.outputs)

  def factory(self, name: str, inputs: Sequence[str], outputs: Sequence[str], aux: Mapping[str, Sequence[str]] | None = None) -> Function:
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
      output, sparsity = self._factory_output(spec, all_inputs, all_outputs)
      ret_outputs.append(output)
      ret_output_names.append(spec.replace(":", "_"))
      ret_sparsities.append(sparsity)
    return Function(name, ret_inputs, ret_outputs, inputs, ret_output_names, ret_sparsities)

  @staticmethod
  def _factory_input(name: str, all_inputs: Mapping[str, Expr], spec: str) -> Expr:
    if name not in all_inputs:
      raise ValueError(f"unknown factory input {name!r} in output {spec!r}")
    return all_inputs[name]

  @staticmethod
  def _factory_expr_output(name: str, all_outputs: Mapping[str, Expr], spec: str) -> Expr:
    if name not in all_outputs:
      raise ValueError(f"unknown factory output {name!r} in output {spec!r}")
    return all_outputs[name]

  def _factory_output(self, spec: str, all_inputs: Mapping[str, Expr], all_outputs: Mapping[str, Expr]) -> tuple[Expr, SparsityType | None]:
    if spec in all_outputs:
      return all_outputs[spec], None
    parts = spec.split(":")
    if len(parts) == 3 and parts[0] == "jac":
      return jacobian(self._factory_expr_output(parts[1], all_outputs, spec), self._factory_input(parts[2], all_inputs, spec)), None
    if len(parts) == 3 and parts[0] == "fwd":
      out_name, in_name = parts[1], parts[2]
      return (
        jvp(
          self._factory_expr_output(out_name, all_outputs, spec),
          self._factory_input(in_name, all_inputs, spec),
          self._factory_input(f"fwd:{in_name}", all_inputs, spec),
        ),
        None,
      )
    if len(parts) == 3 and parts[0] == "spjac":
      sj = sparse_jacobian(self._factory_expr_output(parts[1], all_outputs, spec), self._factory_input(parts[2], all_inputs, spec))
      return sj.values, sj.sparsity
    if len(parts) == 3 and parts[0] == "grad":
      return gradient(self._factory_expr_output(parts[1], all_outputs, spec), self._factory_input(parts[2], all_inputs, spec)), None
    if len(parts) == 3 and parts[0] == "adj":
      out_name, in_name = parts[1], parts[2]
      return (
        vjp(
          (self._factory_expr_output(out_name, all_outputs, spec),),
          (self._factory_input(in_name, all_inputs, spec),),
          (self._factory_input(f"lam:{out_name}", all_inputs, spec),),
        )[0],
        None,
      )
    if len(parts) == 4 and parts[0] == "hess":
      y = self._factory_expr_output(parts[1], all_outputs, spec)
      x0 = self._factory_input(parts[2], all_inputs, spec)
      x1 = self._factory_input(parts[3], all_inputs, spec)
      if parts[2] == parts[3]:
        return hessian(y, x0), None
      return jacobian(gradient(y, x0).reshape((x0.size,)), x1), None
    if len(parts) == 4 and parts[0] == "sphess":
      y = self._factory_expr_output(parts[1], all_outputs, spec)
      x0 = self._factory_input(parts[2], all_inputs, spec)
      x1 = self._factory_input(parts[3], all_inputs, spec)
      if parts[2] == parts[3]:
        sh = sparse_hessian(y, x0)
      else:
        sh = sparse_jacobian(simplify(gradient(y, x0).reshape((x0.size,))), x1)
      return sh.values, sh.sparsity
    raise ValueError(f"unknown factory output {spec!r}")
