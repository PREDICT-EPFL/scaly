"""Expression builders that need a ``Function``.

``ir/expr.py`` owns the expression vocabulary and every builder that only needs a node; a builder
that has to look inside a callee belongs to the frontend instead: ``vmap`` and ``scan``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, overload

import numpy as np

from ..ir.expr import Expr, ExprOp, as_expr, common_lowering
from ..ir.types import TensorType, dtypes
from .model import ConcreteFunction, Function


def vmap(callee: Any, length: int, inputs: Any, output: int = 0) -> Expr:
  """Create an ``ExprOp.VMAP`` node: ``length`` independent calls of ``callee`` whose i-th argument list is
  sliced out of outer tensors with per-input ``(start, stride)`` strides.

  ``inputs`` is either a sequence of ``(outer_tensor, start, stride)`` tuples ordered to match
  ``callee.inputs``, or a mapping from callee input name to the same tuple. The i-th iteration reads
  ``outer[start + i*stride : start + i*stride + callee.inputs[k].size]`` for callee input ``k``.
  Iterations are independent: ``stride=0`` broadcasts the same slice every iteration.

  Only the outer tensors must be rank-1. Callee formals and outputs may be rank-2 (as well as scalar
  or rank-1); each iteration reads a flat slice of ``formal.size`` values and the produced node has
  shape ``(length * callee.outputs[output].size,)``, with iteration outputs concatenated flat.
  """
  if not isinstance(callee, Function):
    raise TypeError(f"vmap callee must be an scaly Function, got {type(callee).__name__}")
  length = int(length)
  if length < 0:
    raise ValueError(f"vmap length must be non-negative, got {length}")
  if not 0 <= output < len(callee.outputs):
    raise ValueError(f"vmap output index {output} out of range for callee with {len(callee.outputs)} outputs")
  out_expr = callee.outputs[output]

  if isinstance(inputs, Mapping):
    extra = [n for n in inputs if n not in callee.input_names]
    if extra:
      raise ValueError(f"vmap inputs reference unknown callee input names {extra}; callee accepts {list(callee.input_names)}")
    missing = [n for n in callee.input_names if n not in inputs]
    if missing:
      raise ValueError(f"vmap inputs missing entries for callee inputs {missing}")
    specs = tuple(inputs[n] for n in callee.input_names)
  else:
    specs = tuple(inputs)
    if len(specs) != len(callee.inputs):
      raise ValueError(f"vmap expects {len(callee.inputs)} input specs, got {len(specs)}")
  outers: list[Expr] = []
  starts: list[int] = []
  strides: list[int] = []
  for i, spec in enumerate(specs):
    outer, start, stride = spec
    outer = as_expr(outer)
    formal = callee.inputs[i]
    if len(outer.shape) != 1:
      raise NotImplementedError(f"vmap currently requires rank-1 outer tensors, got {outer.shape} for input {i}")
    start = int(start)
    stride = int(stride)
    if start < 0:
      raise ValueError(f"vmap input {i} start must be non-negative, got {start}")
    if stride < 0:
      raise ValueError(f"vmap input {i} stride must be non-negative, got {stride}")
    if length > 0:
      end = start + (length - 1) * stride + formal.size
      if end > outer.size:
        raise ValueError(
          f"vmap input {i} reads past outer tensor of size {outer.size}: start={start}, stride={stride}, length={length}, slice_size={formal.size}"
        )
    outers.append(outer)
    starts.append(start)
    strides.append(stride)

  diff = out_expr.type.diff and any(o.type.diff for o in outers)
  return Expr(
    ExprOp.VMAP,
    tuple(outers),
    TensorType((length * out_expr.size,), out_expr.type.dtype, diff=diff),
    attrs={
      "callee": callee,
      "output": int(output),
      "length": length,
      "starts": tuple(starts),
      "strides": tuple(strides),
      "slice_size": int(out_expr.size),
    },
    lowering=common_lowering(*outers) if outers else "auto",
  )


def scan(body: Any, init: Any, xs: Sequence[tuple[Any, int, int]] = (), *, length: int, index: bool = False) -> tuple[Expr, ...]:
  """Run ``body`` ``length`` times in sequence, threading a carry: a loop in the generated C, not an unrolling.

  ``body`` is a ``Function`` whose first input is the carry and whose first output is the next
  carry, with the same shape and dtype. Its other inputs are sliced from ``xs`` exactly as ``vmap``
  slices: step ``k`` reads ``outer[start + k*stride : start + k*stride + formal.size]``, and a
  ``stride`` of zero passes the same slice at every step. Its other outputs are stacked flat, one
  slice per step. Returns ``(final_carry, *ys)``; ``final_carry`` is ``init`` when ``length`` is zero.

  With ``index=True`` the body's second input is the step number, an ``int64`` scalar running
  ``0, 1, ..., length - 1``, and the sliced inputs follow it. The generated loop passes its own
  counter: no table of step numbers is stored. The index carries no derivative.

  The number of steps is fixed when the graph is built, which is what makes the code size and the
  derivative's workspace (reverse mode stores the carry at every step) known ahead of time.
  """
  if not isinstance(body, Function):
    raise TypeError(f"scan body must be an scaly Function, got {type(body).__name__}")
  if index:
    _check_index_input(body, "scan")
    xs = ((step_numbers(length), 0, 1), *xs)
  if not body.inputs or not body.outputs:
    raise ValueError("scan body needs the carry as its first input and the next carry as its first output")
  init = as_expr(init)
  carry, nxt = body.inputs[0], body.outputs[0]
  if nxt.shape != carry.shape or nxt.type.dtype != carry.type.dtype:
    raise ValueError(f"scan body must return a carry like its input: {carry.type.dtype}{carry.shape} -> {nxt.type.dtype}{nxt.shape}")
  if init.shape != carry.shape or init.type.dtype != carry.type.dtype:
    raise ValueError(f"scan init {init.type.dtype}{init.shape} does not match the carry {carry.type.dtype}{carry.shape}")
  specs = tuple(xs)
  if len(specs) != len(body.inputs) - 1:
    given = len(specs) - 1 if index else len(specs)
    raise ValueError(
      f"scan body takes {len(body.inputs) - 1 - int(index)} sliced inputs after the carry{' and the index' if index else ''}, got {given}"
    )
  length = int(length)
  if length < 0:
    raise ValueError(f"scan length must be non-negative, got {length}")
  outers = tuple(as_expr(outer) for outer, _, _ in specs)
  starts = tuple(int(start) for _, start, _ in specs)
  strides = tuple(int(stride) for _, _, stride in specs)
  return tuple(_scan_node(body, init, outers, starts, strides, length, k) for k in range(len(body.outputs)))


def step_numbers(length: int) -> Expr:
  """The ``int64`` constant ``0, 1, ..., length - 1`` a scan slices one entry per step from to hand its
  body the step number. Lowering recognizes an affine ``int64`` constant read one entry per step and
  computes the entry from the loop counter, so the table itself is never stored."""
  return Expr.const(np.arange(int(length), dtype=np.int64), dtype=dtypes.int64)


def _check_index_input(body: ConcreteFunction, what: str) -> None:
  if len(body.inputs) < 2 or body.inputs[1].shape != () or body.inputs[1].type.dtype != dtypes.int64:
    got = f"{body.inputs[1].type.dtype}{body.inputs[1].shape}" if len(body.inputs) > 1 else "nothing"
    raise ValueError(f"{what} with index=True needs an int64 scalar as the body's second input, got {got}")


def _scan_node(
  body: ConcreteFunction, init: Expr, outers: tuple[Expr, ...], starts: tuple[int, ...], strides: tuple[int, ...], length: int, output: int
) -> Expr:
  """One output of a scan: the final carry (0), a stacked output (1..), or with ``output=-1`` the carry
  entering every step, stacked, which reverse mode reads backwards. A negative stride walks backwards."""
  for i, (outer, start, stride) in enumerate(zip(outers, starts, strides, strict=True)):
    formal = body.inputs[i + 1]
    if len(outer.shape) != 1:
      raise NotImplementedError(f"scan requires rank-1 outer tensors, got {outer.shape} for input {i + 1}")
    last = start + (length - 1) * stride
    if length and (min(start, last) < 0 or max(start, last) + formal.size > outer.size):
      raise ValueError(f"scan input {i + 1} reads outside its outer tensor of size {outer.size}: start={start}, stride={stride}, length={length}")
  if output == 0:
    shape: tuple[int, ...] = body.inputs[0].shape
    produced = body.outputs[0]
  elif output == -1:
    shape, produced = (length * body.inputs[0].size,), body.outputs[0]
  else:
    produced = body.outputs[output]
    shape = (length * produced.size,)
  diff = produced.type.diff and (init.type.diff or any(o.type.diff for o in outers))
  args = (init, *outers)
  return Expr(
    ExprOp.SCAN,
    args,
    TensorType(shape, produced.type.dtype, diff=diff),
    attrs={"callee": body, "output": int(output), "length": int(length), "starts": starts, "strides": strides},
    lowering=common_lowering(*args),
  )


def while_loop(cond: Any, body: Any, init: Any, *, max_iter: int, index: bool = False, params: Sequence[Any] = ()) -> tuple[Expr, Expr]:
  """Apply ``body`` to the carry while ``cond`` holds, at most ``max_iter`` times.

  ``cond`` maps the carry to one ``bool``; ``body`` maps the carry to the next carry, with the same
  shape and dtype. Returns ``(carry, n_iter)``, where ``n_iter`` counts the steps taken as a
  ``float64`` scalar (exact for any count a loop can reach). The bound is fixed when the graph is
  built, so the generated code and the workspace of its derivative are known ahead of time.

  ``params`` are tensors every step reads and none changes: a solver's problem data, a matrix
  factor, tolerances. ``body`` takes them after the carry (and the step number), and ``cond`` after
  the carry, in the order given. They are passed to each step as they are, not copied into the
  carry, and reverse mode accumulates their cotangents over the steps taken.

  Both outputs are differentiable in the usual sense for an iteration: the number of steps is
  treated as locally constant, which it is except where the input crosses a switching boundary.
  Reverse mode stores the carry at each of the at most ``max_iter`` steps. For a solver, attaching
  the implicit-function derivative with ``sc.custom_derivative`` avoids differentiating the steps.

  With ``index=True`` the body takes the step number as a second input, an ``int64`` scalar
  counting from zero; the condition does not.
  """
  if not isinstance(cond, Function) or not isinstance(body, Function):
    raise TypeError("while_loop cond and body must be scaly Functions")
  init = as_expr(init)
  params = tuple(as_expr(param) for param in params)
  # Every shape a template leaves open is determined here: the carry is init's, then the step number and the params.
  step = (TensorType((), dtypes.int64),) if index else ()
  cond = cond if cond.is_concrete else cond.instantiate(init, *params)
  body = body if body.is_concrete else body.instantiate(init, *step, *params)
  if index:
    _check_index_input(body, "while_loop")
  first = 1 + int(index)
  if len(body.inputs) != first + len(params) or len(body.outputs) != 1 or len(cond.inputs) != 1 + len(params) or len(cond.outputs) != 1:
    extra = " and the step number" if index else ""
    raise ValueError(
      f"while_loop body takes the carry{extra} and {len(params)} params, cond the carry and the params, and each returns one output; "
      f"got a body of {len(body.inputs)} inputs and a cond of {len(cond.inputs)}"
    )
  carry = body.inputs[0]
  for label, e in (("init", init), ("body output", body.outputs[0]), ("cond input", cond.inputs[0])):
    if e.shape != carry.shape or e.type.dtype != carry.type.dtype:
      raise ValueError(f"while_loop {label} {e.type.dtype}{e.shape} does not match the carry {carry.type.dtype}{carry.shape}")
  for i, param in enumerate(params):
    for label, formal in (("body", body.inputs[first + i]), ("cond", cond.inputs[1 + i])):
      if formal.shape != param.shape or formal.type.dtype != param.type.dtype:
        raise ValueError(f"while_loop param {i} is {param.type.dtype}{param.shape}, but {label} takes {formal.type.dtype}{formal.shape}")
  flag = cond.outputs[0]
  if flag.size != 1 or not flag.type.dtype.is_bool:
    raise ValueError(f"while_loop cond must return one bool, got {flag.type.dtype}{flag.shape}")
  max_iter = int(max_iter)
  if max_iter < 0:
    raise ValueError(f"while_loop max_iter must be non-negative, got {max_iter}")
  return _while_node(cond, body, init, max_iter, 0, params, index), _while_node(cond, body, init, max_iter, 1, params, index)


def _while_node(
  cond: ConcreteFunction, body: ConcreteFunction, init: Expr, max_iter: int, output: int, params: Sequence[Expr] = (), index: bool = False
) -> Expr:
  """One output of a while loop: the carry (0), the step count (1), or with ``output=-1`` the carry
  entering each of the ``max_iter`` possible steps, stacked; slots past the last step taken hold the
  final carry. Reverse mode reads the last one. With ``index`` the body's second input is the step
  number; ``params`` follow it (and the carry, for ``cond``)."""
  carry = body.inputs[0]
  params = tuple(params)
  diff = (init.type.diff or any(p.type.diff for p in params)) and body.outputs[0].type.diff
  if output == 0:
    type_ = TensorType(carry.shape, carry.type.dtype, diff=diff)
  elif output == 1:
    type_ = TensorType((), dtypes.float64, diff=False)
  else:
    type_ = TensorType((max_iter * carry.size,), carry.type.dtype, diff=diff)
  args = (init, *params)
  return Expr(
    ExprOp.WHILE,
    args,
    type_,
    attrs={"callee": body, "cond": cond, "max_iter": int(max_iter), "output": int(output), "index": bool(index)},
    lowering=common_lowering(*args),
  )


def while_parts(expr: Expr) -> tuple[ConcreteFunction, ConcreteFunction, Expr, tuple[Expr, ...], int, bool]:
  """``(cond, body, init, params, max_iter, index)`` of a while-loop node."""
  return (
    expr.attrs["cond"],
    expr.attrs["callee"],
    expr.args[0],
    tuple(expr.args[1:]),
    int(expr.attrs["max_iter"]),
    bool(expr.attrs.get("index", False)),
  )


@overload
def custom_derivative[**PS, **PN, SO, NO](
  fn: ConcreteFunction[PS, PN, SO, NO], *, jvp: Any = None, vjp: Any = None, sparsity: Any = None
) -> ConcreteFunction[PS, PN, SO, NO]: ...


@overload
def custom_derivative[**PS, **PN, SO, NO](
  fn: Function[PS, PN, SO, NO], *, jvp: Any = None, vjp: Any = None, sparsity: Any = None
) -> Function[PS, PN, SO, NO]: ...


def custom_derivative(fn: Any, *, jvp: Any = None, vjp: Any = None, sparsity: Any = None) -> Any:
  """A copy of ``fn`` whose derivatives come from the given Functions instead of from its body.

  ``jvp`` takes ``(*inputs, *input_tangents)`` and returns one tangent per output. ``vjp`` takes
  ``(*inputs, *outputs, *output_cotangents)`` and returns one cotangent per input; receiving the
  outputs lets an implicit-function rule use the solution without solving again. Either may be
  omitted, and that direction then differentiates the body as usual. The typical use is a solver
  loop: its derivative through the iterations is replaced by the implicit-function derivative at
  the solution, which costs one linear solve and is exact there.

  Sparsity patterns come from the body unless ``sparsity`` gives them: a function of ``(output
  index, input index)`` returning the pattern of that output with respect to that input (a
  ``SparsityType``, a boolean mask, a SciPy sparse matrix, or ``None`` for no dependence). A loop's
  structural pattern through run-time indices is conservative and can be costly to compute; the
  rule's author usually knows the real one. A copy keeps its source's pattern unless it is given
  new rules without one.
  """
  if not isinstance(fn, Function):
    raise TypeError(f"custom_derivative needs a scaly Function, got {type(fn).__name__}")
  for label, rule in (("jvp", jvp), ("vjp", vjp)):
    if rule is not None and not isinstance(rule, Function):
      raise TypeError(f"custom {label} must be a scaly Function")
  if not fn.is_concrete:
    return fn._lift(lambda instance: custom_derivative(instance, jvp=jvp, vjp=vjp, sparsity=sparsity), f"{fn.name}_cd")
  # A template rule is instantiated at the leaves it takes: one parameter per leaf, flat.
  types_in, types_out = [e.type for e in fn.inputs], [e.type for e in fn.outputs]
  jvp = jvp if jvp is None or jvp.is_concrete else jvp.instantiate(*types_in, *types_in)
  vjp = vjp if vjp is None or vjp.is_concrete else vjp.instantiate(*types_in, *types_out, *types_out)
  shapes_in = [e.shape for e in fn.inputs]
  shapes_out = [e.shape for e in fn.outputs]
  for label, rule, takes, gives in (
    ("jvp", jvp, shapes_in + shapes_in, shapes_out),
    ("vjp", vjp, shapes_in + shapes_out + shapes_out, shapes_in),
  ):
    if rule is None:
      continue
    got_in, got_out = [e.shape for e in rule.inputs], [e.shape for e in rule.outputs]
    if got_in != takes or got_out != gives:
      raise ValueError(f"custom {label} for {fn.name!r} must map shapes {takes} -> {gives}, got {got_in} -> {got_out}")
  copy = fn._with_outputs(fn.outputs)
  if hasattr(fn, "descriptor"):
    copy.descriptor = fn.descriptor
  copy.custom_jvp = jvp if jvp is not None else fn.custom_jvp
  copy.custom_vjp = vjp if vjp is not None else fn.custom_vjp
  # A declared pattern describes the rules it came with: new rules without a pattern drop it.
  inherited = getattr(fn, "custom_sparsity", None) if jvp is None and vjp is None else None
  copy.custom_sparsity = sparsity if sparsity is not None else inherited
  # Procedures and derivative helpers are named after their Function, so the copy needs a name of
  # its own: in one graph with ``fn``, sharing a name would let one derivative stand for both.
  rules = ",".join(r.name if r is not None else "-" for r in (copy.custom_jvp, copy.custom_vjp))
  copy.name = f"{fn.name}_cd{hashlib.sha1(rules.encode()).hexdigest()[:8]}"
  return copy
