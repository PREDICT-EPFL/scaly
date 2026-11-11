"""Expression-dialect verifier: op-indexed pattern tables that report the first invalid node.

Phase 2 of the roadmap. Inspired by tinygrad's ``spec.py``/``type_verify`` but kept
small and focused on Scaly's expression dialect. The contract:

- Each ``Rule`` runs for a single ``ExprOp`` value (or for any op when
  ``op is None``) and either returns ``None`` (the node is well-formed by that
  rule) or a string describing the violation.
- ``Spec`` is a table of rules grouped by op. ``Spec.check(node)`` returns the
  first violation found.
- ``verify_expr(root, spec=spec_expr)`` walks the DAG topologically and
  raises ``VerifyError`` at the first invalid node, naming the node, the op,
  and the failing rule.

Function construction runs the verifier, including for graphs rewritten before lowering.
Passes can also call it explicitly after graph rewrites or AD transforms.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .expr import (
  COMMON_ELEMENTWISE_BINARY,
  COMMON_ELEMENTWISE_TERNARY,
  COMMON_ELEMENTWISE_UNARY,
  OP_INFO,
  Expr,
  ExprOp,
  _pattern_text,
  _type_text,
  check_stored_pattern,
  elementwise_dtype,
  print_pieces,
  topo,
)
from .spec import Rule, Spec, VerifyError
from .support import index_shape
from .types import CanonicalPattern, DType, TensorType, broadcast_shape, dtypes


def verify_expr(root: Expr | Iterable[Expr], spec: "Spec | None" = None) -> None:
  """Topologically walk the DAG below ``root`` and raise on the first violation.

  ``spec`` defaults to ``spec_expr``, the full expression-dialect contract.
  """
  if spec is None:
    spec = spec_expr
  outputs = (root,) if isinstance(root, Expr) else tuple(root)
  for node in topo(outputs):
    result = spec.check(node)
    if result is not None:
      rule, diag = result
      label = node.name or f"%{node.id}"
      raise VerifyError(f"verify_expr: node {label} op={ExprOp(node.op).value} failed rule {rule.description!r}: {diag}")


# ---------------------------------------------------------------------------
# Shared rules: invariants every expression-dialect node must satisfy.
# ---------------------------------------------------------------------------


def _shape_nonneg(expr: Expr) -> str | None:
  if any(d < 0 for d in expr.type.shape):
    return f"shape {expr.type.shape} contains a negative dimension"
  return None


def _dtype_is_dtype(expr: Expr) -> str | None:
  if not isinstance(expr.type.dtype, DType):
    return f"type.dtype is {type(expr.type.dtype).__name__}, expected DType"
  return None


def _arity_matches(expr: Expr) -> str | None:
  info = OP_INFO.get(ExprOp(expr.op))
  if info is None or info.arity is None:
    return None
  if len(expr.args) != info.arity:
    return f"op {expr.op} expects {info.arity} args, got {len(expr.args)}"
  return None


def _stored_pattern(expr: Expr) -> str | None:
  pattern = expr.type.pattern
  if pattern is not None:
    if not isinstance(pattern, CanonicalPattern):
      return "type.pattern must be canonical"
    if expr.type.ndim < 2 or expr.shape[-2:] != pattern.shape:
      return f"type.pattern shape {pattern.shape} disagrees with last two axes {expr.shape[-2:]}"
  try:
    check_stored_pattern(expr.op, expr.args, expr.type, expr.attrs)
  except ValueError as exc:
    return str(exc)
  return None


spec_expr_shared = Spec(
  [
    Rule(None, "shape-nonnegative", _shape_nonneg),
    Rule(None, "dtype-is-DType", _dtype_is_dtype),
    Rule(None, "arity-matches-op", _arity_matches),
    Rule(None, "stored-pattern", _stored_pattern),
  ]
)


# ---------------------------------------------------------------------------
# Per-op rules.
# ---------------------------------------------------------------------------


def _input_has_name(expr: Expr) -> str | None:
  if not expr.name:
    return "INPUT node has no name"
  return None


def _const_value_present(expr: Expr) -> str | None:
  if expr.value is None:
    return "CONST node has no value"
  if expr.value.shape != expr.type.shape:
    return f"CONST value shape {expr.value.shape} disagrees with type shape {expr.type.shape}"
  expected_np = expr.type.dtype.numpy()
  if expr.value.dtype != expected_np:
    return f"CONST value dtype {expr.value.dtype} disagrees with type dtype {expr.type.dtype}"
  return None


def _unary_shape_dtype(expr: Expr) -> str | None:
  if not expr.args:
    return "unary op has no arg"
  arg = expr.args[0]
  if arg.shape != expr.shape:
    return f"unary elementwise op output shape {expr.shape} != input shape {arg.shape}"
  if arg.type.dtype != expr.type.dtype:
    return f"unary elementwise dtype {expr.type.dtype} != input dtype {arg.type.dtype}"
  return None


def _elementwise_shape_dtype(expr: Expr) -> str | None:
  shapes = [arg.shape for arg in expr.args]
  try:
    expected = shapes[0]
    for shape in shapes[1:]:
      expected = broadcast_shape(expected, shape)
  except ValueError as e:
    return f"elementwise operands not broadcastable: {e}"
  if expected != expr.shape:
    return f"elementwise output shape {expr.shape} != broadcast of {shapes}={expected}"
  if expr.op == ExprOp.CAST:
    return "a cast to bool is spelled x != 0" if expr.type.dtype.is_bool else None
  try:
    dtype = elementwise_dtype(expr.op, *(arg.type.dtype for arg in expr.args))
  except TypeError as e:
    return str(e)
  if dtype != expr.type.dtype:
    return f"elementwise output dtype {expr.type.dtype} != {dtype}"
  return None


def _reduce_shape(expr: Expr) -> str | None:
  axes = expr.attrs.get("axes")
  if expr.attrs.get("kind") not in {"add", "max", "min"}:
    return "REDUCE requires kind add, max or min"
  if not isinstance(axes, tuple) or any(not isinstance(i, int) or i < 0 or i >= len(expr.args[0].shape) for i in axes):
    return "REDUCE requires normalized axes"
  if tuple(sorted(set(axes))) != axes:
    return "REDUCE axes must be sorted and unique"
  shape = tuple(n for i, n in enumerate(expr.args[0].shape) if i not in axes)
  if expr.shape != shape or expr.type.dtype != expr.args[0].type.dtype:
    return f"REDUCE output must have shape {shape} and its input dtype"
  return None


def _reshape_size(expr: Expr) -> str | None:
  if not expr.args:
    return "RESHAPE missing arg"
  src = expr.args[0]
  if int(np.prod(expr.shape, dtype=int)) != src.size:
    return f"RESHAPE source size {src.size} != target size {int(np.prod(expr.shape, dtype=int))}"
  if "shape" not in expr.attrs:
    return "RESHAPE missing 'shape' attr"
  return None


def _transpose_axes(expr: Expr) -> str | None:
  axes = expr.attrs.get("axes")
  if axes is None:
    return "TRANSPOSE missing 'axes' attr"
  if not expr.args:
    return "TRANSPOSE missing arg"
  src = expr.args[0]
  if sorted(axes) != list(range(len(src.shape))):
    return f"TRANSPOSE axes {axes} are not a permutation of range({len(src.shape)})"
  expected = tuple(src.shape[i] for i in axes)
  if expected != expr.shape:
    return f"TRANSPOSE output shape {expr.shape} != permuted shape {expected}"
  return None


def _slice_index(expr: Expr) -> str | None:
  index = expr.attrs.get("index")
  src = expr.args[0]
  if not isinstance(index, tuple) or len(index) != len(src.shape):
    return "SLICE requires a normalized 'index' tuple with one component per source axis"
  expected = []
  for item, dim in zip(index, src.shape, strict=True):
    if isinstance(item, int):
      if not -dim <= item < dim:
        return f"SLICE index {item} out of bounds for axis of size {dim}"
    elif isinstance(item, slice):
      try:
        expected.append(len(range(*item.indices(dim))))
      except (TypeError, ValueError) as exc:
        return f"SLICE invalid slice: {exc}"
    else:
      return f"SLICE unsupported index component {item!r}"
  if expr.shape != tuple(expected):
    return f"SLICE output shape {expr.shape} != indexed shape {tuple(expected)}"
  if expr.type.dtype != src.type.dtype:
    return "SLICE must preserve source dtype"
  return None


def _matmul_shape(expr: Expr) -> str | None:
  if len(expr.args) != 2:
    return f"MATMUL has {len(expr.args)} args"
  x, y = expr.args
  if x.type.dtype != y.type.dtype:
    return f"MATMUL operands have mismatched dtypes: {x.type.dtype} vs {y.type.dtype}"
  if not x.shape or not y.shape or x.shape[-1] != y.shape[-2 if len(y.shape) > 1 else -1]:
    return f"MATMUL shape mismatch: {x.shape}@{y.shape}"
  try:
    shape = broadcast_shape(x.shape[:-2], y.shape[:-2])
  except ValueError:
    return f"MATMUL batch mismatch: {x.shape}@{y.shape}"
  shape += (() if len(x.shape) == 1 else (x.shape[-2],)) + (() if len(y.shape) == 1 else (y.shape[-1],))
  if expr.shape != shape:
    return f"MATMUL shape mismatch: expected {shape}, got {expr.shape}"
  return None


def _call_attrs(expr: Expr) -> str | None:
  callee = expr.attrs.get("callee")
  if callee is None:
    return "CALL missing 'callee' attr"
  out_idx = expr.attrs.get("output")
  if out_idx is None:
    return "CALL missing 'output' attr"
  if not 0 <= int(out_idx) < len(callee.results):
    return f"CALL output index {out_idx} out of range for callee {callee.name!r}"
  if len(expr.args) != len(callee.inputs):
    return f"CALL has {len(expr.args)} args but callee {callee.name!r} expects {len(callee.inputs)}"
  for i, (actual, formal) in enumerate(zip(expr.args, callee.inputs, strict=True)):
    if actual.shape != formal.shape:
      return f"CALL arg {i} shape {actual.shape} != callee formal shape {formal.shape}"
    if actual.type.pattern is not formal.type.pattern or actual.type.dtype != formal.type.dtype:
      return f"CALL arg {i} must have the exact callee pattern and dtype: {_type_text(actual.type)} != {_type_text(formal.type)}"
  expected_out = callee.results[int(out_idx)]
  if expected_out.shape != expr.shape:
    return f"CALL output shape {expr.shape} != callee output shape {expected_out.shape}"
  if expected_out.type.pattern is not expr.type.pattern or expected_out.type.dtype != expr.type.dtype:
    return f"CALL output must preserve the callee pattern and dtype: {_type_text(expr.type)} != {_type_text(expected_out.type)}"
  return None


def _vmap_attrs(expr: Expr) -> str | None:
  """Verify VMAP's flat outer storage: outers are rank-1 and every window fits inside its outer.

  Callee formals and outputs may be rank-2; a window spans its formal's size.
  """
  callee = expr.attrs.get("callee")
  if callee is None:
    return "VMAP missing 'callee' attr"
  for k in ("length", "starts", "strides", "slice_size", "output"):
    if k not in expr.attrs:
      return f"VMAP missing {k!r} attr"
  length = int(expr.attrs["length"])
  if length < 0:
    return f"VMAP length must be non-negative, got {length}"
  starts = expr.attrs["starts"]
  strides = expr.attrs["strides"]
  if len(expr.args) != len(callee.inputs) or len(starts) != len(callee.inputs) or len(strides) != len(callee.inputs):
    return f"VMAP args/starts/strides length mismatch vs callee {callee.name!r} inputs"
  for i, (outer, start, stride, formal) in enumerate(zip(expr.args, starts, strides, callee.inputs, strict=True)):
    start, stride = int(start), int(stride)
    if len(outer.shape) != 1:
      return f"VMAP outer arg must be rank-1, got {outer.shape}"
    if start < 0 or stride < 0:
      return f"VMAP input {i} start and stride must be non-negative, got {start} and {stride}"
    if length > 0 and start + (length - 1) * stride + formal.type.storage_size > outer.size:
      return f"VMAP input {i} reads past its outer of size {outer.size}: start={start}, stride={stride}, length={length}, formal_size={formal.size}"
  expected_size = length * int(expr.attrs["slice_size"])
  output = int(expr.attrs["output"])
  if not 0 <= output < len(callee.results) or int(expr.attrs["slice_size"]) != callee.results[output].type.storage_size:
    return "VMAP slice_size must match the selected output's stored size"
  if expr.shape != (expected_size,):
    return f"VMAP output shape {expr.shape} != ({expected_size},)"
  return None


def _cond_attrs(expr: Expr) -> str | None:
  branches = expr.attrs.get("branches")
  if not isinstance(branches, tuple) or not branches:
    return "COND requires a nonempty tuple of branches"
  if not expr.args or expr.args[0].shape or not (expr.args[0].type.dtype.is_bool or expr.args[0].type.dtype.is_integer):
    return "COND selector must be a bool or integer scalar"
  if expr.args[0].type.dtype.is_bool and len(branches) != 2:
    return "COND boolean selector requires two branches"
  output = expr.attrs.get("output")
  if type(output) is not int or not 0 <= output < len(branches[0].outputs):
    return "COND output index out of range"
  for branch in branches:
    if len(branch.inputs) != len(expr.args) - 1 or len(branch.outputs) != len(branches[0].outputs):
      return "COND branch input and output counts must match"
    for actual, formal in zip(expr.args[1:], branch.inputs, strict=True):
      if actual.shape != formal.shape or actual.type.dtype != formal.type.dtype or actual.type.pattern is not formal.type.pattern:
        return "COND operands must match every branch input shape, dtype and pattern"
    for expected, result in zip(branches[0].outputs, branch.outputs, strict=True):
      if expected.shape != result.shape or expected.type.dtype != result.type.dtype or expected.type.pattern is not result.type.pattern:
        return "COND branch outputs must match in shape, dtype and pattern"
  expected = branches[0].outputs[output]
  if expr.shape != expected.shape or expr.type.dtype != expected.type.dtype or expr.type.pattern is not expected.type.pattern:
    return "COND result must preserve the branch shape, dtype and pattern"
  return None


def loop_result_type(attrs: Mapping[str, Any], output: int) -> TensorType | None:
  """The type of result ``output`` of a ``LOOP``: final carries, stacked ys, the trip count, then trajectories."""
  callee, n_carry, length = attrs["callee"], int(attrs["n_carry"]), int(attrs["length"])
  n_ys = len(callee.outputs) - n_carry
  if 0 <= output < n_carry:
    return callee.inputs[output].type
  if n_carry <= output < n_carry + n_ys or n_carry + n_ys < output <= 2 * n_carry + n_ys:
    source = callee.outputs[output] if output < n_carry + n_ys else callee.inputs[output - n_carry - n_ys - 1]
    return TensorType((length * source.type.storage_size,), source.type.dtype)
  return TensorType((), dtypes.int64) if output == n_carry + n_ys else None


def _loop_attrs(expr: Expr) -> str | None:
  """Verify ``LOOP``: ``args = (*inits, *outers[, trip])`` against ``body(*carries, [k], *xs) -> (*carries, *ys)``."""
  for k in ("callee", "cond", "n_carry", "length", "starts", "strides", "index", "reverse", "in_place", "output"):
    if k not in expr.attrs:
      return f"LOOP missing {k!r} attr"
  callee, cond, n_carry, length = expr.attrs["callee"], expr.attrs["cond"], int(expr.attrs["n_carry"]), int(expr.attrs["length"])
  starts, strides, index = expr.attrs["starts"], expr.attrs["strides"], bool(expr.attrs["index"])
  n_outer = len(starts)
  if length < 0 or n_carry < 0 or len(strides) != n_outer or len(expr.args) not in (n_carry + n_outer, n_carry + n_outer + 1):
    return f"LOOP length {length}, n_carry {n_carry}, {n_outer} starts, {len(strides)} strides and {len(expr.args)} args disagree"
  formals = callee.inputs
  if len(formals) != n_carry + index + n_outer or len(callee.outputs) < n_carry:
    return (
      f"LOOP body {callee.name!r} must take {n_carry} carries, {'the step number, ' if index else ''}{n_outer} slices and return the carries first"
    )

  def same(a: TensorType, b: TensorType) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.pattern is b.pattern

  for i, (init, formal, out) in enumerate(zip(expr.args[:n_carry], formals, callee.outputs, strict=False)):
    if not same(init.type, formal.type) or not same(out.type, formal.type):
      return f"LOOP carry {i} must keep its type: init {_type_text(init.type)}, body {_type_text(formal.type)} -> {_type_text(out.type)}"
  if index and (formals[n_carry].shape != () or formals[n_carry].type.dtype != dtypes.int64):
    return "LOOP step number formal must be an int64 scalar"
  for i, (outer, start, stride, formal) in enumerate(zip(expr.args[n_carry:], starts, strides, formals[n_carry + index :], strict=False)):
    start, stride = int(start), int(stride)
    if len(outer.shape) != 1 or outer.type.dtype != formal.type.dtype:
      return f"LOOP outer {i} must be rank-1 {formal.type.dtype}, got {_type_text(outer.type)}"
    if start < 0 or stride < 0 or length > 0 and start + (length - 1) * stride + formal.type.storage_size > outer.size:
      return f"LOOP outer {i} window start={start}, stride={stride} reads past its outer of size {outer.size} in {length} steps"
  if len(expr.args) > n_carry + n_outer and (expr.args[-1].shape != () or expr.args[-1].type.dtype != dtypes.int64):
    return "LOOP trip operand must be an int64 scalar"
  if cond is not None:
    if len(cond.inputs) != len(formals) or not all(same(c.type, f.type) for c, f in zip(cond.inputs, formals, strict=True)) or len(cond.outputs) != 1:
      return f"LOOP condition {cond.name!r} must take the body's inputs and return one value"
    if cond.outputs[0].shape != () or cond.outputs[0].type.dtype != dtypes.bool_:
      return f"LOOP condition {cond.name!r} must return a bool scalar"
    if expr.attrs["reverse"]:
      return "LOOP with a condition cannot run in reverse"
  if not all(0 <= int(i) < n_carry for i in expr.attrs["in_place"]) or len(set(expr.attrs["in_place"])) != len(expr.attrs["in_place"]):
    return f"LOOP in_place {expr.attrs['in_place']} must name distinct carries"
  expected = loop_result_type(expr.attrs, int(expr.attrs["output"]))
  if expected is None or not same(expr.type, expected):
    return f"LOOP result {expr.attrs['output']} {_type_text(expr.type)} != {'out of range' if expected is None else _type_text(expected)}"
  return None


def _scatter_indices(expr: Expr) -> str | None:
  if len(expr.args) != 3 or expr.args[1].type.dtype.name != "int64":
    return "SCATTER requires base, int64 indices and values operands"
  if expr.attrs.get("mode") not in ("fill", "promise_in_bounds") or expr.attrs.get("combine") not in ("set", "add", "max", "min"):
    return "SCATTER requires a valid mode and combination"
  axis = expr.attrs.get("axis")
  if axis is not None and (not isinstance(axis, int) or not 0 <= axis < len(expr.shape)):
    return "SCATTER axis must be a normalized base axis"
  expected = index_shape(expr.shape, expr.args[1].shape, axis)
  if axis is not None and expr.args[2].shape != expected:
    return "SCATTER values shape must match the indexed shape"
  if int(np.prod(expected, dtype=int)) != expr.args[2].size:
    return "SCATTER index and values sizes differ"
  if expr.shape != expr.args[0].shape or expr.type.dtype != expr.args[0].type.dtype or expr.type.dtype != expr.args[2].type.dtype:
    return "SCATTER base shape and value dtypes must match the result"
  return None


def _storage_conversion(expr: Expr) -> str | None:
  if expr.type.dtype != expr.args[0].type.dtype:
    return "stored-value conversions must preserve dtype"
  if expr.op == ExprOp.PACK and (pattern := expr.attrs.get("pattern")) is not expr.type.pattern:
    actual = _pattern_text(pattern) if isinstance(pattern, CanonicalPattern) else f" {pattern!r}"
    return f"PACK attribute must match the result pattern: got{actual}, expected{_pattern_text(expr.type.pattern)}"
  return None


def _gather_indices(expr: Expr) -> str | None:
  if len(expr.args) != 2 or expr.args[1].type.dtype.name != "int64":
    return "GATHER requires an int64 index operand"
  if expr.attrs.get("mode") not in ("fill", "promise_in_bounds"):
    return "GATHER requires a valid indexing mode"
  axis = expr.attrs.get("axis")
  if axis is not None and (not isinstance(axis, int) or not 0 <= axis < len(expr.args[0].shape)):
    return "GATHER axis must be a normalized source axis"
  if index_shape(expr.args[0].shape, expr.args[1].shape, expr.attrs.get("axis")) != expr.shape:
    return f"GATHER output shape {expr.shape} != indices shape {expr.args[1].shape}"
  return None


def _stack_shapes(expr: Expr) -> str | None:
  if not expr.args:
    return "STACK requires at least one arg"
  base = expr.args[0].shape
  for a in expr.args[1:]:
    if a.shape != base:
      return f"STACK arg shape {a.shape} differs from first {base}"
  return None


def _concat_shapes(expr: Expr) -> str | None:
  if not expr.args:
    return "CONCAT requires at least one arg"
  axis = expr.attrs.get("axis", 0)
  base = expr.args[0].shape
  if not 0 <= axis < len(base):
    return f"CONCAT axis {axis} out of bounds for shape {base}"
  for a in expr.args[1:]:
    if len(a.shape) != len(base):
      return f"CONCAT arg rank {len(a.shape)} != first arg rank {len(base)}"
    for i, (da, db) in enumerate(zip(a.shape, base, strict=True)):
      if i != axis and da != db:
        return f"CONCAT off-axis dim {i} mismatch: {da} vs {db}"
  return None


def _print_format(expr: Expr) -> str | None:
  fmt = expr.attrs.get("format")
  if not isinstance(fmt, str):
    return "PRINT missing string 'format' attr"
  try:
    placeholders = len(print_pieces(fmt)) - 1
  except ValueError as exc:
    return str(exc)
  if not expr.args or placeholders != len(expr.args):
    return f"PRINT format has {placeholders} placeholders for {len(expr.args)} values"
  if expr.type != expr.args[0].type:
    return f"PRINT type {expr.type} differs from its first value's {expr.args[0].type}"
  return None


# Build rule lists for elementwise op classes.
_elementwise_rules = [
  Rule(op, "elementwise-shape-dtype", _elementwise_shape_dtype)
  for op in COMMON_ELEMENTWISE_UNARY | COMMON_ELEMENTWISE_BINARY | COMMON_ELEMENTWISE_TERNARY
]


spec_expr = Spec(
  [
    *spec_expr_shared.any,
    Rule(ExprOp.INPUT, "input-has-name", _input_has_name),
    Rule(ExprOp.CONST, "const-value-present", _const_value_present),
    Rule(ExprOp.PACK, "stored-value-conversion", _storage_conversion),
    Rule(ExprOp.VALUES, "stored-value-conversion", _storage_conversion),
    *_elementwise_rules,
    Rule(ExprOp.STOP_GRADIENT, "stop-gradient-shape-dtype", _unary_shape_dtype),
    Rule(ExprOp.REDUCE, "reduce-shape-kind", _reduce_shape),
    Rule(ExprOp.RESHAPE, "reshape-size", _reshape_size),
    Rule(ExprOp.TRANSPOSE, "transpose-axes", _transpose_axes),
    Rule(ExprOp.SLICE, "slice-index", _slice_index),
    Rule(ExprOp.MATMUL, "matmul-shape", _matmul_shape),
    Rule(ExprOp.CALL, "call-attrs", _call_attrs),
    Rule(ExprOp.EXTERN, "extern-attrs", _call_attrs),
    Rule(ExprOp.VMAP, "vmap-attrs", _vmap_attrs),
    Rule(ExprOp.COND, "cond-attrs", _cond_attrs),
    Rule(ExprOp.LOOP, "loop-attrs", _loop_attrs),
    Rule(ExprOp.GATHER, "gather-indices", _gather_indices),
    Rule(ExprOp.SCATTER, "scatter-indices", _scatter_indices),
    Rule(ExprOp.STACK, "stack-shapes", _stack_shapes),
    Rule(ExprOp.CONCAT, "concat-shapes", _concat_shapes),
    Rule(ExprOp.PRINT, "print-format", _print_format),
  ]
)
