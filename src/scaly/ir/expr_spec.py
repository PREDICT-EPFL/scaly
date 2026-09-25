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

The verifier is opt-in: existing construction-time checks in ``expr.py`` keep
the happy path fast. ``verify_expr`` is the explicit, defensive check passes
should run after non-trivial graph rewrites or AD transforms — and the
negative-test harness for those passes.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .expr import COMMON_ELEMENTWISE_BINARY, COMMON_ELEMENTWISE_UNARY, COMPARE_OPS, Expr, ExprOp, OP_INFO, topo
from .spec import Rule, Spec, VerifyError
from .types import DType, broadcast_shape


def verify_expr(root: Expr | Iterable[Expr], spec: "Spec | None" = None) -> None:
  """Topologically walk the DAG below ``root`` and raise on the first violation.

  ``spec`` defaults to ``spec_expr`` — the full expression-dialect contract.
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


def _sparsity_shape_matches(expr: Expr) -> str | None:
  sp = expr.type.sparsity
  if sp is None:
    return None
  if expr.type.shape != sp.shape:
    return f"sparsity shape {sp.shape} disagrees with tensor shape {expr.type.shape}"
  if sp.nnz != len(sp.rows) or sp.nnz != len(sp.cols):
    return f"sparsity has inconsistent nnz={sp.nnz} rows={len(sp.rows)} cols={len(sp.cols)}"
  return None


spec_expr_shared = Spec(
  [
    Rule(None, "shape-nonnegative", _shape_nonneg),
    Rule(None, "dtype-is-DType", _dtype_is_dtype),
    Rule(None, "arity-matches-op", _arity_matches),
    Rule(None, "sparsity-shape-matches", _sparsity_shape_matches),
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


def _binary_shape_dtype(expr: Expr) -> str | None:
  if len(expr.args) != 2:
    return f"binary op has {len(expr.args)} args"
  x, y = expr.args
  if x.type.dtype != y.type.dtype:
    return f"binary op operands have mismatched dtypes: {x.type.dtype} vs {y.type.dtype}"
  if x.type.dtype != expr.type.dtype:
    return f"binary op output dtype {expr.type.dtype} != operand dtype {x.type.dtype}"
  try:
    expected = broadcast_shape(x.shape, y.shape)
  except ValueError as e:
    return f"binary op operands not broadcastable: {e}"
  if expected != expr.shape:
    return f"binary op output shape {expr.shape} != broadcast({x.shape}, {y.shape})={expected}"
  return None


def _sum_shape(expr: Expr) -> str | None:
  if expr.shape != ():
    return f"SUM output must be scalar, got shape {expr.shape}"
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


def _matmul_shape(expr: Expr) -> str | None:
  if len(expr.args) != 2:
    return f"MATMUL has {len(expr.args)} args"
  x, y = expr.args
  if x.type.dtype != y.type.dtype:
    return f"MATMUL operands have mismatched dtypes: {x.type.dtype} vs {y.type.dtype}"
  if len(x.shape) == 1 and len(y.shape) == 1:
    if x.shape[0] != y.shape[0] or expr.shape != ():
      return f"MATMUL vec/vec shape mismatch: {x.shape}@{y.shape} -> {expr.shape}"
  elif len(x.shape) == 2 and len(y.shape) == 1:
    if x.shape[1] != y.shape[0] or expr.shape != (x.shape[0],):
      return f"MATMUL mat/vec shape mismatch: {x.shape}@{y.shape} -> {expr.shape}"
  elif len(x.shape) == 1 and len(y.shape) == 2:
    if x.shape[0] != y.shape[0] or expr.shape != (y.shape[1],):
      return f"MATMUL vec/mat shape mismatch: {x.shape}@{y.shape} -> {expr.shape}"
  elif len(x.shape) == 2 and len(y.shape) == 2:
    if x.shape[1] != y.shape[0] or expr.shape != (x.shape[0], y.shape[1]):
      return f"MATMUL mat/mat shape mismatch: {x.shape}@{y.shape} -> {expr.shape}"
  else:
    return f"MATMUL unsupported ranks {x.shape}@{y.shape}"
  return None


def _call_attrs(expr: Expr) -> str | None:
  callee = expr.attrs.get("callee")
  if callee is None:
    return "CALL missing 'callee' attr"
  out_idx = expr.attrs.get("output")
  if out_idx is None:
    return "CALL missing 'output' attr"
  if not 0 <= int(out_idx) < len(callee.outputs):
    return f"CALL output index {out_idx} out of range for callee {callee.name!r}"
  if len(expr.args) != len(callee.inputs):
    return f"CALL has {len(expr.args)} args but callee {callee.name!r} expects {len(callee.inputs)}"
  for i, (actual, formal) in enumerate(zip(expr.args, callee.inputs, strict=True)):
    if actual.shape != formal.shape:
      return f"CALL arg {i} shape {actual.shape} != callee formal shape {formal.shape}"
  expected_out = callee.outputs[int(out_idx)]
  if expected_out.shape != expr.shape:
    return f"CALL output shape {expr.shape} != callee output shape {expected_out.shape}"
  return None


def _vmap_attrs(expr: Expr) -> str | None:
  """Verify VMAP's flat outer storage; callee formals/outputs may be rank-2, but outers are rank-1."""
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
  if len(starts) != len(callee.inputs) or len(strides) != len(callee.inputs):
    return f"VMAP starts/strides length mismatch vs callee {callee.name!r} inputs"
  for outer in expr.args:
    if len(outer.shape) != 1:
      return f"VMAP outer arg must be rank-1, got {outer.shape}"
  expected_size = length * int(expr.attrs["slice_size"])
  if expr.shape != (expected_size,):
    return f"VMAP output shape {expr.shape} != ({expected_size},)"
  return None


def _scatter_indices(expr: Expr) -> str | None:
  if "indices" not in expr.attrs:
    return "SCATTER missing 'indices' attr"
  idx = expr.attrs["indices"]
  if not expr.args:
    return "SCATTER missing values arg"
  if int(np.asarray(idx).size) != expr.args[0].size:
    return f"SCATTER indices size {np.asarray(idx).size} != values size {expr.args[0].size}"
  return None


def _gather_indices(expr: Expr) -> str | None:
  if "indices" not in expr.attrs:
    return "GATHER missing 'indices' attr"
  idx = np.asarray(expr.attrs["indices"])
  if tuple(idx.shape) != expr.shape:
    return f"GATHER output shape {expr.shape} != indices shape {tuple(idx.shape)}"
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


def _broadcast_all(expr: Expr) -> str | None:
  shape: tuple[int, ...] = ()
  try:
    for arg in expr.args:
      shape = broadcast_shape(shape, arg.shape)
  except ValueError as e:
    return f"operands not broadcastable: {e}"
  if shape != expr.shape:
    return f"output shape {expr.shape} != broadcast of operand shapes {[a.shape for a in expr.args]}={shape}"
  return None


def _compare_types(expr: Expr) -> str | None:
  x, y = expr.args
  if x.type.dtype != y.type.dtype:
    return f"comparison operands have mismatched dtypes: {x.type.dtype} vs {y.type.dtype}"
  if not expr.type.dtype.is_bool:
    return f"comparison result must be bool, got {expr.type.dtype}"
  return _broadcast_all(expr)


def _logical_types(expr: Expr) -> str | None:
  if not all(a.type.dtype.is_bool for a in expr.args) or not expr.type.dtype.is_bool:
    return f"logical op needs bool operands and result, got {[str(a.type.dtype) for a in expr.args]} -> {expr.type.dtype}"
  return _broadcast_all(expr)


def _isfinite_types(expr: Expr) -> str | None:
  if not expr.args[0].type.dtype.is_floating or not expr.type.dtype.is_bool:
    return f"isfinite maps a floating operand to bool, got {expr.args[0].type.dtype} -> {expr.type.dtype}"
  return _broadcast_all(expr)


def _select_types(expr: Expr) -> str | None:
  cond, a, b = expr.args
  if not cond.type.dtype.is_bool:
    return f"SELECT condition must be bool, got {cond.type.dtype}"
  if a.type.dtype != b.type.dtype or a.type.dtype != expr.type.dtype:
    return f"SELECT branches and result must share a dtype, got {a.type.dtype}, {b.type.dtype} -> {expr.type.dtype}"
  return _broadcast_all(expr)


def _cast_types(expr: Expr) -> str | None:
  if expr.args[0].shape != expr.shape:
    return f"CAST output shape {expr.shape} != input shape {expr.args[0].shape}"
  if expr.type.dtype.is_bool:
    return "CAST to bool is spelled as a comparison with zero"
  return None


def _reduce_shape(expr: Expr) -> str | None:
  if expr.shape != ():
    return f"{expr.op} output must be a scalar, got shape {expr.shape}"
  if expr.args[0].size == 0:
    return f"{expr.op} of an empty operand has no value"
  if expr.args[0].type.dtype != expr.type.dtype:
    return f"{expr.op} dtype {expr.type.dtype} != operand dtype {expr.args[0].type.dtype}"
  return None


# Build rule lists for elementwise op classes.
_unary_rules = [Rule(op, "unary-shape-dtype-match", _unary_shape_dtype) for op in COMMON_ELEMENTWISE_UNARY]
_binary_rules = [Rule(op, "binary-shape-dtype-match", _binary_shape_dtype) for op in COMMON_ELEMENTWISE_BINARY]


spec_expr = Spec(
  [
    *spec_expr_shared.any,
    Rule(ExprOp.INPUT, "input-has-name", _input_has_name),
    Rule(ExprOp.CONST, "const-value-present", _const_value_present),
    *_unary_rules,
    *_binary_rules,
    *(Rule(op, "compare-types", _compare_types) for op in COMPARE_OPS),
    *(Rule(op, "logical-types", _logical_types) for op in (ExprOp.AND, ExprOp.OR, ExprOp.NOT)),
    Rule(ExprOp.ISFINITE, "isfinite-types", _isfinite_types),
    Rule(ExprOp.SELECT, "select-types", _select_types),
    Rule(ExprOp.CAST, "cast-types", _cast_types),
    Rule(ExprOp.SUM, "sum-output-scalar", _sum_shape),
    *(Rule(op, "reduce-output-scalar", _reduce_shape) for op in (ExprOp.MAX, ExprOp.MIN)),
    Rule(ExprOp.RESHAPE, "reshape-size", _reshape_size),
    Rule(ExprOp.TRANSPOSE, "transpose-axes", _transpose_axes),
    Rule(ExprOp.MATMUL, "matmul-shape", _matmul_shape),
    Rule(ExprOp.CALL, "call-attrs", _call_attrs),
    Rule(ExprOp.VMAP, "vmap-attrs", _vmap_attrs),
    Rule(ExprOp.GATHER, "gather-indices", _gather_indices),
    Rule(ExprOp.SCATTER, "scatter-indices", _scatter_indices),
    Rule(ExprOp.STACK, "stack-shapes", _stack_shapes),
    Rule(ExprOp.CONCAT, "concat-shapes", _concat_shapes),
  ]
)


__all__ = [
  "spec_expr",
  "spec_expr_shared",
  "verify_expr",
]
