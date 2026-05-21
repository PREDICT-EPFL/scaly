"""Semantic IR verifier: op-indexed pattern tables that report the first invalid node.

Phase 2 of the roadmap. Inspired by tinygrad's ``spec.py``/``type_verify`` but kept
small and focused on Alloy's semantic IR. The contract:

- Each ``VerifyRule`` runs for a single ``Ops`` value (or for any op when
  ``op is None``) and either returns ``None`` (the node is well-formed by that
  rule) or a string describing the violation.
- ``Spec`` is a table of rules grouped by op. ``Spec.check(node)`` returns the
  first violation found.
- ``verify_expr(root, spec=spec_semantic)`` walks the DAG topologically and
  raises ``VerifyError`` at the first invalid node, naming the node, the op,
  and the failing rule.

The verifier is opt-in: existing construction-time checks in ``expr.py`` keep
the happy path fast. ``verify_expr`` is the explicit, defensive check passes
should run after non-trivial graph rewrites or AD transforms — and the
negative-test harness for those passes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .expr import Expr, topo
from .ops import COMMON_ELEMENTWISE_BINARY, COMMON_ELEMENTWISE_UNARY, OP_INFO, Ops
from .types import DType, broadcast_shape


class VerifyError(Exception):
  """A semantic IR node violated one of the active spec rules."""


@dataclass(frozen=True, slots=True)
class VerifyRule:
  """A single verifier rule.

  ``check`` returns ``None`` when the node passes and a string diagnostic when
  it does not. ``description`` is shown in the error message to identify the
  rule that fired.
  """

  op: Ops | None
  description: str
  check: Callable[[Expr], str | None]

  def applies(self, expr: Expr) -> bool:
    return self.op is None or expr.op == self.op


class Spec:
  """Op-indexed table of ``VerifyRule``s, similar in shape to ``PatternMatcher``."""

  def __init__(self, rules: Iterable[VerifyRule]) -> None:
    self.any: list[VerifyRule] = []
    self.by_op: dict[Ops, list[VerifyRule]] = defaultdict(list)
    for rule in rules:
      if rule.op is None:
        self.any.append(rule)
      else:
        self.by_op[rule.op].append(rule)

  def candidates(self, op: Ops) -> Iterable[VerifyRule]:
    yield from self.any
    yield from self.by_op.get(op, ())

  def check(self, expr: Expr) -> tuple[VerifyRule, str] | None:
    for rule in self.candidates(Ops(expr.op)):
      diag = rule.check(expr)
      if diag is not None:
        return rule, diag
    return None

  def merge(self, *others: "Spec") -> "Spec":
    rules = list(self.any)
    for op, ops_rules in self.by_op.items():
      rules.extend(ops_rules)
    for other in others:
      rules.extend(other.any)
      for op, ops_rules in other.by_op.items():
        rules.extend(ops_rules)
    return Spec(rules)


def verify_expr(root: Expr | Iterable[Expr], spec: "Spec | None" = None) -> None:
  """Topologically walk the DAG below ``root`` and raise on the first violation.

  ``spec`` defaults to ``spec_semantic`` — the full semantic IR contract.
  """
  if spec is None:
    spec = spec_semantic
  outputs = (root,) if isinstance(root, Expr) else tuple(root)
  for node in topo(outputs):
    result = spec.check(node)
    if result is not None:
      rule, diag = result
      label = node.name or f"%{node.id}"
      raise VerifyError(f"verify_expr: node {label} op={Ops(node.op).value} failed rule {rule.description!r}: {diag}")


# ---------------------------------------------------------------------------
# Shared rules: invariants every semantic IR node must satisfy.
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
  info = OP_INFO.get(Ops(expr.op))
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


spec_semantic_shared = Spec(
  [
    VerifyRule(None, "shape-nonnegative", _shape_nonneg),
    VerifyRule(None, "dtype-is-DType", _dtype_is_dtype),
    VerifyRule(None, "arity-matches-op", _arity_matches),
    VerifyRule(None, "sparsity-shape-matches", _sparsity_shape_matches),
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


def _map_attrs(expr: Expr) -> str | None:
  callee = expr.attrs.get("callee")
  if callee is None:
    return "MAP missing 'callee' attr"
  for k in ("length", "starts", "strides", "slice_size", "output"):
    if k not in expr.attrs:
      return f"MAP missing {k!r} attr"
  length = int(expr.attrs["length"])
  if length < 0:
    return f"MAP length must be non-negative, got {length}"
  starts = expr.attrs["starts"]
  strides = expr.attrs["strides"]
  if len(starts) != len(callee.inputs) or len(strides) != len(callee.inputs):
    return f"MAP starts/strides length mismatch vs callee {callee.name!r} inputs"
  for outer in expr.args:
    if len(outer.shape) != 1:
      return f"MAP outer arg must be rank-1, got {outer.shape}"
  expected_size = length * int(expr.attrs["slice_size"])
  if expr.shape != (expected_size,):
    return f"MAP output shape {expr.shape} != ({expected_size},)"
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


# Build rule lists for elementwise op classes.
_unary_rules = [VerifyRule(op, "unary-shape-dtype-match", _unary_shape_dtype) for op in COMMON_ELEMENTWISE_UNARY]
_binary_rules = [VerifyRule(op, "binary-shape-dtype-match", _binary_shape_dtype) for op in COMMON_ELEMENTWISE_BINARY]


spec_semantic = Spec(
  [
    *spec_semantic_shared.any,
    VerifyRule(Ops.INPUT, "input-has-name", _input_has_name),
    VerifyRule(Ops.CONST, "const-value-present", _const_value_present),
    *_unary_rules,
    *_binary_rules,
    VerifyRule(Ops.SUM, "sum-output-scalar", _sum_shape),
    VerifyRule(Ops.RESHAPE, "reshape-size", _reshape_size),
    VerifyRule(Ops.TRANSPOSE, "transpose-axes", _transpose_axes),
    VerifyRule(Ops.MATMUL, "matmul-shape", _matmul_shape),
    VerifyRule(Ops.CALL, "call-attrs", _call_attrs),
    VerifyRule(Ops.MAP, "map-attrs", _map_attrs),
    VerifyRule(Ops.GATHER, "gather-indices", _gather_indices),
    VerifyRule(Ops.SCATTER, "scatter-indices", _scatter_indices),
    VerifyRule(Ops.STACK, "stack-shapes", _stack_shapes),
    VerifyRule(Ops.CONCAT, "concat-shapes", _concat_shapes),
  ]
)


__all__ = [
  "Spec",
  "VerifyError",
  "VerifyRule",
  "spec_semantic",
  "spec_semantic_shared",
  "verify_expr",
]
