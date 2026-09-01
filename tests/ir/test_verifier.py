"""Phase 2: ``spec_expr`` verifier tests.

Two categories:

1. **Positive**: every graph the existing public API produces verifies cleanly
   under ``spec_expr``. The existing benchmark/test fixtures keep them
   honest; this file pins the contract for the small corpus that future
   AD/rewrite passes should not regress.
2. **Negative**: hand-crafted invalid ``Expr`` nodes are rejected with a
   diagnostic that names the failing rule. These guard the verifier itself —
   future Program IR work and AD rewrites will rely on them when refactoring.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.ir.expr import Expr, ExprOp, binary, unary
from alloy.ir.expr_spec import spec_expr, verify_expr
from alloy.ir.spec import VerifyError
from alloy.ir.types import TensorType, dtypes


def test_simple_scalar_graph_verifies() -> None:
  x = al.sym("x", 3)
  y = (x.sin() + x * x).sum()
  verify_expr(y)


def test_matmul_named_call_verifies() -> None:
  a = al.sym("a", (3, 4))
  b = al.sym("b", (4, 2))
  fn = al.Function._from_exprs("mm", [a, b], [a @ b], ["a", "b"], ["c"])
  c = al.sym("c", (3, 4))
  d = al.sym("d", (4, 2))
  out = fn((c, d))
  verify_expr(out)


def test_vmap_graph_verifies() -> None:
  stage_in = al.sym("u", 2)
  stage = al.Function._from_exprs("stage", [stage_in], [stage_in.sin().sum()], ["u"], ["y"])
  batch = al.sym("batch", 8)
  mapped = al.vmap(stage, length=4, inputs=[(batch, 0, 2)])
  verify_expr(mapped)


def test_jacobian_factory_output_verifies() -> None:
  x = al.sym("x", 3)
  y = (x.sin() + x * x).sum()
  fn = al.Function._from_exprs("f", [x], [y], ["x"], ["y"])
  jac = al.jacobian(fn, "y", "x")
  verify_expr(jac.outputs)


def _forge_negative_shape(expr: Expr, shape: tuple[int, ...]) -> Expr:
  """Mutate a frozen ``Expr``'s tensor type to a shape the constructors refuse.

  Used to test the verifier's defensive contract: legitimate ``Expr`` construction
  already rejects this, so we have to bypass it to prove the rule fires when a
  bug-prone pass (e.g. a future AD/lowering bug) builds a malformed node.
  """
  bad_type = TensorType.__new__(TensorType)
  object.__setattr__(bad_type, "shape", shape)
  object.__setattr__(bad_type, "dtype", expr.type.dtype)
  object.__setattr__(bad_type, "sparsity", None)
  object.__setattr__(bad_type, "diff", expr.type.diff)
  object.__setattr__(expr, "type", bad_type)
  return expr


def test_negative_shape_caught() -> None:
  x = al.sym("x", 3)
  bad = Expr(ExprOp.NEG, (x,), TensorType((3,), dtype=dtypes.float64, diff=True))
  _forge_negative_shape(bad, (-1,))
  with pytest.raises(VerifyError, match="shape-nonnegative"):
    verify_expr(bad)


def test_unary_dtype_mismatch_caught() -> None:
  x = al.sym("x", 3, dtype=dtypes.float32)
  # forge a node whose output dtype differs from its arg's dtype
  bad = Expr(ExprOp.NEG, (x,), TensorType((3,), dtype=dtypes.float64, diff=True))
  with pytest.raises(VerifyError, match="unary-shape-dtype-match"):
    verify_expr(bad)


def test_binary_shape_mismatch_caught() -> None:
  x = al.sym("x", 3)
  y = al.sym("y", 4)
  bad = Expr(ExprOp.ADD, (x, y), TensorType((3,), dtype=dtypes.float64, diff=True))
  with pytest.raises(VerifyError, match="binary-shape-dtype-match"):
    verify_expr(bad)


def test_reshape_size_mismatch_caught() -> None:
  x = al.sym("x", 6)
  bad = Expr(ExprOp.RESHAPE, (x,), TensorType((4,), dtype=dtypes.float64, diff=True), attrs={"shape": (4,)})
  with pytest.raises(VerifyError, match="reshape-size"):
    verify_expr(bad)


def test_transpose_axes_must_be_permutation() -> None:
  x = al.sym("x", (2, 3))
  bad = Expr(ExprOp.TRANSPOSE, (x,), TensorType((3, 2), dtype=dtypes.float64, diff=True), attrs={"axes": (0, 0)})
  with pytest.raises(VerifyError, match="transpose-axes"):
    verify_expr(bad)


def test_matmul_contracting_dim_mismatch_caught() -> None:
  a = al.sym("a", (3, 4))
  b = al.sym("b", (5, 2))
  bad = Expr(ExprOp.MATMUL, (a, b), TensorType((3, 2), dtype=dtypes.float64, diff=True))
  with pytest.raises(VerifyError, match="matmul-shape"):
    verify_expr(bad)


def test_call_arg_shape_mismatch_caught() -> None:
  x = al.sym("x", 3)
  fn = al.Function._from_exprs("f", [x], [x.sum()], ["x"], ["y"])
  bad_arg = al.sym("z", 5)
  bad = Expr(
    ExprOp.CALL,
    (bad_arg,),
    TensorType((), dtype=dtypes.float64, diff=True),
    attrs={"callee": fn, "output": 0},
  )
  with pytest.raises(VerifyError, match="call-attrs"):
    verify_expr(bad)


def test_vmap_rank1_outer_required() -> None:
  stage_in = al.sym("u", 2)
  stage = al.Function._from_exprs("stage", [stage_in], [stage_in.sin().sum()], ["u"], ["y"])
  bad_outer = al.sym("batch", (4, 2))  # not rank-1
  bad = Expr(
    ExprOp.VMAP,
    (bad_outer,),
    TensorType((4,), dtype=dtypes.float64, diff=True),
    attrs={
      "callee": stage,
      "output": 0,
      "length": 4,
      "starts": (0,),
      "strides": (2,),
      "slice_size": 1,
    },
  )
  with pytest.raises(VerifyError, match="vmap-attrs"):
    verify_expr(bad)


def test_const_value_dtype_must_match() -> None:
  arr = np.array([1.0, 2.0], dtype=np.float32)
  # forge a CONST whose declared type dtype is float64 but value dtype is float32
  bad = Expr(ExprOp.CONST, (), TensorType((2,), dtype=dtypes.float64, diff=False), value=arr)
  with pytest.raises(VerifyError, match="const-value-present"):
    verify_expr(bad)


def test_verify_walks_subgraph_and_names_first_failure() -> None:
  good = al.sym("x", 3)
  inner_bad = _forge_negative_shape(Expr(ExprOp.NEG, (good,), TensorType((3,), dtype=dtypes.float64, diff=True)), (-1,))
  # parent's type is fine, but the verifier should already have raised on the inner child.
  parent = Expr(ExprOp.NEG, (inner_bad,), TensorType((3,), dtype=dtypes.float64, diff=True))
  with pytest.raises(VerifyError) as excinfo:
    verify_expr(parent)
  msg = str(excinfo.value)
  assert "shape-nonnegative" in msg


def test_verifier_smoke_on_workload_graphs() -> None:
  """Smoke: a small structurally-rich graph (slice + matmul + sum) verifies, including its Jacobian."""
  z = al.sym("z", 6)
  A = al.const(np.eye(4, 6))
  res = A @ z + z[:4]
  fn = al.Function._from_exprs("f", [z], [res.sum()], ["z"], ["y"])
  verify_expr(fn.outputs)
  jac = al.jacobian(fn, "y", "z")
  verify_expr(jac.outputs)


def test_binary_helper_round_trip_verifies() -> None:
  x = al.sym("x", 3)
  y = al.sym("y", 3)
  z = binary(ExprOp.ADD, x, y)
  verify_expr(z)
  w = unary(ExprOp.SIN, x)
  verify_expr(w)


def test_spec_expr_check_returns_none_on_valid() -> None:
  x = al.sym("x", 3)
  assert spec_expr.check(x) is None
