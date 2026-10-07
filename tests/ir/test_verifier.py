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

from dataclasses import fields

import numpy as np
import pytest

from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
import scaly as sc
from scaly.ir.expr import Expr, ExprOp, binary, unary
from scaly.ir.expr_spec import spec_expr, verify_expr
from scaly.ir.spec import VerifyError
from scaly.ir.types import TensorType, dtypes


def test_simple_scalar_graph_verifies() -> None:
  x = sc.sym("x", 3)
  y = (x.sin() + x * x).sum()
  verify_expr(y)


def test_matmul_named_call_verifies() -> None:
  @sc.function(sc.group(sc.arg("a", (3, 4)), sc.arg("b", (4, 2))), outputs=sc.arg("c"), name="mm")
  def fn(ab: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    a, b = ab
    return a @ b

  c = sc.sym("c", (3, 4))
  d = sc.sym("d", (4, 2))
  out = fn((c, d))
  verify_expr(out)


def test_vmap_graph_verifies() -> None:
  @sc.function(sc.arg("u", 2), outputs=sc.arg("y"))
  def stage(u: sc.Expr) -> sc.Expr:
    return u.sin().sum()

  batch = sc.sym("batch", 8)
  mapped = _mapped_call(stage, length=4, inputs=[(batch, 0, 2)])
  verify_expr(mapped)


def test_jacobian_factory_output_verifies() -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="f")
  def fn(x: sc.Expr) -> sc.Expr:
    return (x.sin() + x * x).sum()

  jac = sc.jacobian(fn, "y", "x")
  verify_expr(as_concrete(jac).outputs)


def _forge_negative_shape(expr: Expr, shape: tuple[int, ...]) -> Expr:
  """A copy of ``expr`` outside the interning cache, typed with a shape the constructors refuse.

  Used to test the verifier's defensive contract: legitimate ``Expr`` construction
  already rejects this, so we have to bypass it to prove the rule fires when a
  bug-prone pass (e.g. a future AD/lowering bug) builds a malformed node. Mutating ``expr``
  itself would change the interned node every later construction returns.
  """
  bad_type = TensorType.__new__(TensorType)
  object.__setattr__(bad_type, "shape", shape)
  object.__setattr__(bad_type, "dtype", expr.type.dtype)
  object.__setattr__(bad_type, "diff", expr.type.diff)
  forged = object.__new__(Expr)
  for f in fields(Expr):
    object.__setattr__(forged, f.name, getattr(expr, f.name))
  object.__setattr__(forged, "type", bad_type)
  return forged


def test_negative_shape_caught() -> None:
  x = sc.sym("x", 3)
  bad = _forge_negative_shape(Expr(ExprOp.NEG, (x,), TensorType((3,), dtype=dtypes.float64, diff=True)), (-1,))
  with pytest.raises(VerifyError, match="shape-nonnegative"):
    verify_expr(bad)


def test_unary_dtype_mismatch_caught() -> None:
  x = sc.sym("x", 3, dtype=dtypes.float32)
  # forge a node whose output dtype differs from its arg's dtype
  bad = Expr(ExprOp.NEG, (x,), TensorType((3,), dtype=dtypes.float64, diff=True))
  with pytest.raises(VerifyError, match="unary-shape-dtype-match"):
    verify_expr(bad)


def test_binary_shape_mismatch_caught() -> None:
  x = sc.sym("x", 3)
  y = sc.sym("y", 4)
  bad = Expr(ExprOp.ADD, (x, y), TensorType((3,), dtype=dtypes.float64, diff=True))
  with pytest.raises(VerifyError, match="binary-shape-dtype-match"):
    verify_expr(bad)


def test_reshape_size_mismatch_caught() -> None:
  x = sc.sym("x", 6)
  bad = Expr(ExprOp.RESHAPE, (x,), TensorType((4,), dtype=dtypes.float64, diff=True), attrs={"shape": (4,)})
  with pytest.raises(VerifyError, match="reshape-size"):
    verify_expr(bad)


def test_transpose_axes_must_be_permutation() -> None:
  x = sc.sym("x", (2, 3))
  bad = Expr(ExprOp.TRANSPOSE, (x,), TensorType((3, 2), dtype=dtypes.float64, diff=True), attrs={"axes": (0, 0)})
  with pytest.raises(VerifyError, match="transpose-axes"):
    verify_expr(bad)


def test_matmul_contracting_dim_mismatch_caught() -> None:
  a = sc.sym("a", (3, 4))
  b = sc.sym("b", (5, 2))
  bad = Expr(ExprOp.MATMUL, (a, b), TensorType((3, 2), dtype=dtypes.float64, diff=True))
  with pytest.raises(VerifyError, match="matmul-shape"):
    verify_expr(bad)


def test_call_arg_shape_mismatch_caught() -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="f")
  def fn(x: sc.Expr) -> sc.Expr:
    return x.sum()

  bad_arg = sc.sym("z", 5)
  bad = Expr(
    ExprOp.CALL,
    (bad_arg,),
    TensorType((), dtype=dtypes.float64, diff=True),
    attrs={"callee": fn.instantiate(), "output": 0},
  )
  with pytest.raises(VerifyError, match="call-attrs"):
    verify_expr(bad)


def test_vmap_rank1_outer_required() -> None:
  @sc.function(sc.arg("u", 2), outputs=sc.arg("y"))
  def stage(u: sc.Expr) -> sc.Expr:
    return u.sin().sum()

  bad_outer = sc.sym("batch", (4, 2))  # not rank-1
  bad = Expr(
    ExprOp.VMAP,
    (bad_outer,),
    TensorType((4,), dtype=dtypes.float64, diff=True),
    attrs={
      "callee": stage.instantiate(),
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
  good = sc.sym("x", 3)
  inner_bad = _forge_negative_shape(Expr(ExprOp.NEG, (good,), TensorType((3,), dtype=dtypes.float64, diff=True)), (-1,))
  # parent's type is fine, but the verifier should already have raised on the inner child.
  parent = Expr(ExprOp.NEG, (inner_bad,), TensorType((3,), dtype=dtypes.float64, diff=True))
  with pytest.raises(VerifyError) as excinfo:
    verify_expr(parent)
  msg = str(excinfo.value)
  assert "shape-nonnegative" in msg


def test_verifier_smoke_on_workload_graphs() -> None:
  """Smoke: a small structurally-rich graph (slice + matmul + sum) verifies, including its Jacobian."""

  @sc.function(sc.arg("z", 6), outputs=sc.arg("y"), name="f")
  def fn(z: sc.Expr) -> sc.Expr:
    A = sc.const(np.eye(4, 6))
    return (A @ z + z[:4]).sum()

  verify_expr(as_concrete(fn).outputs)
  jac = sc.jacobian(fn, "y", "z")
  verify_expr(as_concrete(jac).outputs)


def test_binary_helper_round_trip_verifies() -> None:
  x = sc.sym("x", 3)
  y = sc.sym("y", 3)
  z = binary(ExprOp.ADD, x, y)
  verify_expr(z)
  w = unary(ExprOp.SIN, x)
  verify_expr(w)


def test_spec_expr_check_returns_none_on_valid() -> None:
  x = sc.sym("x", 3)
  assert spec_expr.check(x) is None
