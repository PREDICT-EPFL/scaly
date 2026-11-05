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
import scaly as sc
from scaly.ir.expr import Expr, ExprOp
from scaly.ir.expr_spec import verify_expr
from scaly.ir.spec import VerifyError
from scaly.ir.types import TensorType, dtypes


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


@pytest.mark.parametrize("index,shape", [(None, (2,)), ((slice(None),), (3,)), ((4,), ())])
def test_slice_contract_rejects_invalid_nodes(index, shape) -> None:
  x = sc.sym("slice_contract_x", 2)
  attrs = {} if index is None else {"index": index}
  bad = Expr(ExprOp.SLICE, (x,), TensorType(shape), attrs=attrs)
  with pytest.raises(VerifyError, match="slice-index"):
    verify_expr(bad)


def test_function_construction_verifies_inner_nodes() -> None:
  x = sc.sym("construction_verify_x", 3)
  bad = Expr(ExprOp.RESHAPE, (x,), TensorType((4,)), attrs={"shape": (4,)})
  with pytest.raises(VerifyError, match="reshape-size"):
    sc.function(sc.arg("construction_verify_x", 3), outputs=sc.arg("out"))(lambda _: bad).instantiate()


@pytest.mark.parametrize("fault", ["solver", "output", "output_name", "arity", "input_shape", "shape", "diff"])
def test_solver_call_contract(fault) -> None:
  from scaly.solvers.model import SolverDescriptor

  descriptor = SolverDescriptor("verify_solver", "test", 2, 0, 0, (("p", (2,)),), (("x", (2,)),), (), 1)
  arg = sc.sym("solver_contract_p", 2, diff=False)
  attrs = {"solver": descriptor, "output": 0, "output_name": "x"}
  args = (arg,)
  shape, diff = (2,), False
  if fault == "solver":
    del attrs["solver"]
  elif fault == "output":
    attrs["output"] = 1
  elif fault == "output_name":
    attrs["output_name"] = "wrong"
  elif fault == "arity":
    args = ()
  elif fault == "input_shape":
    args = (sc.sym("solver_bad_p", 3, diff=False),)
  elif fault == "shape":
    shape = (3,)
  else:
    diff = True
  bad = Expr(ExprOp.SOLVER_CALL, args, TensorType(shape, diff=diff), attrs=attrs)
  with pytest.raises(VerifyError, match="solver-call-signature"):
    verify_expr(bad)


@pytest.mark.parametrize("index", [(slice(None, None, -1), -1), (slice(1, 1), slice(None)), (1, slice(None, None, 2))])
def test_slice_contract_accepts_normalized_index(index) -> None:
  x = sc.sym("slice_positive_x", (3, 4))
  sliced = x[index]
  verify_expr(sliced)
  assert sliced.shape == np.empty(x.shape)[index].shape


def test_solver_call_contract_accepts_descriptor_function() -> None:
  from scaly.solvers.model import SolverDescriptor, descriptor_function

  descriptor = SolverDescriptor("valid_verify_solver", "test", 2, 0, 0, (("p", (2,)),), (("x", (2,)),), (), 1)
  verify_expr(as_concrete(descriptor_function(descriptor)).outputs)
