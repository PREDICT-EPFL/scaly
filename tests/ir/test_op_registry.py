"""The expression op registry: every builtin is registered through it, a registered name is what an
``Expr`` holds and interns by, a name registers once, and an unknown name is refused."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.ir.expr import Expr, ExprOp, OpDef, op_def, register_op, registered_ops
from scaly.ir.expr_spec import verify_expr
from scaly.ir.spec import VerifyError
from scaly.ir.types import TensorType

# Registered at import, once per process, as an extension would: before any Expr uses it.
TOY = register_op("test_registry_toy", arity=1, numpy=lambda x: 2.0 * x)


def test_every_builtin_is_registered_in_enum_order() -> None:
  assert registered_ops()[: len(ExprOp)] == tuple(ExprOp)
  assert all(op_def(op).name is op for op in ExprOp)
  assert op_def("add") is op_def(ExprOp.ADD)


def test_an_extension_op_interns_by_its_registered_name() -> None:
  x = sc.sym("x", 3)
  by_name = Expr("test_registry_toy", (x,), x.type)
  assert Expr(TOY.name, (x,), x.type) is by_name
  assert by_name.op == "test_registry_toy" and isinstance(by_name.op, str)
  assert by_name.structural_key()[0] == "test_registry_toy"
  verify_expr(by_name)
  with pytest.raises(VerifyError, match="expects 1 args"):
    verify_expr(Expr(TOY.name, (x, x), x.type))


def test_a_builtin_built_by_name_is_the_same_node() -> None:
  x = sc.sym("x", 2)
  total = x + x
  assert Expr("add", (x, x), total.type) is total
  assert total.op is ExprOp.ADD


def test_a_name_registers_once_and_an_unknown_one_is_refused() -> None:
  with pytest.raises(ValueError, match="already registered"):
    register_op("add", arity=2)
  with pytest.raises(ValueError, match="already registered"):
    register_op("test_registry_toy", arity=1)
  with pytest.raises(ValueError, match="unknown expression op 'nope'"):
    Expr("nope", (), TensorType())
  assert isinstance(TOY, OpDef) and TOY.numpy is not None
  np.testing.assert_allclose(TOY.numpy(np.ones(2)), [2.0, 2.0])
