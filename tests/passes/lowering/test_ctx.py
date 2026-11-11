"""Check lowering-rule registration at the compiler boundary."""

import numpy as np
import pytest

from scaly.ir.expr import ExprOp
from scaly.passes.lowering import ctx


def test_duplicate_rule_rejected_without_partial_registration(monkeypatch) -> None:
  monkeypatch.setattr(ctx, "_RULES", {op: rule for op, rule in ctx._RULES.items() if op != ExprOp.INPUT})
  original = ctx._RULES[ExprOp.SIN]

  def rule(ctx, node) -> None:
    pass

  with pytest.raises(ValueError, match="already registered.*sin"):
    ctx.lowers(ExprOp.INPUT, ExprOp.SIN)(rule)

  assert ctx._RULES[ExprOp.SIN] is original
  assert ExprOp.INPUT not in ctx._RULES


def test_non_float64_callee_leaves_are_typed_parameters() -> None:
  from scaly.function.concrete import ConcreteFunction
  from scaly.ir.expr import Expr

  x = Expr.sym("x", ())
  z = Expr.sym("z", (), dtype="int64")
  child = ConcreteFunction._from_exprs("typed_child", [z], [z > 2], ["z"], ["y"])
  fn = ConcreteFunction._from_exprs("typed_parent", [x], [child(Expr.const(3, dtype="int64")).cast("float64") + x], ["x"], ["y"])
  assert fn(np.array(1.0)) == 2.0
