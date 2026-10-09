"""Check lowering-rule registration at the compiler boundary."""

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


def test_rule_registration_returns_the_rule(monkeypatch) -> None:
  monkeypatch.setattr(ctx, "_RULES", {op: rule for op, rule in ctx._RULES.items() if op != ExprOp.INPUT})

  def rule(ctx, node) -> None:
    pass

  assert ctx.lowers(ExprOp.INPUT)(rule) is rule
  assert ctx._RULES[ExprOp.INPUT] is rule


@pytest.mark.parametrize("dtype", ["bool", "int32", "int64", "float32"])
@pytest.mark.parametrize("side", ["input", "output"])
def test_non_float64_leaves_refused(dtype, side) -> None:
  from scaly.function.concrete import ConcreteFunction
  from scaly.ir.expr import Expr

  x = Expr.sym("x", (), dtype=dtype if side == "input" else "float64")
  y = Expr.const(1, dtype=dtype) if side == "output" else Expr.const(1.0)
  fn = ConcreteFunction._from_exprs("typed_boundary", [x], [y], ["x"], ["y"])
  with pytest.raises(ctx.LoweringError, match=rf"{side}.*{dtype}.*float64"):
    ctx.lower_function(fn)


def test_non_float64_callee_leaf_refused() -> None:
  from scaly.function.concrete import ConcreteFunction
  from scaly.ir.expr import Expr

  x = Expr.sym("x", ())
  z = Expr.sym("z", (), dtype="int64")
  child = ConcreteFunction._from_exprs("typed_child", [z], [Expr.const(1.0)], ["z"], ["y"])
  fn = ConcreteFunction._from_exprs("typed_parent", [x], [child(Expr.const(1, dtype="int64")) + x], ["x"], ["y"])
  with pytest.raises(ctx.LoweringError, match="typed_child.*input.*int64.*float64"):
    ctx.lower_function(fn)
