"""Check lowering boundaries and rule registration."""

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


@pytest.mark.parametrize("nested", [False, True])
def test_lowering_verifies_only_normalized_graphs(monkeypatch, nested: bool) -> None:
  from scaly.function.concrete import ConcreteFunction
  from scaly.ir.expr import Expr, topo
  from scaly.ir.expr_spec import spec_expr

  x = Expr.sym("verification_x", 2)
  child = ConcreteFunction._from_exprs("verification_child", [x], [(x * 1.0).sin().sum()], ["x"], ["y"])
  fn = ConcreteFunction._from_exprs("verification_parent", [x], [child(x) + x.sum()], ["x"], ["y"]) if nested else child
  checked: list[Expr] = []
  normalized: list[ConcreteFunction] = []
  original = spec_expr.check

  def check(node: Expr):
    checked.append(node)
    return original(node)

  monkeypatch.setattr(spec_expr, "check", check)
  ctx.lower_function(fn, observe_expr=lambda _phase, fun: normalized.append(fun))

  assert len(normalized) == (2 if nested else 1)
  assert checked == [node for fun in normalized for node in topo((*fun.inputs, *fun.results))]


@pytest.mark.parametrize("dtype", ["float32"])
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
