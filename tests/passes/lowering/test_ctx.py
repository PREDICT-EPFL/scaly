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
