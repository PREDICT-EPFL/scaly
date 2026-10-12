"""The elementwise table: one row per op, lazy partials, and constants typed like the primal."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.ir.expr import COMMON_ELEMENTWISE_BINARY, COMMON_ELEMENTWISE_UNARY, OP_INFO, ExprOp, topo

UNARY = {
  ExprOp.NEG: lambda x: -x,
  ExprOp.SIN: lambda x: x.sin(),
  ExprOp.COS: lambda x: x.cos(),
  ExprOp.TAN: lambda x: x.tan(),
  ExprOp.ASIN: lambda x: x.asin(),
  ExprOp.ACOS: lambda x: x.acos(),
  ExprOp.ATAN: lambda x: x.atan(),
  ExprOp.SINH: lambda x: x.sinh(),
  ExprOp.COSH: lambda x: x.cosh(),
  ExprOp.TANH: lambda x: x.tanh(),
  ExprOp.ERF: lambda x: x.erf(),
  ExprOp.EXP: lambda x: x.exp(),
  ExprOp.LOG: lambda x: x.log(),
  ExprOp.SQRT: lambda x: x.sqrt(),
  ExprOp.ABS: lambda x: x.abs(),
}
BINARY = {
  ExprOp.ADD: lambda x, y: x + y,
  ExprOp.SUB: lambda x, y: x - y,
  ExprOp.MUL: lambda x, y: x * y,
  ExprOp.DIV: lambda x, y: x / y,
  ExprOp.POW: lambda x, y: x**y,
  ExprOp.ATAN2: sc.atan2,
}
FLOAT32_CASES = {
  **{op.value: (build, 1) for op, build in UNARY.items()},
  **{op.value: (build, 2) for op, build in BINARY.items()},
  "square": (lambda x: x * x, 1),
}


def test_table_has_one_row_per_elementwise_op():
  from scaly.ad.rules import ELEMENTWISE

  assert set(ELEMENTWISE) == COMMON_ELEMENTWISE_UNARY | COMMON_ELEMENTWISE_BINARY
  for op, row in ELEMENTWISE.items():
    assert (row.partials is not None) == OP_INFO[op].differentiable
    assert row.partials is None or len(row.partials) == OP_INFO[op].arity


@pytest.mark.parametrize("name", sorted(FLOAT32_CASES))
def test_float32_derivatives_stay_float32(name):
  f32 = sc.dtypes.float32
  build, arity = FLOAT32_CASES[name]
  operands = tuple(sc.sym(f"typed_{i}", 2, dtype=f32) for i in range(arity))
  out = build(*operands)
  derivatives = (
    *(sc.jvp(out, wrt, sc.sym("typed_seed", 2, dtype=f32)) for wrt in operands),
    *(sc.jvp_many(out, wrt, sc.sym("typed_seeds", (3, 2), dtype=f32)) for wrt in operands),
    *sc.vjp((out,), operands, (sc.sym("typed_cot", 2, dtype=f32),)),
  )
  assert {node.type.dtype for node in topo(derivatives) if node.type.dtype.is_floating} == {f32}


def test_partials_are_built_only_for_active_operands(monkeypatch):
  def refuse(self):
    raise AssertionError("built log(x) for an inactive exponent")

  x, p = sc.sym("lazy_base", 2), sc.sym("lazy_exponent", 2)
  y = x**p
  monkeypatch.setattr(sc.Expr, "log", refuse)
  sc.jvp(y, x, sc.sym("lazy_seed", 2))
  sc.jvp_many(y, x, sc.sym("lazy_seeds", (3, 2)))
  sc.vjp((y,), (x,), (sc.sym("lazy_cot", 2),))


@pytest.mark.parametrize(
  ("op", "x", "y", "direction", "expected"),
  [
    ("div", 1e200, 1e-100, 1e-300, -1e100),
    ("div", 1e-100, 1e200, 1e300, -1e-200),
    ("atan2", 1e-100, 1e150, 1e300, -1e-100),
    ("log", 1e-310, 1.0, 1e-310, 1.0),
  ],
)
def test_quotient_partials_keep_finite_products(op, x, y, direction, expected):
  """A partial written as a quotient multiplies its numerator into the tangent before dividing."""

  @sc.function(
    sc.group(sc.arg("scale_x", ()), sc.arg("scale_y", ()), sc.arg("scale_t", ())),
    outputs=sc.group(sc.arg("forward"), sc.arg("reverse")),
    name=f"scale_{op}_{np.log10(x):.0f}",
  )
  def products(inputs):
    a, b, t = inputs
    out, wrt = {"div": (a / b, b), "atan2": (sc.atan2(a, b), b), "log": (a.log(), a)}[op]
    return sc.jvp(out, wrt, t), sc.vjp((out,), (wrt,), (t,))[0]

  for value in products((np.array(x), np.array(y), np.array(direction))):
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=0)
