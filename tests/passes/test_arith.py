"""The shared arithmetic identities hold in expression graphs, scalarized code and loop bodies alike."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import scaly as sc
from scaly.ir import program as p
from scaly.ir.expr import topo
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.ir.types import dtypes
from scaly.passes.arith import fold_program
from scaly.passes.lowering import lower_function, main_proc
from scaly.passes.program.fold_arith import fold_arith


@pytest.fixture(autouse=True)
def scalar_libm(monkeypatch):
  monkeypatch.setenv("SCALY_VECTOR_LIBM", "none")


DATA = np.array([0.5, -1.25, 2.0])

# name -> (build, reference, operations that must disappear)
CASES: dict[str, tuple[Callable[[sc.Expr], sc.Expr], Callable[[np.ndarray], np.ndarray], set[str]]] = {
  "add_zero": (lambda x: x + 0.0, lambda v: v, {"add"}),
  "zero_add": (lambda x: 0.0 + x, lambda v: v, {"add"}),
  "sub_zero": (lambda x: x - 0.0, lambda v: v, {"sub"}),
  "zero_sub": (lambda x: 0.0 - x, lambda v: -v, {"sub"}),
  "mul_one": (lambda x: x * 1.0, lambda v: v, {"mul"}),
  "one_mul": (lambda x: 1.0 * x, lambda v: v, {"mul"}),
  "div_one": (lambda x: x / 1.0, lambda v: v, {"div"}),
  "pow_one": (lambda x: x**1.0, lambda v: v, {"pow"}),
  "mul_zero": (lambda x: x * 0.0, lambda v: 0 * v, {"mul"}),
  "zero_div": (lambda x: 0.0 / x, lambda v: 0 * v, {"div"}),
  "sub_self": (lambda x: x - x, lambda v: 0 * v, {"sub"}),
  "div_self": (lambda x: x / x, lambda v: v / v, {"div"}),
  "neg_neg": (lambda x: -(-x), lambda v: v, {"neg"}),
  "add_neg": (lambda x: x + (-x.sin()), lambda v: v - np.sin(v), {"neg"}),
  "neg_add": (lambda x: (-x.sin()) + x, lambda v: v - np.sin(v), {"neg"}),
  "sub_neg": (lambda x: x - (-x.sin()), lambda v: v + np.sin(v), {"neg"}),
  "neg_mul_neg": (lambda x: (-x) * (-x.sin()), lambda v: v * np.sin(v), {"neg"}),
  "neg_mul": (lambda x: x + (-x) * x.sin(), lambda v: v - v * np.sin(v), {"neg", "add"}),
  "mul_neg": (lambda x: x + x * (-x.sin()), lambda v: v - v * np.sin(v), {"neg", "add"}),
  "neg_div": (lambda x: x + (-x) / x.sin(), lambda v: v - v / np.sin(v), {"neg", "add"}),
  "div_neg": (lambda x: x + x / (-x.sin()), lambda v: v - v / np.sin(v), {"neg", "add"}),
  "minus_one_mul": (lambda x: -1.0 * x, lambda v: -v, {"mul"}),
  "pow_zero": (lambda x: x**0.0, lambda v: np.ones_like(v), {"pow"}),
  "pow_two": (lambda x: x**2.0, lambda v: v * v, {"pow"}),
  "constants": (lambda x: x + sc.const(3.0) / sc.const(2.0), lambda v: v + 1.5, {"div"}),
  "uniform_tensor": (lambda x: x * sc.const([1.0, 1.0, 1.0]) + sc.const([0.0, 0.0, 0.0]), lambda v: v, {"mul", "add"}),
}


def _function(name: str, build: Callable[[sc.Expr], sc.Expr], shape: int = 3) -> sc.Function:
  @sc.function(sc.arg("x", shape), outputs=sc.arg("y", ...), name=name)
  def fn(x: sc.Expr) -> sc.Expr:
    return build(x)

  return fn


def _proc_ops(proc: ProgramNode) -> set[str]:
  return {str(n.op) for stmt in proc.args[proc.attrs["param_count"] :] for n in walk_program(stmt)}


@pytest.mark.parametrize("name", sorted(CASES))
def test_expression_graph(name: str) -> None:
  build, reference, gone = CASES[name]
  x = sc.sym("x", 3)
  y = sc.simplify(build(x))
  assert not gone & {str(n.op) for n in topo([y])}
  np.testing.assert_allclose(_function(f"arith_expr_{name}", lambda x: sc.simplify(build(x)))(DATA), reference(DATA), rtol=1e-15, atol=0)


@pytest.mark.parametrize("name", sorted(CASES))
def test_scalarized_procedure(name: str) -> None:
  build, reference, gone = CASES[name]
  fn = _function(f"arith_scalar_{name}", lambda x: build(x).scalar())
  proc = main_proc(lower_function(fn))
  assert proc.attrs.get("scalarized")
  assert not gone & _proc_ops(proc)
  np.testing.assert_allclose(fn(DATA), reference(DATA), rtol=1e-15, atol=0)


@pytest.mark.parametrize("name", sorted(CASES))
def test_loop_body(name: str) -> None:
  build, reference, gone = CASES[name]
  fn = _function(f"arith_loop_{name}", lambda x: build(x).block())
  proc = main_proc(lower_function(fn))
  assert not proc.attrs.get("scalarized") and "for" in _proc_ops(proc)
  assert not gone & _proc_ops(proc)
  np.testing.assert_allclose(fn(DATA), reference(DATA), rtol=1e-15, atol=0)


def test_mixed_constant_tensor_folds_per_element_only_where_the_element_is_known() -> None:
  build = lambda x, hint: (x * sc.const([1.0, 0.0, 2.0]) + sc.const([0.0, 1.0, 0.0]) * x.sin()).with_lowering(hint)
  scalar = _function("mixed_scalar", lambda x: build(x, "scalar"))
  ops = [n.op for stmt in main_proc(lower_function(scalar)).args[1:] for n in walk_program(stmt)]
  assert ops.count(ProgramOp.MUL) == 1 and ops.count(ProgramOp.SIN) == 1
  loop = _function("mixed_loop", lambda x: build(x, "block"))
  assert {"mul", "add", "sin", "for"} <= _proc_ops(main_proc(lower_function(loop)))
  expected = DATA * [1.0, 0.0, 2.0] + [0.0, 1.0, 0.0] * np.sin(DATA)
  for fn in (scalar, loop):
    np.testing.assert_allclose(fn(DATA), expected, rtol=1e-15, atol=0)


def test_int64_index_arithmetic_folds_with_c_truncation() -> None:
  x, y = (p.buffer(name, dtypes.float64, (4,)) for name in ("ix", "iy"))
  i = p.var("i")
  index = p.add(p.mul(i, p.const_int(1)), p.div(p.const_int(-7), p.const_int(2)))
  loop = p.for_(p.range_("i", 3, 4), [p.store(p.view(y, [index]), p.load(p.view(x, [p.sub(i, p.const_int(0))])))])
  prog = fold_arith(p.program([p.proc("index", [x, y], [loop])]))
  store = prog.args[0].args[2].args[1]
  target, value = store.args
  assert target.args[0].op == ProgramOp.ADD and target.args[0].args[0] is i
  assert target.args[0].args[1].op == ProgramOp.CONST_INT and target.args[0].args[1].attrs["value"] == -3
  assert value.args[0].args[0] is i


def test_known_invalid_constants_stay_runtime_operations() -> None:
  zero = p.const_float(0.0)
  assert fold_program(ProgramNode(ProgramOp.DIV, (zero, zero))).op == ProgramOp.DIV
  x = sc.sym("x", 2)
  kept = sc.simplify(sc.const([0.0, 1.0]) / sc.const([0.0, 0.0]))
  assert kept.op == sc.ExprOp.DIV
  assert sc.simplify(sc.const(0.0) / sc.const(0.0)).op == sc.ExprOp.DIV
  assert sc.simplify(sc.const(1000.0).exp()).op == sc.ExprOp.EXP
  assert sc.simplify(sc.const(0.0) / x).op == sc.ExprOp.CONST
  fn = _function("invalid_loop", lambda x: (x + sc.const(0.0) / sc.const(0.0)).block(), shape=2)
  assert "div" in _proc_ops(main_proc(lower_function(fn)))
  assert np.isnan(fn(np.ones(2))).all()


def test_integer_constant_evaluation_refuses_values_outside_the_dtype() -> None:
  from scaly.ir.program import ProgramNode, ProgramOp, const_int
  from scaly.ir.types import dtypes
  from scaly.passes.arith import fold_program

  big = ProgramNode(ProgramOp.MUL, (const_int(1 << 40), const_int(1 << 40)), dtype=dtypes.int64)
  assert fold_program(big).op == ProgramOp.MUL
  assert fold_program(ProgramNode(ProgramOp.ADD, (const_int(1 << 40), const_int(1)), dtype=dtypes.int64)).attrs["value"] == (1 << 40) + 1
