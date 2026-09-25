"""Comparisons, logic, ``select``, ``isfinite``, ``copysign`` and casts compile to C that matches NumPy.

Every case runs through the JIT and the universal ``double`` ABI, both as loops and scalarized, on a
grid of the values where C and NumPy could disagree: signed zeros, infinities and NaN.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_api_header, render_c_source
from scaly.ir.program import ProgramOp
from scaly.ir.types import Lowering
from scaly.passes.lowering import lower_function
from scaly.passes.program._common import _walk

SPECIAL = np.array([-np.inf, -1.5, -0.0, 0.0, 1.5, np.inf, np.nan])
GRID_X, GRID_Y = (a.reshape(-1) for a in np.meshgrid(SPECIAL, SPECIAL, indexing="ij"))
N = GRID_X.size

# name -> (scaly builder, NumPy reference)
BINARY: dict[str, tuple[Callable[[sc.Expr, sc.Expr], sc.Expr], Callable[[np.ndarray, np.ndarray], np.ndarray]]] = {
  "lt": (lambda x, y: x < y, np.less),
  "le": (lambda x, y: x <= y, np.less_equal),
  "gt": (lambda x, y: x > y, np.greater),
  "ge": (lambda x, y: x >= y, np.greater_equal),
  "eq": (sc.equal, np.equal),
  "ne": (sc.not_equal, np.not_equal),
  "and": (lambda x, y: (x < 1.0) & (y > -1.0), lambda x, y: (x < 1.0) & (y > -1.0)),
  "or": (lambda x, y: (x < 1.0) | (y > -1.0), lambda x, y: (x < 1.0) | (y > -1.0)),
  "not": (lambda x, y: ~(x <= y), lambda x, y: ~(x <= y)),
  "isfinite": (lambda x, y: sc.isfinite(x) & sc.isfinite(y), lambda x, y: np.isfinite(x) & np.isfinite(y)),
  "select": (lambda x, y: sc.where(x < y, x, y), lambda x, y: np.where(x < y, x, y)),
  "select_nan_branch": (lambda x, y: sc.where(sc.isfinite(x), x, 0.0), lambda x, y: np.where(np.isfinite(x), x, 0.0)),
  "copysign": (sc.copysign, np.copysign),
  "cast_bool_float": (lambda x, y: sc.cast(x < y, "float64") * 2.0, lambda x, y: (x < y).astype(np.float64) * 2.0),
  "cast_float_bool": (lambda x, y: sc.cast(x, "bool"), lambda x, y: x != 0),
}


def _compiled(name: str, build: Callable[[sc.Expr, sc.Expr], sc.Expr], lowering: Lowering) -> sc.Function:
  x = sc.sym("x", N).with_lowering(lowering)
  y = sc.sym("y", N).with_lowering(lowering)
  # A callee keeps the op inside a procedure that can scalarize; the entry itself does not by default.
  inner = sc.Function._from_exprs(f"ctl_{name}_{lowering}_inner", [x, y], [build(x, y)], ["x", "y"], ["out"])
  xo, yo = sc.sym("x", N), sc.sym("y", N)
  return sc.Function._from_exprs(f"ctl_{name}_{lowering}", [xo, yo], [inner((xo, yo))], ["x", "y"], ["out"])


@pytest.mark.parametrize("lowering", ["block", "scalar"])
@pytest.mark.parametrize("name", sorted(BINARY))
def test_compiled_control_ops_match_numpy_on_special_values(name: str, lowering: Lowering) -> None:
  build, reference = BINARY[name]
  fun = _compiled(name, build, lowering)
  with np.errstate(invalid="ignore"):
    expected = reference(GRID_X, GRID_Y)
  got = fun((GRID_X, GRID_Y))
  assert got.dtype == expected.dtype
  np.testing.assert_array_equal(got, expected)
  if name == "copysign":
    np.testing.assert_array_equal(np.signbit(got), np.signbit(expected))


def test_scalar_lowering_expands_control_ops_and_loop_lowering_keeps_loops() -> None:
  def ops(lowering: Lowering) -> set[ProgramOp]:
    prog = lower_function(_compiled("probe", BINARY["select"][0], lowering))
    inner = prog.args[0]
    return {n.op for stmt in inner.args for n in _walk(stmt)}

  assert ProgramOp.FOR not in ops("scalar") and ProgramOp.SELECT in ops("scalar")
  assert {ProgramOp.FOR, ProgramOp.SELECT, ProgramOp.LT} <= ops("block")


def test_bool_inputs_and_outputs_cross_the_double_abi() -> None:
  flag = sc.sym("flag", 4, dtype="bool")
  x = sc.sym("x", 4)
  helper_in = sc.sym("h", 4, dtype="bool")
  helper = sc.Function._from_exprs("ctl_bool_helper", [helper_in], [~helper_in], ["h"], ["negated"])
  fun = sc.Function._from_exprs(
    "ctl_bool_abi", [flag, x], [flag & (x > 0.0), sc.where(flag, x, -x), helper(flag), flag], ["flag", "x"], ["both", "chosen", "negated", "echo"]
  )
  flags = np.array([True, False, True, False])
  xv = np.array([1.0, 2.0, -3.0, -4.0])
  both, chosen, negated, echo = fun((flags, xv))
  assert both.dtype == negated.dtype == echo.dtype == np.bool_ and chosen.dtype == np.float64
  np.testing.assert_array_equal(both, flags & (xv > 0))
  np.testing.assert_array_equal(chosen, np.where(flags, xv, -xv))
  np.testing.assert_array_equal(negated, ~flags)
  np.testing.assert_array_equal(echo, flags)
  # Nonzero reads as true at the boundary, as a C caller passing 2.0 or -1.0 would expect.
  assert fun._flat_numerical_call(np.array([2.0, 0.0, -1.0, 0.0]), xv)[3].tolist() == [True, False, True, False]
  source, header = render_c_source(fun), render_c_api_header(fun)
  assert "int ctl_bool_abi(const double** arg, double** res" in source
  assert "uint8_t" in source and "uint8_t" not in header


def test_float_only_functions_render_without_bool_conversions() -> None:
  x = sc.sym("x", 3)
  fun = sc.Function._from_exprs("ctl_plain", [x], [x * 2.0], ["x"], ["y"])
  source = render_c_source(fun)
  assert "uint8_t" not in source.split("may_alias));")[1]
  assert "(double)" not in source


def test_expr_has_no_python_truth_value_and_equality_stays_structural() -> None:
  x = sc.sym("x", 3)
  with pytest.raises(TypeError, match="no Python truth value"):
    bool(x < 1.0)
  with pytest.raises(TypeError, match="no Python truth value"):
    if x:
      pass
  assert (x == sc.sym("x", 3)) is True  # hash-consed, so structural equality is identity
  assert (x == sc.sym("y", 3)) is False
  assert sc.equal(x, 1.0).op == sc.ExprOp.EQ


def test_predicate_types_and_python_scalars() -> None:
  x = sc.sym("x", 3)
  y = sc.sym("y", 3, dtype="float32")
  assert (x < 1).type.dtype == sc.dtypes.bool_ and not (x < 1).type.diff
  assert (y < 1).args[1].type.dtype == sc.dtypes.float32  # the Python number takes the Expr's dtype
  assert (1.0 < x).op == sc.ExprOp.LT and (1.0 < x).args[1] is x
  assert (x > 2.0).args[0].op == sc.ExprOp.CONST  # built as 2.0 < x
  assert sc.where(x < 0, x, 0.0).type.diff
  assert sc.cast(x, "float64") is x
  assert sc.cast(x, "bool").op == sc.ExprOp.NE
  assert sc.cast(x < 0, "float64").type.dtype == sc.dtypes.float64
  assert not sc.cast(x < 0, "float64").type.diff
  assert sc.cast(x, "float32").type.diff
  with pytest.raises(TypeError, match="mixed-dtype"):
    _ = x < y
  with pytest.raises(TypeError, match="bool operands"):
    _ = x & (x < 1)
  with pytest.raises(TypeError, match="floating operand"):
    sc.isfinite(x < 1)
  with pytest.raises(TypeError, match="bool operands"):
    sc.where(x, x, x)
  assert sc.where(sc.sym("c", (2, 1), dtype="bool"), sc.sym("a", 3), 0.0).shape == (2, 3)


REDUCTION_CASES = {
  "mixed": np.array([1.0, 5.0, -3.0, 5.0, 2.0, 0.0]),
  "nan": np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0]),
  "nan_first": np.array([np.nan, 1.0, 3.0, 4.0, 5.0, 6.0]),
  "nan_last": np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan]),
  "all_negative": -np.arange(1.0, 7.0),
  "infinities": np.array([-np.inf, 0.0, np.inf, 1.0, -1.0, 2.0]),
  "all_equal": np.full(6, 2.5),
  "signed_zeros": np.array([-0.0, 0.0, -0.0, 0.0, -0.0, 0.0]),
}


@pytest.mark.parametrize("lowering", ["block", "scalar"])
@pytest.mark.parametrize("case", sorted(REDUCTION_CASES))
def test_reductions_match_numpy_including_nan(case: str, lowering: Lowering) -> None:
  x = sc.sym("x", 6).with_lowering(lowering)
  inner = sc.Function._from_exprs(f"red_{case}_{lowering}_in", [x], [sc.stack([x.max(), x.min(), sc.norm_inf(x), sc.norm_1(x)])], ["x"], ["y"])
  xo = sc.sym("x", 6)
  fun = sc.Function._from_exprs(f"red_{case}_{lowering}", [xo], [inner(xo)], ["x"], ["y"])
  v = REDUCTION_CASES[case]
  np.testing.assert_array_equal(fun(v), [np.max(v), np.min(v), np.max(np.abs(v)), np.sum(np.abs(v))])


def test_reductions_of_one_element_and_large_inputs() -> None:
  x1, xl = sc.sym("x", 1), sc.sym("x", 5000)
  one = sc.Function._from_exprs("red_one", [x1], [x1.max(), x1.min()], ["x"], ["mx", "mn"])
  assert [float(v) for v in one(np.array([-4.0]))] == [-4.0, -4.0]
  big = sc.Function._from_exprs("red_big", [xl], [xl.max(), xl.min()], ["x"], ["mx", "mn"])
  v = np.random.default_rng(1).normal(size=5000)
  assert [float(r) for r in big(v)] == [v.max(), v.min()]
  body = render_c_source(big).split("RESULT;")[-1]
  assert body.count("for (") == 2  # one reduction loop each, whatever the size
  with pytest.raises(ValueError, match="empty"):
    sc.sym("e", 0).max()
