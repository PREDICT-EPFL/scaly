"""Edge cases of ``atan2``, ``copysign``, ``maximum``/``minimum``, the comparisons, logic, ``isfinite``, ``where`` and ``cast``.

Values only: what the compiled C returns next to what constant folding returns and what NumPy returns, on
signed zeros, subnormals, extreme magnitudes, infinities and NaN of either sign, in block and scalar lowering
and inside ``vmap``; float32, int64 and bool dtypes; broadcasting; folding of ``where``; and every refusal the
builders raise. Derivatives are in ``tests/ad/test_elementwise_derivative_edge_cases.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

import scaly as sc
from scaly.ir.expr import topo
from scaly.ir.types import Lowering
from scaly.passes.expr import simplify

SPECIAL = np.array([-np.inf, -1e308, -1.5, -5e-324, -0.0, 0.0, 5e-324, 1e-310, 1.5, 1e308, np.inf, np.nan, -np.nan])
GRID_X, GRID_Y = (a.reshape(-1) for a in np.meshgrid(SPECIAL, SPECIAL, indexing="ij"))
N = GRID_X.size

Binary = Callable[[sc.Expr, sc.Expr], sc.Expr]
Reference = Callable[[np.ndarray, np.ndarray], np.ndarray]

# name -> (scaly builder, NumPy reference). ``maximum``/``minimum`` follow C's fmax/fmin, as the derivative
# rule documents: NaN beside a number yields the number.
GRID_OPS: dict[str, tuple[Binary, Reference]] = {
  "atan2": (sc.atan2, np.arctan2),
  "copysign": (sc.copysign, np.copysign),
  "maximum": (sc.maximum, np.fmax),
  "minimum": (sc.minimum, np.fmin),
  "lt": (lambda x, y: x < y, np.less),
  "le": (lambda x, y: x <= y, np.less_equal),
  "gt": (lambda x, y: x > y, np.greater),
  "ge": (lambda x, y: x >= y, np.greater_equal),
  "eq": (sc.equal, np.equal),
  "ne": (sc.not_equal, np.not_equal),
  "and_or_not": (lambda x, y: (x < y) & ~(x > 0.0) | sc.isfinite(y), lambda x, y: (x < y) & ~(x > 0.0) | np.isfinite(y)),
  "select": (lambda x, y: sc.where(x <= y, x, y), lambda x, y: np.where(x <= y, x, y)),
  "select_nonfinite": (lambda x, y: sc.where(sc.isfinite(x), y, x), lambda x, y: np.where(np.isfinite(x), y, x)),
  "cast_to_bool": (lambda x, y: sc.cast(x, "bool") & sc.cast(y, "bool"), lambda x, y: x.astype(bool) & y.astype(bool)),
  "cast_from_bool": (lambda x, y: sc.cast(sc.not_equal(x, y), "float64") - sc.cast(x < y, "float64"), lambda x, y: (x != y) * 1.0 - (x < y) * 1.0),
}
ULP = {"atan2": 1}  # a transcendental: NumPy's and C's libm may round differently
NAN_SIGN = {"copysign", "select_nonfinite"}  # the sign bit of a NaN result is defined


def _fn(name: str, inputs: list[sc.Expr], outputs: list[sc.Expr]) -> sc.Function:
  return sc.Function._from_exprs(name, inputs, outputs, [str(e.name) for e in inputs], [f"out{i}" for i in range(len(outputs))])


def _grid_outputs(variant: str) -> list[np.ndarray]:
  names = sorted(GRID_OPS)
  if variant == "folded":
    outs = [GRID_OPS[n][0](sc.const(GRID_X), sc.const(GRID_Y)) for n in names]
    # Each value below must come from NumPy folding in Python, not from C.
    assert all(simplify(o).op == sc.ExprOp.CONST for o in outs)
    return list(_fn("ee_grid_folded", [sc.sym("z", 1)], outs)(np.zeros(1)))
  if variant == "vmap":
    p = sc.sym("p", 2)
    body = sc.stack([sc.cast(GRID_OPS[n][0](p[0], p[1]), "float64") for n in names])
    z = sc.sym("z", 2 * N)
    mapped = sc.vmap(_fn("ee_grid_vmap_body", [p], [body]), N, [(z, 0, 2)])
    flat = _fn("ee_grid_vmap", [z], [mapped])(np.stack([GRID_X, GRID_Y], axis=1).reshape(-1))
    return list(flat.reshape(N, len(names)).T)
  lowering: Lowering = "scalar" if variant == "scalar" else "block"
  x, y = sc.sym("x", N).with_lowering(lowering), sc.sym("y", N).with_lowering(lowering)
  # A callee keeps the ops inside a procedure that can scalarize; the entry itself does not by default.
  inner = _fn(f"ee_grid_{variant}_inner", [x, y], [GRID_OPS[n][0](x, y) for n in names])
  xo, yo = sc.sym("x", N), sc.sym("y", N)
  return list(_fn(f"ee_grid_{variant}", [xo, yo], list(inner((xo, yo))))((GRID_X, GRID_Y)))


def _assert_values(
  got: np.ndarray, expected: np.ndarray, msg: str, *, ulp: int = 0, nan_sign: bool = False, zero_ties: np.ndarray | None = None
) -> None:
  """Equal values, NaN in the same places and the same sign of zero. ``zero_ties`` masks entries whose sign
  is the platform's choice (C leaves ``fmax(-0.0, +0.0)`` unspecified)."""
  assert got.dtype == expected.dtype, msg
  if expected.dtype == np.bool_:
    np.testing.assert_array_equal(got, expected, err_msg=msg)
    return
  nan = np.isnan(expected)
  np.testing.assert_array_equal(np.isnan(got), nan, err_msg=f"{msg}: NaN positions")
  if ulp:
    np.testing.assert_array_max_ulp(got[~nan], expected[~nan], maxulp=ulp)
  else:
    np.testing.assert_array_equal(got[~nan], expected[~nan], err_msg=msg)
  signed = ~nan if zero_ties is None else ~nan & ~zero_ties
  np.testing.assert_array_equal(np.signbit(got[signed]), np.signbit(expected[signed]), err_msg=f"{msg}: sign of zero")
  if nan_sign:
    np.testing.assert_array_equal(np.signbit(got[nan]), np.signbit(expected[nan]), err_msg=f"{msg}: sign of NaN")


def _zero_ties(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  return (x == 0) & (y == 0)


@pytest.mark.parametrize("variant", ["block", "scalar", "vmap", "folded"])
def test_special_value_grid_matches_numpy_in_every_lowering_and_when_folded(variant: str) -> None:
  for name, got in zip(sorted(GRID_OPS), _grid_outputs(variant), strict=True):
    with np.errstate(invalid="ignore"):
      expected = GRID_OPS[name][1](GRID_X, GRID_Y)
    if variant == "vmap":
      expected = expected.astype(np.float64)
    ties = _zero_ties(GRID_X, GRID_Y) if name in ("maximum", "minimum") else None
    _assert_values(got, expected, f"{variant} {name}", ulp=ULP.get(name, 0), nan_sign=name in NAN_SIGN, zero_ties=ties)


def test_atan2_of_signed_zeros_and_infinities_is_exact() -> None:
  y = np.array([0.0, -0.0, 0.0, -0.0, np.inf, np.inf, -np.inf, -np.inf, 5e-324, -5e-324, 1.0, -1.0])
  x = np.array([0.0, 0.0, -0.0, -0.0, np.inf, -np.inf, np.inf, -np.inf, -1.0, -1.0, -np.inf, np.inf])
  expected = [0.0, -0.0, np.pi, -np.pi, np.pi / 4, 3 * np.pi / 4, -np.pi / 4, -3 * np.pi / 4, np.pi, -np.pi, np.pi, -0.0]
  ys, xs = sc.sym("y", y.size), sc.sym("x", x.size)
  got = _fn("ee_atan2_exact", [ys, xs], [sc.atan2(ys, xs), ys.atan2(xs)])((y, x))
  for g in got:
    np.testing.assert_array_equal(g, expected)
    np.testing.assert_array_equal(np.signbit(g), np.signbit(expected))


def test_folded_maximum_and_minimum_beside_nan_match_the_compiled_fmax() -> None:
  a, b = np.array([np.nan, 1.0, np.nan]), np.array([1.0, np.nan, np.nan])
  p, q = sc.sym("p", 3), sc.sym("q", 3)
  compiled = _fn("ee_fmax_nan_rt", [p, q], [sc.maximum(p, q), sc.minimum(p, q)])((a, b))
  folded = _fn("ee_fmax_nan_fold", [p], [sc.maximum(sc.const(a), sc.const(b)), sc.minimum(sc.const(a), sc.const(b))])(np.zeros(3))
  np.testing.assert_array_equal(compiled[0], np.fmax(a, b))
  np.testing.assert_array_equal(folded[0], compiled[0])
  np.testing.assert_array_equal(folded[1], compiled[1])


def test_a_negative_nan_constant_keeps_its_sign() -> None:
  x = sc.sym("x", 2)
  minus_nan = sc.const(np.array([-np.nan, -np.nan]))
  f = _fn("ee_neg_nan_const", [x], [sc.copysign(x, minus_nan), sc.copysign(sc.const(np.array([1.0, np.nan])), sc.const(-1.0))])
  runtime, folded = f(np.array([2.0, 3.0]))
  np.testing.assert_array_equal(runtime, [-2.0, -3.0])
  assert np.signbit(folded).all()


def test_signed_zero_constants_keep_their_sign_inside_a_scalar_expanded_callee() -> None:
  a_vals, b_vals = np.array([1.0, 1.0, -0.0, 0.0]), np.array([-0.0, 0.0, 1.0, 1.0])
  a, b = sc.sym("a", 4).with_lowering("scalar"), sc.sym("b", 4).with_lowering("scalar")
  inner = _fn("ee_sz_inner", [a, b], [sc.copysign(a, b), sc.atan2(a, b)])
  m = sc.sym("m", 1).with_lowering("scalar")
  # The constant arguments reach the inner body once it expands into ``mid``, where they fold.
  mid = _fn("ee_sz_mid", [m], [*inner((sc.const(a_vals), sc.const(b_vals))), m])
  z = sc.sym("z", 1)
  signs, angles, _ = mid(z)
  got = _fn("ee_sz", [z], [signs, angles])(np.zeros(1))
  for g, e in zip(got, (np.copysign(a_vals, b_vals), np.arctan2(a_vals, b_vals)), strict=True):
    np.testing.assert_array_equal(g, e)
    np.testing.assert_array_equal(np.signbit(g), np.signbit(e))


def test_where_between_signed_zeros_keeps_the_chosen_sign() -> None:
  x = sc.sym("x", 3)
  equal = sc.where(x < 0.0, 0.0, 0.0)  # branches with the same bits fold, to the shape the condition broadcasts to
  assert simplify(equal).op == sc.ExprOp.CONST and simplify(equal).shape == (3,)
  got, zeros = _fn("ee_where_sz", [x], [sc.where(x < 0.0, -0.0, 0.0), equal])(np.array([-1.0, 1.0, -2.0]))
  np.testing.assert_array_equal(np.signbit(got), [True, False, True])
  np.testing.assert_array_equal(zeros, np.zeros(3))


F32 = np.array([-np.inf, -3.4e38, -1.5, -1e-45, -0.0, 0.0, 1e-45, 0.1, 1.5, 3.4e38, np.inf, np.nan], dtype=np.float32)
F32_X, F32_Y = (a.reshape(-1) for a in np.meshgrid(F32, F32, indexing="ij"))


def test_float32_ops_round_like_numpy_float32() -> None:
  n = F32_X.size
  a, b = sc.sym("a", n, dtype="float32"), sc.sym("b", n, dtype="float32")
  outs = [sc.copysign(a, b), sc.maximum(a, b), sc.minimum(a, b), sc.where(a < b, a, b), a <= b, sc.equal(a, b), sc.isfinite(a), sc.atan2(a, b)]
  got = _fn("ee_f32_grid", [a, b], outs)((F32_X, F32_Y))
  with np.errstate(invalid="ignore"):
    expected = [
      np.copysign(F32_X, F32_Y),
      np.fmax(F32_X, F32_Y),
      np.fmin(F32_X, F32_Y),
      np.where(F32_X < F32_Y, F32_X, F32_Y),
      F32_X <= F32_Y,
      F32_X == F32_Y,
      np.isfinite(F32_X),
    ]
  ties = _zero_ties(F32_X, F32_Y)
  for i, (g, e) in enumerate(zip(got[:-1], expected, strict=True)):
    _assert_values(g, e.astype(np.float64) if e.dtype != np.bool_ else e, f"float32 output {i}", zero_ties=ties if i in (1, 2) else None)
  angle, reference = got[-1].astype(np.float32), np.arctan2(F32_X, F32_Y)
  nan = np.isnan(reference)
  np.testing.assert_array_equal(np.isnan(angle), nan)
  np.testing.assert_array_max_ulp(angle[~nan], reference[~nan], maxulp=1)


def test_float32_parameters_round_at_the_abi_and_python_numbers_take_float32() -> None:
  just_below = np.nextafter(np.float32(0.1), np.float32(0.0))
  xs = np.array([0.1, np.float64(np.float32(0.1)), just_below, 1e39, -1e-46, 2.0**24 + 1.0])
  x = sc.sym("x", xs.size, dtype="float32")
  # In float64, float32(0.1) > 0.1: every comparison here would flip if the constant stayed float64.
  outs = [
    sc.cast(x, "float64"),
    x <= 0.1,
    sc.equal(x, 0.1),
    0.1 > x,
    sc.cast(sc.maximum(x, 0.1), "float64"),
    sc.cast(sc.where(x < 0.1, 0.1, x), "float64"),
  ]
  f = _fn("ee_f32_abi", [x], outs)
  # The flat call hands C the doubles as they are, so C's conversion to float does the rounding.
  got = f._flat_numerical_call(xs)
  with np.errstate(over="ignore"):
    x32 = xs.astype(np.float32)
    for g, via_python in zip(got, f(xs), strict=True):
      np.testing.assert_array_equal(g, via_python)
  tenth = np.float32(0.1)
  np.testing.assert_array_equal(got[0], x32.astype(np.float64))
  np.testing.assert_array_equal(got[1], x32 <= tenth)
  np.testing.assert_array_equal(got[2], x32 == tenth)
  np.testing.assert_array_equal(got[3], tenth > x32)
  np.testing.assert_array_equal(got[4], np.fmax(x32, tenth).astype(np.float64))
  np.testing.assert_array_equal(got[5], np.where(x32 < tenth, tenth, x32).astype(np.float64))
  assert got[1][:3].tolist() == [True, True, True] and got[2][:3].tolist() == [True, True, False]


def test_a_float32_atan2_output_is_a_float32_value() -> None:
  a, b = sc.sym("a", 4, dtype="float32"), sc.sym("b", 4, dtype="float32")
  got = _fn("ee_f32_atan2_out", [a, b], [sc.atan2(a, b)])((np.array([0.1, -0.3, 2.0, 7.0]), np.array([0.7, 0.9, -3.0, 1.0])))
  np.testing.assert_array_equal(got, got.astype(np.float32).astype(np.float64))


F32_LIBM: dict[str, Callable[[np.ndarray], np.ndarray]] = {
  "sin": np.sin,
  "cos": np.cos,
  "tan": np.tan,
  "asin": np.arcsin,
  "acos": np.arccos,
  "atan": np.arctan,
  "sinh": np.sinh,
  "cosh": np.cosh,
  "tanh": np.tanh,
  "erf": lambda x: np.asarray(np.frompyfunc(math.erf, 1, 1)(x), dtype=np.float32),
  "exp": np.exp,
  "log": np.log,
  "sqrt": np.sqrt,
  "abs": np.abs,
  "floor": np.floor,
  "ceil": np.ceil,
}


def test_float32_libm_calls_give_float32_values_within_an_ulp_of_numpy() -> None:
  """libm computes in double: each float32 result, and the float arithmetic around it, must still be float32."""
  n = F32_X.size
  a, b = sc.sym("a", n, dtype="float32"), sc.sym("b", n, dtype="float32")
  tenth = sc.const(0.1, dtype="float32")
  outs = [getattr(a, name)() for name in F32_LIBM] + [a**b, a.floor() * tenth + a.abs().sqrt()]
  got = _fn("ee_f32_libm", [a, b], outs)((F32_X, F32_Y))
  with np.errstate(all="ignore"):
    expected = [f(F32_X) for f in F32_LIBM.values()] + [F32_X**F32_Y, np.floor(F32_X) * np.float32(0.1) + np.sqrt(np.abs(F32_X))]
  for name, g, e in zip([*F32_LIBM, "pow", "floor_mul_add_sqrt"], got, expected, strict=True):
    assert e.dtype == np.float32, name
    np.testing.assert_array_equal(g, g.astype(np.float32).astype(np.float64), err_msg=f"{name}: not a float32 value")
    g32, finite = g.astype(np.float32), np.isfinite(e)
    np.testing.assert_array_equal(g32[~finite], e[~finite], err_msg=name)
    np.testing.assert_array_max_ulp(g32[finite], e[finite], maxulp=0 if name == "floor_mul_add_sqrt" else 1)


PY_NUMBER = {
  "maximum": lambda x: sc.maximum(x, 1.0),
  "where": lambda x: sc.where(x < 0.0, x, 2.0),
  "less": lambda x: sc.less(1.0, x),
  "atan2": lambda x: sc.atan2(x, 1.0),
  "atan2_left": lambda x: sc.atan2(1.0, x),
  "copysign": lambda x: sc.copysign(x, -1.0),
  "copysign_left": lambda x: sc.copysign(2.0, x),
  "mul": lambda x: x * 2.0,
  "rsub": lambda x: 1 - x,
  "div": lambda x: x / 3.0,
  "rdiv": lambda x: 1.0 / x,
  "pow": lambda x: x**2,
  "rpow": lambda x: 2.0**x,
  "numpy_scalar": lambda x: x * np.float64(0.5),
}


@pytest.mark.parametrize("name", list(PY_NUMBER))
def test_a_python_number_beside_float32_takes_its_dtype(name: str) -> None:
  """``docs/how_it_works/ir.md``: a Python number beside an ``Expr`` takes that ``Expr``'s dtype."""
  x = sc.sym("x", 2, dtype="float32")
  out = PY_NUMBER[name](x)
  assert all(a.type.dtype == sc.dtypes.float32 for a in out.args if a.type.dtype != sc.dtypes.bool_)


def test_a_python_number_beside_an_integer_expression_never_truncates() -> None:
  """An integer joins an ``int64`` expression; a float does not silently truncate to one, and ``/`` and
  ``**`` are not integer operations, so each stays float64 and is refused as mixed-dtype."""
  k = sc.sym("k", 2, dtype="int64")
  assert (k * 2).type.dtype == (3 + k).type.dtype == (k - np.int64(1)).type.dtype == sc.dtypes.int64
  for build in (lambda: k * 2.5, lambda: k + 0.5, lambda: k / 2, lambda: 2 / k, lambda: k**2, lambda: 2**k):
    with pytest.raises(TypeError, match="mixed-dtype"):
      build()
  with pytest.raises(TypeError, match="mixed-dtype"):
    _ = sc.sym("x", 2, dtype="float32") * True  # a Python bool is not a number here


INT_CASES = np.array([-(2.0**62), -2.5, -1.5, -0.5, -0.0, 0.5, 1.9, 2.0**53, 2.0**62])


@pytest.mark.parametrize("variant", ["runtime", "folded"])
def test_int_casts_truncate_toward_zero_and_int64_arithmetic_stays_exact(variant: str) -> None:
  z = sc.sym("z", INT_CASES.size)
  x = z if variant == "runtime" else sc.const(INT_CASES)
  i = sc.cast(x, "int64")
  clamped = sc.cast(sc.maximum(sc.minimum(x, 1e9), -1e9), "int32")
  outs = [
    sc.cast(i, "float64"),
    sc.cast(i + 1, "float64"),  # 2**53 + 1 rounds to even: 2**53
    sc.cast(i + 3, "float64"),  # 2**53 + 3 rounds to even: 2**53 + 4
    (i + 1) > i,  # exact in int64 even where the float64 images are equal
    sc.equal(sc.cast(i + 1, "float64"), sc.cast(i, "float64")),
    sc.cast(i, "bool"),
    sc.cast(sc.cast(x > 0.0, "int64") - sc.cast(x < 0.0, "int64"), "float64"),
    sc.cast(clamped, "float64"),
    sc.cast(sc.cast(clamped, "int64"), "float64"),
  ]
  if variant == "folded":
    assert all(simplify(o).op == sc.ExprOp.CONST for o in outs)
  got = _fn(f"ee_int_casts_{variant}", [z], outs)(INT_CASES)
  i_ref = INT_CASES.astype(np.int64)
  c_ref = np.clip(INT_CASES, -1e9, 1e9).astype(np.int32)
  expected = [
    i_ref.astype(np.float64),
    (i_ref + 1).astype(np.float64),
    (i_ref + 3).astype(np.float64),
    (i_ref + 1) > i_ref,
    (i_ref + 1).astype(np.float64) == i_ref.astype(np.float64),
    i_ref != 0,
    np.sign(INT_CASES),
    c_ref.astype(np.float64),
    c_ref.astype(np.float64),
  ]
  for k, (g, e) in enumerate(zip(got, expected, strict=True)):
    np.testing.assert_array_equal(g, e, err_msg=str(k))
  assert got[0].tolist()[:5] == [-(2.0**62), -2.0, -1.0, 0.0, 0.0] and not np.signbit(got[0][3:5]).any()
  assert got[4].tolist() == [True, False, False, False, False, False, False, True, True]


def test_an_int64_constant_above_two_to_the_53_is_exact() -> None:
  x = sc.sym("x", 3)
  k = 2**53 + 1
  got = _fn("ee_i64_big_const", [x], [sc.cast(sc.cast(x, "int64") + sc.const(k, dtype="int64"), "float64")])(np.array([-1.0, 0.0, 2.0]))
  np.testing.assert_array_equal(got, (np.array([-1, 0, 2], dtype=np.int64) + k).astype(np.float64))


def test_casts_to_and_from_bool_and_casts_of_casts() -> None:
  values = np.array([np.nan, -np.nan, -0.0, 0.0, 5e-324, -np.inf, 1e-310, 2.0])
  x = sc.sym("x", values.size)
  outs = [
    sc.cast(x, "bool"),
    x.cast("bool"),
    sc.cast(sc.cast(x, "bool"), "float32").cast("float64"),
    sc.cast(sc.cast(x, "bool"), "int64").cast("float64"),
    sc.cast(sc.cast(x, "float32"), "float64"),  # a float64 subnormal rounds to a float32 zero of its sign
    sc.cast(sc.cast(sc.cast(x, "float32"), "bool"), "float64"),
  ]
  got = _fn("ee_bool_casts", [x], outs)(values)
  truthy, as_f32 = values.astype(bool), values.astype(np.float32)
  for k, (g, e) in enumerate(
    zip(got, [truthy, truthy, truthy * 1.0, truthy * 1.0, as_f32.astype(np.float64), as_f32.astype(bool) * 1.0], strict=True)
  ):
    np.testing.assert_array_equal(g, e, err_msg=str(k))
  np.testing.assert_array_equal(np.signbit(got[4][2:]), np.signbit(as_f32[2:]))
  assert got[0].tolist() == [True, True, False, False, True, True, True, True]  # NaN of either sign is true, -0.0 is false
  assert got[5][6] == 0.0  # 1e-310 is nonzero in float64 but zero in float32
  # Structure: a cast to the same dtype is the node itself, a cast to bool is a comparison, and a
  # round trip through float32 keeps both casts, since it rounds.
  assert sc.cast(x, "float64") is x and x.cast(sc.dtypes.float64) is x
  assert sc.cast(x, "bool").op == sc.ExprOp.NE and sc.cast(x, "bool") is x.cast("bool")
  trip = sc.cast(sc.cast(x, "float32"), "float64")
  assert simplify(trip).op == sc.ExprOp.CAST and simplify(trip).args[0].op == sc.ExprOp.CAST


TRUTHY = np.array([0.0, -0.0, 1.0, 2.0, -1.0, np.nan, np.inf, 5e-324])


def test_bool_parameters_read_nonzero_as_true_and_logic_follows_its_truth_tables() -> None:
  av, bv = (m.reshape(-1) for m in np.meshgrid(TRUTHY, TRUTHY, indexing="ij"))
  a, b = sc.sym("a", av.size, dtype="bool"), sc.sym("b", bv.size, dtype="bool")
  s = sc.sym("s", (), dtype="bool")
  outs = [
    a & b,
    a | b,
    ~a,
    sc.not_equal(a, b),
    sc.equal(~(a & b), ~a | ~b),  # De Morgan, evaluated in C
    a < b,
    sc.logical_and(True, b) | sc.logical_or(a, False),
    ~sc.logical_not(True) & a,
    s & a,
    sc.where(a, b, s),
  ]
  got = _fn("ee_bool_logic", [a, b, s], outs)._flat_numerical_call(av, bv, np.array(-0.0))
  ta, tb = av != 0, bv != 0
  expected = [ta & tb, ta | tb, ~ta, ta != tb, np.ones_like(ta), ~ta & tb, tb | ta, ta, np.zeros_like(ta), np.where(ta, tb, False)]
  for k, (g, e) in enumerate(zip(got, expected, strict=True)):
    assert g.dtype == np.bool_, k
    np.testing.assert_array_equal(g, e, err_msg=str(k))
  assert got[2][::8].tolist() == [True, True, False, False, False, False, False, False]  # only 0.0 and -0.0 read as false
  (nan_scalar,) = _fn("ee_bool_scalar", [a, s], [s & a])._flat_numerical_call(av, np.array(np.nan))
  np.testing.assert_array_equal(nan_scalar, ta)


def test_broadcasting_shapes_and_values() -> None:
  c = sc.sym("c", (2, 1), dtype="bool")
  u, x, s, w = sc.sym("u", (2, 1)), sc.sym("x", 3), sc.sym("s", ()), sc.sym("w", 1)
  cases: list[tuple[sc.Expr, Callable[..., np.ndarray], tuple[int, ...]]] = [
    (sc.where(c, x, s), lambda c, u, x, s, w: np.where(c, x, s), (2, 3)),
    (sc.where(s > 0.0, w, x), lambda c, u, x, s, w: np.where(s > 0.0, w, x), (3,)),
    (sc.where(c, 1.0, u), lambda c, u, x, s, w: np.where(c, 1.0, u), (2, 1)),
    (sc.atan2(u, x), lambda c, u, x, s, w: np.arctan2(u, x), (2, 3)),
    (sc.copysign(x, s), lambda c, u, x, s, w: np.copysign(x, s), (3,)),
    (sc.copysign(s, u), lambda c, u, x, s, w: np.copysign(s, u), (2, 1)),
    (sc.maximum(w, s), lambda c, u, x, s, w: np.fmax(w, s), (1,)),
    (sc.minimum(u, x), lambda c, u, x, s, w: np.fmin(u, x), (2, 3)),
    (u < x, lambda c, u, x, s, w: u < x, (2, 3)),
    (sc.equal(w, s), lambda c, u, x, s, w: np.equal(w, s), (1,)),
    (c & (x > 0.0), lambda c, u, x, s, w: c & (x > 0.0), (2, 3)),
    (sc.isfinite(u) | ~sc.isfinite(x), lambda c, u, x, s, w: np.isfinite(u) | ~np.isfinite(x), (2, 3)),
  ]
  for expr, _, shape in cases:
    assert expr.shape == shape
  f = _fn("ee_broadcast", [c, u, x, s, w], [e for e, _, _ in cases])
  values = (np.array([[True], [False]]), np.array([[-0.0], [np.nan]]), np.array([-1.0, 0.5, np.inf]), np.array(-2.0), np.array([-0.0]))
  got = f(values)
  for k, (g, (_, reference, shape)) in enumerate(zip(got, cases, strict=True)):
    with np.errstate(invalid="ignore"):
      e = reference(*values)
    assert g.shape == shape, k
    _assert_values(g, e, f"broadcast case {k}")


def _ops(expr: sc.Expr) -> set[sc.ExprOp]:
  return {sc.ExprOp(n.op) for n in topo([simplify(expr)])}


def test_where_folds_constant_conditions_and_equal_branches_and_keeps_the_rest() -> None:
  x = sc.sym("x", 3)
  k = sc.less(sc.const(1.0), sc.const(2.0))
  mask = sc.const(np.array([True, False, True]), dtype="bool")
  cases = [
    (sc.where(k, x, -x), True),
    (sc.where(~k, x, -x), True),
    (sc.where(mask, x, -x), False),
    (sc.where(x > 0.0, x.sin(), x.sin()), True),
    # A uniform condition whose chosen branch is narrower than the result must still broadcast.
    (sc.where(k, 1.0, x), False),
    (sc.where(x > 0.0, sc.where(x > 1.0, 2.0, x), sc.where(x < -1.0, -2.0, -x)), False),
    (sc.cast(sc.where(x > 0.0, x > 1.0, ~(x < -1.0)), "float64"), False),
    (sc.cast(sc.where(x > 0.0, sc.cast(x, "int64"), sc.cast(-x, "int64") * 2), "float64"), False),
    (sc.cast(~~(x > 0.0), "float64"), True),
  ]
  for k_, (expr, folds) in enumerate(cases):
    assert (sc.ExprOp.SELECT not in _ops(expr) and sc.ExprOp.NOT not in _ops(expr)) == folds, k_
  assert sc.where(k, 1.0, x).shape == (3,)
  xv = np.array([-2.5, 0.5, 1.5])
  got = _fn("ee_where_fold", [x], [e for e, _ in cases])(xv)
  expected = [
    xv,
    -xv,
    np.where([True, False, True], xv, -xv),
    np.sin(xv),
    np.ones(3),
    np.select([xv > 1.0, xv > 0.0, xv < -1.0], [2.0, xv, -2.0], -xv),
    np.where(xv > 0.0, xv > 1.0, ~(xv < -1.0)) * 1.0,
    np.where(xv > 0.0, xv.astype(np.int64), (-xv).astype(np.int64) * 2) * 1.0,
    (xv > 0.0) * 1.0,
  ]
  for k_, (g, e) in enumerate(zip(got, expected, strict=True)):
    np.testing.assert_array_equal(g, e, err_msg=str(k_))


def test_nan_and_inf_in_the_unchosen_branch_do_not_reach_the_value() -> None:
  x = sc.sym("x", 4)
  y = sc.where(x > 0.0, x.log(), 1.0 / x) + sc.where(sc.isfinite(1.0 / x), 0.0, x.sqrt())
  xv = np.array([-1.0, 0.0, -0.0, 2.0])
  got = _fn("ee_where_unchosen", [x], [y])(xv)
  with np.errstate(divide="ignore", invalid="ignore"):
    expected = np.where(xv > 0.0, np.log(xv), 1.0 / xv) + np.where(np.isfinite(1.0 / xv), 0.0, np.sqrt(xv))
  np.testing.assert_array_equal(got, expected)
  assert got.tolist()[:3] == [-1.0, np.inf, -np.inf]


REFUSALS: dict[str, tuple[Callable[[], object], type[Exception], str]] = {
  "compare_mixed": (lambda: sc.sym("x", 2) < sc.sym("f", 2, dtype="float32"), TypeError, "mixed-dtype"),
  "compare_int_float": (lambda: sc.less(sc.sym("i", 2, dtype="int64"), sc.sym("x", 2)), TypeError, "mixed-dtype"),
  "atan2_mixed": (lambda: sc.atan2(sc.sym("x", 2), sc.sym("f", 2, dtype="float32")), TypeError, "mixed-dtype"),
  "copysign_mixed": (lambda: sc.copysign(sc.sym("f", 2, dtype="float32"), sc.sym("x", 2)), TypeError, "mixed-dtype"),
  "maximum_mixed": (lambda: sc.maximum(sc.sym("x", 2), sc.sym("f", 2, dtype="float32")), TypeError, "mixed-dtype"),
  "where_branches_mixed": (lambda: sc.where(sc.sym("c", 2, dtype="bool"), sc.sym("x", 2), sc.sym("f", 2, dtype="float32")), TypeError, "mixed-dtype"),
  "and_float": (lambda: sc.logical_and(sc.sym("x", 2), sc.sym("c", 2, dtype="bool")), TypeError, "bool operands"),
  "or_int": (lambda: sc.sym("c", 2, dtype="bool") | sc.sym("i", 2, dtype="int64"), TypeError, "bool operands"),
  "not_float": (lambda: ~sc.sym("x", 2), TypeError, "bool operands"),
  "where_float_cond": (lambda: sc.where(sc.sym("x", 2), 1.0, 0.0), TypeError, "bool operands"),
  "where_python_true": (lambda: sc.where(True, sc.sym("x", 2), 0.0), TypeError, "Python bool"),
  "where_identity_eq": (lambda: sc.where(sc.sym("x", 2) == 0, 1.0, 0.0), TypeError, "== is identity"),
  "isfinite_int": (lambda: sc.isfinite(sc.sym("i", 2, dtype="int64")), TypeError, "floating operand"),
  "isfinite_bool": (lambda: sc.sym("c", 2, dtype="bool").isfinite(), TypeError, "floating operand"),
  "cast_unknown": (lambda: sc.cast(sc.sym("x", 2), "complex64"), ValueError, "unknown dtype"),
  "cast_not_a_dtype": (lambda: sc.cast(sc.sym("x", 2), 3), TypeError, "cannot interpret"),  # ty: ignore[invalid-argument-type]
  "where_shapes": (lambda: sc.where(sc.sym("c", 4, dtype="bool"), sc.sym("x", 3), 0.0), ValueError, r"cannot broadcast shapes \(4,\) and \(3,\)"),
  "atan2_shapes": (lambda: sc.atan2(sc.sym("x", (2, 3)), sc.sym("y", 2)), ValueError, "cannot broadcast"),
  "compare_shapes": (lambda: sc.sym("x", 3) <= sc.sym("y", 4), ValueError, "cannot broadcast"),
  "and_shapes": (lambda: sc.sym("c", 3, dtype="bool") & sc.sym("d", (3, 2), dtype="bool"), ValueError, "cannot broadcast"),
  "truth_value": (lambda: bool(sc.maximum(sc.sym("x", 1), 0.0) > 0.0), TypeError, "no Python truth value"),
}


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_builders_refuse_invalid_operands(name: str) -> None:
  build, error, match = REFUSALS[name]
  with pytest.raises(error, match=match):
    build()
