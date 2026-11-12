"""Predicates, ``where``, ``cast`` and integer ``//`` and ``%`` against NumPy.

Values only: what the compiled C returns next to what constant folding returns and what NumPy returns,
on signed zeros, subnormals, extreme magnitudes, infinities and NaN, in block and scalar lowering and
inside ``vmap``; int64, int32 and bool values between float64 leaves; broadcasting; and every refusal
the builders raise. Derivatives are in ``tests/ad/test_select_derivatives.py``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_source
from scaly.passes.expr import simplify
from tests.compilers import c_compiler

SPECIAL = np.array([-np.inf, -1e308, -1.5, -5e-324, -0.0, 0.0, 5e-324, 1e-310, 1.5, 1e308, np.inf, np.nan])
GRID_X, GRID_Y = (a.reshape(-1) for a in np.meshgrid(SPECIAL, SPECIAL, indexing="ij"))
N = GRID_X.size

Binary = Callable[[sc.Expr, sc.Expr], sc.Expr]
Reference = Callable[[np.ndarray, np.ndarray], np.ndarray]

GRID_OPS: dict[str, tuple[Binary, Reference]] = {
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


def _float(e: sc.Expr) -> sc.Expr:
  return e if e.type.dtype == sc.dtypes.float64 else sc.cast(e, "float64")


def _grid_outputs(variant: str) -> np.ndarray:
  names = sorted(GRID_OPS)
  if variant == "folded":
    outs = [_float(GRID_OPS[n][0](sc.const(GRID_X), sc.const(GRID_Y))) for n in names]
    # Each value below comes from NumPy folding in Python, not from C.
    assert all(simplify(o).op == sc.ExprOp.CONST for o in outs)

    @sc.function(sc.arg("z", 1), outputs=sc.arg("y"), name="sc_grid_folded")
    def folded(z: sc.Expr) -> sc.Expr:
      return sc.stack(outs)

    return np.asarray(folded(np.zeros(1)))
  if variant == "vmap":

    @sc.function(sc.arg("p", 2), outputs=sc.arg("y"), name="sc_grid_vmap_body")
    def body(p: sc.Expr) -> sc.Expr:
      return sc.stack([_float(GRID_OPS[n][0](p[0], p[1])) for n in names])

    @sc.function(sc.arg("z", (N, 2)), outputs=sc.arg("y"), name="sc_grid_vmap")
    def mapped(z: sc.Expr) -> sc.Expr:
      return sc.vmap(body, N)(z)

    return np.asarray(mapped(np.stack([GRID_X, GRID_Y], axis=1))).reshape(N, len(names)).T

  @sc.function(sc.arg("x", N), sc.arg("y", N), outputs=sc.arg("out"), name=f"sc_grid_{variant}")
  def grid(x: sc.Expr, y: sc.Expr) -> sc.Expr:
    return getattr(sc.stack([_float(GRID_OPS[n][0](x, y)) for n in names]), variant)()

  return np.asarray(grid(GRID_X, GRID_Y))


def _assert_values(got: np.ndarray, expected: np.ndarray, msg: str) -> None:
  """Equal values, NaN in the same places and the same sign of zero."""
  expected = expected.astype(np.float64)
  nan = np.isnan(expected)
  np.testing.assert_array_equal(np.isnan(got), nan, err_msg=f"{msg}: NaN positions")
  np.testing.assert_array_equal(got[~nan], expected[~nan], err_msg=msg)
  np.testing.assert_array_equal(np.signbit(got[~nan]), np.signbit(expected[~nan]), err_msg=f"{msg}: sign of zero")


@pytest.mark.parametrize("variant", ["block", "scalar", "vmap", "folded"])
def test_special_value_grid_matches_numpy_in_every_lowering_and_when_folded(variant: str) -> None:
  for name, got in zip(sorted(GRID_OPS), _grid_outputs(variant), strict=True):
    with np.errstate(invalid="ignore"):
      expected = GRID_OPS[name][1](GRID_X, GRID_Y)
    _assert_values(np.asarray(got), expected, f"{variant} {name}")


INTS = np.array([-(2.0**63), -(2.0**40), -9.0, -7.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 7.0, 2.0**40, 2.0**63 - 1024])
INT_X, INT_Y = (a.reshape(-1) for a in np.meshgrid(INTS, INTS, indexing="ij"))


@pytest.mark.parametrize("dtype", ["int64", "int32"])
@pytest.mark.parametrize("variant", ["block", "scalar", "folded"])
def test_integer_floordiv_and_mod_follow_numpy_floor_semantics(dtype: str, variant: str) -> None:
  """NumPy rounds the quotient down, gives the remainder the divisor's sign, and returns 0 for a zero
  divisor; ``INT64_MIN // -1`` wraps. C truncates toward zero and traps on both."""
  x = INT_X if dtype == "int64" else np.clip(INT_X, -(2.0**31), 2.0**31 - 1)
  y = INT_Y if dtype == "int64" else np.clip(INT_Y, -(2.0**31), 2.0**31 - 1)
  xi, yi = x.astype(dtype), y.astype(dtype)
  with np.errstate(divide="ignore", over="ignore"):
    expected = [np.floor_divide(xi, yi), np.remainder(xi, yi), xi // 3, xi % -3, 5 // yi]

  def build(a: sc.Expr, b: sc.Expr) -> tuple[sc.Expr, ...]:
    i, j = sc.cast(a, dtype), sc.cast(b, dtype)
    return tuple(sc.cast(e, "float64") for e in (i // j, i % j, i // 3, i % -3, 5 // j))

  if variant == "folded":
    outs = build(sc.const(x), sc.const(y))

    @sc.function(sc.arg("z", 1), outputs=sc.group(*(sc.arg(f"o{k}") for k in range(5))), name=f"sc_int_folded_{dtype}")
    def folded(z: sc.Expr) -> tuple[sc.Expr, ...]:
      return outs

    got = folded(np.zeros(1))
  else:

    @sc.function(sc.arg("x", x.size), sc.arg("y", y.size), outputs=sc.group(*(sc.arg(f"o{k}") for k in range(5))), name=f"sc_int_{variant}_{dtype}")
    def runtime(a: sc.Expr, b: sc.Expr) -> tuple[sc.Expr, ...]:
      return tuple(getattr(e, variant)() for e in build(a, b))

    got = runtime(x, y)
  for k, (g, e) in enumerate(zip(got, expected, strict=True)):
    np.testing.assert_array_equal(g, e.astype(np.float64), err_msg=f"output {k}")


CAST_CASES = np.array([-(2.0**62), -2.5, -1.5, -0.5, -0.0, 0.5, 1.9, 2.0**53, 2.0**62])


def test_uint8_casts_are_rejected_before_lowering() -> None:
  for x in (sc.const([0, 255]).block(), sc.sym("bytes", 2).block()):
    with pytest.raises(ValueError, match="unsupported expression dtype uint8"):
      sc.cast(x, "uint8")
    with pytest.raises(ValueError, match="unsupported expression dtype uint8"):
      x.cast(sc.DType("u1", 8, "uint8_t", is_integer=True, is_unsigned=True))


@pytest.mark.parametrize("variant", ["block", "scalar"])
def test_int_casts_truncate_toward_zero_and_int64_arithmetic_stays_exact(variant: str) -> None:
  @sc.function(sc.arg("z", CAST_CASES.size), outputs=sc.group(*(sc.arg(f"o{k}") for k in range(8))), name=f"sc_int_casts_{variant}")
  def casts(x: sc.Expr) -> tuple[sc.Expr, ...]:
    i = sc.cast(x, "int64")
    clamped = sc.cast(sc.maximum(sc.minimum(x, 1e9), -1e9), "int32")
    outs = [
      sc.cast(i, "float64"),
      sc.cast(i + 1, "float64"),  # 2**53 + 1 rounds to even: 2**53
      sc.cast(i + 3, "float64"),  # 2**53 + 3 rounds to even: 2**53 + 4
      sc.cast((i + 1) > i, "float64"),  # exact in int64 even where the float64 images are equal
      sc.cast(sc.equal(sc.cast(i + 1, "float64"), sc.cast(i, "float64")), "float64"),
      sc.cast(sc.cast(i, "bool"), "float64"),
      sc.cast(sc.cast(x > 0.0, "int64") - sc.cast(x < 0.0, "int64"), "float64"),
      sc.cast(sc.cast(clamped, "int64"), "float64"),
    ]
    return tuple(getattr(o, variant)() for o in outs)

  got = [np.asarray(g) for g in casts(CAST_CASES)]
  i_ref = CAST_CASES.astype(np.int64)
  expected = [
    i_ref.astype(np.float64),
    (i_ref + 1).astype(np.float64),
    (i_ref + 3).astype(np.float64),
    np.ones(CAST_CASES.size),
    ((i_ref + 1).astype(np.float64) == i_ref.astype(np.float64)) * 1.0,
    (i_ref != 0) * 1.0,
    np.sign(CAST_CASES),
    np.clip(CAST_CASES, -1e9, 1e9).astype(np.int32).astype(np.float64),
  ]
  for k, (g, e) in enumerate(zip(got, expected, strict=True)):
    np.testing.assert_array_equal(g, e, err_msg=str(k))
  assert not np.signbit(got[0][3:5]).any()


def test_casts_to_and_from_bool_and_their_structure() -> None:
  values = np.array([np.nan, -np.nan, -0.0, 0.0, 5e-324, -np.inf, 1e-310, 2.0])

  @sc.function(sc.arg("x", values.size), outputs=sc.group(sc.arg("a"), sc.arg("b"), sc.arg("c")), name="sc_bool_casts")
  def casts(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    truth = sc.cast(x, "bool")
    return sc.cast(truth, "float64"), sc.cast(sc.cast(truth, "int64"), "float64"), sc.cast(~x.cast("bool"), "float64")

  truthy = values.astype(bool)
  got = [np.asarray(g) for g in casts(values)]
  np.testing.assert_array_equal(got[0], truthy * 1.0)
  np.testing.assert_array_equal(got[1], truthy * 1.0)
  np.testing.assert_array_equal(got[2], ~truthy * 1.0)
  assert got[0].tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]  # NaN of either sign is true, -0.0 is false
  x = sc.sym("x", 3)
  assert sc.cast(x, "float64") is x and x.cast(sc.dtypes.float64) is x
  assert sc.cast(x, "bool").op == sc.ExprOp.NE and sc.cast(x, "bool") is x.cast("bool")
  assert sc.cast(x, "int64").op == sc.ExprOp.CAST and sc.cast(x, "int64").type.dtype == sc.dtypes.int64


def test_broadcasting_shapes_and_values() -> None:
  cases: list[tuple[Callable[..., sc.Expr], Callable[..., np.ndarray], tuple[int, ...]]] = [
    (lambda c, u, x, s, w: sc.where(c > 0.0, x, s), lambda c, u, x, s, w: np.where(c > 0.0, x, s), (2, 3)),
    (lambda c, u, x, s, w: sc.where(s > 0.0, w, x), lambda c, u, x, s, w: np.where(s > 0.0, w, x), (3,)),
    (lambda c, u, x, s, w: sc.where(c > 0.0, 1.0, u), lambda c, u, x, s, w: np.where(c > 0.0, 1.0, u), (2, 1)),
    (lambda c, u, x, s, w: _float(u < x), lambda c, u, x, s, w: u < x, (2, 3)),
    (lambda c, u, x, s, w: _float(sc.equal(w, s)), lambda c, u, x, s, w: np.equal(w, s), (1,)),
    (lambda c, u, x, s, w: _float((c > 0.0) & (x > 0.0)), lambda c, u, x, s, w: (c > 0.0) & (x > 0.0), (2, 3)),
    (lambda c, u, x, s, w: _float(sc.isfinite(u) | ~sc.isfinite(x)), lambda c, u, x, s, w: np.isfinite(u) | ~np.isfinite(x), (2, 3)),
  ]
  values = (np.array([[1.0], [-1.0]]), np.array([[-0.0], [np.nan]]), np.array([-1.0, 0.5, np.inf]), np.array(-2.0), np.array([-0.0]))
  shapes = [(2, 1), (2, 1), (3,), (), (1,)]
  symbols = [sc.sym(name, shape) for name, shape in zip("cuxsw", shapes, strict=True)]
  for build, _, shape in cases:
    assert build(*symbols).shape == shape

  @sc.function(
    sc.group(*(sc.arg(n, s) for n, s in zip("cuxsw", shapes, strict=True))), outputs=sc.group(*(sc.arg(f"o{k}") for k in range(len(cases))))
  )
  def broadcast(args: tuple[sc.Expr, ...]) -> tuple[sc.Expr, ...]:
    return tuple(build(*args) for build, _, _ in cases)

  for k, (g, (_, reference, shape)) in enumerate(zip(map(np.asarray, broadcast(values)), cases, strict=True)):
    with np.errstate(invalid="ignore"):
      e = reference(*values)
    assert np.shape(g) == shape, k
    _assert_values(np.asarray(g), np.asarray(e), f"broadcast case {k}")


@pytest.mark.parametrize("operation", ["isfinite", "maximum", "reduce_max"])
def test_nonfinite_operations_refuse_fast_math(tmp_path: Path, operation: str) -> None:
  """``isfinite`` needs infinities and NaN, so a build that assumes neither stops at an ``#error``."""

  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"), name="sc_fast_math")
  def finite(x: sc.Expr) -> sc.Expr:
    return {
      "isfinite": lambda: sc.where(sc.isfinite(x), x, 0.0),
      "maximum": lambda: sc.maximum(x, 0.0),
      "reduce_max": lambda: sc.reduce_max(x),
    }[operation]()

  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"), name="sc_no_isfinite")
  def plain(x: sc.Expr) -> sc.Expr:
    return sc.where(x > 0.0, x, 0.0)

  for fn, refused in ((finite, True), (plain, False)):
    source = tmp_path / f"{fn.name}.c"
    source.write_text(render_c_source(fn))
    for flags in ((), ("-ffinite-math-only",)):
      built = subprocess.run([*c_compiler(), *flags, "-c", str(source), "-o", str(tmp_path / "out.o")], capture_output=True, text=True)
      assert (built.returncode != 0) == (refused and bool(flags)), built.stderr
      assert not refused or not flags or "-ffast-math" in built.stderr


REFUSALS: dict[str, tuple[Callable[[], object], type[Exception], str]] = {
  "where_branches_mixed": (lambda: sc.where(sc.sym("x", 2) > 0.0, sc.sym("y", 2), sc.cast(sc.sym("z", 2), "int64")), TypeError, "sc.cast"),
  "int_beside_python_float": (lambda: sc.cast(sc.sym("x", 2), "int64") < 0.5, TypeError, "cannot combine float"),
  "and_float": (lambda: sc.sym("x", 2) & (sc.sym("y", 2) > 0.0), TypeError, "bool operands"),
  "where_float_cond": (lambda: sc.where(sc.sym("x", 2), 1.0, 0.0), TypeError, "bool condition"),
  "where_python_bool": (lambda: sc.where(sc.sym("x", 2) == 0, 1.0, 0.0), TypeError, "sc.equal"),
  "and_python_bool": (lambda: (sc.sym("x", 2) == 0) & (sc.sym("y", 2) > 0.0), TypeError, "sc.equal"),
  "or_python_bool": (lambda: (sc.sym("y", 2) > 0.0) | (sc.sym("x", 2) == 0), TypeError, "sc.equal"),
  "cast_python_bool": (lambda: sc.cast(sc.sym("x", 2) == 0, "float64"), TypeError, "sc.equal"),
  "sqrt_int": (lambda: sc.cast(sc.sym("x", 2), "int64").sqrt(), TypeError, "sqrt takes floating operands"),
  "div_int": (lambda: sc.cast(sc.sym("x", 2), "int64") / 2, TypeError, "/ takes floating operands"),
  "sum_bool": (lambda: (sc.sym("x", 2) > 0.0).sum(), TypeError, "sum takes floating values"),
  "matmul_int": (lambda: sc.cast(sc.sym("x", 2), "int64") @ sc.cast(sc.sym("y", 2), "int64"), TypeError, "@ takes floating"),
  "isfinite_int": (lambda: sc.isfinite(sc.cast(sc.sym("x", 2), "int64")), TypeError, "floating operand"),
  "floordiv_float": (lambda: sc.sym("x", 2) // 2.0, TypeError, "integer operands"),
  "cast_unknown": (lambda: sc.cast(sc.sym("x", 2), "complex64"), ValueError, "unknown dtype"),
  "where_shapes": (lambda: sc.where(sc.sym("c", 4) > 0.0, sc.sym("x", 3), 0.0), ValueError, "broadcast"),
  "truth_value": (lambda: bool(sc.sym("x", 1) > 0.0), TypeError, "sc.where"),
}


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_builders_refuse_invalid_operands(name: str) -> None:
  build, error, match = REFUSALS[name]
  with pytest.raises(error, match=match):
    build()


def test_double_equals_stays_identity() -> None:
  x = sc.sym("x", 2)
  assert (x == x) is True and (x == sc.sym("y", 2)) is False
  assert len({x, x * 1.0, x}) == 2


@pytest.mark.parametrize("variant", ["block", "scalar"])
def test_int64_selects_and_bool_constants_keep_their_width(variant: str) -> None:
  """C reads ``50000`` as an ``int``: a product of two selects between such literals must still be int64."""

  @sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("p"), sc.arg("b")), name=f"sc_int_width_{variant}")
  def width(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    a = sc.where(x > 0.0, np.int64(50000), np.int64(60000))
    b = sc.where(x > 1.0, np.int64(50000), np.int64(60000))
    truth = (x > 0.0) & np.bool_(True) | sc.where(x > 1.0, np.bool_(True), np.bool_(False))
    return getattr(sc.cast(a * b, "float64"), variant)(), getattr(sc.cast(truth, "float64"), variant)()

  product, truth = width(np.array([-1.0, 0.5, 2.0]))
  np.testing.assert_array_equal(product, [3.6e9, 3.0e9, 2.5e9])
  np.testing.assert_array_equal(truth, [0.0, 1.0, 1.0])
