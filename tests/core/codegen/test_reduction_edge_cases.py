"""Edge cases of the extremum reductions against NumPy: ``reduce_max``, ``reduce_min``, ``norm_inf``,
``norm_1``, ``segment_max`` and ``segment_min``, compiled as loops and scalarized, and folded.

NaN in every lane and in the tail, infinities, signed zeros, subnormals and the largest doubles;
scalar, 2-D and 3-D operands and strided views of them; elementwise producers fused into the
reduction; reductions inside ``vmap`` and ``scan`` bodies; empty, unsorted, repeated and
out-of-range segment ids. The derivatives are in ``tests/core/ad/test_reduction_derivative_edge_cases.py``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_source
from scaly.ir.expr import ExprOp
from scaly.ir.types import Lowering

BIG = np.finfo(np.float64).max
TINY = np.finfo(np.float64).smallest_subnormal


def _fn(name: str, inputs: Sequence[sc.Expr], outputs: Sequence[sc.Expr]) -> sc.Function:
  return sc.Function.from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"out{k}" for k in range(len(outputs))])


def _special_rows(n: int) -> np.ndarray:
  """Rows of length ``n`` whose extrema a compiled loop could get wrong."""
  rng = np.random.default_rng(n)
  rows = [np.full(n, v) for v in (np.nan, -np.inf, np.inf, -0.0, 0.0)]
  for k in sorted({0, 1, 2, 3, n // 2, n - 1}):  # every lane of four, the middle and the tail
    for scale, value in ((None, np.nan), (-np.inf, -BIG), (np.inf, BIG), (-1.0, -0.0), (-1.0, 0.0), (1.0, -0.0), (1.0, 0.0)):
      row = rng.standard_normal(n) if scale is None else scale * rng.uniform(1.0, 2.0, n)
      row[k] = value
      rows.append(row)
  patterns = [
    [1.0, np.nan, 2.0, np.nan],
    [np.inf, np.nan, -np.inf],
    [np.nan, -np.inf, np.inf, np.nan],
    [np.inf, -np.inf],
    [TINY, -TINY, 1e-310, -2.2e-308, 0.0],
    [BIG, -BIG, 1e300, -1e-300],
    [7.0, 1.0, -3.0, 7.0, -3.0],
    [-0.0, 0.0],
  ]
  return np.array(rows + [np.resize(p, n) for p in patterns])


def _stats(rows: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
  with np.errstate(invalid="ignore", over="ignore"):
    return np.array([[np.max(r), np.min(r), np.max(np.abs(r)), np.sum(np.abs(r))] for r in rows])


def _assert_stats(got: np.ndarray, rows: Sequence[np.ndarray] | np.ndarray, *, zero_signs: Literal["fixed", "all"] = "fixed") -> None:
  """Extrema exactly, ``norm_1`` (if present) to rounding, since NumPy sums pairwise, and the sign of
  every zero whose sign is fixed: a zero extremum among zeros of both signs may be either, as the
  lowering notes."""
  want = _stats(rows)
  want = want[:, : np.asarray(got).size // len(rows)]
  got = np.asarray(got).reshape(want.shape)
  np.testing.assert_array_equal(got[:, :3], want[:, :3])
  np.testing.assert_allclose(got[:, 3:], want[:, 3:], rtol=1e-14)
  fixed = ~np.isnan(want)
  if zero_signs == "fixed":
    mixed = np.array([0 < np.signbit(r[r == 0]).sum() < np.count_nonzero(r == 0) for r in rows])
    fixed[:, :2] &= ~((want[:, :2] == 0) & mixed[:, None])
  np.testing.assert_array_equal(np.signbit(got)[fixed], np.signbit(want)[fixed])


@pytest.mark.parametrize("lowering", ["block", "scalar"])
@pytest.mark.parametrize("n", [5, 13])
def test_extrema_of_special_rows_in_calls_vmap_and_scan(n: int, lowering: Lowering) -> None:
  """Below eight entries one accumulator runs; from eight, four lanes and a tail. Each row goes
  through a plain call, a ``vmap`` and a ``scan`` step of the same body."""
  rows = _special_rows(n)
  k = len(rows)
  u, c = sc.sym("u", n).with_lowering(lowering), sc.sym("c", 1).with_lowering(lowering)
  stats = sc.stack([u.max(), u.min(), sc.norm_inf(u), sc.norm_1(u)])
  inner = _fn(f"re_rows_{n}_{lowering}_in", [u], [stats])
  step = _fn(f"re_rows_{n}_{lowering}_step", [c, u], [c + u.max(), stats])
  flat = sc.sym("flat", k * n)
  calls = sc.concat([inner(flat[i * n : (i + 1) * n]) for i in range(k)])
  final, ys = sc.scan(step, sc.const(np.zeros(1)), [(flat, 0, n)], length=k)
  fun = _fn(f"re_rows_{n}_{lowering}", [flat], [calls, sc.vmap(inner, k, [(flat, 0, n)]), ys, final])
  got = fun(rows.reshape(-1))
  for out in got[:3]:
    _assert_stats(out, rows)
  total = np.zeros(1)
  for row in rows:
    total = total + np.max(row)
  np.testing.assert_array_equal(got[3], total)


VIEWS: dict[str, Callable[[Any], Any]] = {  # the same view of an Expr and of an array
  "whole": lambda a: a,
  "transposed": lambda a: a.T,
  "permuted": lambda a: a.transpose((1, 2, 0))[2],
  "strided": lambda a: a[:, 1:, ::2],
  "reversed": lambda a: a[1, ::-1, 1:3],
  "reshaped": lambda a: a.reshape((6, 4))[::2].T,
  "plane": lambda a: a[0].T,
  "column": lambda a: a[:, :, 1:2],
  "entry": lambda a: a[1, 2, 3],
}


@pytest.mark.parametrize("lowering", ["block", "scalar"])
def test_extrema_of_scalars_blocks_and_strided_views(lowering: Lowering) -> None:
  """A rank-0 operand, a 3-D block and views that read it out of order or skip entries: a NaN that a
  view skips must not reach its extremum."""
  x, s = sc.sym("x", (2, 3, 4)).with_lowering(lowering), sc.sym("s", ()).with_lowering(lowering)
  parts = [*(VIEWS[v](x) for v in VIEWS), s]
  inner = _fn(f"re_views_{lowering}_in", [x, s], [sc.concat([sc.stack([p.max(), p.min(), sc.norm_inf(p), sc.norm_1(p)]) for p in parts])])
  xo, so = sc.sym("x", (2, 3, 4)), sc.sym("s", ())
  fun = _fn(f"re_views_{lowering}", [xo, so], [inner((xo, so))])
  rng = np.random.default_rng(7)
  finite = rng.standard_normal((2, 3, 4))
  finite[1, 2, 3], finite[0, 1, 2] = 50.0, -50.0
  special = finite.copy()
  special[0, 0, 1], special[1, 0, 0], special[0, 2, 3] = np.nan, np.inf, -np.inf
  zeros = np.full((2, 3, 4), -0.0)
  zeros[1, 1, 1] = -1.0
  for xv, sv in ((finite, 3.0), (special, np.nan), (zeros, -0.0)):
    rows = [np.asarray(VIEWS[v](xv)).reshape(-1) for v in VIEWS] + [np.array([sv])]
    _assert_stats(fun((xv, np.asarray(sv))), rows)


def test_extrema_of_a_long_vector_with_the_extremum_or_nan_in_the_tail() -> None:
  n = 100_003  # three entries past the last block of four
  x = sc.sym("x", n)
  fun = _fn("re_long", [x], [sc.stack([x.max(), x.min(), sc.norm_inf(x), (0.5 * x - 1.0).max()])])
  v = np.random.default_rng(11).standard_normal(n)
  v[-1], v[-2] = 10.0, -10.0
  np.testing.assert_array_equal(fun(v), [10.0, -10.0, 10.0, 4.0])
  v[-3] = np.nan
  assert np.isnan(fun(v)).all()
  w = np.full(n, -np.inf)
  w[n // 2] = -BIG
  np.testing.assert_array_equal(fun(w), [-BIG, -np.inf, np.inf, 0.5 * -BIG - 1.0])


def test_integer_and_bool_extrema_reduce_in_their_own_type() -> None:
  """``int64`` values below ``2**53`` cross the ``double`` ABI exactly, in any lane or the tail."""
  x, b = sc.sym("x", 11, dtype="int64"), sc.sym("b", 9, dtype="bool")
  fun = _fn("re_int_bool", [x, b], [x.max(), x.min(), sc.norm_inf(x), b.max(), b.min()])
  assert [o.type.dtype.name for o in fun.outputs] == ["int64"] * 3 + ["bool"] * 2
  base = np.array([3, -9, 4, 2**52, -1, -(2**53), 0, 7, 7, -3, 11], dtype=np.int64)
  for xv, bv in (
    (base, np.zeros(9, dtype=bool)),
    (base[::-1].copy(), np.eye(9, dtype=bool)[8]),
    (np.full(11, -5, dtype=np.int64), np.ones(9, dtype=bool)),
  ):
    got = fun._flat_numerical_call(xv, bv)
    np.testing.assert_array_equal(got, [xv.max(), xv.min(), np.abs(xv).max(), bv.max(), bv.min()])


def test_empty_operands_are_refused_when_built() -> None:
  for shape in ((0,), (0, 3), (2, 0, 4)):
    e = sc.sym("e", shape)
    for build, op in ((sc.reduce_max, "max"), (lambda a: a.min(), "min"), (sc.norm_inf, "max")):
      with pytest.raises(ValueError, match=f"^{op} of an empty expression has no value$"):
        build(e)
  with pytest.raises(ValueError, match="empty"):
    sc.reduce_min(np.zeros(0))
  assert sc.norm_1(sc.sym("e", 0)).shape == ()  # the empty sum is zero, not a refusal


@pytest.mark.parametrize("n", [5, 13])
def test_folded_extrema_of_constants_agree_with_compiled_and_numpy(n: int) -> None:
  """Constant operands fold in Python with NumPy; symbols run in C. The two agree on every value, and
  on the sign of every zero that is fixed. Among zeros of both signs the fold returns NumPy's zero and
  the compiled loop the first one it saw, which may differ."""
  rows = _special_rows(n)

  def stats(r: sc.Expr) -> sc.Expr:
    return sc.stack([r.max(), r.min(), sc.norm_inf(r), sc.norm_1(r)])

  # ``norm_1`` of the largest doubles overflows, and a fold that would overflow is left to run time.
  folded = [sc.simplify(sc.stack([c.max(), c.min(), sc.norm_inf(c)])) for c in map(sc.const, rows)]
  assert all(f.op == ExprOp.CONST for f in folded)
  _assert_stats(np.array([f.value for f in folded]), rows, zero_signs="all")
  flat = sc.sym("flat", rows.size)
  sym_stats = sc.concat([stats(flat[i * n : (i + 1) * n]) for i in range(len(rows))])
  fun = _fn(f"re_fold_{n}", [flat], [sym_stats, sc.concat([stats(sc.const(r)) for r in rows])])
  compiled, from_constants = fun(rows.reshape(-1))
  _assert_stats(compiled, rows)
  _assert_stats(from_constants, rows, zero_signs="all")


def test_folded_zero_extrema_keep_their_sign_through_code_generation() -> None:
  """``max`` of negative zeros is ``-0``, and ``norm_inf`` of them ``+0``; folded, both reach the
  generated C as constants, where they must stay distinct."""
  c = sc.const(np.full(3, -0.0))
  fun = _fn("re_fold_zero_sign", [sc.sym("d", 1)], [sc.stack([c.max(), sc.norm_inf(c)])])
  np.testing.assert_array_equal(np.signbit(fun(np.zeros(1))), [True, False])


SEGMENTS: dict[str, tuple[np.ndarray, int]] = {  # 12 values each: ids, num_segments
  "unsorted_gaps": (np.array([5, 0, 5, 2, 0, 5, 7, 2, 0, 7, 5, 2]), 9),
  "sorted_runs": (np.array([0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 4, 4]), 6),
  "one_segment": (np.zeros(12, dtype=np.int64), 1),
  "own_segments": (np.random.default_rng(0).permutation(12), 12),
  "own_segments_gaps": (2 * np.random.default_rng(1).permutation(12), 25),
}
FILLS = (None, 0.0, np.nan, np.inf, -np.inf)


def _segment_ref(op: str, values: np.ndarray, ids: np.ndarray, n: int, fill: float | None) -> np.ndarray:
  """``fill`` seeds every bin and each value replaces its bin's entry when larger (smaller) or NaN, as
  the lowering and the fold both do; so ``fill`` is what an empty bin reads."""
  out = np.full(n, (-np.inf if op == "max" else np.inf) if fill is None else fill)
  with np.errstate(invalid="ignore"):
    (np.maximum if op == "max" else np.minimum).at(out, ids.reshape(-1), values.reshape(-1))
  return out


def _segment_values() -> list[np.ndarray]:
  rng = np.random.default_rng(5)
  rows = [rng.standard_normal(12) for _ in range(4)]
  rows[1][[0, 6]] = np.nan  # one NaN in two different bins
  rows[2][[10, 11]] = np.nan  # a whole run is NaN
  rows[3][[2, 5, 9]] = [np.inf, -np.inf, np.inf]
  return [*rows, np.full(12, -0.0), np.resize([TINY, -BIG, BIG, -TINY, 0.0], 12)]


@pytest.mark.parametrize("lowering", ["block", "scalar"])
def test_segment_extrema_with_gaps_runs_fills_and_nan(lowering: Lowering) -> None:
  """Empty bins read ``fill``; a NaN stays in its own bin; 2-D values reduce in flat order; a single
  value reduces into one bin of several."""
  v, m = sc.sym("v", 12).with_lowering(lowering), sc.sym("m", (3, 4)).with_lowering(lowering)
  cases: list[tuple[str, np.ndarray, int, float | None]] = [
    (op, ids, n, fill) for ids, n in SEGMENTS.values() for fill in FILLS for op in ("max", "min")
  ]
  builds = {"max": sc.segment_max, "min": sc.segment_min}
  outs = [builds[op](v, ids, n, fill=fill) for op, ids, n, fill in cases]
  ids2 = SEGMENTS["unsorted_gaps"][0].reshape(3, 4)
  outs += [sc.segment_max(m, ids2, 9), sc.segment_min(m.T, ids2, 9, fill=0.0), sc.segment_max(v[3:4], [2], 4), sc.segment_min(v[11], [0], 1)]
  inner = _fn(f"re_seg_{lowering}_in", [v, m], outs)
  vo, mo = sc.sym("v", 12), sc.sym("m", (3, 4))
  fun = _fn(f"re_seg_{lowering}", [vo, mo], list(inner((vo, mo))))
  for values in _segment_values():
    mv = values[::-1].reshape(3, 4).copy()
    got = fun((values, mv))
    want = [_segment_ref(op, values, ids, n, fill) for op, ids, n, fill in cases]
    want += [
      _segment_ref("max", mv, ids2, 9, None),
      _segment_ref("min", mv.T, ids2, 9, 0.0),
      _segment_ref("max", values[3:4], np.array([2]), 4, None),
      _segment_ref("min", values[11:], np.array([0]), 1, None),
    ]
    # Where every value is a negative zero, a bin with the default infinite fill has a fixed sign.
    signed = [fill is None for *_, fill in cases] + [True, False, True, True]
    for g, w, sign in zip(got, want, signed, strict=True):
      np.testing.assert_array_equal(g, w)
      if sign and np.signbit(values).all():
        np.testing.assert_array_equal(np.signbit(g), np.signbit(w))


def test_folded_segment_extrema_of_constants_match_numpy() -> None:
  for values in _segment_values():
    for ids, n in SEGMENTS.values():
      for fill in FILLS:
        for op, build in (("max", sc.segment_max), ("min", sc.segment_min)):
          folded = sc.simplify(build(sc.const(values), ids, n, fill=fill))
          assert folded.op == ExprOp.CONST
          np.testing.assert_array_equal(folded.value, _segment_ref(op, values, ids, n, fill))


def test_segment_extrema_refuse_bad_ids() -> None:
  v = sc.sym("v", 3)
  for build, op in ((sc.segment_max, "segment_max"), (sc.segment_min, "segment_min")):
    with pytest.raises(IndexError, match=r"indices must be in \[0, 3\)"):
      build(v, [0, 3, 1], 3)
    with pytest.raises(IndexError, match=r"indices must be in \[0, 2\)"):
      build(v, [0, -1, 1], 2)
    with pytest.raises(IndexError, match=r"indices must be in \[0, 0\)"):
      build(v, [0, 0, 0], 0)
    with pytest.raises(ValueError, match=f"^{op} has 2 segment ids for 3 values$"):
      build(v, [0, 1], 2)
    with pytest.raises(ValueError, match=f"^{op} has 4 segment ids for 6 values$"):
      build(sc.sym("m", (2, 3)), np.zeros((2, 2), dtype=np.int64), 1)
    assert build(v, [0, 0, 0], 5).shape == (5,)  # bins past the largest id are empty, not refused


PRODUCERS: dict[str, tuple[Callable[[sc.Expr, sc.Expr], sc.Expr], Callable[[np.ndarray, np.ndarray], np.floating]]] = {
  "product_max": (lambda x, y: (x * y).max(), lambda x, y: np.max(x * y)),
  "quotient_min": (lambda x, y: (x / y).min(), lambda x, y: np.min(x / y)),
  "shifted_norm_inf": (lambda x, y: sc.norm_inf(x - 3.0), lambda x, y: np.max(np.abs(x - 3.0))),
  "masked_norm_inf": (lambda x, y: sc.norm_inf(sc.where(y > 0.0, x, 0.0)), lambda x, y: np.max(np.abs(np.where(y > 0.0, x, 0.0)))),
  "negated_min": (lambda x, y: (-x).min(), lambda x, y: np.min(-x)),
  "sqrt_max": (lambda x, y: x.sqrt().max(), lambda x, y: np.max(np.sqrt(x))),
}


@pytest.mark.parametrize("lowering", ["block", "scalar"])
@pytest.mark.parametrize("n", [7, 13, 64])
def test_fused_producers_propagate_the_nan_they_make(n: int, lowering: Lowering) -> None:
  """Elementwise producers fused into the reduction make their NaN (``0 * inf``, ``0 / 0``,
  ``inf / inf``) inside the loop, in any lane or the tail; a NaN that ``where`` masks out must not
  appear. ``sqrt``, costly, is stored first instead, and its NaN must propagate all the same."""
  x, y = sc.sym("x", n).with_lowering(lowering), sc.sym("y", n).with_lowering(lowering)
  inner = _fn(f"re_fused_{n}_{lowering}_in", [x, y], [sc.stack([build(x, y) for build, _ in PRODUCERS.values()])])
  xo, yo = sc.sym("x", n), sc.sym("y", n)
  fun = _fn(f"re_fused_{n}_{lowering}", [xo, yo], [inner((xo, yo))])
  if lowering == "block":
    assert len(re.findall(rf"double \w+\[{n}\]", render_c_source(fun))) <= 1, "a producer other than sqrt was stored, not fused"
  rng = np.random.default_rng(n)
  for k in sorted({0, 1, 2, 3, n // 2, n - 2, n - 1}):
    for xk, yk in ((0.0, np.inf), (0.0, 0.0), (np.nan, -1.0), (-1.0, 2.0), (np.inf, np.inf), (-np.inf, 1.0)):
      xv, yv = rng.uniform(0.5, 2.0, n), rng.uniform(0.5, 2.0, n) * rng.choice([-1.0, 1.0], n)
      xv[k], yv[k] = xk, yk
      with np.errstate(all="ignore"):
        want = [reference(xv, yv) for _, reference in PRODUCERS.values()]
      np.testing.assert_array_equal(fun((xv, yv)), want, err_msg=f"x[{k}]={xk}, y[{k}]={yk}")
