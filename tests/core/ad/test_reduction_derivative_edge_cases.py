"""Derivatives of the extremum reductions at their edge cases: ``reduce_max``, ``reduce_min``,
``norm_inf``, ``segment_max`` and ``segment_min``.

One-hot at a unique extremum; the ``sc.options(nonsmooth=...)`` conventions at ties, including
three-way ties, ties across signed zeros, ties in a transposed view and ties in one segment but not
another; empty segments; the refusal under ``"error"`` and that the choice is fixed when a graph is
built; forward against reverse mode, Hessians, finite differences away from ties and the Jacobian
sparsity of segment extrema. The values are in ``tests/core/codegen/test_reduction_edge_cases.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference

MODES = ["split", "first"]


def _fn(name: str, inputs: Sequence[sc.Expr], outputs: Sequence[sc.Expr]) -> sc.Function:
  return sc.Function.from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"out{k}" for k in range(len(outputs))])


def _three_ways(name: str, out: sc.Expr, x: sc.Expr, mode: str) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """The Jacobian of ``out`` three ways: batched forward mode (``sc.jacobian``), one ``jvp`` per seed
  and one ``vjp`` per cotangent. The seeds and cotangents are run-time inputs, so no constant zero
  lets the simplifier drop a product that could be NaN."""
  w, t = sc.sym("w", out.shape), sc.sym("t", x.shape)
  with sc.options(nonsmooth=mode):
    jac = sc.jacobian(out, x)
    (rev,) = sc.vjp((out,), (x,), (w,))
    fwd = sc.jvp(out, x, t)
  fun = _fn(name, [x, w, t], [jac, rev, fwd])

  def run(xv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zw, zt = np.zeros(out.shape), np.zeros(x.shape)
    dense = fun((xv, zw, zt))[0].reshape(out.size, x.size)
    rows = np.array([fun((xv, np.eye(out.size)[i].reshape(out.shape), zt))[1].reshape(-1) for i in range(out.size)])
    cols = np.array([fun((xv, zw, np.eye(x.size)[j].reshape(x.shape)))[2].reshape(-1) for j in range(x.size)]).T
    return dense, rows, cols

  return run


def _share(values: np.ndarray, mode: str) -> np.ndarray:
  """How the derivative of the largest of ``values`` spreads over them: equally over the tied
  entries, or all to the first."""
  hit = values == values.max()
  return hit / hit.sum() if mode == "split" else np.eye(values.size)[np.argmax(hit)]


def _extremum_row(values: np.ndarray, index: np.ndarray, kind: str, size: int, mode: str) -> np.ndarray:
  """d(reduction of ``values``)/dx, where ``values`` is a view whose entry ``k`` is ``x.flat[index.flat[k]]``."""
  v, idx = values.reshape(-1), index.reshape(-1)
  weights = {"max": lambda: _share(v, mode), "min": lambda: _share(-v, mode), "norm_inf": lambda: _share(np.abs(v), mode) * np.sign(v)}[kind]()
  row = np.zeros(size)
  np.add.at(row, idx, weights)
  return row


def _segment_rows(values: np.ndarray, index: np.ndarray, ids: np.ndarray, n: int, kind: str, size: int, mode: str) -> np.ndarray:
  v, idx, ids = values.reshape(-1), index.reshape(-1), np.asarray(ids).reshape(-1)
  rows = np.zeros((n, size))
  for b in range(n):
    members = np.flatnonzero(ids == b)
    if members.size:  # an empty bin reads its fill, a constant
      rows[b] = _extremum_row(v[members], idx[members], kind, size, mode)
  return rows


def _assert_three_ways(got: tuple[np.ndarray, np.ndarray, np.ndarray], want: np.ndarray) -> None:
  for jac in got:
    np.testing.assert_array_equal(jac, want)


@pytest.mark.parametrize("mode", MODES)
def test_unique_extrema_have_one_hot_derivatives_in_every_view(mode: str) -> None:
  """Away from ties the conventions agree: a one-hot row (signed for ``norm_inf``) at the entry the
  view picks, in any lane of four or the tail, matching finite differences."""
  x = sc.sym("x", (3, 5))
  ids = np.array([[2, 0, 1, 3, 0], [1, 2, 3, 0, 2], [3, 1, 0, 1, 3]])
  index = np.arange(15).reshape(3, 5)
  xv = np.random.default_rng(3).permutation(np.linspace(-2.0, 2.0, 15)).reshape(3, 5)
  xv[2, 4], xv[0, 2], xv[1, 3] = 5.0, -4.0, -7.0
  parts: list[tuple[sc.Expr, np.ndarray, np.ndarray, str]] = [
    (x.max(), xv, index, "max"),
    (x.min(), xv, index, "min"),
    (sc.norm_inf(x), xv, index, "norm_inf"),
    (x.T.max(), xv.T, index.T, "max"),
    (x[:, ::2].min(), xv[:, ::2], index[:, ::2], "min"),
    (sc.norm_inf(x[1:, 1:]), xv[1:, 1:], index[1:, 1:], "norm_inf"),
  ]
  segments = sc.concat([sc.segment_max(x, ids, 4, fill=-100.0), sc.segment_min(x.T, ids.T, 4, fill=100.0)])
  out = sc.concat([sc.stack([p[0] for p in parts]), segments])
  want = np.vstack(
    [
      np.array([_extremum_row(v, i, kind, 15, mode) for _, v, i, kind in parts]),
      _segment_rows(xv, index, ids, 4, "max", 15, mode),
      _segment_rows(xv.T, index.T, ids.T, 4, "min", 15, mode),
    ]
  )
  _assert_three_ways(_three_ways(f"rxd_unique_{mode}", out, x, mode)(xv), want)
  values = _fn(f"rxd_unique_{mode}_value", [x], [out])
  np.testing.assert_allclose(want, finite_difference(lambda v: values(v), xv), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("mode", MODES)
def test_tied_extrema_split_equally_or_go_to_the_lowest_index_of_the_view(mode: str) -> None:
  """A three-way maximum across lanes and the tail, a two-way minimum, a five-way ``norm_inf`` tie of
  both signs, a tie between ``-0.0`` and ``+0.0``, and a tie in a transposed view, where "lowest
  index" means lowest in the view's order."""
  x = sc.sym("x", 20)
  a, y, m = x[:13], x[13:16], x[16:].reshape((2, 2))
  av = np.array([-5.0, 1.0, 5.0, 2.0, -1.0, 0.5, 3.0, 5.0, -2.0, -5.0, 4.0, 0.25, 5.0])
  xv = np.concatenate([av, [-0.0, 0.0, -1.0], [1.0, 5.0, 5.0, 0.0]])
  index = np.arange(20)
  im = index[16:].reshape(2, 2)
  parts = [(a.max(), "max", 0), (a.min(), "min", 0), (sc.norm_inf(a), "norm_inf", 0), (y.max(), "max", 1), (sc.norm_inf(y), "norm_inf", 1)]
  views = [(xv[:13], index[:13]), (xv[13:16], index[13:16])]
  want = [_extremum_row(*views[v], kind, 20, mode) for _, kind, v in parts]
  want += [_extremum_row(xv[16:].reshape(2, 2), im, "max", 20, mode), _extremum_row(xv[16:].reshape(2, 2).T, im.T, "max", 20, mode)]
  out = sc.stack([*(p for p, _, _ in parts), m.max(), m.T.max()])
  _assert_three_ways(_three_ways(f"rxd_ties_{mode}", out, x, mode)(xv), np.array(want))
  if mode == "split":
    assert list(want[0][[2, 7, 12]]) == [1 / 3] * 3 and list(want[3][13:15]) == [0.5, 0.5]
  else:
    assert want[3][13] == 1.0 and want[5][17] == 1.0 and want[6][18] == 1.0  # -0.0 wins as first; the view's order decides


@pytest.mark.parametrize("mode", MODES)
def test_segment_ties_empty_segments_and_views(mode: str) -> None:
  """Ties in some bins and not others, unsorted ids whose lowest tied index is not the first run,
  a tie across signed zeros, empty bins (a zero row), and a transposed operand."""
  x = sc.sym("x", 9)
  ids = np.array([3, 1, 3, 0, 3, 1, 4, 4, 0])
  xv = np.array([7.0, 2.0, 7.0, 5.0, 7.0, -1.0, -0.0, 0.0, 5.0])
  index = np.arange(9)
  m, im = xv.reshape(3, 3).T, index.reshape(3, 3).T
  out = sc.concat([sc.segment_max(x, ids, 6), sc.segment_min(x, ids, 6), sc.segment_max(x.reshape((3, 3)).T, ids, 6, fill=0.5)])
  want = np.vstack(
    [
      _segment_rows(xv, index, ids, 6, "max", 9, mode),
      _segment_rows(xv, index, ids, 6, "min", 9, mode),
      _segment_rows(m, im, ids, 6, "max", 9, mode),
    ]
  )
  _assert_three_ways(_three_ways(f"rxd_seg_{mode}", out, x, mode)(xv), want)
  assert not want[[2, 5, 8, 11]].any()
  if mode == "split":
    np.testing.assert_array_equal(want[3][[0, 2, 4]], [1 / 3] * 3)


@pytest.mark.parametrize("mode", MODES)
def test_hessians_of_extrema_are_their_pieces_hessians(mode: str) -> None:
  """Piecewise-linear extrema have a zero Hessian at a tie; ``max(x * x)`` has ``2 w`` on the
  diagonal, ``w`` the convention's weights."""
  x = sc.sym("x", 5)
  ids = np.array([0, 1, 0, 1, 2])
  linear = x.max() + x.min() + sc.norm_inf(x) + sc.segment_max(x, ids, 3).sum() + sc.segment_min(x, ids, 4, fill=9.0).sum()
  with sc.options(nonsmooth=mode):
    outs = [sc.hessian(linear, x), sc.gradient((x * x).max(), x), sc.hessian((x * x).max(), x)]
  fun = _fn(f"rxd_hess_{mode}", [x], outs)
  tie = np.array([3.0, -3.0, 3.0, -3.0, 1.0])
  hess_linear, grad_sq, hess_sq = fun(tie)
  np.testing.assert_array_equal(hess_linear, np.zeros((5, 5)))
  w = np.array([0.25, 0.25, 0.25, 0.25, 0.0]) if mode == "split" else np.eye(5)[0]
  np.testing.assert_array_equal(grad_sq, 2 * w * tie)
  np.testing.assert_array_equal(hess_sq, np.diag(2 * w))


@pytest.mark.parametrize(
  "mode",
  ["split", "first"],
)
def test_a_bin_that_its_fill_wins_has_a_zero_derivative(mode: str) -> None:
  """``fill`` seeds every bin, so a bin whose values all lie below it reads the constant ``fill``: its
  derivative is zero, as finite differences say, and the other bins' are untouched."""
  x = sc.sym("x", 4)
  out = sc.concat([sc.segment_max(x, [0, 0, 1, 1], 2, fill=0.0), sc.segment_min(x, [1, 0, 1, 0], 2, fill=-5.0)])
  xv = np.array([-1.0, -2.0, 3.0, 1.0])
  want = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
  values = _fn(f"rxd_fill_{mode}_value", [x], [out])
  np.testing.assert_allclose(finite_difference(lambda v: values(v), xv), want, atol=1e-12)
  _assert_three_ways(_three_ways(f"rxd_fill_{mode}", out, x, mode)(xv), want)


@pytest.mark.parametrize(
  "mode",
  ["split", "first"],
)
def test_a_nan_in_one_bin_leaves_the_other_bins_derivatives_exact(mode: str) -> None:
  """NaN propagates within its bin. The other bins' derivatives, and every entry the sparsity pattern
  calls a structural zero, stay exact."""
  x = sc.sym("x", 5)
  ids = np.array([0, 0, 1, 1, 2])
  out = sc.segment_max(x, ids, 3)
  got = _three_ways(f"rxd_nan_{mode}", out, x, mode)(np.array([np.nan, -2.0, 3.0, 1.0, 4.0]))
  structural_zero = ~np.eye(3, dtype=bool)[ids].T
  for jac in got:
    np.testing.assert_array_equal(jac[1:], [[0, 0, 1, 0, 0], [0, 0, 0, 0, 1]])
    np.testing.assert_array_equal(jac[structural_zero], 0.0)


def test_error_mode_refuses_every_extremum_when_the_derivative_is_built() -> None:
  x, p = sc.sym("x", 4), sc.sym("p", 4)
  ids = [0, 1, 0, 1]
  builds: list[tuple[str, Callable[[sc.Expr], sc.Expr]]] = [
    ("max", sc.reduce_max),
    ("min", lambda v: v.min()),
    ("max", sc.norm_inf),
    ("segment_max", lambda v: sc.segment_max(v, ids, 3)),
    ("segment_min", lambda v: sc.segment_min(v, ids, 2, fill=0.0)),
  ]
  with sc.options(nonsmooth="error"):
    for op, build in builds:
      out = build(x)  # the value itself is fine
      for derive in (
        lambda: sc.gradient(out.sum(), x),
        lambda: sc.jacobian(out, x),
        lambda: sc.jvp(out, x, sc.sym("t", 4)),
        lambda: sc.vjp((out,), (x,), (sc.sym("w", out.shape),)),
        lambda: sc.sparse_jacobian(out.reshape((out.size,)), x),
      ):
        with pytest.raises(NotImplementedError, match=rf"^derivative of nonsmooth op '{op}' refused under sc.options\(nonsmooth='error'\)$"):
          derive()
      sc.gradient(build(p).sum() * x.sum(), x)  # a term that does not depend on x is never differentiated
    sc.gradient(sc.norm_1(x), x)  # abs has one convention, abs'(0) = 0


def test_the_convention_is_fixed_when_a_derived_function_is_built() -> None:
  """A derived Function built under ``"first"`` keeps it, and one built under the default keeps
  ``"split"``, even when compiled and called inside ``"error"``."""
  x = sc.sym("x", 4)
  f = _fn("rxd_baked", [x], [x.max() + sc.segment_min(x, [1, 0, 1, 0], 2).sum()])
  with sc.options(nonsmooth="first"):
    g_first = sc.gradient(f, "out0", "x")
  g_split = sc.gradient(f, "out0", "x")
  tie = np.array([2.0, 1.0, 2.0, 1.0])
  with sc.options(nonsmooth="error"):
    np.testing.assert_array_equal(g_first(tie), [1.0 + 1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(g_split(tie), [0.5 + 0.5, 0.5, 0.5 + 0.5, 0.5])
    with pytest.raises(NotImplementedError, match="nonsmooth='error'"):
      sc.gradient(f, "out0", "x")


ROWS = np.array([[2.0, 2.0, 1.0, 1.0], [5.0, -1.0, 5.0, 5.0], [-0.0, 0.0, -3.0, 0.0]])


def _body_grad(row: np.ndarray, mode: str) -> np.ndarray:
  """d/du of ``max(u) + sum(segment_min(u, [0, 1, 0, 1], 2))``."""
  index = np.arange(4)
  return _extremum_row(row, index, "max", 4, mode) + _segment_rows(row, index, np.array([0, 1, 0, 1]), 2, "min", 4, mode).sum(axis=0)


@pytest.mark.parametrize("mode", MODES)
def test_tie_conventions_hold_inside_call_vmap_and_scan_bodies(mode: str) -> None:
  u, c = sc.sym("u", 4), sc.sym("c", 1)
  body = u.max() + sc.segment_min(u, [0, 1, 0, 1], 2).sum()
  inner, step = _fn(f"rxd_body_{mode}", [u], [body]), _fn(f"rxd_step_{mode}", [c, u], [c + body])
  flat = sc.sym("flat", 12)
  mapped = sc.vmap(inner, 3, [(flat, 0, 4)]).sum()
  called = sc.stack([inner(flat[4 * i : 4 * i + 4]) for i in range(3)]).sum()
  (final,) = sc.scan(step, sc.const(np.zeros(1)), [(flat, 0, 4)], length=3)
  with sc.options(nonsmooth=mode):
    outs = [sc.gradient(e, flat) for e in (mapped, called, final.sum())] + [
      sc.jacobian(e.reshape((1,)), flat).reshape((12,)) for e in (mapped, called)
    ]
  want = np.concatenate([_body_grad(r, mode) for r in ROWS])
  for got in _fn(f"rxd_maps_{mode}", [flat], outs)(ROWS.reshape(-1)):
    np.testing.assert_array_equal(got, want)


def _mapped_and_scanned(tag: str) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  u, c = sc.sym("u", 4), sc.sym("c", 1)
  inner, step = _fn(f"rxd_{tag}_body", [u], [u.max()]), _fn(f"rxd_{tag}_step", [c, u], [c + u.max()])
  flat = sc.sym("flat", 4)
  (final,) = sc.scan(step, sc.const(np.zeros(1)), [(flat, 0, 4)], length=1)
  return flat, sc.vmap(inner, 1, [(flat, 0, 4)]).sum(), final.sum()


def test_one_callee_differentiated_under_two_conventions_keeps_each() -> None:
  flat, mapped, scanned = _mapped_and_scanned("twice")
  outs = []
  for mode in MODES:
    with sc.options(nonsmooth=mode):
      outs += [sc.gradient(mapped, flat), sc.gradient(scanned, flat), sc.jacobian(mapped.reshape((1,)), flat).reshape((4,))]
  got = _fn("rxd_twice", [flat], outs)(ROWS[0])
  np.testing.assert_array_equal(got, [_extremum_row(ROWS[0], np.arange(4), "max", 4, mode) for mode in MODES for _ in range(3)])


def test_error_mode_refuses_a_callee_already_differentiated_under_split() -> None:
  flat, mapped, scanned = _mapped_and_scanned("refused")
  derivatives = (lambda: sc.gradient(mapped, flat), lambda: sc.gradient(scanned, flat), lambda: sc.jvp(mapped, flat, sc.sym("t", 4)))
  for derive in derivatives:
    derive()
  with sc.options(nonsmooth="error"):
    for derive in derivatives:
      with pytest.raises(NotImplementedError, match="nonsmooth='error'"):
        derive()


def test_segment_extremum_sparsity_is_the_segment_structure() -> None:
  """Each bin depends on its own values only, whatever the operand's layout; an empty bin on none. The
  gradient of a piecewise-linear extremum depends on nothing, so the Hessian pattern is empty."""
  x = sc.sym("x", (3, 3))
  ids = np.array([[3, 1, 3], [0, 3, 1], [4, 4, 0]])
  index = np.arange(9).reshape(3, 3)

  def pattern(values_index: np.ndarray) -> np.ndarray:
    out = np.zeros((6, 9), dtype=bool)
    out[ids.reshape(-1), values_index.reshape(-1)] = True
    return out

  def dense(sp: sc.SparsityType) -> np.ndarray:
    out = np.zeros(sp.shape, dtype=bool)
    out[list(sp.rows), list(sp.cols)] = True
    return out

  for build in (sc.segment_max, sc.segment_min):
    np.testing.assert_array_equal(dense(sc.jacobian_sparsity(build(x, ids, 6), x)), pattern(index))
    np.testing.assert_array_equal(dense(sc.jacobian_sparsity(build(x.T, ids, 6), x)), pattern(index.T))
    np.testing.assert_array_equal(dense(sc.jacobian_sparsity(build(x * x.T, ids, 6), x)), pattern(index) | pattern(index.T))
    assert sc.jacobian_sparsity(sc.gradient(x.max() + sc.norm_inf(x) + build(x, ids, 6).sum(), x).reshape((9,)), x).nnz == 0
  tie = np.array([[7.0, 2.0, 7.0], [5.0, 7.0, -1.0], [-0.0, 0.0, 5.0]])
  out = sc.segment_max(x, ids, 6, fill=0.0).reshape((6,))
  fun = _fn("rxd_sparse_seg", [x], [sc.sparse_jacobian(out, x).to_dense(), sc.jacobian(out, x)])
  for jac in fun(tie):
    np.testing.assert_array_equal(jac, _segment_rows(tie, index, ids, 6, "max", 9, "split"))


@pytest.mark.parametrize("mode", ["split", "first"])
def test_a_nan_extremum_gives_no_entry_a_share(mode: str) -> None:
  """No entry equals a NaN result, so none gets any of its derivative: zero, not 0 / 0, under either
  convention, in reverse and in forward mode, and the finite terms beside it keep theirs."""
  x = sc.sym("x", 4)
  with sc.options(nonsmooth=mode):
    extrema = x.max() + sc.segment_min(x, [0, 1, 1, 0], 2).sum()
    outs = [sc.gradient(extrema + (x * x).sum(), x), sc.gradient(extrema, x), sc.jacobian(extrema.reshape((1,)), x)]
  g, g_extrema, j = _fn(f"rxd_nan_share_{mode}", [x], outs)(np.array([1.0, np.nan, 3.0, -2.0]))
  np.testing.assert_array_equal(g, [2.0, np.nan, 6.0, -4.0 + 1.0])
  np.testing.assert_array_equal(g_extrema, [0.0, 0.0, 0.0, 1.0])  # the max is NaN; the finite bin's min is x[3]
  np.testing.assert_array_equal(j.reshape(-1), g_extrema)
