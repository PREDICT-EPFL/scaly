"""``ragged_add`` and ``ragged_dot``: loops of run-time length over fixed index maps, checked against
NumPy, in every derivative mode, in sparsity, and in the in-place proof, whose interval test must
agree with an entry-by-entry comparison."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.derivatives import gradient, hessian, jacobian
from scaly.ad.forward import jvp
from scaly.codegen import render_c_module
from scaly.ir.expr import ragged_add, ragged_dot
from scaly.passes import lowering
from scaly.passes.lowering import _may_overlap, _Ragged

RNG = np.random.default_rng(808)
DST = np.array([5, 0, 3, 3, 1, 2, 6, 6])
SRC = np.array([4, 3, 2, 1, 0, 4, 1, 2])
LO, HI = np.array([0, 2, 4, 7, 3]), np.array([2, 2, 7, 8, 5])  # an empty group, overlapping groups


def _fn(name, inputs, outputs):
  return sc.Function._from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _numpy(base, src, scale, dmap=DST, smap=SRC):
  out, dots = base.copy(), np.zeros(LO.size)
  for g, (lo, hi) in enumerate(zip(LO, HI, strict=True)):
    for p in range(lo, hi):
      d = p if dmap is None else dmap[p]
      q = p if smap is None else smap[p]
      out[d] += src[q] * scale[g]
      dots[g] += src[q] * base[d]
  return out, dots


def _parts(dmap=DST, smap=SRC):
  base, src, scale = sc.sym("base", 8), sc.sym("src", 8), sc.sym("scale", LO.size)
  lo, hi = sc.sym("lo", LO.size, dtype="int64"), sc.sym("hi", LO.size, dtype="int64")
  y = ragged_add(base, src, lo, hi, scale, dst_map=dmap, src_map=smap)
  d = ragged_dot(src, base, lo, hi, a_map=smap, b_map=dmap)
  return (base, src, scale, lo, hi), y, d


POINT = (RNG.standard_normal(8), RNG.standard_normal(8), RNG.standard_normal(LO.size), LO.astype(float), HI.astype(float))


@pytest.mark.parametrize("maps", ["both", "identity"])
def test_values(maps: str) -> None:
  dmap, smap = (DST, SRC) if maps == "both" else (None, None)
  inputs, y, d = _parts(dmap, smap)
  got = _fn(f"rv_{maps}", inputs, [y, d])._flat_numerical_call(*POINT)
  want = _numpy(*POINT[:3], dmap, smap)
  np.testing.assert_allclose(got[0], want[0], rtol=1e-13, atol=1e-15)  # a fused multiply-add rounds once
  np.testing.assert_allclose(got[1], want[1], rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_derivatives(monkeypatch: pytest.MonkeyPatch, which: int) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  inputs, y, d = _parts()
  f = sc.sumsqr(y) + (d * d.sin()).sum() + (y[:5] * d).sum()
  wrt = inputs[which]
  fn = _fn(f"rd{which}", inputs, [f, gradient(f, wrt), jacobian(f.reshape((1,)), wrt), hessian(f, wrt)])
  _, g, j, h = fn._flat_numerical_call(*POINT)

  def value(z: np.ndarray) -> np.ndarray:
    pt = list(POINT)
    pt[which] = z
    return fn._flat_numerical_call(*pt)[0].reshape(1)

  np.testing.assert_allclose(g, finite_difference(value, POINT[which]).reshape(-1), rtol=1e-6, atol=1e-8)
  np.testing.assert_allclose(j.reshape(-1), g, rtol=1e-12, atol=1e-14)
  grad = _fn(f"rdg{which}", inputs, [gradient(f, wrt)])

  def gvalue(z: np.ndarray) -> np.ndarray:
    pt = list(POINT)
    pt[which] = z
    return grad._flat_numerical_call(*pt)[0].reshape(-1)

  np.testing.assert_allclose(h, finite_difference(gvalue, POINT[which]), rtol=1e-6, atol=1e-7)
  tangent = RNG.standard_normal(POINT[which].shape)
  one = _fn(f"rdt{which}", inputs, [jvp(f, wrt, sc.const(tangent))])._flat_numerical_call(*POINT)[0]
  np.testing.assert_allclose(one, g @ tangent, rtol=1e-12, atol=1e-14)


def test_sparsity_is_conservative() -> None:
  inputs, y, d = _parts()
  base, src, scale = inputs[:3]
  for out in (y, d):
    for wrt in (base, src, scale):
      mask = sc.jacobian_sparsity(out, wrt).to_mask()
      jac = _fn("rsp", inputs, [jacobian(out, wrt)])._flat_numerical_call(*POINT)[0]
      assert not np.any((jac != 0) & ~mask)


def test_the_inner_loop_has_run_time_bounds() -> None:
  inputs, y, _ = _parts()
  src = str(render_c_module(_fn("rc", inputs, [y])).source)
  assert "for (long long rp_" in src and "< arg[4]" not in src


def _positions(lo, hi, table):
  return _Ragged(np.asarray(lo).reshape(1, -1), np.asarray(hi).reshape(1, -1), None if table is None else np.asarray(table))


@pytest.mark.parametrize(
  ("a", "b", "overlap"),
  [
    (([0], [3], None), ([3], [5], None), False),  # adjacent ranges
    (([0], [3], None), ([2], [5], None), True),
    (([0, 5], [2, 7], None), ([2], [5], None), False),  # intervals meet, entries do not
    (([0], [2], [10, 11, 12]), ([0], [3], None), False),  # a mapped range far away
    (([0], [2], [1, 4, 9]), ([4], [5], None), True),
    (([1], [1], None), ([0], [9], None), False),  # an empty range
  ],
)
def test_interval_test_agrees_with_enumeration(a, b, overlap: bool) -> None:
  x, y = _positions(*a), _positions(*b)
  assert _may_overlap(x, y) is overlap
  assert _may_overlap(x, y.explicit()) is overlap
  assert _may_overlap(x.explicit(), y.explicit()) is overlap


def test_ragged_updates_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
  """A carry ``[x | y]`` where each step adds a run of ``x`` into ``y``: in place, and the same
  numbers as the two-slot loop. Reading a run the step writes keeps two slots."""
  n, steps = 6, 4
  lo_t, hi_t = np.array([[0], [1], [2], [3]]), np.array([[3], [4], [5], [6]])

  def build(tag: str, *, clash: bool) -> sc.Function:
    c, lo, hi, w = sc.sym("c", 2 * n), sc.sym("lo", 1, dtype="int64"), sc.sym("hi", 1, dtype="int64"), sc.sym("w", 1)
    dmap = np.arange(n) + (0 if clash else n)
    body = sc.Function._from_exprs(
      f"rip_{int(clash)}_{tag}", [c, lo, hi, w], [ragged_add(c, c, lo, hi, w, dst_map=dmap)], ["c", "lo", "hi", "w"], ["n"]
    )
    c0, ws = sc.sym("c0", 2 * n), sc.sym("ws", steps)
    tables = [(sc.const(lo_t.reshape(-1), dtype="int64"), 0, 1), (sc.const(hi_t.reshape(-1), dtype="int64"), 0, 1), (ws, 0, 1)]
    (fin,) = sc.scan(body, c0, tables, length=steps)
    return _fn(f"rip_run_{int(clash)}_{tag}", [c0, ws], [fin])

  c0, ws = RNG.standard_normal(2 * n), RNG.standard_normal(steps)
  for clash in (False, True):
    fn = build("ip", clash=clash)
    assert ("_inplace_raw" in str(render_c_module(fn).source)) is not clash
    got = fn._flat_numerical_call(c0, ws)[0]
    monkeypatch.setattr(lowering, "DONATE_CARRIES", False)
    ref = build("two", clash=clash)._flat_numerical_call(c0, ws)[0]
    monkeypatch.setattr(lowering, "DONATE_CARRIES", True)
    np.testing.assert_array_equal(got, ref)
    want = c0.copy()
    for k in range(steps):
      old = want.copy()
      for p in range(lo_t[k, 0], hi_t[k, 0]):
        want[p + (0 if clash else n)] += old[p] * ws[k]
    np.testing.assert_allclose(got, want, rtol=1e-14)


def test_validation() -> None:
  v, lo = sc.sym("v", 4), sc.sym("lo", 2, dtype="int64")
  with pytest.raises(TypeError, match="int64"):
    ragged_add(v, v, sc.sym("flo", 2), lo, sc.sym("s", 2))
  with pytest.raises(ValueError, match="one entry per group"):
    ragged_dot(v, v, lo, sc.sym("hi3", 3, dtype="int64"))
  with pytest.raises(ValueError, match="one scale per group"):
    ragged_add(v, v, lo, lo, sc.sym("s3", 3))
  with pytest.raises(ValueError, match="non-negative"):
    ragged_add(v, v, lo, lo, sc.sym("s", 2), dst_map=[-1, 0])
