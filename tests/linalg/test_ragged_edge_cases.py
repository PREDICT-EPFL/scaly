"""Edge cases of ``ragged_add`` and ``ragged_dot``, loops of run-time length over fixed index maps.

Group layouts at the edges (no groups at all, every group empty, one group over everything, groups
in reverse order, groups overlapping onto one destination), maps left as ``None`` or spelled out,
source and base being one array, non-finite scales, range lengths around the four-way blocked sum,
refused inputs, derivatives in every mode, and scans whose ranges come from a table sliced by the
step number, which must agree bit for bit whether the carry is updated in place or kept in two
slots. Values are checked against NumPy loops, exactly wherever the arithmetic is exact (the data
are multiples of 1/8 for that reason).
"""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.derivatives import gradient, hessian, jacobian
from scaly.codegen import render_c_module
from scaly.ir.expr import Expr
from scaly.linalg.ops import ragged_add, ragged_dot
from scaly.linalg.ops.ragged import RAGGED_ADD, RAGGED_DOT
from scaly.ir.types import TensorType
from scaly.passes import lowering

DST = np.array([3, 3, 0, 3, 7, 7, 1, 3])  # repeated destinations
SRC = np.array([7, 0, 0, 5, 2, 2, 6, 1])
LAYOUTS = {
  "none": ([], []),
  "all_empty": ([0, 3, 8, 5], [0, 3, 8, 5]),
  "whole": ([0], [8]),
  "reversed": ([6, 3, 0], [8, 6, 3]),
  "overlapping": ([0, 2, 2, 7], [5, 8, 4, 8]),
}


def _fn(name, inputs, outputs):
  return sc.Function.from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _eighths(shape, seed: int) -> np.ndarray:
  """Small multiples of 1/8: the few sums and products the tests form stay exact in any order."""
  return np.random.default_rng(seed).integers(-24, 25, size=shape) / 8.0


def _np_add(base, src, lo, hi, scale, dst_map=None, src_map=None) -> np.ndarray:
  out = base.copy()
  for g, (a, b) in enumerate(zip(lo, hi, strict=True)):
    for p in range(a, b):
      out[p if dst_map is None else dst_map[p]] += src[p if src_map is None else src_map[p]] * scale[g]
  return out


def _np_dot(a, b, lo, hi, a_map=None, b_map=None) -> np.ndarray:
  out = np.zeros(len(lo))
  for g, (s, e) in enumerate(zip(lo, hi, strict=True)):
    for p in range(s, e):
      out[g] += a[p if a_map is None else a_map[p]] * b[p if b_map is None else b_map[p]]
  return out


def _bounds(name: str, groups: int) -> tuple[Expr, Expr]:
  return sc.sym(f"lo_{name}", groups, dtype="int64"), sc.sym(f"hi_{name}", groups, dtype="int64")


def test_group_layouts_match_numpy() -> None:
  """Each layout with the maps, with ``None`` maps and with the identity spelled out (the same bits
  as ``None``), and with the source (or both dot operands) the base itself, which every lane reads
  as it was before the update."""
  base, src = sc.sym("base", 8), sc.sym("src", 8)
  inputs, point, outs, want = [base, src], [_eighths(8, 1), _eighths(8, 2)], [], []
  bv, sv = point
  eye = np.arange(8)
  for k, (name, (lo_v, hi_v)) in enumerate(LAYOUTS.items()):
    lo, hi = _bounds(name, len(lo_v))
    scale = sc.sym(f"s_{name}", len(lo_v))
    wv = _eighths(len(lo_v), 10 + k)
    inputs += [lo, hi, scale]
    point += [np.array(lo_v, dtype=float), np.array(hi_v, dtype=float), wv]
    outs += [
      ragged_add(base, src, lo, hi, scale, dst_map=DST, src_map=SRC),
      ragged_add(base, src, lo, hi, scale),
      ragged_add(base, src, lo, hi, scale, dst_map=eye, src_map=eye),
      ragged_add(base, base, lo, hi, scale, dst_map=DST, src_map=SRC),
      ragged_dot(src, base, lo, hi, a_map=SRC, b_map=DST),
      ragged_dot(src, base, lo, hi),
      ragged_dot(src, base, lo, hi, a_map=eye, b_map=eye),
      ragged_dot(base, base, lo, hi, a_map=SRC, b_map=DST),
    ]
    want += [
      _np_add(bv, sv, lo_v, hi_v, wv, DST, SRC),
      _np_add(bv, sv, lo_v, hi_v, wv),
      _np_add(bv, sv, lo_v, hi_v, wv),
      _np_add(bv, bv, lo_v, hi_v, wv, DST, SRC),
      _np_dot(sv, bv, lo_v, hi_v, SRC, DST),
      _np_dot(sv, bv, lo_v, hi_v),
      _np_dot(sv, bv, lo_v, hi_v),
      _np_dot(bv, bv, lo_v, hi_v, SRC, DST),
    ]
  got = _fn("rge_layouts", inputs, outs)._flat_numerical_call(*point)
  for g, w in zip(got, want, strict=True):
    np.testing.assert_array_equal(g, w)


def test_range_lengths_around_the_blocked_sum() -> None:
  """``ragged_dot`` sums four partial sums over blocks of four and then a tail: every length 0 to 9
  at every offset 0 to 3, so each block and tail boundary is crossed from each residue."""
  lo_v = np.array([off for off in range(4) for _ in range(10)])
  hi_v = lo_v + np.tile(np.arange(10), 4)
  a, b = sc.sym("a", 13), sc.sym("b", 13)
  lo, hi = _bounds("len", lo_v.size)
  rev = np.arange(13)[::-1].copy()
  outs = [ragged_dot(a, b, lo, hi), ragged_dot(a, b, lo, hi, a_map=rev), ragged_dot(a, a, lo, hi, b_map=rev)]
  av, bv = _eighths(13, 3), _eighths(13, 4)
  got = _fn("rge_len", [a, b, lo, hi], outs)._flat_numerical_call(av, bv, lo_v, hi_v)
  np.testing.assert_array_equal(got[0], _np_dot(av, bv, lo_v, hi_v))
  np.testing.assert_array_equal(got[1], _np_dot(av, bv, lo_v, hi_v, a_map=rev))
  np.testing.assert_array_equal(got[2], _np_dot(av, av, lo_v, hi_v, b_map=rev))
  assert np.all(got[0][::10] == 0.0)  # the empty ranges


def test_non_finite_scales_and_entries() -> None:
  """A zero scale times an infinite source is NaN, as in IEEE arithmetic; a NaN or infinite scale on
  an empty group, and a NaN entry outside every range, change nothing."""
  lo_v, hi_v = [0, 2, 4, 4, 6, 1], [2, 4, 4, 6, 6, 2]
  base, src, scale = sc.sym("base", 8), sc.sym("src", 8), sc.sym("scale", 6)
  lo, hi = _bounds("nf", 6)
  outs = [ragged_add(base, src, lo, hi, scale), ragged_add(base, src, lo, hi, scale, dst_map=DST, src_map=SRC), ragged_dot(base, base, lo, hi)]
  bv, sv = _eighths(8, 5), np.array([0.5, np.inf, 1.25, -2.0, -np.inf, 0.75, 3.0, -1.5])
  bv[7] = np.nan  # outside every range of the identity maps
  wv = np.array([0.0, np.inf, np.nan, -np.inf, np.nan, 0.5])  # groups 2 and 4 are empty
  got = _fn("rge_nonfinite", [base, src, lo, hi, scale], outs)._flat_numerical_call(bv, sv, lo_v, hi_v, wv)
  with np.errstate(invalid="ignore"):
    want = [_np_add(bv, sv, lo_v, hi_v, wv), _np_add(bv, sv, lo_v, hi_v, wv, DST, SRC), _np_dot(bv, bv, lo_v, hi_v)]
  for g, w in zip(got, want, strict=True):
    np.testing.assert_array_equal(g, w)
  assert np.isnan(got[0][1]) and got[0][2] == np.inf and got[0][4] == np.inf  # inf * 0, 1.25 * inf, -inf * -inf
  assert np.all(np.isfinite(got[2]))


def test_refusals() -> None:
  v, m, lo = sc.sym("v", 4), sc.sym("m", (2, 2)), sc.sym("lo", 2, dtype="int64")
  s = sc.sym("s", 2)
  with pytest.raises(ValueError, match="src_map entries must be non-negative"):
    ragged_add(v, v, lo, lo, s, src_map=[0, -1])
  with pytest.raises(ValueError, match="a_map entries must be non-negative"):
    ragged_dot(v, v, lo, lo, a_map=[-3])
  with pytest.raises(ValueError, match="b_map entries must be non-negative"):
    ragged_dot(v, v, lo, lo, b_map=np.array([[0, 1], [2, -1]]))
  with pytest.raises(ValueError, match="needs vectors"):
    ragged_add(m, v, lo, lo, s)
  with pytest.raises(ValueError, match="needs vectors"):
    ragged_add(v, m, lo, lo, s)
  with pytest.raises(ValueError, match="needs vectors"):
    ragged_dot(v, m, lo, lo)
  with pytest.raises(ValueError, match="one scale per group"):
    ragged_add(v, v, lo, lo, sc.sym("s21", (2, 1)))
  for bad in (sc.sym("lo22", (2, 2), dtype="int64"), sc.sym("flo", 2), sc.sym("ilo", 2, dtype="int32"), sc.sym("klo", (), dtype="int64")):
    with pytest.raises(TypeError, match="rank-1 int64"):
      ragged_add(v, v, bad, lo, s)
    with pytest.raises(TypeError, match="rank-1 int64"):
      ragged_dot(v, v, lo, bad)
  with pytest.raises(TypeError, match="mixed-dtype"):
    ragged_add(sc.sym("f", 4, dtype="float32"), v, lo, lo, sc.sym("s32", 2, dtype="float32"))
  with pytest.raises(TypeError, match="mixed-dtype"):
    ragged_dot(sc.sym("f", 4, dtype="float32"), v, lo, lo)
  maps = {"dst_map": None, "src_map": None}
  for node, message in (
    (Expr(RAGGED_DOT, (v, v, lo, lo), TensorType((4,)), attrs={"a_map": None, "b_map": None}), "one value per group"),
    (Expr(RAGGED_ADD, (v, v, lo, lo, sc.sym("s3", 3)), TensorType((4,)), attrs=maps), "one scale per group"),
    (Expr(RAGGED_ADD, (v, v, sc.sym("flo", 2), lo, s), TensorType((4,)), attrs=maps), "int64 lo and hi"),
  ):
    with pytest.raises(sc.VerifyError, match=message):
      sc.verify_expr(node)


# --- derivatives ---------------------------------------------------------------------------------


@pytest.mark.parametrize("wrt", ["x", "scale"])
def test_derivatives_when_source_and_base_are_one_array(monkeypatch: pytest.MonkeyPatch, wrt: str) -> None:
  """``x`` is the base and the source, and both operands of the dot, over empty, overlapping and
  reversed groups: every path through ``x`` is differentiated, in each mode."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  lo_v, hi_v = np.array([6, 3, 0, 2, 8, 2]), np.array([8, 3, 5, 8, 8, 4])
  x, scale = sc.sym("x", 8), sc.sym("scale", 6)
  lo, hi = _bounds("al", 6)
  y = ragged_add(x, x, lo, hi, scale, dst_map=DST, src_map=SRC)
  d = ragged_dot(x, x.sin(), lo, hi, a_map=SRC, b_map=DST)
  f = sc.sumsqr(y) + (d * d.cos()).sum() + (y[:6] * d * scale).sum()
  var = {"x": x, "scale": scale}[wrt]
  inputs = [x, scale, lo, hi]
  fn = _fn(f"rge_alias_{wrt}", inputs, [gradient(f, var), jacobian(f.reshape((1,)), var), hessian(f, var)])
  xv, sv = np.linspace(-0.9, 0.8, 8), np.linspace(0.3, -0.7, 6)
  g, j, h = fn._flat_numerical_call(xv, sv, lo_v, hi_v)

  def numpy(xz: np.ndarray, sz: np.ndarray) -> np.ndarray:
    yv = _np_add(xz, xz, lo_v, hi_v, sz, DST, SRC)
    dv = _np_dot(xz, np.sin(xz), lo_v, hi_v, SRC, DST)
    return np.array([yv @ yv + dv @ np.cos(dv) + np.sum(yv[:6] * dv * sz)])

  def at(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (z, sv) if wrt == "x" else (xv, z)

  point = xv if wrt == "x" else sv
  np.testing.assert_allclose(g, finite_difference(lambda z: numpy(*at(z)), point).reshape(-1), rtol=1e-6, atol=1e-7)
  np.testing.assert_allclose(j.reshape(-1), g, rtol=1e-12, atol=1e-14)
  grad = _fn(f"rge_alias_g_{wrt}", inputs, [gradient(f, var)])
  np.testing.assert_allclose(h, finite_difference(lambda z: grad._flat_numerical_call(*at(z), lo_v, hi_v)[0], point), rtol=1e-6, atol=1e-7)


def test_derivatives_of_degenerate_layouts(monkeypatch: pytest.MonkeyPatch) -> None:
  """No groups: the output is the base. Every group empty: no derivative reaches the source or the
  scale, exactly, in either mode."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  x, src = sc.sym("x", 5), sc.sym("src", 5)
  lo0, hi0 = _bounds("zero", 0)
  loe, hie = _bounds("empty", 3)
  s0, se = sc.sym("s0", 0), sc.sym("se", 3)
  y0 = ragged_add(x, x.sin(), lo0, hi0, s0)
  ye = ragged_add(x, src * src, loe, hie, se, dst_map=[4, 3, 2, 1, 0], src_map=[0, 0, 1, 1, 2])
  de = ragged_dot(src, x, loe, hie)
  f = sc.sumsqr(y0) + sc.sumsqr(ye) + (de * de).sum() + ragged_dot(x, src, lo0, hi0).sum()
  outs = [gradient(f, x), gradient(f, src), gradient(f, se), gradient(f, s0), jacobian(ye, src), jacobian(ye, se), jacobian(de, x), jacobian(y0, x)]
  inputs = [x, src, lo0, hi0, s0, loe, hie, se]
  xv, sv = np.linspace(-1.0, 1.0, 5), np.linspace(0.5, 2.5, 5)
  empty = np.array([0.0, 3.0, 5.0])
  got = _fn("rge_degenerate", inputs, outs)._flat_numerical_call(
    xv, sv, np.zeros(0), np.zeros(0), np.zeros(0), empty, empty, np.array([np.nan, 2.0, 3.0])
  )
  np.testing.assert_array_equal(got[0], 4.0 * xv)  # 2 x from each of the two squared copies of x
  for k in (1, 2, 4, 5, 6):
    np.testing.assert_array_equal(got[k], np.zeros_like(got[k]))
  assert got[3].shape == (0,)
  np.testing.assert_array_equal(got[7], np.eye(5))


# --- scans whose ranges come from a table sliced by the step number ------------------------------

N_IP = 12
SHIFT = (np.arange(N_IP) + 6) % N_IP  # reads p + 6 (mod 12) for a write at p
STEP_GROUPS = [
  [(0, 3), (8, 8), (4, 6)],  # the empty group sits at 8, whose source entry 2 is written
  [(3, 5), (0, 4), (6, 6)],  # overlapping writes; an empty group again at a written source entry
  [(5, 6), (5, 5), (0, 0)],
  [(7, 9), (0, 0), (0, 0)],  # writes 7 and 8, which other steps read: across steps only
  [(2, 6), (1, 2), (11, 11)],
]


def _run_both(monkeypatch: pytest.MonkeyPatch, build, point, *, expect_in_place: bool):
  """Run the function built with the carry updated in place (when proven) and with two carry slots;
  the two must agree exactly, and the proof must decide as expected."""
  fn = build("ip")
  assert ("_inplace_raw" in str(render_c_module(fn).source)) == expect_in_place
  got = fn._flat_numerical_call(*point)
  monkeypatch.setattr(lowering, "DONATE_CARRIES", False)
  ref = build("two")._flat_numerical_call(*point)
  monkeypatch.setattr(lowering, "DONATE_CARRIES", True)
  for a, b in zip(got, ref, strict=True):
    np.testing.assert_array_equal(a, b)
  return got


@pytest.mark.parametrize("clash", [False, True])
def test_step_tables_decide_in_place(monkeypatch: pytest.MonkeyPatch, clash: bool) -> None:
  """``c[p] += c[p + 6] * w[g]`` over each step's groups. Empty groups, overlapping writes and entries
  shared only between different steps leave the carry in place. With ``clash`` one step also runs
  ``p = 11``, writing 11 and reading 5 while another group of that step writes 5: two slots."""
  groups = [list(g) for g in STEP_GROUPS]
  if clash:
    groups[2][1] = (11, 12)
  lo_t, hi_t = (np.array([[r[i] for r in step] for step in groups]) for i in (0, 1))
  steps, width = lo_t.shape
  c, lo, hi, w = sc.sym("c", N_IP), *_bounds("st", width), sc.sym("w", width)

  def build(tag: str) -> sc.Function:
    body = _fn(f"rge_st_{int(clash)}_body_{tag}", [c, lo, hi, w], [ragged_add(c, c, lo, hi, w, src_map=SHIFT)])
    c0, ws = sc.sym("c0", N_IP), sc.sym("ws", steps * width)
    tables = [(sc.const(t.reshape(-1), dtype="int64"), 0, width) for t in (lo_t, hi_t)]
    (fin,) = sc.scan(body, c0, [*tables, (ws, 0, width)], length=steps)
    return _fn(f"rge_st_{int(clash)}_{tag}", [c0, ws], [fin])

  c0, ws = _eighths(N_IP, 7), _eighths(steps * width, 8)
  (got,) = _run_both(monkeypatch, build, (c0, ws), expect_in_place=not clash)
  want = c0.copy()
  for k in range(steps):
    want = _np_add(want, want.copy(), lo_t[k], hi_t[k], ws[k * width : (k + 1) * width], src_map=SHIFT)
  np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("clash", [False, True])
def test_mapped_writes_are_bounded_by_their_map(monkeypatch: pytest.MonkeyPatch, clash: bool) -> None:
  """The first update writes ``dst_map[p]`` for ``p`` in ``[0, 3)``; the second reads the carry as the
  step began at ``[9, 12)``. With ``clash`` the map sends the first update's writes to 9, 10 and 11,
  far from the range ``p`` itself covers, and the carry keeps two slots."""
  dst1 = np.r_[[9, 10, 11] if clash else [0, 1, 2], np.zeros(N_IP - 3, dtype=np.int64)]
  dst2 = np.r_[np.zeros(9, dtype=np.int64), [3, 4, 5]]
  c, x, w = sc.sym("c", N_IP), sc.sym("x", N_IP), sc.sym("w", 2)
  lo1, hi1, lo2, hi2 = (sc.const(np.array([v]), dtype="int64") for v in (0, 3, 9, 12))

  def build(tag: str) -> sc.Function:
    u1 = ragged_add(c, x, lo1, hi1, w[:1], dst_map=dst1)
    body = _fn(f"rge_map_{int(clash)}_body_{tag}", [c, x, w], [ragged_add(u1, c, lo2, hi2, w[1:], dst_map=dst2)])
    c0, xs, ws = sc.sym("c0", N_IP), sc.sym("xs", N_IP), sc.sym("ws", 6)
    (fin,) = sc.scan(body, c0, [(xs, 0, 0), (ws, 0, 2)], length=3)
    return _fn(f"rge_map_{int(clash)}_{tag}", [c0, xs, ws], [fin])

  c0, xs, ws = _eighths(N_IP, 9), _eighths(N_IP, 10), _eighths(6, 11)
  (got,) = _run_both(monkeypatch, build, (c0, xs, ws), expect_in_place=not clash)
  want = c0.copy()
  for k in range(3):
    old = want.copy()
    want = _np_add(want, xs, [0], [3], ws[2 * k : 2 * k + 1], dst_map=dst1)
    want = _np_add(want, old, [9], [12], ws[2 * k + 1 : 2 * k + 2], dst_map=dst2)
  np.testing.assert_array_equal(got, want)


def test_derivatives_through_a_scan_of_step_tables(monkeypatch: pytest.MonkeyPatch) -> None:
  """Reverse mode stores every carry, forward mode carries the tangents; both agree with finite
  differences across empty groups and overlapping writes."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  lo_t, hi_t = (np.array([[r[i] for r in step] for step in STEP_GROUPS]) for i in (0, 1))
  steps, width = lo_t.shape
  c, lo, hi, w, x = sc.sym("c", N_IP), *_bounds("sd", width), sc.sym("w", width), sc.sym("x", N_IP)
  body = _fn("rge_sd_body", [c, lo, hi, w, x], [ragged_add(c, x * c.sin(), lo, hi, w, src_map=SHIFT)])
  c0, ws, xs = sc.sym("c0", N_IP), sc.sym("ws", steps * width), sc.sym("xs", N_IP)
  tables = [(sc.const(t.reshape(-1), dtype="int64"), 0, width) for t in (lo_t, hi_t)]
  (fin,) = sc.scan(body, c0, [*tables, (ws, 0, width), (xs, 0, 0)], length=steps)
  f = sc.sumsqr(fin) + ragged_dot(fin, xs, sc.const([0, 4], dtype="int64"), sc.const([3, 12], dtype="int64")).sum()
  fn = _fn("rge_sd", [c0, ws, xs], [gradient(f, c0), gradient(f, ws), jacobian(f.reshape((1,)), ws), hessian(f, ws)])
  cv, wv, xv = np.linspace(-1.0, 1.0, N_IP), np.linspace(0.2, 0.9, steps * width), np.linspace(0.5, 1.5, N_IP)

  def numpy(cz: np.ndarray, wz: np.ndarray) -> np.ndarray:
    cc = cz.copy()
    for k in range(steps):
      cc = _np_add(cc, xv * np.sin(cc), lo_t[k], hi_t[k], wz[k * width : (k + 1) * width], src_map=SHIFT)
    return np.array([cc @ cc + _np_dot(cc, xv, [0, 4], [3, 12]).sum()])

  gc, gw, jw, hw = fn._flat_numerical_call(cv, wv, xv)
  np.testing.assert_allclose(gc, finite_difference(lambda z: numpy(z, wv), cv).reshape(-1), rtol=1e-6, atol=1e-7)
  np.testing.assert_allclose(gw, finite_difference(lambda z: numpy(cv, z), wv).reshape(-1), rtol=1e-6, atol=1e-7)
  np.testing.assert_allclose(jw.reshape(-1), gw, rtol=1e-12, atol=1e-13)
  grad = _fn("rge_sd_g", [c0, ws, xs], [gradient(f, ws)])
  np.testing.assert_allclose(hw, finite_difference(lambda z: grad._flat_numerical_call(cv, z, xv)[0], wv), rtol=1e-6, atol=1e-7)
  assert np.all(gw.reshape(steps, width)[3, 1:] == 0.0)  # empty groups get no derivative


def test_a_loop_body_with_no_groups() -> None:
  c = sc.sym("c", 4)
  none = sc.const(np.zeros(0, dtype=np.int64), dtype="int64")
  body = _fn("rge_none_body", [c], [ragged_add(c, c, none, none, sc.const(np.zeros(0)))])
  c0 = sc.sym("c0", 4)
  (fin,) = sc.scan(body, c0, [], length=3)
  np.testing.assert_array_equal(_fn("rge_none", [c0], [fin])._flat_numerical_call(np.arange(4.0))[0], np.arange(4.0))


def test_a_while_loop_of_no_iterations_with_ranges_at_the_step_number() -> None:
  c, k, cc = sc.sym("c", 6), sc.sym("k", (), dtype="int64"), sc.sym("cc", 6)
  body = _fn("rge_w0_body", [c, k], [ragged_add(c, c, k.reshape((1,)), (k + 2).reshape((1,)), sc.const([0.5]), dst_map=np.arange(6) % 3 + 3)])
  cond = _fn("rge_w0_cond", [cc], [sc.less(cc.sum(), 100.0)])
  c0 = sc.sym("c0", 6)
  fin, count = sc.while_loop(cond, body, c0, max_iter=0, index=True)
  got, steps = _fn("rge_w0", [c0], [fin, count])._flat_numerical_call(np.arange(6.0))
  np.testing.assert_array_equal(got, np.arange(6.0))
  assert steps == 0
