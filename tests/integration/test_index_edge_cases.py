"""Edge cases of the indexed updates: ``index_add``/``index_set`` at indices fixed when the graph is
built, and ``take``/``put_add``/``put`` at ``int64`` indices known only at run time.

Indices at and past both ends of the axis (up to ``2**62`` and the ``int64`` extremes), every lane
padded, empty and complete index lists, repeated indices, leading axes, ``float32``, indices cast
from floats, updates whose values read the array they update, and refused inputs. Derivatives with
repeated and padded lanes go through every mode. Loops (``scan``, ``while_loop``, ``vmap``) slice
their index tables by step number and must agree bit for bit whether the carry is updated in place
or kept in two slots. Values are checked against NumPy, exactly wherever the arithmetic is exact
(the data are multiples of 1/8 for that reason).
"""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.derivatives import gradient, hessian, jacobian
from scaly.codegen import render_c_module
from scaly.ir.expr import Expr, ExprOp
from scaly.ir.types import TensorType
from scaly.passes import lowering
from scaly.passes.lowering import _normalize_function, _put_scratch, lower_function, main_proc, update_chain

I64 = np.iinfo(np.int64)


def _fn(name, inputs, outputs):
  return sc.Function.from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _eighths(shape, seed: int) -> np.ndarray:
  """Small multiples of 1/8: the few sums and products the tests form stay exact in any order."""
  return np.random.default_rng(seed).integers(-24, 25, size=shape) / 8.0


def _normal(*shapes, seed: int) -> list[np.ndarray]:
  rng = np.random.default_rng(seed)
  return [rng.standard_normal(shape) for shape in shapes]


def _np_take(x: np.ndarray, idx: np.ndarray, fill: float = 0.0) -> np.ndarray:
  out = np.full((*x.shape[:-1], idx.size), fill)
  for j, i in enumerate(idx):
    if 0 <= i < x.shape[-1]:
      out[..., j] = x[..., i]
  return out


def _np_put(base: np.ndarray, idx: np.ndarray, values: np.ndarray, *, add: bool) -> np.ndarray:
  out = base.copy()
  for j, i in enumerate(idx):
    if 0 <= i < base.shape[-1]:
      out[..., i] = out[..., i] + values[..., j] if add else values[..., j]
  return out


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


# --- index_add / index_set: indices fixed when the graph is built --------------------------------


def _index_update_cases(x: Expr, m: Expr, v: Expr) -> list[Expr]:
  return [
    sc.index_add(x, [2, 2, 2, 0], v),  # a repeated index accumulates in order
    sc.index_add(x, [], np.zeros(0)),
    sc.index_set(x, [], np.zeros(0)),
    sc.index_add(x, [5, 4, 3, 2, 1, 0], x),  # the values are the base itself: every read sees the old entry
    sc.index_set(x, [5, 4, 3, 2, 1, 0], x),
    sc.index_set(x, [0, 5], v[:2]),
    sc.index_add(m, [[5, 0], [3, 5]], v.reshape((2, 2))),  # flat indices into a matrix, given as a matrix
    sc.index_add(x, [3], 2.5),
    sc.index_set(x, [1], np.array(1.5)),
    sc.index_add(x, np.array([[1, 2], [4, 1]]), np.array([[0.5, 1.0], [2.0, 4.0]])),
    sc.index_add(x, [4, 1, 4, 4], v) - (x + sc.scatter(v, [4, 1, 4, 4], (6,))),
  ]


def _index_update_numpy(xv: np.ndarray, mv: np.ndarray, vv: np.ndarray) -> list[np.ndarray]:
  def added(base, idx, values):
    out = base.copy().reshape(-1)
    np.add.at(out, np.asarray(idx).reshape(-1), np.asarray(values).reshape(-1))
    return out.reshape(base.shape)

  def assigned(base, idx, values):
    out = base.copy()
    out[idx] = values
    return out

  return [
    added(xv, [2, 2, 2, 0], vv),
    xv,
    xv,
    xv + xv[::-1],
    xv[::-1],
    assigned(xv, [0, 5], vv[:2]),
    added(mv, [5, 0, 3, 5], vv),
    added(xv, [3], [2.5]),
    assigned(xv, [1], 1.5),
    added(xv, [1, 2, 4, 1], [0.5, 1.0, 2.0, 4.0]),
    np.zeros(6),
  ]


@pytest.mark.parametrize("hint", ["scalar", "block"])
def test_index_updates_match_numpy(hint: str) -> None:
  """Every case under both lowering hints: the scalar expansion resolves each fixed index when the
  code is generated, the block code keeps the loops."""
  x, m, v = sc.sym("x", 6), sc.sym("m", (2, 3)), sc.sym("v", 4)
  outs = [e.scalar() if hint == "scalar" else e.block() for e in _index_update_cases(x, m, v)]
  fn = _fn(f"ie_idx_{hint}", [x, m, v], outs)
  if hint == "scalar":
    assert main_proc(lower_function(fn)).attrs["scalarize_mode"] == "procedure"
  point = (_eighths(6, 1), _eighths((2, 3), 2), _eighths(4, 3))
  for got, want in zip(fn._flat_numerical_call(*point), _index_update_numpy(*point), strict=True):
    np.testing.assert_array_equal(got, want)


def test_index_update_refusals() -> None:
  x = sc.sym("x", 4)
  with pytest.raises(IndexError, match=r"\[0, 4\)"):
    sc.index_add(x, [4], [1.0])
  with pytest.raises(IndexError, match=r"\[0, 4\)"):
    sc.index_set(x, [-1], [1.0])
  with pytest.raises(IndexError, match=r"\[0, 4\)"):
    sc.index_add(sc.sym("m", (2, 2)), [[0, 4]], [1.0, 2.0])  # flat: a matrix of 4 entries
  with pytest.raises(ValueError, match="distinct"):
    sc.index_set(x, [[1], [1]], [1.0, 2.0])
  with pytest.raises(ValueError, match="2 indices for 1 values"):
    sc.index_add(x, [0, 1], 3.0)
  with pytest.raises(ValueError, match="0 indices for 1 values"):
    sc.index_set(x, [], 3.0)
  with pytest.raises(TypeError, match="mixed-dtype"):
    sc.index_add(sc.sym("f", 4, dtype="float32"), [0], sc.sym("d", 1))
  v2 = sc.sym("v2", 2)
  for op, idx, values, message in (
    (ExprOp.INDEX_SET, [1, 1], v2, "distinct"),
    (ExprOp.INDEX_ADD, [0, 4], v2, r"\[0, 4\)"),
    (ExprOp.INDEX_ADD, [0], v2, "one rank-1 value per index"),
    (ExprOp.INDEX_SET, [0, 1], sc.sym("v22", (1, 2)), "one rank-1 value per index"),
  ):
    with pytest.raises(sc.VerifyError, match=message):
      sc.verify_expr(Expr(op, (x, values), TensorType((4,)), attrs={"indices": np.array(idx)}))


def test_index_update_derivatives() -> None:
  """A value added three times at one entry and then partly overwritten; and a set of every entry,
  which leaves no derivative in the base at all."""
  c, v = sc.sym("c", 5), sc.sym("v", 4)
  y = sc.index_set(sc.index_add(c * c.sin(), [3, 3, 0, 3], v * v), [3, 1], sc.stack([v[0] * c[2], c[3]]))
  f = sc.sumsqr(y) + (y * y.cos()).sum()
  every = sc.index_set(c, [4, 2, 0, 1, 3], v.sum() * sc.const(np.arange(1.0, 6.0)))
  outs = [f.reshape((1,))]
  for wrt in (c, v):
    outs += [gradient(f, wrt), jacobian(f.reshape((1,)), wrt), hessian(f, wrt)]
  fn = _fn("ie_idx_d", [c, v], [*outs, jacobian(every, c), gradient((every * every).sum(), v)])
  cv, vv = np.array([0.3, -1.1, 0.8, 0.45, -0.6]), np.array([0.7, -0.2, 0.35, 1.2])
  got = fn._flat_numerical_call(cv, vv)

  def numpy(cz: np.ndarray, vz: np.ndarray) -> np.ndarray:
    yv = cz * np.sin(cz)
    np.add.at(yv, [3, 3, 0, 3], vz * vz)
    yv[[3, 1]] = [vz[0] * cz[2], cz[3]]
    return np.array([yv @ yv + np.sum(yv * np.cos(yv))])

  np.testing.assert_allclose(got[0], numpy(cv, vv), rtol=1e-14)
  grads = _fn("ie_idx_g", [c, v], [gradient(f, c), gradient(f, v)])
  for k, (wrt, point) in enumerate(((c, cv), (v, vv))):
    g, j, h = got[1 + 3 * k : 4 + 3 * k]

    def at(z: np.ndarray, k: int = k) -> tuple[np.ndarray, np.ndarray]:
      return (z, vv) if k == 0 else (cv, z)

    np.testing.assert_allclose(g, finite_difference(lambda z: numpy(*at(z)), point).reshape(-1), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(j.reshape(-1), g, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(h, finite_difference(lambda z: grads._flat_numerical_call(*at(z))[k], point), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(h, h.T, rtol=1e-12, atol=1e-13)
  np.testing.assert_array_equal(got[7], np.zeros((5, 5)))
  assert sc.jacobian_sparsity(every, c).nnz == 0
  np.testing.assert_allclose(got[8], np.full(4, 2.0 * vv.sum() * 55.0), rtol=1e-14)


def _chain(c: Expr, v: Expr, w: Expr) -> Expr:
  """``+v`` at 0 and 2, then entry 2 overwritten, then two more additions at 2, then entry 4 set
  from the new entry 2: each link reads only its predecessor, away from what it writes."""
  u1 = sc.index_add(c, [0, 2], v)
  u2 = sc.index_set(u1, [2], w)
  u3 = sc.index_add(u2, [2, 2], np.array([1.0, 1.0]))
  return sc.index_set(u3, [4], u3[2:3] * 2.0)


def _np_chain(c: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
  c = c.copy()
  c[[0, 2]] += v
  c[2] = w[0] + 2.0
  c[4] = c[2] * 2.0
  return c


def test_index_update_chains_in_scan_and_while(monkeypatch: pytest.MonkeyPatch) -> None:
  """A chain where later links overwrite what earlier ones wrote updates the carry in place, in a
  scan and in a while loop; a swap (values reading the entries they replace) keeps two slots."""
  c, v, w = sc.sym("c", 5), sc.sym("v", 2), sc.sym("w", 1)
  cc = sc.sym("cc", 5)

  def build(tag: str, kind: str) -> sc.Function:
    c0, vs, ws = sc.sym("c0", 5), sc.sym("vs", 6), sc.sym("ws", 3)
    if kind == "while":
      body = _fn(f"ie_chw_body_{tag}", [c], [_chain(c, sc.const([1.0, 0.5]), sc.const([0.25]))])
      cond = _fn(f"ie_chw_cond_{tag}", [cc], [sc.less(cc[0], 3.5)])
      fin, count = sc.while_loop(cond, body, c0, max_iter=10)
      return _fn(f"ie_chw_{tag}", [c0, vs, ws], [fin, count])
    swapped = sc.index_add(c, [0, 2], v)
    nxt = _chain(c, v, w) if kind == "chain" else sc.index_set(swapped, [0, 2], sc.stack([swapped[2], swapped[0]]))
    body = _fn(f"ie_ch_{kind}_body_{tag}", [c, v, w], [nxt])
    (fin,) = sc.scan(body, c0, [(vs, 0, 2), (ws, 0, 1)], length=3)
    return _fn(f"ie_ch_{kind}_{tag}", [c0, vs, ws], [fin])

  c0, vs, ws = _eighths(5, 4), _eighths(6, 5), _eighths(3, 6)
  (got,) = _run_both(monkeypatch, lambda tag: build(tag, "chain"), (c0, vs, ws), expect_in_place=True)
  want = c0
  for k in range(3):
    want = _np_chain(want, vs[2 * k : 2 * k + 2], ws[k : k + 1])
  np.testing.assert_array_equal(got, want)
  (got,) = _run_both(monkeypatch, lambda tag: build(tag, "swap"), (c0, vs, ws), expect_in_place=False)
  want = c0.copy()
  for k in range(3):
    want[[0, 2]] += vs[2 * k : 2 * k + 2]
    want[[0, 2]] = want[[2, 0]]
  np.testing.assert_array_equal(got, want)
  got, count = _run_both(monkeypatch, lambda tag: build(tag, "while"), (c0 * 0.0, vs, ws), expect_in_place=True)
  want, steps = np.zeros(5), 0
  while want[0] < 3.5 and steps < 10:
    want, steps = _np_chain(want, np.array([1.0, 0.5]), np.array([0.25])), steps + 1
  np.testing.assert_array_equal(got, want)
  assert count == steps == 4


@pytest.mark.parametrize("length", [5, 6])
def test_a_read_at_the_step_number_clashes_at_one_step_only(monkeypatch: pytest.MonkeyPatch, length: int) -> None:
  """``c[5] += c[k]``: a read of the entry the update writes happens only at the last step of a
  six-step scan, and that one step keeps the whole loop on two slots."""
  c, k = sc.sym("c", 6), sc.sym("k", (), dtype="int64")

  def build(tag: str) -> sc.Function:
    body = _fn(f"ie_one_{length}_body_{tag}", [c, k], [sc.index_add(c, [5], sc.take(c, k.reshape((1,))))])
    c0 = sc.sym("c0", 6)
    (fin,) = sc.scan(body, c0, [], length=length, index=True)
    return _fn(f"ie_one_{length}_{tag}", [c0], [fin])

  c0 = _eighths(6, 7)
  (got,) = _run_both(monkeypatch, build, (c0,), expect_in_place=length == 5)
  want = c0.copy()
  for step in range(length):
    want[5] += want[step]
  np.testing.assert_array_equal(got, want)


# --- take / put_add / put: indices known at run time --------------------------------------------


def _runtime_cases(n: int) -> dict[str, np.ndarray]:
  return {
    "edges": np.array([-1, 0, n - 1, n, 0, n - 1, n, -1]),
    "huge": np.array([2**40, -(2**40), 2**62, -(2**62), n - 1, 0]),  # exact as the doubles the ABI carries
    "padded": np.array([n, -1, n + 1, -(n + 1)]),
    "empty": np.zeros(0, dtype=np.int64),
  }


@pytest.mark.parametrize("shape", [(5,), (1,), (2, 3, 4)])
def test_boundary_indices_match_numpy(shape: tuple[int, ...]) -> None:
  """Each index list as a run-time input and as an ``int64`` constant; the constants also carry the
  ``int64`` extremes, which no double reaches exactly. A fill of NaN or infinity reads back as such."""
  n, lead = shape[-1], shape[:-1]
  cases = {**_runtime_cases(n), "extremes": np.array([I64.min, I64.max, 0, I64.min + 1, n - 1])}
  tag = "x".join(map(str, shape))
  x = sc.sym("x", shape)
  inputs, outs, want = [x], [], []
  xv = _eighths(shape, 8)
  point = [xv]
  for name, iv in cases.items():
    v = sc.sym(f"v_{name}", (*lead, iv.size))
    vv = _eighths((*lead, iv.size), 9 + len(point))
    inputs.append(v)
    point.append(vv)
    sources = [sc.const(iv, dtype="int64")]
    if name != "extremes":
      sources.append(sc.sym(f"i_{name}", iv.size, dtype="int64"))
      inputs.append(sources[-1])
      point.append(iv.astype(float))
    for idx in sources:
      outs += [sc.take(x, idx, fill=np.nan), sc.take(x, idx, fill=-np.inf), sc.put_add(x, idx, v), sc.put(x, idx, v)]
      want += [_np_take(xv, iv, np.nan), _np_take(xv, iv, -np.inf), _np_put(xv, iv, vv, add=True), _np_put(xv, iv, vv, add=False)]
  got = _fn(f"ie_bound_{tag}", inputs, outs)._flat_numerical_call(*point)
  for g, w in zip(got, want, strict=True):
    np.testing.assert_array_equal(g, w)


def test_in_range_matches_the_checked_version_bit_for_bit() -> None:
  """Valid indices with a repeat on a rank-3 array: the unchecked code gives the same bits as the
  checked code, values and derivatives."""
  x, v, idx = sc.sym("x", (2, 3, 5)), sc.sym("v", (2, 3, 4)), sc.sym("idx", 4, dtype="int64")
  outs = []
  for in_range in (False, True):
    t = sc.take(x, idx, fill=7.0, in_range=in_range)
    z = sc.put(sc.put_add(x.sin(), idx, v * t, in_range=in_range), idx, t.cos() * v, in_range=in_range)
    f = sc.sumsqr(z) + sc.take(z, idx, in_range=in_range).exp().sum()
    outs += [t, z, gradient(f, x), gradient(f, v)]
  got = _fn("ie_inrange", [x, v, idx], outs)._flat_numerical_call(*_normal((2, 3, 5), (2, 3, 4), seed=31), [4.0, 0.0, 4.0, 2.0])
  for a, b in zip(got[:4], got[4:], strict=True):
    np.testing.assert_array_equal(a, b)


def test_indices_cast_from_floats_truncate_toward_zero() -> None:
  """``cast("int64")`` truncates as C does: -0.5 and -0.99 become index 0, -1.5 becomes -1."""
  x, f, v = sc.sym("x", 5), sc.sym("f", 7), sc.sym("v", 7)
  idx = f.cast("int64")
  fv = np.array([2.9, -0.5, -1.5, 4.99, 5.0, -0.99, 1e12 + 0.5])
  iv = np.trunc(fv).astype(np.int64)
  assert list(iv[:6]) == [2, 0, -1, 4, 5, 0]
  xv, vv = _eighths(5, 10), _eighths(7, 11)
  got = _fn("ie_cast", [x, f, v], [sc.take(x, idx, fill=-9.0), sc.put_add(x, idx, v), sc.put(x, idx, v)])._flat_numerical_call(xv, fv, vv)
  np.testing.assert_array_equal(got[0], _np_take(xv, iv, -9.0))
  np.testing.assert_array_equal(got[1], _np_put(xv, iv, vv, add=True))
  np.testing.assert_array_equal(got[2], _np_put(xv, iv, vv, add=False))


def test_updates_whose_values_read_the_array_they_update() -> None:
  """The values are read from the array before any lane writes it: a rotation, a reversal and an
  accumulation of an array into itself, with repeated and padded lanes. Derivatives through them
  agree with finite differences, forward mode with reverse mode."""
  x, i1, i2, i5 = sc.sym("x", 5), sc.sym("i1", 4, dtype="int64"), sc.sym("i2", 4, dtype="int64"), sc.sym("i5", 5, dtype="int64")
  rot = sc.put(x, i1, sc.take(x, i2))
  rev = sc.put(x, i5, x)
  acc = sc.put_add(x, i5, x)
  f = sc.sumsqr(rot.sin()) + (acc * rev).sum() + (sc.take(acc, i1, fill=0.5) * sc.take(rot, i2, fill=-0.25)).sum()
  iv = (np.array([0, 1, 2, 0]), np.array([1, 2, 0, -1]), np.array([4, 3, 2, 1, 0]))
  xv = _eighths(5, 12)
  cases = [(iv[0], iv[1], iv[2]), (np.array([0, 9, 1, 1]), np.array([4, 0, 3, 2]), np.array([0, 0, 4, 9, -1]))]
  for case, (a, b, c) in enumerate(cases):
    fn = _fn(f"ie_alias_{case}", [x, i1, i2, i5], [rot, rev, acc, f.reshape((1,)), gradient(f, x), jacobian(f.reshape((1,)), x)])
    got = fn._flat_numerical_call(xv, a, b, c)
    rot_np = _np_put(xv, a, _np_take(xv, b), add=False)
    rev_np, acc_np = _np_put(xv, c, xv, add=False), _np_put(xv, c, xv, add=True)
    np.testing.assert_array_equal(got[0], rot_np)
    np.testing.assert_array_equal(got[1], rev_np)
    np.testing.assert_array_equal(got[2], acc_np)

    def numpy(z: np.ndarray, a=a, b=b, c=c) -> np.ndarray:
      r = _np_put(z, a, _np_take(z, b), add=False)
      s = _np_put(z, c, z, add=True)
      return np.array([np.sum(np.sin(r) ** 2) + s @ _np_put(z, c, z, add=False) + _np_take(s, a, 0.5) @ _np_take(r, b, -0.25)])

    np.testing.assert_allclose(got[3], numpy(xv), rtol=1e-14)
    np.testing.assert_allclose(got[4], finite_difference(numpy, xv).reshape(-1), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(got[5].reshape(-1), got[4], rtol=1e-12, atol=1e-14)


def test_runtime_index_refusals() -> None:
  x, m = sc.sym("x", 4), sc.sym("m", (2, 3, 4))
  for bad in (sc.sym("fi", 2), sc.sym("i32", 2, dtype="int32"), sc.sym("b", 2, dtype="bool"), sc.sym("k", (), dtype="int64")):
    for build in (lambda i: sc.take(x, i), lambda i: sc.put_add(x, i, sc.sym("v2", 2)), lambda i: sc.put(x, i, sc.sym("v2", 2))):
      with pytest.raises(TypeError, match="rank-1 int64"):
        build(bad)
  i3 = sc.sym("i3", 3, dtype="int64")
  with pytest.raises(ValueError, match=r"needs values of shape \(3,\)"):
    sc.put(x, i3, 0.0)  # values are not broadcast
  with pytest.raises(ValueError, match=r"needs values of shape \(2, 3, 3\)"):
    sc.put_add(m, i3, sc.sym("v323", (3, 2, 3)))
  with pytest.raises(ValueError, match="scalar"):
    sc.put(sc.sym("s", ()), i3, sc.sym("v3", 3))
  with pytest.raises(TypeError, match="mixed-dtype"):
    sc.put_add(sc.sym("f", 4, dtype="float32"), i3, sc.sym("v3", 3))
  f_idx = sc.sym("fidx", 3)
  for node, message in (
    (Expr(ExprOp.TAKE, (x, i3), TensorType((4,)), attrs={"fill": 0.0}), r"by 3 indices gives \(3,\)"),
    (Expr(ExprOp.TAKE, (x, f_idx), TensorType((3,)), attrs={"fill": 0.0}), "rank-1 int64"),
    (Expr(ExprOp.PUT, (x, i3, sc.sym("v2", 2)), TensorType((4,))), r"needs values \(3,\)"),
    (Expr(ExprOp.PUT_ADD, (x, f_idx, sc.sym("v3", 3)), TensorType((4,))), "rank-1 int64"),
  ):
    with pytest.raises(sc.VerifyError, match=message):
      sc.verify_expr(node)


def test_float32_arrays() -> None:
  """``float32`` arithmetic in lane order, as NumPy's ``float32`` gives it; the fill here is exact
  in ``float32``."""
  x, v, idx = sc.sym("x", 5, dtype="float32"), sc.sym("v", 4, dtype="float32"), sc.sym("idx", 4, dtype="int64")
  xv, vv = (a.astype(np.float32) for a in _normal(5, 4, seed=32))
  iv = np.array([3, 5, 3, -1])
  outs = [sc.take(x, idx, fill=0.375), sc.put_add(x, idx, v), sc.put(x, idx, v), sc.index_add(x, [3, 3, 0], v[:3])]
  got = _fn("ie_f32", [x, v, idx], outs)._flat_numerical_call(xv, vv, iv)
  added, placed, fixed = xv.copy(), xv.copy(), xv.copy()
  for j, i in enumerate(iv):
    if 0 <= i < 5:
      added[i] = added[i] + vv[j]
      placed[i] = vv[j]
  for j, i in enumerate([3, 3, 0]):
    fixed[i] = fixed[i] + vv[j]
  want = [np.where((iv >= 0) & (iv < 5), xv[np.clip(iv, 0, 4)], np.float32(0.375)), added, placed, fixed]
  for g, w in zip(got, want, strict=True):
    assert w.dtype == np.float32
    np.testing.assert_array_equal(g, w.astype(np.float64))


def test_a_float32_fill_reads_as_float32() -> None:
  x, idx = sc.sym("x", 2, dtype="float32"), sc.sym("idx", 1, dtype="int64")
  (got,) = _fn("ie_f32_fill", [x, idx], [sc.take(x, idx, fill=0.1)])._flat_numerical_call(np.zeros(2), np.array([5.0]))
  assert got[0] == float(np.float32(0.1))


# --- derivatives with repeated and padded lanes ---------------------------------------------------

IDX_PUT = np.array([4, 1, 4, 7])  # 4 written twice (lane 2 wins), lane 3 padded
IDX_ADD = np.array([1, -1, 1, 0])  # 1 twice, lane 1 padded
IDX_DISTINCT = np.array([4, 1, 5, 5])  # distinct inside the axis, padded twice


WEIGHTS = np.arange(30.0).reshape(2, 3, 5) / 30.0


def _repeated(b: Expr, v: Expr, idx_put: Expr, idx_add: Expr) -> Expr:
  z = sc.put(b.sin(), idx_put, v * sc.take(b, idx_add))
  z2 = sc.put_add(z * z, idx_add, v.cos())
  return (z2 * sc.const(WEIGHTS)).sum() + sc.sumsqr(sc.take(z2, idx_put, fill=3.0))


def _numpy_repeated(bv: np.ndarray, vv: np.ndarray, idx_put: np.ndarray) -> np.ndarray:
  z = _np_put(np.sin(bv), idx_put, vv * _np_take(bv, IDX_ADD), add=False)
  z2 = _np_put(z * z, IDX_ADD, np.cos(vv), add=True)
  return np.array([np.sum(z2 * WEIGHTS) + np.sum(_np_take(z2, idx_put, 3.0) ** 2)])


@pytest.mark.parametrize("source", ["symbol", "constant", "constant_distinct"])
def test_derivatives_with_repeated_and_padded_lanes(monkeypatch: pytest.MonkeyPatch, source: str) -> None:
  """A put whose index repeats gives the derivative to the last lane writing an entry only, and a
  padded lane gets none, in forward and reverse mode. Constant indices take a shortcut when they
  are distinct inside the axis, and must agree with the run-time path."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  idx_put = IDX_DISTINCT if source == "constant_distinct" else IDX_PUT
  b, v = sc.sym("b", (2, 3, 5)), sc.sym("v", (2, 3, 4))
  ip, ia = sc.sym("ip", 4, dtype="int64"), sc.sym("ia", 4, dtype="int64")
  if source != "symbol":
    ip, ia = sc.const(idx_put, dtype="int64"), sc.const(IDX_ADD, dtype="int64")
  inputs = [b, v, *(e for e in (ip, ia) if e.op == ExprOp.INPUT)]
  indices = [idx_put, IDX_ADD] if source == "symbol" else []
  f = _repeated(b, v, ip, ia)
  lanes = (sc.put(b, ip, v) * sc.const(WEIGHTS)).sum()
  outs = [gradient(f, b), gradient(f, v), jacobian(f.reshape((1,)), b), jacobian(f.reshape((1,)), v), hessian(f, v)]
  outs += [gradient(lanes, v), jacobian(lanes.reshape((1,)), v)]
  fn = _fn(f"ie_rep_{source}", inputs, outs)
  bv, vv = (0.5 * a for a in _normal((2, 3, 5), (2, 3, 4), seed=33))
  gb, gv, jb, jv, hv, lane_rev, lane_fwd = fn._flat_numerical_call(bv, vv, *indices)
  fd_b = finite_difference(lambda z: _numpy_repeated(z.reshape(2, 3, 5), vv, idx_put), bv.reshape(-1))
  np.testing.assert_allclose(gb.reshape(-1), fd_b.reshape(-1), rtol=1e-6, atol=1e-7)
  fd_v = finite_difference(lambda z: _numpy_repeated(bv, z.reshape(2, 3, 4), idx_put), vv.reshape(-1))
  np.testing.assert_allclose(gv.reshape(-1), fd_v.reshape(-1), rtol=1e-6, atol=1e-7)
  np.testing.assert_allclose(jb.reshape(-1), gb.reshape(-1), rtol=1e-12, atol=1e-14)
  np.testing.assert_allclose(jv.reshape(-1), gv.reshape(-1), rtol=1e-12, atol=1e-14)
  grad_v = _fn(f"ie_rep_gv_{source}", inputs, [gradient(f, v)])
  fd = finite_difference(lambda z: grad_v._flat_numerical_call(bv, z.reshape(2, 3, 4), *indices)[0].reshape(-1), vv.reshape(-1))
  np.testing.assert_allclose(hv.reshape(24, 24), fd, rtol=1e-6, atol=1e-7)
  # Only the lane whose write survives sees its weight: IDX_PUT's lane 0 is overwritten by lane 2,
  # and lanes aimed outside the axis write nothing.
  survivors = {"symbol": [1, 2], "constant": [1, 2], "constant_distinct": [0, 1]}[source]
  want = np.zeros((2, 3, 4))
  for j in survivors:
    want[..., j] = WEIGHTS[..., idx_put[j]]
  np.testing.assert_array_equal(lane_rev, want)
  np.testing.assert_array_equal(lane_fwd.reshape(2, 3, 4), want)


# --- loops that slice their index tables by step number -------------------------------------------


def test_vmap_slices_an_index_table_per_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  n, lanes, m = 5, 3, 4
  xb, ib, vb = sc.sym("xb", n), sc.sym("ib", lanes, dtype="int64"), sc.sym("vb", lanes)
  inner = _fn("ie_vm_inner", [xb, ib, vb], [sc.put(sc.put_add(xb, ib, vb * sc.take(xb, ib, fill=2.0)), ib, vb.sin())])
  table = np.array([[0, 0, 5], [4, -1, 4], [1, 2, 3], [7, 7, 7]])  # repeats, padding, distinct, all padded
  xs, vs = sc.sym("xs", m * n), sc.sym("vs", m * lanes)
  y = sc.vmap(inner, m, [(xs, 0, n), (sc.const(table.reshape(-1), dtype="int64"), 0, lanes), (vs, 0, lanes)])
  f = (y * y.cos()).sum()
  fn = _fn("ie_vm", [xs, vs], [y, jacobian(y, vs), gradient(f, xs), gradient(f, vs)])
  xv, vv = _normal(m * n, m * lanes, seed=34)

  def numpy(xz: np.ndarray, vz: np.ndarray) -> np.ndarray:
    rows = []
    for r in range(m):
      x, v = xz[r * n : (r + 1) * n], vz[r * lanes : (r + 1) * lanes]
      z = _np_put(x, table[r], v * _np_take(x, table[r], 2.0), add=True)
      rows.append(_np_put(z, table[r], np.sin(v), add=False))
    return np.concatenate(rows)

  got = fn._flat_numerical_call(xv, vv)
  np.testing.assert_allclose(got[0], numpy(xv, vv), rtol=1e-14, atol=1e-15)
  np.testing.assert_allclose(got[1], finite_difference(lambda z: numpy(xv, z), vv), rtol=1e-6, atol=1e-7)

  def value(xz: np.ndarray, vz: np.ndarray) -> np.ndarray:
    y = numpy(xz, vz)
    return np.array([np.sum(y * np.cos(y))])

  np.testing.assert_allclose(got[2], finite_difference(lambda z: value(z, vv), xv).reshape(-1), rtol=1e-6, atol=1e-7)
  np.testing.assert_allclose(got[3], finite_difference(lambda z: value(xv, z), vv).reshape(-1), rtol=1e-6, atol=1e-7)


def test_reads_of_what_earlier_steps_wrote_stay_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
  """Step ``k`` writes ``c[k]`` from ``c[k - 1]`` (written by the step before) and ``c[min(k + 2, 6)]``
  (padded near the end): positions shared only across steps never force two slots."""
  n = 6
  c, k, x = sc.sym("c", n), sc.sym("k", (), dtype="int64"), sc.sym("x", 1)

  def build(tag: str) -> sc.Function:
    ahead = sc.minimum(k + 2, sc.const(n, dtype="int64")).reshape((1,))
    value = sc.take(c, (k - 1).reshape((1,)), fill=1.0) * 2.0 + sc.take(c, ahead, fill=0.5) + x
    body = _fn(f"ie_prev_body_{tag}", [c, k, x], [sc.put(c, k.reshape((1,)), value)])
    c0, xs = sc.sym("c0", n), sc.sym("xs", n)
    (fin,) = sc.scan(body, c0, [(xs, 0, 1)], length=n, index=True)
    return _fn(f"ie_prev_{tag}", [c0, xs], [fin])

  c0, xs = _eighths(n, 13), _eighths(n, 14)
  (got,) = _run_both(monkeypatch, build, (c0, xs), expect_in_place=True)
  want = c0.copy()
  for step in range(n):
    ahead = min(step + 2, n)
    want[step] = (want[step - 1] if step else 1.0) * 2.0 + (want[ahead] if ahead < n else 0.5) + xs[step]
  np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(("reads", "in_place"), [((4, 5, 3), True), ((4, 4, 3), False)])
def test_row_positions_of_a_reshaped_carry(monkeypatch: pytest.MonkeyPatch, reads: tuple[int, ...], in_place: bool) -> None:
  """A ``(2, 3)`` carry whose step ``k`` writes column ``k`` of both rows (flat ``k`` and ``3 + k``)
  and reads flat entry ``reads[k]`` of its flat view: entry 4 at step 1 is row 1 of the column
  that step writes."""
  c, k = sc.sym("c", (2, 3)), sc.sym("k", (), dtype="int64")
  table = sc.const(np.array(reads), dtype="int64")

  def build(tag: str) -> sc.Function:
    read = sc.take(c.reshape((6,)), sc.take(table, k.reshape((1,)), fill=-1.0))
    body = _fn(f"ie_rows_{int(in_place)}_body_{tag}", [c, k], [sc.put(c, k.reshape((1,)), read.reshape((1, 1)) * sc.const([[1.0], [-2.0]]) + 0.5)])
    c0 = sc.sym("c0", (2, 3))
    (fin,) = sc.scan(body, c0, [], length=3, index=True)
    return _fn(f"ie_rows_{int(in_place)}_{tag}", [c0], [fin])

  c0 = _eighths((2, 3), 15)
  (got,) = _run_both(monkeypatch, build, (c0,), expect_in_place=in_place)
  want = c0.copy()
  for step in range(3):
    r = want.reshape(-1)[reads[step]]
    want[:, step] = [r + 0.5, -2.0 * r + 0.5]
  np.testing.assert_array_equal(got, want)


def test_padded_entries_of_an_index_table_drop_their_writes(monkeypatch: pytest.MonkeyPatch) -> None:
  """The write index is read from a three-entry table with fill -1, past which the scan runs two
  more steps: those writes are dropped, never aimed at entry 0, which every step reads."""
  c, k = sc.sym("c", 4), sc.sym("k", (), dtype="int64")
  table = sc.const(np.array([1, 2, 3]), dtype="int64")

  def build(tag: str) -> sc.Function:
    target = sc.take(table, k.reshape((1,)), fill=-1.0)
    body = _fn(f"ie_tab_body_{tag}", [c, k], [sc.put_add(c, target, sc.take(c, sc.const([0], dtype="int64")) + 1.0)])
    c0 = sc.sym("c0", 4)
    (fin,) = sc.scan(body, c0, [], length=5, index=True)
    return _fn(f"ie_tab_{tag}", [c0], [fin])

  c0 = _eighths(4, 16)
  (got,) = _run_both(monkeypatch, build, (c0,), expect_in_place=True)
  np.testing.assert_array_equal(got, c0 + np.array([0.0, 1.0, 1.0, 1.0]) * (c0[0] + 1.0))


def test_scratch_slots_cover_the_widest_update(monkeypatch: pytest.MonkeyPatch) -> None:
  """Two updates of a ``(2, 5)`` carry, three and five lanes wide, many padded: the in-place carry
  keeps a scratch slot for each padded lane of the wider one, row by row."""
  rows, n, steps = 2, 5, 4
  t3 = np.array([[0, 5, 5], [9, 1, 1], [5, 5, 5], [-1, 2, 4]])
  t5 = np.array([[5, 5, 5, 5, 3], [0, 6, 6, 6, 6], [4, 4, -3, 5, 5], [5, 5, 5, 5, 5]])
  c, i3, i5, u = sc.sym("c", (rows, n)), sc.sym("i3", 3, dtype="int64"), sc.sym("i5", 5, dtype="int64"), sc.sym("u", rows * 8)
  uu = u.reshape((rows, 8))
  nxt = sc.put(sc.put_add(c, i3, uu[:, :3]), i5, uu[:, 3:])
  assert _put_scratch(update_chain(_normalize_function(_fn("ie_scr_probe", [c, i3, i5, u], [nxt]))) or []) == rows * 5

  def build(tag: str) -> sc.Function:
    body = _fn(f"ie_scr_body_{tag}", [c, i3, i5, u], [nxt])
    c0, us = sc.sym("c0", (rows, n)), sc.sym("us", steps * rows * 8)
    tables = [(sc.const(t3.reshape(-1), dtype="int64"), 0, 3), (sc.const(t5.reshape(-1), dtype="int64"), 0, 5), (us, 0, rows * 8)]
    (fin,) = sc.scan(body, c0, tables, length=steps)
    return _fn(f"ie_scr_{tag}", [c0, us], [fin])

  c0, us = _eighths((rows, n), 17), _eighths(steps * rows * 8, 18)
  (got,) = _run_both(monkeypatch, build, (c0, us), expect_in_place=True)
  want = c0.copy()
  for step in range(steps):
    vals = us[step * rows * 8 : (step + 1) * rows * 8].reshape(rows, 8)
    want = _np_put(_np_put(want, t3[step], vals[:, :3], add=True), t5[step], vals[:, 3:], add=False)
  np.testing.assert_array_equal(got, want)


def test_a_while_loop_of_no_iterations_updating_at_the_step_number() -> None:
  c, k, cc = sc.sym("c", 4), sc.sym("k", (), dtype="int64"), sc.sym("cc", 4)
  body = _fn("ie_w0_body", [c, k], [sc.put(c, k.reshape((1,)), sc.const([1.0]))])
  cond = _fn("ie_w0_cond", [cc], [sc.less(cc.sum(), 40.0)])
  c0 = sc.sym("c0", 4)
  fin, count = sc.while_loop(cond, body, c0, max_iter=0, index=True)
  got, steps = _fn("ie_w0", [c0], [fin, count])._flat_numerical_call(np.arange(4.0))
  np.testing.assert_array_equal(got, np.arange(4.0))
  assert steps == 0
