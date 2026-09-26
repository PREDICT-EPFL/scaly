"""``take``, ``put_add`` and ``put``: indexing with ``int64`` indices known only at run time.

Values are checked against NumPy, including repeated indices and indices outside ``[0, n)``,
which read the fill or drop their value. Derivatives are checked against finite differences,
forward mode against reverse mode, and multi-seed forward mode in strict mode. The typical use,
a loop that slices a padded index table by its step number, is checked against SciPy.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.derivatives import gradient, hessian, jacobian
from scaly.ad.forward import jvp
from scaly.codegen import render_c_module
from scaly.passes.lowering import lower_function, main_proc

RNG = np.random.default_rng(202)


def _fn(name, inputs, outputs):
  return sc.Function._from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _np_take(x: np.ndarray, idx: np.ndarray, fill: float = 0.0) -> np.ndarray:
  n = x.shape[-1]
  inside = (idx >= 0) & (idx < n)
  out = x[..., np.where(inside, idx, 0)] if n else np.zeros((*x.shape[:-1], idx.size))
  return np.where(inside, out, fill)


def _np_put(base: np.ndarray, idx: np.ndarray, values: np.ndarray, *, add: bool) -> np.ndarray:
  out = base.copy()
  n = base.shape[-1]
  for j, i in enumerate(idx):
    if 0 <= i < n:
      out[..., i] = out[..., i] + values[..., j] if add else values[..., j]
  return out


INDEX_CASES = {
  "plain": np.array([3, 0, 5, 1]),
  "repeats": np.array([2, 2, 4, 2, 0]),
  "padded": np.array([1, 6, 3, -1, 6, 1000, -7]),
  "all_out": np.array([6, -1, 6]),
  "empty": np.zeros(0, dtype=np.int64),
}


@pytest.mark.parametrize("case", list(INDEX_CASES))
@pytest.mark.parametrize("rows", [None, 3])
def test_values_match_numpy(case: str, rows: int | None) -> None:
  idx_v = INDEX_CASES[case]
  shape = (6,) if rows is None else (rows, 6)
  lanes = (idx_v.size,) if rows is None else (rows, idx_v.size)
  x, v = sc.sym("x", shape), sc.sym("v", lanes)
  idx = sc.sym("idx", idx_v.size, dtype="int64")
  outs = [sc.take(x, idx, fill=-2.5), sc.put_add(x, idx, v), sc.put(x, idx, v)]
  xv, vv = RNG.standard_normal(shape), RNG.standard_normal(lanes)
  got = _fn(f"vals_{case}_{rows}", [x, idx, v], outs)._flat_numerical_call(xv, idx_v.astype(float), vv)
  np.testing.assert_array_equal(got[0], _np_take(xv, idx_v, -2.5))
  np.testing.assert_allclose(got[1], _np_put(xv, idx_v, vv, add=True), rtol=0, atol=1e-15)
  np.testing.assert_array_equal(got[2], _np_put(xv, idx_v, vv, add=False))


def test_take_from_an_empty_axis_reads_the_fill() -> None:
  x, idx = sc.sym("x", 0), sc.sym("idx", 3, dtype="int64")
  got = _fn("take_empty", [x, idx], [sc.take(x, idx, fill=7.0)])._flat_numerical_call(np.zeros(0), np.array([0.0, 1.0, -1.0]))[0]
  np.testing.assert_array_equal(got, [7.0, 7.0, 7.0])


def test_indices_computed_in_the_graph() -> None:
  """Indices are ordinary ``int64`` expressions: arithmetic on an input, a table read at run time."""
  x, k = sc.sym("x", 8), sc.sym("k", (), dtype="int64")
  cols = k * 2 + sc.const(np.arange(3), dtype="int64")  # 2k, 2k+1, 2k+2
  perm = sc.const(np.array([7, 6, 5, 4, 3, 2, 1, 0]), dtype="int64")
  y = sc.take(x, sc.take(perm, cols, fill=0.0).cast("int64"))
  xv = np.arange(8.0) * 1.5
  for kv in range(4):
    got = _fn("idx_graph", [x, k], [y])._flat_numerical_call(xv, np.array(float(kv)))[0]
    want = [xv[7 - c] if c < 8 else 0.0 for c in range(2 * kv, 2 * kv + 3)]
    np.testing.assert_array_equal(got, want)


def test_chained_updates_and_neighbouring_elementwise_code() -> None:
  """Updates chained with elementwise code on both sides, so loop fusion and hoisting see them."""
  x, v, w = sc.sym("x", 7), sc.sym("v", 3), sc.sym("w", 2)
  i1, i2, i3 = (sc.sym(f"i{k}", n, dtype="int64") for k, n in ((1, 3), (2, 2), (3, 4)))
  z = sc.put(sc.put_add(2.0 * x, i1, v * 3.0), i2, w.sin())
  out = sc.take(z, i3) * 2.0 + sc.take(x + 1.0, i3)
  pts = (RNG.standard_normal(7), RNG.standard_normal(3), RNG.standard_normal(2))
  iv = (np.array([1, 1, 6]), np.array([0, 7]), np.array([0, 1, 6, 7]))
  got = _fn("chain", [x, v, w, i1, i2, i3], [z, out])._flat_numerical_call(*pts, *(a.astype(float) for a in iv))
  z_np = _np_put(_np_put(2.0 * pts[0], iv[0], pts[1] * 3.0, add=True), iv[1], np.sin(pts[2]), add=False)
  np.testing.assert_allclose(got[0], z_np, rtol=1e-15)
  np.testing.assert_allclose(got[1], _np_take(z_np, iv[2]) * 2.0 + _np_take(pts[0] + 1.0, iv[2]), rtol=1e-15)


# --- derivatives ---------------------------------------------------------------------------------

IDX_A = np.array([4, 1, 7, 1, -1])  # a repeat and a padded lane
IDX_B = np.array([0, 5, 7, 2])  # distinct, one padded (n = 7)


def _nonlinear():
  x, v = sc.sym("x", 7), sc.sym("v", 5)
  ia, ib = sc.sym("ia", 5, dtype="int64"), sc.sym("ib", 4, dtype="int64")
  t = sc.take(x, ia)
  z = sc.put_add(x * x, ia, (t * v).sin())
  z = sc.put(z, ib, sc.take(z, ia)[:4] * sc.take(x, ib).cos())
  return x, v, ia, ib, sc.sumsqr(z) + sc.take(z, ib).exp().sum()


def _numpy_nonlinear(xv: np.ndarray, vv: np.ndarray) -> float:
  t = _np_take(xv, IDX_A)
  z = _np_put(xv * xv, IDX_A, np.sin(t * vv), add=True)
  z = _np_put(z, IDX_B, _np_take(z, IDX_A)[:4] * np.cos(_np_take(xv, IDX_B)), add=False)
  return float(z @ z + np.exp(_np_take(z, IDX_B)).sum())


POINT = (RNG.standard_normal(7) * 0.5, RNG.standard_normal(5) * 0.5)
INDICES = (IDX_A.astype(float), IDX_B.astype(float))


def test_nonlinear_value() -> None:
  x, v, ia, ib, f = _nonlinear()
  np.testing.assert_allclose(_fn("nl_val", [x, v, ia, ib], [f])._flat_numerical_call(*POINT, *INDICES)[0], _numpy_nonlinear(*POINT), rtol=1e-14)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("kind", ["jacobian", "gradient", "hessian"])
def test_nonlinear_derivatives(monkeypatch: pytest.MonkeyPatch, which: int, kind: str) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  x, v, ia, ib, f = _nonlinear()
  wrt = (x, v)[which]
  expr = {"jacobian": lambda: jacobian(f.reshape((1,)), wrt), "gradient": lambda: gradient(f, wrt), "hessian": lambda: hessian(f, wrt)}[kind]()
  got = _fn(f"nl_{kind}_{which}", [x, v, ia, ib], [expr])._flat_numerical_call(*POINT, *INDICES)[0]
  if kind == "hessian":
    grad = _fn(f"nl_g_{which}", [x, v, ia, ib], [gradient(f, wrt)])

    def fun(z: np.ndarray) -> np.ndarray:
      pt = list(POINT)
      pt[which] = z
      return grad._flat_numerical_call(*pt, *INDICES)[0].reshape(-1)
  else:

    def fun(z: np.ndarray) -> np.ndarray:
      pt = list(POINT)
      pt[which] = z
      return np.array([_numpy_nonlinear(*pt)])

  fd = finite_difference(fun, POINT[which])
  np.testing.assert_allclose(got.reshape(-1), np.asarray(fd).reshape(-1), rtol=1e-6, atol=1e-7)


def test_single_seed_tangent_of_a_padded_take_is_zero() -> None:
  """The fill is a constant: a padded lane has no tangent, whatever the fill."""
  x, idx = sc.sym("x", 4), sc.sym("idx", 3, dtype="int64")
  dx = sc.const(np.array([1.0, 2.0, 3.0, 4.0]))
  tangent = jvp(sc.take(x, idx, fill=5.0), x, dx)
  got = _fn("jvp_fill", [x, idx], [tangent])._flat_numerical_call(np.zeros(4), np.array([2.0, 4.0, -1.0]))[0]
  np.testing.assert_array_equal(got, [3.0, 0.0, 0.0])


def test_forward_and_reverse_agree_on_batched_rows() -> None:
  x, v, idx = sc.sym("x", (3, 5)), sc.sym("v", (3, 4)), sc.sym("idx", 4, dtype="int64")
  y = sc.put_add(x.sin(), idx, v * sc.take(x, idx)) * x
  out = sc.stack([sc.sumsqr(y[r]) for r in range(3)])
  exprs = [jacobian(out, x), sc.stack([gradient(out[r], x).reshape((15,)) for r in range(3)])]
  exprs += [jacobian(out, v), sc.stack([gradient(out[r], v).reshape((12,)) for r in range(3)])]
  got = _fn("batch_fr", [x, v, idx], exprs)._flat_numerical_call(
    RNG.standard_normal((3, 5)), RNG.standard_normal((3, 4)), np.array([4.0, 0.0, 5.0, 4.0])
  )
  np.testing.assert_allclose(got[0], got[1], rtol=1e-12, atol=1e-14)
  np.testing.assert_allclose(got[2], got[3], rtol=1e-12, atol=1e-14)
  assert np.all(got[2].reshape(3, 3, 4)[[0, 1, 2], [1, 2, 0]] == 0.0)  # rows do not mix


def test_sparsity_is_conservative_and_row_local() -> None:
  """The pattern holds the true pattern at any index values, and a row never reaches another row."""
  x, v, idx = sc.sym("x", (2, 4)), sc.sym("v", (2, 3)), sc.sym("idx", 3, dtype="int64")
  y = sc.put(sc.take(x, idx) * v, idx, x[:, :3] * v)
  mask = sc.jacobian_sparsity(y, x).to_mask()
  assert mask.shape == (6, 8)
  assert not mask[:3, 4:].any() and not mask[3:, :4].any()
  mask_v = sc.jacobian_sparsity(y, v).to_mask()
  for iv in (np.array([0, 1, 2]), np.array([2, 2, 1]), np.array([5, 0, -1])):
    point = (RNG.standard_normal((2, 4)), RNG.standard_normal((2, 3)), iv.astype(float))
    jx, jv = _fn("sp_j", [x, v, idx], [jacobian(y, x), jacobian(y, v)])._flat_numerical_call(*point)
    assert not np.any((jx != 0) & ~mask)
    assert not np.any((jv != 0) & ~mask_v) and np.any(jv != 0)


# --- the typical use: a loop reading a padded table by its step number --------------------------


def _padded_csr(a: sparse.csr_array) -> tuple[np.ndarray, np.ndarray, int]:
  width = int(np.diff(a.indptr).max())
  cols = np.full((a.shape[0], width), a.shape[1], dtype=np.int64)  # pad with n: the dump index
  pos = np.full((a.shape[0], width), a.nnz, dtype=np.int64)  # pad past the values
  for r in range(a.shape[0]):
    lo, hi = a.indptr[r], a.indptr[r + 1]
    cols[r, : hi - lo] = a.indices[lo:hi]
    pos[r, : hi - lo] = np.arange(lo, hi)
  return cols.reshape(-1), pos.reshape(-1), width


def test_scan_spmv_and_transpose_product_match_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
  """``A x`` row by row and ``A^T y`` by accumulating rows into a carry, reading a padded CSR
  table by step number; values of ``A`` and ``x`` are inputs and both products are differentiated."""
  a = sparse.random_array((9, 12), density=0.3, random_state=3, format="csr")
  a.sort_indices()
  cols_t, pos_t, width = _padded_csr(a)
  vals, x, y = sc.sym("vals", a.nnz), sc.sym("x", 12), sc.sym("y", 9)
  c, k, xb, vb, yk = sc.sym("c", 12), sc.sym("k", (), dtype="int64"), sc.sym("xb", 12), sc.sym("vb", a.nnz), sc.sym("yk", ())
  lane = k * width + sc.const(np.arange(width), dtype="int64")
  cols = sc.take(sc.const(cols_t, dtype="int64"), lane)
  row_vals = sc.take(vb, sc.take(sc.const(pos_t, dtype="int64"), lane))
  body = sc.Function._from_exprs(
    "spmv_row",
    [c, k, xb, vb, yk],
    [sc.put_add(c, cols, row_vals * yk), sc.stack([(row_vals * sc.take(xb, cols)).sum()])],
    ["c", "k", "xb", "vb", "yk"],
    ["n", "ax"],
  )
  aty, ax = sc.scan(body, sc.const(np.zeros(12)), [(x, 0, 0), (vals, 0, 0), (y, 0, 1)], length=9, index=True)
  f = sc.sumsqr(ax) + (aty * aty.sin()).sum()
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  fn = _fn("spmv", [vals, x, y], [ax, aty, gradient(f, vals), gradient(f, x), hessian(f, x), hessian(f, y)])
  pt = (a.data, RNG.standard_normal(12), RNG.standard_normal(9))
  got = fn._flat_numerical_call(*pt)
  np.testing.assert_allclose(got[0], a @ pt[1], rtol=1e-13)
  np.testing.assert_allclose(got[1], a.T @ pt[2], rtol=1e-13)

  def value(data: np.ndarray, xv: np.ndarray) -> np.ndarray:
    m = sparse.csr_array((data, a.indices, a.indptr), shape=a.shape)
    aty_np = m.T @ pt[2]
    return np.array([np.sum((m @ xv) ** 2) + np.sum(aty_np * np.sin(aty_np))])

  np.testing.assert_allclose(got[2], finite_difference(lambda d: value(d, pt[1]), pt[0]).reshape(-1), rtol=1e-6, atol=1e-7)
  np.testing.assert_allclose(got[3], finite_difference(lambda z: value(pt[0], z), pt[1]).reshape(-1), rtol=1e-6, atol=1e-7)
  m = a.toarray()
  np.testing.assert_allclose(got[4], 2.0 * m.T @ m, rtol=1e-12, atol=1e-12)  # f is quadratic in x
  u = m.T @ pt[2]
  np.testing.assert_allclose(got[5], m @ np.diag(2 * np.cos(u) - u * np.sin(u)) @ m.T, rtol=1e-10, atol=1e-12)


# --- generated code and validation ---------------------------------------------------------------


def test_padded_lanes_write_a_slot_each() -> None:
  x, v, idx = sc.sym("x", 6), sc.sym("v", 4), sc.sym("idx", 4, dtype="int64")
  src = str(render_c_module(_fn("dump_c", [x, v, idx], [sc.put_add(x, idx, v)])).source)
  assert re.search(r"\(6 \+ j_\w+\)", src), "a padded lane addresses the slot after the entries by its own lane"
  loops = src.split("dump_c(")[1].split("for (long long j_", 1)[1]
  assert "if (" not in loops, "the write is unconditional"


def test_runtime_indices_keep_a_scalar_hint_working() -> None:
  x, idx = sc.sym("x", 5), sc.sym("idx", 2, dtype="int64")
  fn = _fn("scalar_hint", [x, idx], [(sc.take(x.sin(), idx) * 2.0).scalar()])
  assert main_proc(lower_function(fn)).attrs["scalarize_mode"] == "disabled"
  np.testing.assert_allclose(fn._flat_numerical_call(np.arange(5.0), np.array([4.0, 9.0]))[0], [2 * np.sin(4.0), 0.0])


def test_validation() -> None:
  x = sc.sym("x", 4)
  with pytest.raises(TypeError, match="rank-1 int64"):
    sc.take(x, sc.sym("fi", 2))
  with pytest.raises(TypeError, match="rank-1 int64"):
    sc.take(x, sc.sym("i2", (2, 2), dtype="int64"))
  with pytest.raises(ValueError, match="scalar"):
    sc.take(sc.sym("s", ()), sc.sym("ii", 2, dtype="int64"))
  with pytest.raises(ValueError, match=r"needs values of shape \(3,\)"):
    sc.put_add(x, sc.sym("i3", 3, dtype="int64"), sc.sym("v2", 2))
  with pytest.raises(ValueError, match=r"needs values of shape \(2, 3\)"):
    sc.put(sc.sym("m", (2, 4)), sc.sym("i3", 3, dtype="int64"), sc.sym("v3", 3))
  sc.verify_expr(sc.put(sc.sym("m", (2, 4)), sc.sym("i3", 3, dtype="int64"), sc.sym("v23", (2, 3))))
  sc.verify_expr(sc.take(sc.sym("m", (2, 4)), [1, 2, 9]))
