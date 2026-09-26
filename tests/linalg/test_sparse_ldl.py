"""The generated sparse ``L D L^T`` and its solves, against dense factorizations and SciPy, with the
in-place loops, the implicit derivatives and the sharing of one factorization checked on the
generated code."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.derivatives import gradient, hessian, jacobian
from scaly.ad.forward import jvp
from scaly.codegen import render_c_module
from scaly.linalg import SparseLDL, SparseMatrix, sparse_ldl
from scaly.linalg.sparse_factor import Schedule, _call
from scaly.linalg.symbolic import Ordering

RNG = np.random.default_rng(707)


def _fn(name, inputs, outputs):
  return sc.Function._from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _kkt(n: int, m: int, delta: float, seed: int) -> sparse.csc_array:
  rng = np.random.default_rng(seed)
  p = sparse.random_array((n, n), density=3.0 / n, random_state=rng)
  p = p @ p.T + sparse.eye_array(n)
  a = sparse.random_array((m, n), density=3.0 / n, random_state=rng)
  return sparse.csc_array(sparse.bmat([[p, a.T], [a, -delta * sparse.eye_array(m)]]))


def _mpc(stages: int, nx: int = 3, nu: int = 1) -> sparse.csc_array:
  rng = np.random.default_rng(stages)
  z = stages * (nx + nu) + nx
  h = sparse.eye_array(z)
  rows = []
  for k in range(stages):
    blk = sparse.lil_array((nx, z))
    off = k * (nx + nu)
    blk[:, off : off + nx + nu] = rng.standard_normal((nx, nx + nu))
    blk[:, off + nx + nu : off + 2 * nx + nu] = -np.eye(nx)
    rows.append(blk)
  c = sparse.vstack(rows)
  return sparse.csc_array(sparse.bmat([[h, c.T], [c, -1e-6 * sparse.eye_array(c.shape[0])]]))


MATRICES = {"kkt": lambda: _kkt(20, 8, 1e-4, 1), "mpc": lambda: _mpc(6), "diag": lambda: sparse.csc_array(sparse.diags_array(np.arange(1.0, 6.0)))}


def _triangle(k: sparse.csc_array, which: str) -> sparse.csc_array:
  t = {"lower": sparse.tril(k), "upper": sparse.triu(k), "full": k}[which]
  t = sparse.csc_array(t)
  t.sort_indices()
  return t


def _values(mat: SparseMatrix, a: sparse.csc_array) -> np.ndarray:
  return np.asarray(a.tocsr()[mat.coordinates()]).reshape(-1)


@pytest.mark.parametrize("schedule", ["scan", "unroll"])
@pytest.mark.parametrize("which", ["lower", "upper", "full"])
@pytest.mark.parametrize("ordering", ["natural", "rcm", "mmd", "auto"])
@pytest.mark.parametrize("name", list(MATRICES))
def test_factor_and_solve(name: str, ordering: Ordering, which: str, schedule: Schedule) -> None:
  k = MATRICES[name]()
  t = _triangle(k, which)
  mat = SparseMatrix.symbol("K", t)
  fact = SparseLDL(mat, ordering=ordering, schedule=schedule)
  assert fact.schedule == schedule
  b, bm = sc.sym("b", k.shape[0]), sc.sym("bm", (k.shape[0], 3))
  fn = _fn(f"sl_{name}_{ordering}_{which}_{schedule}", [mat.values, b, bm], [fact.l_values, fact.d, fact.solve(b), fact.solve(bm)])
  bv, bmv = RNG.standard_normal(k.shape[0]), RNG.standard_normal((k.shape[0], 3))
  lv, d, x, xm = fn._flat_numerical_call(_values(mat, t), bv, bmv)
  s = fact.symbolic
  n = s.n
  unit_l = np.eye(n)
  unit_l[s.l_rows, np.repeat(np.arange(n), s.col_counts)] = lv
  dense = k.toarray()
  # Componentwise backward error of the factorization: |L D L^T - P K P^T| <= eps |L| |D| |L^T|.
  bound = np.abs(unit_l) @ np.diag(np.abs(d)) @ np.abs(unit_l.T)
  assert np.all(np.abs(unit_l @ np.diag(d) @ unit_l.T - dense[np.ix_(s.perm, s.perm)]) <= 1e-13 * bound + 1e-15)
  np.testing.assert_allclose(x, np.linalg.solve(dense, bv), rtol=1e-8, atol=1e-10)
  np.testing.assert_allclose(xm, np.linalg.solve(dense, bmv), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("delta", [1e-4, 1e-8, 1e-10, 1e-13])
def test_accuracy_across_regularization(delta: float) -> None:
  """Backward error at the level of rounding, and the forward error SciPy's pivoted LU gets."""
  k = _kkt(60, 30, delta, 3)
  t = _triangle(k, "lower")
  mat = SparseMatrix.symbol("K", t)
  b = sc.sym("b", 90)
  (x,) = _fn(f"acc_{int(-np.log10(delta))}", [mat.values, b], [sparse_ldl(mat).solve(b)])._flat_numerical_call(
    _values(mat, t), bv := RNG.standard_normal(90)
  )
  backward = np.abs(k @ x - bv).max() / (abs(k).sum(axis=1).max() * np.abs(x).max() + np.abs(bv).max())
  assert backward < 1e-13
  ref = spsolve(k.tocsc(), bv)
  forward_ref = np.abs(ref - np.linalg.solve(k.toarray(), bv)).max() / np.abs(ref).max()
  assert np.abs(x - ref).max() / np.abs(ref).max() < max(1e-8, 1e3 * forward_ref)


def _scan_procs(src: str) -> tuple[list[str], list[str]]:
  names = set(re.findall(r"void (\w+?)_raw\(", src))
  loops = sorted(n for n in names if re.search(r"_(f\d+|sf|sb)$", n))
  in_place = sorted(n for n in names if n.endswith("_inplace"))
  return loops, in_place


def test_every_loop_runs_in_place_and_one_factorization_serves_every_solve() -> None:
  k = MATRICES["kkt"]()
  t = _triangle(k, "lower")
  mat = SparseMatrix.symbol("K", t)
  fact = SparseLDL(mat, name="share", schedule="scan")
  b1, b2 = sc.sym("b1", k.shape[0]), sc.sym("b2", k.shape[0])
  src = str(render_c_module(_fn("share_run", [mat.values, b1, b2], [fact.solve(b1), fact.solve(b2)])).body)
  loops, in_place = _scan_procs(src)
  assert not loops, f"loops without the in-place proof: {loops}"
  factor_loops = [n for n in in_place if re.search(r"_f\d+_inplace", n)]
  assert len(factor_loops) == len(fact.segments)
  entry = src[src.index("int share_run(") :]
  assert sum(entry.count(f"{n}_raw(") for n in factor_loops) == len(fact.segments), "one factorization, however many solves"


@pytest.mark.parametrize("schedule", ["scan", "unroll"])
def test_implicit_derivatives(monkeypatch: pytest.MonkeyPatch, schedule: Schedule) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  k = _kkt(12, 5, 1e-2, 5)
  t = _triangle(k, "lower")
  mat = SparseMatrix.symbol("K", t)
  fact = SparseLDL(mat, schedule=schedule)
  b = sc.sym("b", 17)
  x = fact.solve(b)
  f = sc.sumsqr(x) + (x * b).sum()
  kv, bv = _values(mat, t), RNG.standard_normal(17)
  fn = _fn(f"impl_{schedule}", [mat.values, b], [f, gradient(f, mat.values), gradient(f, b), jacobian(f.reshape((1,)), mat.values), hessian(f, b)])
  val, gk, gb, jk, hb = fn._flat_numerical_call(kv, bv)
  scale = np.abs(gk).max()
  fd_k = finite_difference(lambda z: fn._flat_numerical_call(z, bv)[0].reshape(1), kv).reshape(-1)
  np.testing.assert_allclose(gk, fd_k, rtol=1e-5, atol=1e-6 * scale)
  np.testing.assert_allclose(jk.reshape(-1), gk, rtol=1e-9, atol=1e-11 * scale)
  kinv = np.linalg.inv(k.toarray())
  np.testing.assert_allclose(gb, 2 * kinv.T @ (kinv @ bv) + kinv @ bv + kinv.T @ bv, rtol=1e-9)
  np.testing.assert_allclose(hb, 2 * kinv.T @ kinv + kinv + kinv.T, rtol=1e-8, atol=1e-10)
  one = _fn(f"impl_jvp_{schedule}", [mat.values, b], [jvp(x, mat.values, sc.const(np.ones(mat.nnz)))])._flat_numerical_call(kv, bv)[0]
  dk = (t + sparse.tril(t, -1).T).toarray()
  np.testing.assert_allclose(one, -kinv @ ((dk != 0) * 1.0 @ (kinv @ bv)), rtol=1e-9, atol=1e-12)
  # The derivatives never differentiate the factorization: every loop in the gradient is in place.
  loops, _ = _scan_procs(str(render_c_module(_fn(f"impl_grad_{schedule}", [mat.values, b], [gradient(f, mat.values)])).body))
  assert not loops


def test_full_pattern_derivative_reads_the_lower_entry() -> None:
  """With both triangles stored, the factorization reads the lower entry of a mirrored pair; the
  upper one gets no derivative."""
  k = MATRICES["kkt"]()
  t = _triangle(k, "full")
  mat = SparseMatrix.symbol("K", t)
  b = sc.sym("b", k.shape[0])
  x = sparse_ldl(mat).solve(b)
  (g,) = _fn("full_grad", [mat.values, b], [gradient(sc.sumsqr(x), mat.values)])._flat_numerical_call(
    _values(mat, t), RNG.standard_normal(k.shape[0])
  )
  rows, cols = mat.coordinates()
  assert np.all(g[rows < cols] == 0.0) and np.any(g[rows > cols] != 0.0)


def test_validation() -> None:
  with pytest.raises(ValueError, match="square"):
    SparseLDL(SparseMatrix.symbol("r", np.ones((2, 3), dtype=bool)))
  with pytest.raises(ValueError, match=r"diagonal entry stored; columns \[1\] have none"):
    SparseLDL(SparseMatrix.symbol("nd", np.array([[1, 1], [1, 0]], dtype=bool)))
  # Columns are named in the input's numbering, whatever the ordering.
  arrow = np.eye(5, dtype=bool)
  arrow[0, :] = arrow[:, 0] = True
  arrow[3, 3] = False
  for ordering in ("natural", "rcm", "mmd"):
    with pytest.raises(ValueError, match=r"columns \[3\] have none"):
      SparseLDL(SparseMatrix.symbol("ar", arrow), ordering=ordering)
  fact = SparseLDL(SparseMatrix.symbol("d", np.eye(3, dtype=bool)))
  with pytest.raises(ValueError, match="length 3"):
    fact.solve(sc.sym("b", 4))


@pytest.mark.parametrize("name", list(MATRICES))
@pytest.mark.parametrize("ordering", ["natural", "mmd"])
def test_padded_groups_run_empty_ranges(name: str, ordering: Ordering) -> None:
  """A padded group of a column update is an empty range, so padding costs no multiply-adds."""
  k = MATRICES[name]()
  fact = SparseLDL(SparseMatrix.symbol("K", _triangle(k, "lower")), ordering=ordering, schedule="scan")
  s = fact.symbolic
  assert fact.segments
  for seg in fact.segments:
    tables = fact._factor_tables(seg)
    g = max(seg.u, 1)
    lo, hi = tables[3].reshape(seg.length, g), tables[4].reshape(seg.length, g)
    real = np.arange(g)[None, :] < np.diff(s.r_ptr)[seg.start : seg.stop, None]
    assert np.all(hi[~real] == lo[~real]) and np.all(hi[real] > lo[real])
    assert int((hi - lo).sum()) == int(s.u_ptr[seg.stop] - s.u_ptr[seg.start])


# --- schedules, refinement, health, sparsity (T2-9) --------------------------------------------


def test_unrolled_schedule_is_straight_line_code() -> None:
  k = MATRICES["mpc"]()
  mat = SparseMatrix.symbol("K", _triangle(k, "lower"))
  fact = SparseLDL(mat, schedule="unroll", name="unr")
  b = sc.sym("b", k.shape[0])
  src = str(render_c_module(_fn("unr_run", [mat.values, b], [fact.solve(b)])).body)
  assert not fact.segments and "unr_f0" not in src and "unr_sf" not in src and "unr_sb" not in src


def test_auto_schedule_follows_the_option() -> None:
  mat = SparseMatrix.symbol("K", _triangle(MATRICES["kkt"](), "lower"))
  work = SparseLDL(mat, schedule="scan").work
  assert work == SparseLDL(mat, schedule="scan").symbolic.update_lanes + SparseLDL(mat, schedule="scan").symbolic.nnz_l
  with sc.options(sparse_unroll=work):
    assert SparseLDL(mat).schedule == "unroll"
  with sc.options(sparse_unroll=work - 1):
    assert SparseLDL(mat).schedule == "scan"
  with pytest.raises(ValueError, match="schedule must be one of"):
    SparseLDL(mat, schedule="loop")  # ty: ignore[invalid-argument-type]


def _full(t: sparse.csc_array) -> sparse.csr_array:
  return (t + sparse.tril(t, -1).T).tocsr()


@pytest.mark.parametrize("schedule", ["scan", "unroll"])
def test_refinement_fixed_and_adaptive(schedule: Schedule) -> None:
  """At a tiny regularization one solve leaves a residual far above rounding; refinement removes
  it."""
  k = _kkt(12, 6, 1e-11, 11)
  t = _triangle(k, "lower")
  mat = SparseMatrix.symbol("K", t)
  fact = SparseLDL(mat, schedule=schedule)
  b = sc.sym("b", 18)
  outs = [fact.solve(b), fact.solve(b, refine=3), fact.solve(b, refine=4, tol=1e30), fact.solve(b, refine=6, tol=1e-15)]
  fn = _fn(f"refine_{schedule}", [mat.values, b], outs)
  full = _full(t)
  for seed in range(5):
    kv, bv = _values(mat, t), np.random.default_rng(seed).standard_normal(18)
    x0, x3, a0, a6 = fn._flat_numerical_call(kv, bv)
    res = lambda x: np.abs(full @ x - bv).max()  # noqa: E731, B023
    assert res(x3) < 1e-3 * res(x0) and res(a6) < 1e-3 * res(x0)
    np.testing.assert_array_equal(a0, x0)  # a tolerance met at once: no step runs
  with pytest.raises(ValueError, match="refine"):
    fact.solve(b, refine=-1)
  with pytest.raises(ValueError, match="tol"):
    fact.solve(b, refine=2, tol=0.0)


@pytest.mark.parametrize("schedule", ["scan", "unroll"])
def test_refinement_step_counts_with_an_inexact_factor(schedule: Schedule) -> None:
  """Refining against ``K`` with the factor of ``1.2 K`` contracts the error by exactly 1/6 per
  step, so ``k`` steps give ``(1 - 6^{-(k+1)}) K^{-1} b`` and the adaptive loop's step count can be
  read off the result: it stops at the first residual ``6^{-(k+1)} ||b||`` below
  ``tol * max(1, ||b||_inf)`` or after ``refine`` steps."""
  k = _kkt(10, 4, 1e-2, 16)
  t = _triangle(k, "lower")
  mat = SparseMatrix.symbol("K", t)
  fact = SparseLDL(mat, schedule=schedule)
  inexact = SparseLDL(mat.with_values(mat.values * 1.2), symbolic=fact.symbolic, schedule=schedule)
  b = sc.sym("b", 14)
  cases = {(0, None): 0, (1, None): 1, (3, None): 3, (2, 1e-300): 2, (6, 1e-3): None}
  outs = [_call(fact._solver_function(refine, tol), inexact.values, mat.values, b) for refine, tol in cases]
  fn = _fn(f"refine_count_{schedule}", [mat.values, b], outs)
  kv = _values(mat, t)
  exact = lambda bv: np.linalg.solve(k.toarray(), bv)  # noqa: E731
  for scale, adaptive_steps in ((100.0, 3), (0.01, 1)):
    bv = scale * RNG.uniform(0.5, 1.0, 14) * np.sign(RNG.standard_normal(14))
    bv[0] = scale
    got = fn._flat_numerical_call(kv, bv)
    for steps, value in zip(cases.values(), got, strict=True):
      steps = adaptive_steps if steps is None else steps
      np.testing.assert_allclose(value, (1.0 - 6.0 ** -(steps + 1)) * exact(bv), rtol=1e-9, atol=1e-12 * scale)


def test_refined_solve_keeps_the_implicit_derivative() -> None:
  k = _kkt(10, 4, 1e-3, 12)
  t = _triangle(k, "lower")
  mat = SparseMatrix.symbol("K", t)
  fact = SparseLDL(mat, schedule="scan")
  b = sc.sym("b", 14)
  kv, bv = _values(mat, t), RNG.standard_normal(14)
  grads = []
  options: list[dict[str, Any]] = [{}, {"refine": 2}, {"refine": 2, "tol": 1e-14}]
  for opts in options:
    f = sc.sumsqr(fact.solve(b, **opts))
    grads.append(_fn(f"rg{len(grads)}", [mat.values, b], [gradient(f, mat.values), gradient(f, b)])._flat_numerical_call(kv, bv))
  for g in grads[1:]:
    for got, ref in zip(g, grads[0], strict=True):
      np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10 * np.abs(ref).max())


def test_inertia_and_health() -> None:
  k = _kkt(9, 4, 1e-3, 13)
  t = _triangle(k, "lower")
  mat = SparseMatrix.symbol("K", t)
  signs = np.r_[np.ones(9), -np.ones(4)]
  for schedule in ("scan", "unroll"):
    fact = SparseLDL(mat, schedule=schedule)
    b = sc.sym("b", 13)
    x = fact.solve(b)
    outs = [
      fact.inertia(),
      fact.health(),
      fact.health(signs=signs),
      fact.health(signs=-signs),
      fact.health(pivot_tol=1e6),
      fact.health(x=x),
      fact.health(x=x * np.inf),
    ]
    fn = _fn(f"health_{schedule}", [mat.values, b], outs)
    kv, bv = _values(mat, t), RNG.standard_normal(13)
    inertia, *flags = fn._flat_numerical_call(kv, bv)
    np.testing.assert_array_equal(inertia, [9.0, 4.0, 0.0])
    assert [bool(f) for f in flags] == [True, True, False, False, True, False]
    # A NaN entry, and a singular matrix whose second pivot is exactly zero.
    for poison in (np.nan, np.inf):
      bad = kv.copy()
      bad[0] = poison
      inertia, ok, *_ = fn._flat_numerical_call(bad, bv)
      assert not ok
    assert not np.array_equal(fact.symbolic.perm, np.arange(13))  # so ``signs`` must be permuted
  sing = SparseMatrix.symbol("S", np.ones((2, 2), dtype=bool))
  fs = SparseLDL(sing, ordering="natural")
  inertia, ok = _fn("health_sing", [sing.values], [fs.inertia(), fs.health()])._flat_numerical_call(np.ones(sing.nnz))
  np.testing.assert_array_equal(inertia, [1.0, 0.0, 1.0])
  assert not ok
  # An infinite pivot is positive but not healthy.
  dg = SparseMatrix.symbol("Dg", np.eye(3, dtype=bool))
  fd = SparseLDL(dg)
  inertia, ok = _fn("health_inf", [dg.values], [fd.inertia(), fd.health()])._flat_numerical_call(np.array([1.0, np.inf, -2.0]))
  np.testing.assert_array_equal(inertia, [2.0, 1.0, 0.0])
  assert not ok
  with pytest.raises(ValueError, match="signs"):
    fact.health(signs=np.ones(3))
  with pytest.raises(ValueError, match="signs"):
    fact.health(signs=0.5 * signs)


def test_solve_sparsity_is_per_connected_component() -> None:
  """Two decoupled blocks: the solution of one never depends on the other's right-hand side or
  entries, nor on the factor; the compressed Jacobian then needs few colors and is exact."""
  from scaly.ad.sparse import sparse_jacobian
  from scaly.ad.sparsity import jacobian_sparsity

  k1, k2 = _kkt(6, 3, 1e-2, 14), _kkt(5, 2, 1e-2, 15)
  k = sparse.csc_array(sparse.block_diag([k1, k2]))
  t = _triangle(k, "full")
  mat = SparseMatrix.symbol("K", t)
  fact = SparseLDL(mat, schedule="scan")
  b = sc.sym("b", 16)
  x = fact.solve(b)
  comp = np.r_[np.zeros(9), np.ones(7)]
  pattern = jacobian_sparsity(x, b)
  dense = np.zeros(pattern.shape, dtype=bool)
  dense[pattern.rows, pattern.cols] = True
  np.testing.assert_array_equal(dense, comp[:, None] == comp[None, :])
  pk = jacobian_sparsity(x, mat.values)
  rows, cols = mat.coordinates()
  assert np.all(comp[np.asarray(pk.rows)] == comp[rows[np.asarray(pk.cols)]])
  assert set(np.asarray(pk.cols).tolist()) == set(fact.symbolic.a_source.tolist())  # the entries read: lower ones only
  assert x.attrs["callee"].custom_sparsity(0, 0) is None  # the factor: the rules give it no derivative
  jac = sparse_jacobian(x, b)
  assert jac.coloring_width == 9
  kv, bv = _values(mat, t), RNG.standard_normal(16)
  (vals,) = _fn("spj", [mat.values, b], [jac.values])._flat_numerical_call(kv, bv)
  got = np.zeros((16, 16))
  got[jac.sparsity.rows, jac.sparsity.cols] = vals
  np.testing.assert_allclose(got, np.linalg.inv(k.toarray()), rtol=1e-9, atol=1e-12)


# --- review round (T2-R) -----------------------------------------------------------------------


@pytest.mark.parametrize("schedule", ["scan", "unroll"])
@pytest.mark.parametrize("which", ["lower", "upper"])
def test_second_derivatives_in_the_matrix(monkeypatch: pytest.MonkeyPatch, schedule: Schedule, which: str) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  k = _kkt(7, 3, 1e-2, 5)
  t = _triangle(k, which)
  mat = SparseMatrix.symbol("K", t)
  b = sc.sym("b", 10)
  x = SparseLDL(mat, schedule=schedule).solve(b)
  f = sc.sumsqr(x) + (x.sin() * b).sum()
  fn = _fn(f"hk_{schedule}_{which}", [mat.values, b], [gradient(f, mat.values), hessian(f, mat.values), jacobian(gradient(f, mat.values), b)])
  kv, bv = _values(mat, t), np.random.default_rng(1).standard_normal(10)
  _, hk, hkb = fn._flat_numerical_call(kv, bv)
  fd_k = finite_difference(lambda z: fn._flat_numerical_call(z, bv)[0].reshape(-1), kv)
  np.testing.assert_allclose(hk, fd_k, rtol=1e-4, atol=1e-5 * np.abs(hk).max())
  fd_b = finite_difference(lambda z: fn._flat_numerical_call(kv, z)[0].reshape(-1), bv)
  np.testing.assert_allclose(hkb, fd_b, rtol=1e-4, atol=1e-5 * np.abs(hkb).max())


@pytest.mark.parametrize("schedule", ["scan", "unroll"])
def test_matrix_right_hand_side_derivatives(schedule: Schedule) -> None:
  k = _kkt(7, 3, 1e-2, 6)
  t = _triangle(k, "lower")
  mat = SparseMatrix.symbol("K", t)
  bm = sc.sym("B", (10, 3))
  xm = SparseLDL(mat, schedule=schedule).solve(bm)
  f = xm.sin().sum() + sc.sumsqr(xm)
  fn = _fn(f"mrhs_{schedule}", [mat.values, bm], [f, gradient(f, mat.values), gradient(f, bm), hessian(f, mat.values)])
  kv, bv = _values(mat, t), np.random.default_rng(2).standard_normal((10, 3))
  _, gk, gb, hk = fn._flat_numerical_call(kv, bv)
  fd_k = finite_difference(lambda z: fn._flat_numerical_call(z, bv)[0].reshape(1), kv).reshape(-1)
  np.testing.assert_allclose(gk, fd_k, rtol=1e-5, atol=1e-6 * np.abs(gk).max())
  fd_b = finite_difference(lambda z: fn._flat_numerical_call(kv, z.reshape(10, 3))[0].reshape(1), bv.reshape(-1)).reshape(10, 3)
  np.testing.assert_allclose(gb, fd_b, rtol=1e-5, atol=1e-6 * np.abs(gb).max())
  fd_h = finite_difference(lambda z: fn._flat_numerical_call(z, bv)[1].reshape(-1), kv)
  np.testing.assert_allclose(hk, fd_h, rtol=1e-4, atol=1e-5 * np.abs(hk).max())


def test_third_derivatives_go_through_the_loops() -> None:
  k = _kkt(5, 2, 1e-2, 9)
  t = _triangle(k, "lower")
  mat = SparseMatrix.symbol("K", t)
  b = sc.sym("b", 7)
  f = sc.sumsqr(SparseLDL(mat, schedule="scan").solve(b))
  third = jacobian(hessian(f, b).reshape((49,)), b)  # f is quadratic in b
  (value,) = _fn("third", [mat.values, b], [third])._flat_numerical_call(_values(mat, t), np.ones(7))
  np.testing.assert_allclose(value, 0.0, atol=1e-8)


@pytest.mark.parametrize("schedule", ["scan", "unroll"])
def test_every_solve_variant_in_one_graph(schedule: Schedule) -> None:
  """Solves with and without refinement, fixed or adaptive at two tolerances, for a vector and a
  matrix right-hand side, and their gradients: every variant and rule has a name of its own."""
  k = _kkt(8, 3, 1e-6, 17)
  t = _triangle(k, "lower")
  mat = SparseMatrix.symbol("K", t)
  fact = SparseLDL(mat, schedule=schedule)
  b, bm = sc.sym("b", 11), sc.sym("B", (11, 2))
  variants: list[dict[str, Any]] = [
    {},
    {"refine": 1},
    {"refine": 2},
    {"refine": 3, "tol": 1e-1},
    {"refine": 3, "tol": 1e-14},
    {"refine": 0, "tol": 1e-3},
  ]
  xs = [fact.solve(b, **v) for v in variants] + [fact.solve(bm, **v) for v in variants]
  grads = [gradient(sc.sumsqr(x), mat.values) for x in xs]
  kv = _values(mat, t)
  bv, bmv = np.random.default_rng(3).standard_normal(11), np.random.default_rng(4).standard_normal((11, 2))
  out = _fn(f"variants_{schedule}", [mat.values, b, bm], xs + grads)._flat_numerical_call(kv, bv, bmv)
  dense = k.toarray()
  for x, ref in zip(out[: len(xs)], [np.linalg.solve(dense, bv)] * 6 + [np.linalg.solve(dense, bmv)] * 6, strict=True):
    np.testing.assert_allclose(x, ref, rtol=1e-6, atol=1e-9 * np.abs(ref).max())
  for g in out[len(xs) : len(xs) + 6]:
    np.testing.assert_allclose(g, out[len(xs)], rtol=1e-6, atol=1e-9 * np.abs(g).max())


def test_forward_mode_never_differentiates_the_factorization() -> None:
  k = MATRICES["kkt"]()
  t = _triangle(k, "lower")
  mat = SparseMatrix.symbol("K", t)
  fact = SparseLDL(mat, schedule="scan", name="fwdonly")
  b = sc.sym("b", k.shape[0])
  tangent = jvp(fact.solve(b), mat.values, sc.const(np.ones(mat.nnz)))
  src = str(render_c_module(_fn("fwdonly_run", [mat.values, b], [tangent])).body)
  loops = set(re.findall(r"void (fwdonly_f\d+\w*)_raw\(", src))
  assert loops and all(name.endswith("_inplace") for name in loops), loops  # the primal factorization only


def test_a_given_analysis_must_match_the_pattern() -> None:
  k = MATRICES["kkt"]()
  t = _triangle(k, "lower")
  mat = SparseMatrix.symbol("K", t)
  own = SparseLDL(mat, schedule="scan").symbolic
  assert SparseLDL(SparseMatrix.symbol("K2", t), symbolic=own, schedule="scan").symbolic is own
  diag = SparseMatrix.symbol("D", sparse.eye_array(k.shape[0], format="csc"))
  with pytest.raises(ValueError, match="different sparsity pattern"):
    SparseLDL(mat, symbolic=SparseLDL(diag).symbolic)
  with pytest.raises(ValueError, match="is for a"):
    SparseLDL(SparseMatrix.symbol("S", np.eye(3, dtype=bool)), symbolic=own)
