"""The generated sparse ``L D L^T`` and its solves, against dense factorizations and SciPy, with the
in-place loops, the implicit derivatives and the sharing of one factorization checked on the
generated code."""

from __future__ import annotations

import re

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


@pytest.mark.parametrize("which", ["lower", "upper", "full"])
@pytest.mark.parametrize("ordering", ["natural", "rcm", "mmd", "auto"])
@pytest.mark.parametrize("name", list(MATRICES))
def test_factor_and_solve(name: str, ordering: str, which: str) -> None:
  k = MATRICES[name]()
  t = _triangle(k, which)
  mat = SparseMatrix.symbol("K", t)
  fact = SparseLDL(mat, ordering=ordering)
  b, bm = sc.sym("b", k.shape[0]), sc.sym("bm", (k.shape[0], 3))
  fn = _fn(f"sl_{name}_{ordering}_{which}", [mat.values, b, bm], [fact.l_values, fact.d, fact.solve(b), fact.solve(bm)])
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
  loops = sorted(n for n in names if re.search(r"_(f|sf|sb)\d+", n) and not n.endswith("_inplace"))
  in_place = sorted(n for n in names if n.endswith("_inplace"))
  return loops, in_place


def test_every_loop_runs_in_place_and_one_factorization_serves_every_solve() -> None:
  k = MATRICES["kkt"]()
  t = _triangle(k, "lower")
  mat = SparseMatrix.symbol("K", t)
  fact = SparseLDL(mat, name="share")
  b1, b2 = sc.sym("b1", k.shape[0]), sc.sym("b2", k.shape[0])
  src = str(render_c_module(_fn("share_run", [mat.values, b1, b2], [fact.solve(b1), fact.solve(b2)])).body)
  loops, in_place = _scan_procs(src)
  assert not loops, f"loops without the in-place proof: {loops}"
  factor_loops = [n for n in in_place if re.search(r"_f\d+_inplace", n)]
  assert len(factor_loops) == len(fact.segments)
  entry = src[src.index("int share_run(") :]
  assert sum(entry.count(f"{n}_raw(") for n in factor_loops) == len(fact.segments), "one factorization, however many solves"


def test_implicit_derivatives(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  k = _kkt(12, 5, 1e-2, 5)
  t = _triangle(k, "lower")
  mat = SparseMatrix.symbol("K", t)
  fact = SparseLDL(mat)
  b = sc.sym("b", 17)
  x = fact.solve(b)
  f = sc.sumsqr(x) + (x * b).sum()
  kv, bv = _values(mat, t), RNG.standard_normal(17)
  fn = _fn("impl", [mat.values, b], [f, gradient(f, mat.values), gradient(f, b), jacobian(f.reshape((1,)), mat.values), hessian(f, b)])
  val, gk, gb, jk, hb = fn._flat_numerical_call(kv, bv)
  scale = np.abs(gk).max()
  fd_k = finite_difference(lambda z: fn._flat_numerical_call(z, bv)[0].reshape(1), kv).reshape(-1)
  np.testing.assert_allclose(gk, fd_k, rtol=1e-5, atol=1e-6 * scale)
  np.testing.assert_allclose(jk.reshape(-1), gk, rtol=1e-9, atol=1e-11 * scale)
  kinv = np.linalg.inv(k.toarray())
  np.testing.assert_allclose(gb, 2 * kinv.T @ (kinv @ bv) + kinv @ bv + kinv.T @ bv, rtol=1e-9)
  np.testing.assert_allclose(hb, 2 * kinv.T @ kinv + kinv + kinv.T, rtol=1e-8, atol=1e-10)
  one = _fn("impl_jvp", [mat.values, b], [jvp(x, mat.values, sc.const(np.ones(mat.nnz)))])._flat_numerical_call(kv, bv)[0]
  dk = (t + sparse.tril(t, -1).T).toarray()
  np.testing.assert_allclose(one, -kinv @ ((dk != 0) * 1.0 @ (kinv @ bv)), rtol=1e-9, atol=1e-12)
  # The derivatives never differentiate the factorization: every loop in the gradient is in place.
  loops, _ = _scan_procs(str(render_c_module(_fn("impl_grad", [mat.values, b], [gradient(f, mat.values)])).body))
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
  with pytest.raises(ValueError, match="diagonal"):
    SparseLDL(SparseMatrix.symbol("nd", np.array([[1, 1], [1, 0]], dtype=bool)))
  fact = SparseLDL(SparseMatrix.symbol("d", np.eye(3, dtype=bool)))
  with pytest.raises(ValueError, match="length 3"):
    fact.solve(sc.sym("b", 4))
