"""Dense ``cholesky``, ``ldl`` and ``solve_triangular``: generated loops (and unrolled straight-line
code for small orders) checked against NumPy, derivatives in every mode against finite differences,
and structural sparsity against the Jacobian."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.derivatives import gradient, hessian, jacobian
from scaly.ad.forward import jvp
from scaly.codegen import render_c_module
from scaly.linalg import cho_solve, cholesky, ldl, ldl_solve, ldl_unpack, solve, solve_triangular
from scaly.passes.lowering import DENSE_UNROLL

RNG = np.random.default_rng(606)
SIZES = [1, 2, 3, DENSE_UNROLL, DENSE_UNROLL + 1, 13, 20]
FLAGS = [(lower, trans, unit) for lower in (True, False) for trans in (False, True) for unit in (False, True)]


def _fn(name, inputs, outputs):
  return sc.Function._from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _spd(n: int) -> np.ndarray:
  m = RNG.standard_normal((n, n))
  return m @ m.T + n * np.eye(n)


def _quasi_definite(n: int) -> np.ndarray:
  signs = np.where(np.arange(n) % 3 == 2, -1.0, 1.0)
  m = RNG.standard_normal((n, n)) * 0.3
  return (m + m.T) * np.outer(signs, signs) * (np.outer(signs, signs) > 0) + np.diag(signs * n)


def _triangle(a: np.ndarray, lower: bool, unit: bool) -> np.ndarray:
  t = np.tril(a) if lower else np.triu(a)
  if unit:
    t = t - np.diag(np.diag(t)) + np.eye(a.shape[0])
  return t


@pytest.mark.parametrize("n", SIZES)
def test_factorizations_match_numpy_and_read_only_the_lower_triangle(n: int) -> None:
  a = sc.sym("a", (n, n))
  fn = _fn(f"fac{n}", [a], [cholesky(a), ldl(a)])
  s, q = _spd(n), _quasi_definite(n)
  noise = np.triu(RNG.standard_normal((n, n)), 1) * 100.0  # the upper triangle is never read
  chol, _ = fn._flat_numerical_call(s + noise)
  np.testing.assert_allclose(chol, np.linalg.cholesky(s), rtol=1e-12, atol=1e-12)
  _, packed = fn._flat_numerical_call(q + noise)
  unit_l, d = np.tril(packed, -1) + np.eye(n), np.diag(packed)
  np.testing.assert_allclose(unit_l @ np.diag(d) @ unit_l.T, q, rtol=1e-12, atol=1e-12)
  assert np.all(np.triu(packed, 1) == 0) and np.all(np.triu(chol, 1) == 0)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize(("lower", "trans", "unit"), FLAGS)
def test_triangular_solves_match_numpy(n: int, lower: bool, trans: bool, unit: bool) -> None:
  t, b, bm = sc.sym("t", (n, n)), sc.sym("b", n), sc.sym("bm", (n, 3))
  outs = [solve_triangular(t, b, lower=lower, trans=trans, unit_diagonal=unit), solve_triangular(t, bm, lower=lower, trans=trans, unit_diagonal=unit)]
  fn = _fn(f"tri{n}{int(lower)}{int(trans)}{int(unit)}", [t, b, bm], outs)
  tv = RNG.standard_normal((n, n)) * (0.5 / n) + np.diag(1.0 + RNG.random(n))  # well conditioned
  bv, bmv = RNG.standard_normal(n), RNG.standard_normal((n, 3))
  x, xm = fn._flat_numerical_call(tv, bv, bmv)
  op = _triangle(tv, lower, unit)
  op = op.T if trans else op
  np.testing.assert_allclose(x, np.linalg.solve(op, bv), rtol=1e-12, atol=1e-13)
  np.testing.assert_allclose(xm, np.linalg.solve(op, bmv), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("n", [3, 12])
def test_solves_built_from_factors(n: int) -> None:
  a, b, bm = sc.sym("a", (n, n)), sc.sym("b", n), sc.sym("bm", (n, 2))
  unit_l, d = ldl_unpack(ldl(a))
  outs = [cho_solve(cholesky(a), b), ldl_solve(ldl(a), bm), solve(a, b), solve(a, bm, assume="sym"), unit_l, d]
  fn = _fn(f"solves{n}", [a, b, bm], outs)
  s, q = _spd(n), _quasi_definite(n)
  bv, bmv = RNG.standard_normal(n), RNG.standard_normal((n, 2))
  got = fn._flat_numerical_call(s, bv, bmv)
  np.testing.assert_allclose(got[0], np.linalg.solve(s, bv), rtol=1e-11)
  np.testing.assert_allclose(got[2], np.linalg.solve(s, bv), rtol=1e-11)
  got_q = fn._flat_numerical_call(q, bv, bmv)
  np.testing.assert_allclose(got_q[1], np.linalg.solve(q, bmv), rtol=1e-11)
  np.testing.assert_allclose(got_q[3], np.linalg.solve(q, bmv), rtol=1e-11)
  np.testing.assert_allclose(got_q[4] @ np.diag(got_q[5]) @ got_q[4].T, q, rtol=1e-12, atol=1e-12)
  with pytest.raises(ValueError, match="assume"):
    solve(a, b, assume="lu")  # ty: ignore[invalid-argument-type]


def test_cholesky_of_an_indefinite_matrix_is_nan() -> None:
  a = sc.sym("a", (3, 3))
  (out,) = _fn("chol_nan", [a], [cholesky(a)])._flat_numerical_call(np.diag([1.0, -1.0, 1.0]))
  assert np.isnan(out[1, 1])


# --- derivatives --------------------------------------------------------------------------------


def _cases(n: int):
  weights = sc.const(RNG.standard_normal((n, n)))
  cases = {
    "cholesky": (lambda a, b, bm: (cholesky(a) * weights).sum() + sc.sumsqr(cholesky(a) @ b), _spd(n)),
    "ldl": (lambda a, b, bm: ldl(a).sin().sum() + sc.sumsqr(ldl(a) @ b), _quasi_definite(n)),
  }
  for lower, trans, unit in FLAGS:
    cases[f"tri{int(lower)}{int(trans)}{int(unit)}"] = (
      lambda a, b, bm, lower=lower, trans=trans, unit=unit: (
        sc.sumsqr(solve_triangular(a, b, lower=lower, trans=trans, unit_diagonal=unit))
        + solve_triangular(a, bm, lower=lower, trans=trans, unit_diagonal=unit).sin().sum()
      ),
      _spd(n) / n + np.eye(n),
    )
  return cases


@pytest.mark.parametrize("n", [4, DENSE_UNROLL + 2])
@pytest.mark.parametrize("name", ["cholesky", "ldl", *(f"tri{int(a)}{int(b)}{int(c)}" for a, b, c in FLAGS)])
def test_derivatives(monkeypatch: pytest.MonkeyPatch, n: int, name: str) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  build, av = _cases(n)[name]
  a, b, bm = sc.sym("a", (n, n)), sc.sym("b", n), sc.sym("bm", (n, 2))
  f = build(a, b, bm)
  fn = _fn(f"d_{name}{n}", [a, b, bm], [f, gradient(f, a), jacobian(f.reshape((1,)), a), gradient(f, b), hessian(f, b), hessian(f, a)])
  bv, bmv = RNG.standard_normal(n), RNG.standard_normal((n, 2))
  _, ga, ja, gb, hb, ha = fn._flat_numerical_call(av, bv, bmv)
  scale = max(1.0, float(np.abs(ga).max()))
  fda = finite_difference(lambda z: fn._flat_numerical_call(z.reshape(n, n), bv, bmv)[0].reshape(1), av.reshape(-1)).reshape(n, n)
  np.testing.assert_allclose(ga, fda, rtol=1e-5, atol=1e-6 * scale)
  np.testing.assert_allclose(ja.reshape(n, n), ga, rtol=1e-10, atol=1e-12 * scale)
  fdb = finite_difference(lambda z: fn._flat_numerical_call(av, z, bmv)[0].reshape(1), bv).reshape(-1)
  np.testing.assert_allclose(gb, fdb, rtol=1e-5, atol=1e-6 * max(1.0, float(np.abs(gb).max())))
  grad_b = _fn(f"gb_{name}{n}", [a, b, bm], [gradient(f, b)])
  fdh = finite_difference(lambda z: grad_b._flat_numerical_call(av, z, bmv)[0].reshape(-1), bv)
  np.testing.assert_allclose(hb, fdh, rtol=1e-5, atol=1e-6 * max(1.0, float(np.abs(hb).max())))
  np.testing.assert_allclose(ha, ha.T, rtol=1e-8, atol=1e-10 * max(1.0, float(np.abs(ha).max())))


def test_sparsity_is_conservative_and_structural() -> None:
  n = 5
  a, bm = sc.sym("a", (n, n)), sc.sym("bm", (n, 3))
  mask = sc.jacobian_sparsity(cholesky(a), a).to_mask()
  upper_rows = [i * n + j for i in range(n) for j in range(n) if j > i]
  assert not mask[upper_rows].any() and not mask[:, upper_rows].any()
  x = solve_triangular(a, bm, lower=True)
  mask_b = sc.jacobian_sparsity(x, bm).to_mask().reshape(n, 3, n, 3)
  assert not any(mask_b[:, c, :, c2].any() for c in range(3) for c2 in range(3) if c != c2)
  av, bv = _spd(n) / n + np.eye(n), RNG.standard_normal((n, 3))
  jac = _fn("sp_tri", [a, bm], [jacobian(x, a), jacobian(x, bm)])._flat_numerical_call(av, bv)
  assert not np.any((jac[0] != 0) & ~sc.jacobian_sparsity(x, a).to_mask())
  assert not np.any((jac[1] != 0) & ~sc.jacobian_sparsity(x, bm).to_mask())


def test_small_orders_are_straight_line_and_large_ones_loops() -> None:
  srcs = {}
  for n in (DENSE_UNROLL, 24, 48):
    a, b = sc.sym("a", (n, n)), sc.sym("b", n)
    srcs[n] = str(render_c_module(_fn(f"shape{n}", [a, b], [cholesky(a), ldl(a), solve_triangular(a, b)])).body)
  assert "for (" not in srcs[DENSE_UNROLL].split("int shape")[1]
  lines = {n: len(src.splitlines()) for n, src in srcs.items()}
  assert lines[24] == lines[48], "the loop code does not grow with the order"


def test_validation_and_vmap() -> None:
  with pytest.raises(ValueError, match="square"):
    cholesky(sc.sym("r", (2, 3)))
  with pytest.raises(TypeError, match="floating"):
    ldl(sc.sym("i", (2, 2), dtype="int64"))
  with pytest.raises(ValueError, match="right-hand side"):
    solve_triangular(sc.sym("t", (3, 3)), sc.sym("b", 4))
  n, k = 6, 5
  a = sc.sym("a", (n, n))
  callee = _fn("vm_chol", [a], [cholesky(a).reshape((n * n,))])
  stack = sc.sym("stack", k * n * n)
  mats = [_spd(n) for _ in range(k)]
  (got,) = _fn("vm_run", [stack], [sc.vmap(callee, k, [(stack, 0, n * n)])])._flat_numerical_call(np.concatenate([m.reshape(-1) for m in mats]))
  for i, m in enumerate(mats):
    np.testing.assert_allclose(got[i * n * n : (i + 1) * n * n].reshape(n, n), np.linalg.cholesky(m), rtol=1e-12)


def test_exposed_under_scaly_linalg() -> None:
  assert sc.linalg.cholesky is cholesky and sc.linalg.solve_triangular is solve_triangular


@pytest.mark.parametrize("name", ["cholesky", "ldl", *(f"tri{int(a)}{int(b)}{int(c)}" for a, b, c in FLAGS)])
def test_single_seed_tangents_match_the_multi_seed_jacobian(name: str) -> None:
  n = 5
  build, av = _cases(n)[name]
  a, b, bm = sc.sym("a", (n, n)), sc.sym("b", n), sc.sym("bm", (n, 2))
  f = build(a, b, bm)
  direction = RNG.standard_normal((n, n))
  fn = _fn(f"jvp1_{name}", [a, b, bm], [jvp(f, a, sc.const(direction)), jacobian(f.reshape((1,)), a)])
  one, jac = fn._flat_numerical_call(av, RNG.standard_normal(n), RNG.standard_normal((n, 2)))
  np.testing.assert_allclose(one, jac.reshape(-1) @ direction.reshape(-1), rtol=1e-10, atol=1e-12)


SECOND_ORDER = {
  "cholesky": (lambda a, b: cholesky(a).sin().sum() + sc.sumsqr(cholesky(a) @ b), lambda n: _spd(n)),
  "ldl": (lambda a, b: ldl(a).sin().sum() + sc.sumsqr(ldl(a) @ b), lambda n: _quasi_definite(n)),
  "trisolve": (lambda a, b: sc.sumsqr(solve_triangular(a, b, trans=True)).sqrt(), lambda n: _spd(n) / n + np.eye(n)),
  "solve_sym": (lambda a, b: sc.sumsqr(solve(a, b, assume="sym")), lambda n: _quasi_definite(n)),
}


@pytest.mark.parametrize("n", [4, 10])
@pytest.mark.parametrize("name", list(SECOND_ORDER))
def test_second_derivatives_in_the_matrix_match_finite_differences(name: str, n: int) -> None:
  build, make = SECOND_ORDER[name]
  a, b = sc.sym("a", (n, n)), sc.sym("b", n)
  f = build(a, b)
  fn = _fn(f"so_{name}{n}", [a, b], [gradient(f, a), hessian(f, a), jacobian(gradient(f, a).reshape((n * n,)), b)])
  av, bv = make(n), RNG.standard_normal(n)
  _, h, hab = fn._flat_numerical_call(av, bv)
  fd = finite_difference(lambda z: fn._flat_numerical_call(z.reshape(n, n), bv)[0].reshape(-1), av.reshape(-1))
  np.testing.assert_allclose(h.reshape(n * n, n * n), fd, rtol=1e-4, atol=1e-5 * max(1.0, np.abs(h).max()))
  fd_b = finite_difference(lambda z: fn._flat_numerical_call(av, z)[0].reshape(-1), bv)
  np.testing.assert_allclose(hab.reshape(n * n, n), fd_b, rtol=1e-4, atol=1e-5 * max(1.0, np.abs(hab).max()))
