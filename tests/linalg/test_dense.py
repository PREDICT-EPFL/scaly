"""Dense ``cholesky``, ``ldl`` and ``solve_triangular``: generated loops (and unrolled straight-line
code for small orders) checked against NumPy, derivatives in every mode against finite differences,
and structural sparsity against the Jacobian."""

from __future__ import annotations

import re

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.derivatives import gradient, hessian, jacobian
from scaly.ad.forward import jvp
from scaly.codegen import render_c_module, render_c_source
from scaly.linalg import cho_solve, cholesky, ldl, ldl_solve, ldl_unpack, solve, solve_triangular

ORDER = 8  # an order every test builds both ways, and its neighbours

RNG = np.random.default_rng(606)
SIZES = [1, 2, 3, ORDER, ORDER + 1, 13, 20]
FLAGS = [(lower, trans, unit) for lower in (True, False) for trans in (False, True) for unit in (False, True)]


def _fn(name, inputs, outputs):
  return sc.Function.from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


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


@pytest.mark.parametrize("n", [10, 11, 12, 14, 15, 37, 42, 64])
def test_tiled_cholesky_matches_numpy(n: int) -> None:
  """Above the unrolling threshold the Cholesky runs by register tiles; every remainder of the tile
  side (``n % 4``) takes its own edge tiles, and a larger order splits the dot products in quarters."""
  a = sc.sym("a", (n, n))
  s = _spd(n)
  (chol,) = _fn(f"tiled{n}", [a], [cholesky(a)])._flat_numerical_call(s + np.triu(RNG.standard_normal((n, n)), 1))
  np.testing.assert_allclose(chol, np.linalg.cholesky(s), rtol=1e-12, atol=1e-12)
  assert np.all(np.triu(chol, 1) == 0)
  assert np.abs(chol @ chol.T - s).max() <= 64 * np.finfo(float).eps * np.abs(s).max()


def test_cholesky_runs_by_tiles_and_ldl_by_entries() -> None:
  """The looped Cholesky is the tiled kernel (block loops, then dot products per tile); the looped
  ``ldl`` keeps the entry-at-a-time Crout loops."""
  a = sc.sym("a", (20, 20))
  with sc.options(linalg=dict(dense_unroll=0)):
    chol = str(render_c_module(_fn("tiles_chol", [a], [cholesky(a)])).body)
    packed = str(render_c_module(_fn("tiles_ldl", [a], [ldl(a)])).body)
  assert re.search(r"for \(long long tbi_\w+ = 0; tbi_\w+ < 5;", chol) and "fi_" not in chol
  assert "tbi_" not in packed and "fi_" in packed


def test_cholesky_of_an_indefinite_matrix_is_nan() -> None:
  a = sc.sym("a", (3, 3))
  (out,) = _fn("chol_nan", [a], [cholesky(a)])._flat_numerical_call(np.diag([1.0, -1.0, 1.0]))
  assert np.isnan(out[1, 1])
  b = sc.sym("b", (11, 11))  # tiled: the NaN pivot's row and every row after it
  (tiled,) = _fn("chol_nan_tiled", [b], [cholesky(b)])._flat_numerical_call(np.diag([1.0] * 5 + [-1.0] + [1.0] * 5))
  assert np.all(np.isfinite(tiled[:5, :5])) and np.isnan(tiled[5, 5]) and np.isnan(tiled[6:, 5]).all()


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


@pytest.mark.parametrize("n", [4, ORDER + 2])
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


@pytest.mark.parametrize(("target", "chol", "tri"), [("apple-m3", 23, 63), ("generic", 12, 26)])
def test_the_target_makes_small_bodies_straight_line(target: str, chol: int, tri: int) -> None:
  """Without the option, a factorization or solve is straight-line code while its body is under the
  target's ``straight_line_ops`` (4 096 operations on the M3, 682 for scalar C): ``n^3 / 3`` for
  ``cholesky`` and ``ldl``, ``n^2`` for each right-hand side of a solve. Past it the code loops, and
  the loop code does not grow with the order."""

  def body(name: str, n: int, op) -> str:
    a, b = sc.sym("a", (n, n)), sc.sym("b", n)
    return render_c_source(_fn(f"{name}{n}", [a, b], [op(a, b)]), target=target).split(f"int {name}{n}(")[1]

  for name, op, largest in (("chol", lambda a, b: cholesky(a), chol), ("ldl", lambda a, b: ldl(a), chol), ("tri", solve_triangular, tri)):
    assert "for (" not in body(name, largest, op)
    assert "for (" in body(name, largest + 1, op)
    # Orders a multiple of the Cholesky tile and of the blocks apart, both past the order the blocks
    # start at: the loops' remainders are the same.
    assert len(body(name, largest + 25, op).splitlines()) == len(body(name, largest + 49, op).splitlines()), (
      "the loop code does not grow with the order"
    )
  a, bm = sc.sym("a", (tri // 2, tri // 2)), sc.sym("bm", (tri // 2, 5))  # five right-hand sides: five times the body
  assert "for (" in render_c_source(_fn("tri_five", [a, bm], [solve_triangular(a, bm)]), target=target).split("int tri_five(")[1]


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


# --- blocked factorization and solves (C-215) ---------------------------------------------------------


@pytest.mark.parametrize("target", ["apple-m3", "x86-64-v3", "x86-64-v4", "armv8-a", "generic"])
@pytest.mark.parametrize("n", [48, 53, 64, 99])  # whole blocks, a last block of one to seven columns, rows a tile leaves
def test_a_cholesky_in_blocks_is_numpys(target: str, n: int) -> None:
  """From six blocks of the target's register tile the factor goes by blocks of columns (order 48
  with the M3's and AVX2's 8-column tiles, 24 with 4 columns, 96 with AVX-512's 16; never in scalar
  C): the same factor, its upper triangle exactly zero."""
  spd = _spd(n)
  a = sc.sym("a", (n, n))
  fn = _fn(f"blocked_chol_{n}_{target.replace('-', '_')}", [a], [cholesky(a)])
  with sc.target(target):
    got = fn._flat_numerical_call(spd)[0].reshape(n, n)
  np.testing.assert_allclose(got, np.linalg.cholesky(spd), rtol=1e-13, atol=1e-13)
  assert not np.triu(got, 1).any()


@pytest.mark.parametrize("target", ["apple-m3", "armv8-a"])
@pytest.mark.parametrize(("n", "m"), [(32, 8), (33, 9), (67, 13)])  # a last block of one row; right-hand sides a tile leaves
@pytest.mark.parametrize(("lower", "trans", "unit"), FLAGS)
def test_a_solve_in_blocks_is_the_substitution(target: str, n: int, m: int, lower: bool, trans: bool, unit: bool) -> None:
  """With a tile's width of right-hand sides or more, from four blocks, the substitution goes by
  blocks of unknowns, forward or backward, against the triangle or its transpose."""
  tri = _triangle(_spd(n) / n + np.eye(n), lower, unit)
  b = RNG.standard_normal((n, m))
  a, bm = sc.sym("a", (n, n)), sc.sym("b", (n, m))
  name = f"blocked_tri_{n}_{m}_{int(lower)}{int(trans)}{int(unit)}_{target.replace('-', '_')}"
  fn = _fn(name, [a, bm], [solve_triangular(a, bm, lower=lower, trans=trans, unit_diagonal=unit)])
  given = tri + np.diag(RNG.uniform(2.0, 3.0, n)) if unit else tri  # a unit diagonal is not read
  with sc.target(target):
    got = fn._flat_numerical_call(given, b)[0].reshape(n, m)
  np.testing.assert_allclose(got, np.linalg.solve(tri.T if trans else tri, b), rtol=1e-11, atol=1e-11)


def test_a_tile_too_narrow_for_quarters_keeps_the_row_code() -> None:
  """The blocks sum each dot product in four quarters of the columns before them, which takes a
  block width that is a multiple of four: a target whose registers hold only a two-column tile
  factors and solves row by row."""
  starved = sc.Target("starved", vector_bytes=16, vector_registers=16, fma_units=4, fma_latency=4)
  assert starved.product_tile == (14, 2)
  n, m = 48, 8
  spd, b = _spd(n), RNG.standard_normal((n, m))
  a, bm = sc.sym("a", (n, n)), sc.sym("b", (n, m))
  fn = _fn("narrow_tile", [a, bm], [cholesky(a), solve_triangular(a, bm, lower=True)])
  assert "bjb_" not in render_c_source(fn, target=starved) and "tb_" not in render_c_source(fn, target=starved)
  with sc.target(starved):
    factor, solved = fn._flat_numerical_call(spd, b)
  np.testing.assert_allclose(factor.reshape(n, n), np.linalg.cholesky(spd), rtol=1e-13, atol=1e-13)
  np.testing.assert_allclose(solved.reshape(n, m), np.linalg.solve(np.tril(spd), b), rtol=1e-11, atol=1e-11)


def test_blocks_start_where_they_are_faster() -> None:
  """On the M3: a Cholesky factor in blocks from order 48 (at 40 the Crout tiles were as fast), a
  solve from order 32 with eight right-hand sides; a single right-hand side keeps the row-by-row
  code, and so does scalar C."""

  def source(op, n: int, m: int | None, target: str) -> str:
    a, b = sc.sym("a", (n, n)), sc.sym("b", n if m is None else (n, m))
    return render_c_source(_fn(f"starts_{op.__name__}_{n}_{m}_{target.replace('-', '_')}", [a, b], [op(a, b)]), target=target)

  def chol(a, b):
    return cholesky(a)

  assert "bjb_" in source(chol, 48, None, "apple-m3") and "bjb_" not in source(chol, 47, None, "apple-m3")
  assert "bjb_" not in source(chol, 64, None, "generic")
  assert "tb_" in source(solve_triangular, 32, 8, "apple-m3")
  assert "tb_" not in source(solve_triangular, 31, 8, "apple-m3") and "tb_" not in source(solve_triangular, 32, 7, "apple-m3")
  assert "tb_" not in source(solve_triangular, 64, None, "apple-m3") and "tb_" not in source(solve_triangular, 64, 8, "generic")
