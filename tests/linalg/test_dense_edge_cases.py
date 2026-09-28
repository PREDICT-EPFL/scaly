"""Edge cases of the dense ``cholesky``, ``ldl``, ``lu`` and ``solve_triangular`` and the solves built on
them: every order next to the unrolling threshold and the Cholesky tile side, built both looped and
straight-line and checked against each other and against NumPy and SciPy; identity, diagonal,
permutation, tied-pivot, singular, badly scaled, ill-conditioned and non-finite inputs; NaN in
whatever an op must not read; quasi-definite ``ldl``; float32; empty shapes and refusals;
derivatives at order one, across the threshold and through a pivoting general solve; the ops inside
``vmap``, ``scan`` and ``while_loop``; constant matrices."""

from __future__ import annotations

import zlib

import numpy as np
import pytest
import scipy.linalg as sl

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.derivatives import gradient, hessian, jacobian
from scaly.ad.forward import jvp
from scaly.ir.expr import Expr, ExprOp
from scaly.ir.expr_spec import verify_expr
from scaly.ir.spec import VerifyError
from scaly.ir.types import TensorType, dtypes
from scaly.linalg import cho_solve, cholesky, ldl, ldl_solve, ldl_unpack, lu, lu_solve, solve, solve_triangular
from scaly.linalg.ops import CHOLESKY_TILE, DENSE_UNROLL
from scaly.linalg.ops.dense import CHOLESKY, LDL
from scaly.linalg.ops.trisolve import TRISOLVE

FLAGS = [(lower, trans, unit) for lower in (True, False) for trans in (False, True) for unit in (False, True)]
# One below, at and one above the default unrolling threshold and the tile side, and past two tiles.
ORDERS = sorted({1, 2, CHOLESKY_TILE - 1, CHOLESKY_TILE, CHOLESKY_TILE + 1, DENSE_UNROLL - 1, DENSE_UNROLL, DENSE_UNROLL + 1, 2 * CHOLESKY_TILE + 1})
EPS = np.finfo(np.float64).eps


@pytest.fixture
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
  return np.random.default_rng(zlib.crc32(request.node.name.encode()))


def _fn(name, inputs, outputs):
  return sc.Function.from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _forms(n: int) -> tuple[int, int]:
  """The ``dense_unroll`` settings that build order ``n`` as loops and as straight-line code."""
  return n - 1, n


def _close(got, want, rtol: float = 1e-12) -> None:
  want = np.asarray(want)
  np.testing.assert_allclose(got, want, rtol=rtol, atol=rtol * max(1.0, float(np.abs(want).max(initial=0.0))))


def _spd(rng: np.random.Generator, n: int) -> np.ndarray:
  m = rng.standard_normal((n, n))
  return m @ m.T + n * np.eye(n)


def _quasi_definite(rng: np.random.Generator, p: int, q: int, *, negative_first: bool = False) -> np.ndarray:
  """``[[P, B^T], [B, -N]]`` (or its negative block first) with ``P`` and ``N`` positive definite."""
  pos, neg, coupling = _spd(rng, p), -_spd(rng, q), rng.standard_normal((q, p))
  if negative_first:
    return np.block([[neg, coupling], [coupling.T, pos]])
  return np.block([[pos, coupling.T], [coupling, neg]])


def _cyclic(rng: np.random.Generator, n: int) -> np.ndarray:
  """A cyclic shift plus small noise: partial pivoting swaps rows at every column."""
  return np.roll(np.eye(n), 1, axis=1) + 0.1 * rng.standard_normal((n, n))


def _triangle_operand(rng: np.random.Generator, n: int) -> np.ndarray:
  """Both triangles filled, each well conditioned as a triangular matrix, with diagonal entries of either sign."""
  return rng.standard_normal((n, n)) * (0.5 / n) + np.diag(rng.choice([-1.0, 1.0], n) * (1.0 + rng.random(n)))


def _garbage(a: np.ndarray, *, lower: bool = True, unit: bool = False) -> np.ndarray:
  """``a`` with NaN wherever an op reading its ``lower`` triangle (and diagonal unless ``unit``) must not look."""
  unread = ~(np.tril(np.ones(a.shape, dtype=bool)) if lower else np.triu(np.ones(a.shape, dtype=bool)))
  if unit:
    np.fill_diagonal(unread, True)
  return np.where(unread, np.nan, a)


def _unpack_lu(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  n = packed.shape[1]
  return packed[n].astype(np.int64), np.tril(packed[:n], -1) + np.eye(n), np.triu(packed[:n])


def _reconstruct_ldl(packed: np.ndarray) -> np.ndarray:
  unit_l = np.tril(packed, -1) + np.eye(packed.shape[0])
  return unit_l @ np.diag(np.diag(packed)) @ unit_l.T


# --- orders around the thresholds ----------------------------------------------------------------


@pytest.mark.parametrize("n", [*ORDERS, 12])
def test_factorizations_agree_looped_unrolled_and_with_references(rng: np.random.Generator, n: int) -> None:
  """Each order built as loops (the tiled Cholesky) and as straight-line code; 12 forces straight-line
  code past the default threshold. The factorizations read no NaN above the diagonal."""
  s, k, g = _spd(rng, n), _quasi_definite(rng, n - n // 2, n // 2), _cyclic(rng, n)
  got = []
  for unroll in _forms(n):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      a, q, m = sc.sym("a", (n, n)), sc.sym("q", (n, n)), sc.sym("m", (n, n))
      outs = [cholesky(a), ldl(q), lu(m)]
    assert [o.attrs["unroll"] for o in outs] == [unroll >= n] * 3
    got.append(_fn(f"de_fac{n}_{unroll}", [a, q, m], outs)._flat_numerical_call(_garbage(s), _garbage(k), g))
  (chol, packed, fac), (chol_flat, packed_flat, fac_flat) = got
  np.testing.assert_array_equal(fac, fac_flat)  # the same operations in the same order
  _close(chol, chol_flat, 1e-13)
  _close(packed, packed_flat, 1e-13)
  _close(chol, np.linalg.cholesky(s))
  _close(_reconstruct_ldl(packed), k)
  assert np.all(np.triu(chol, 1) == 0) and np.all(np.triu(packed, 1) == 0)
  perm, unit_l, upper = _unpack_lu(fac)
  p, l_ref, u_ref = sl.lu(g)
  np.testing.assert_array_equal(perm, np.argmax(p, axis=0))
  assert n == 1 or np.all(perm != np.arange(n)), "every column of the cyclic shift takes a row swap"
  _close(unit_l, l_ref)
  _close(upper, u_ref)


@pytest.mark.parametrize("n", ORDERS)
def test_triangular_solves_read_only_their_triangle_across_the_threshold(rng: np.random.Generator, n: int) -> None:
  """All eight ``lower``/``trans``/``unit_diagonal`` combinations for a vector, a one-column and a
  five-column right-hand side; each matrix is NaN wherever its solve must not read."""
  t = _triangle_operand(rng, n)
  rhs = {"v": rng.standard_normal(n), "c": rng.standard_normal((n, 1)), "m": rng.standard_normal((n, 5))}
  keys = [(lower, unit) for lower in (True, False) for unit in (False, True)]
  got = []
  for unroll in _forms(n):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      ts = {key: sc.sym(f"t{int(key[0])}{int(key[1])}", (n, n)) for key in keys}
      bs = {name: sc.sym(name, v.shape) for name, v in rhs.items()}
      outs = [
        solve_triangular(ts[lower, unit], bs[name], lower=lower, trans=trans, unit_diagonal=unit) for lower, trans, unit in FLAGS for name in rhs
      ]
    fn = _fn(f"de_tri{n}_{unroll}", [*ts.values(), *bs.values()], outs)
    got.append(fn._flat_numerical_call(*(_garbage(t, lower=lower, unit=unit) for lower, unit in keys), *rhs.values()))
  refs = [sl.solve_triangular(t, v, lower=lower, trans=int(trans), unit_diagonal=unit) for lower, trans, unit in FLAGS for v in rhs.values()]
  for looped, flat, ref in zip(*got, refs, strict=True):
    _close(looped, flat, 1e-13)
    _close(flat, ref)


@pytest.mark.parametrize("n", ORDERS)
def test_solves_built_on_the_factorizations_across_the_threshold(rng: np.random.Generator, n: int) -> None:
  s, k, g = _spd(rng, n), _quasi_definite(rng, n - n // 2, n // 2), _cyclic(rng, n)
  b, bc, bm = rng.standard_normal(n), rng.standard_normal((n, 1)), rng.standard_normal((n, 3))
  got = []
  for unroll in _forms(n):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      a, q, m = sc.sym("a", (n, n)), sc.sym("q", (n, n)), sc.sym("m", (n, n))
      v, c, w = sc.sym("b", n), sc.sym("c", (n, 1)), sc.sym("w", (n, 3))
      outs = [
        cho_solve(cholesky(a), v),
        solve(a, w),
        ldl_solve(ldl(q), c),
        solve(q, w, assume="sym"),
        lu_solve(lu(m), v),
        lu_solve(lu(m), w, trans=True),
        lu_solve(lu(m), c, trans=True),
        solve(m, v, assume="gen"),
        solve(m, c, assume="gen"),
      ]
    got.append(_fn(f"de_fsolve{n}_{unroll}", [a, q, m, v, c, w], outs)._flat_numerical_call(_garbage(s), _garbage(k), g, b, bc, bm))
  refs = [np.linalg.solve(*pair) for pair in ((s, b), (s, bm), (k, bc), (k, bm), (g, b), (g.T, bm), (g.T, bc), (g, b), (g, bc))]
  for looped, flat, ref in zip(*got, refs, strict=True):
    _close(looped, flat, 1e-12)
    _close(flat, ref, 1e-11)


# --- structured matrices -------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 5, DENSE_UNROLL + 1])
def test_diagonal_triangular_and_permutation_matrices_give_exact_results(rng: np.random.Generator, n: int) -> None:
  v = rng.choice([-1.0, 1.0], n) * (0.5 + rng.random(n))
  upper = np.triu(rng.standard_normal((n, n)), 1) + np.diag(v)
  perms = [np.eye(n)[::-1], np.roll(np.eye(n), 1, axis=1)]
  b = rng.standard_normal(n)
  for unroll in _forms(n):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      a, x = sc.sym("a", (n, n)), sc.sym("b", n)
      tri = [solve_triangular(a, x, lower=lower, trans=trans, unit_diagonal=unit) for lower, trans, unit in FLAGS]
      fn = _fn(f"de_exact{n}_{unroll}", [a, x], [cholesky(a), ldl(a), lu(a), solve(a, x, assume="gen"), lu_solve(lu(a), x, trans=True), *tri])
    got = fn._flat_numerical_call(np.diag(np.abs(v)), b)
    np.testing.assert_array_equal(got[0], np.diag(np.sqrt(np.abs(v))))
    for x_tri, (_, _, unit) in zip(got[5:], FLAGS, strict=True):
      np.testing.assert_array_equal(x_tri, b if unit else b / np.abs(v))
    _, packed, fac, x_gen, x_t, *_ = fn._flat_numerical_call(np.diag(v), b)
    np.testing.assert_array_equal(packed, np.diag(v))
    np.testing.assert_array_equal(fac, np.vstack([np.diag(v), np.arange(n)]))
    np.testing.assert_array_equal(x_gen, b / v)
    np.testing.assert_array_equal(x_t, b / v)
    (fac,) = fn._flat_numerical_call(upper, b)[2:3]
    np.testing.assert_array_equal(fac, np.vstack([upper, np.arange(n)]))  # nothing below the diagonal: no swap, no update
    for p in perms:
      _, _, fac, x_gen, x_t, *_ = fn._flat_numerical_call(p, b)
      np.testing.assert_array_equal(fac, np.vstack([np.eye(n), np.argmax(p, axis=0)]))
      np.testing.assert_array_equal(x_gen, b[np.argmax(p, axis=0)])  # P^T b
      np.testing.assert_array_equal(x_t, b[np.argmax(p, axis=1)])  # P b


def _signed_hadamard(rng: np.random.Generator, n: int) -> np.ndarray:
  """A Hadamard matrix with rows shuffled and negated: every pivot search ties between entries of
  opposite sign, and the arithmetic stays exact."""
  return (sl.hadamard(n) * rng.choice([-1.0, 1.0], n)[:, None])[rng.permutation(n)]


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_tied_pivots_choose_the_first_row_as_lapack_does(rng: np.random.Generator, n: int) -> None:
  h = _signed_hadamard(rng, n)
  p, l_ref, u_ref = sl.lu(h)
  explicit = np.ones((5, 5)) + 4 * np.eye(5)
  explicit[:, 0] = [0.5, -3.0, 3.0, -3.0, 1.0]  # -3 in row 1 ties +3 in row 2 and -3 in row 3
  for unroll in _forms(n) if n <= DENSE_UNROLL else (n - 1,):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      a, e = sc.sym("a", (n, n)), sc.sym("e", (5, 5))
      fac, first = _fn(f"de_ties{n}_{unroll}", [a, e], [lu(a), lu(e)])._flat_numerical_call(h, explicit)
    perm, unit_l, upper = _unpack_lu(fac)
    np.testing.assert_array_equal(perm, np.argmax(p, axis=0))
    np.testing.assert_array_equal(unit_l, l_ref)
    np.testing.assert_array_equal(upper, u_ref)
    assert first[5, 0] == 1.0


@pytest.mark.parametrize("n", [4, DENSE_UNROLL + 1])
def test_singular_matrices_give_a_zero_pivot_and_nonfinite_values_after_it(rng: np.random.Generator, n: int) -> None:
  j = n // 2
  zero_column = rng.standard_normal((n, n))
  zero_column[:, j] = 0.0
  duplicate = rng.standard_normal((n, n))
  duplicate[n - 1] = duplicate[1]
  b = rng.standard_normal(n)
  for unroll in _forms(n):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      a, x = sc.sym("a", (n, n)), sc.sym("b", n)
      fn = _fn(f"de_singular{n}_{unroll}", [a, x], [lu(a), solve(a, x, assume="gen"), cholesky(a), ldl(a), solve(a, x)])
    fac, x_gen, *_ = fn._flat_numerical_call(zero_column, b)
    assert fac[j, j] == 0.0 and np.isnan(fac[j + 1 : n, j]).all()  # 0 / 0 below the zero pivot
    assert np.all(np.isfinite(fac[:j, :j])) and not np.isfinite(x_gen).any()
    fac, x_gen, *_ = fn._flat_numerical_call(duplicate, b)
    assert np.any(np.diag(fac[:n]) == 0.0) and not np.isfinite(x_gen).all()
    _, _, chol, packed, x_pos = fn._flat_numerical_call(np.ones((n, n)), b)  # rank one: the second pivot is exactly zero
    assert chol[1, 1] == 0.0 and packed[1, 1] == 0.0
    assert np.isnan(chol[2:, 1]).all() and np.isnan(packed[2:, 1]).all() and not np.isfinite(x_pos).all()
    _, _, chol, _, _ = fn._flat_numerical_call(-np.eye(n), b)
    assert np.isnan(np.diag(chol)).all()  # every pivot is the square root of a negative number


@pytest.mark.parametrize("n", [5, DENSE_UNROLL + 1])
def test_power_of_two_scaling_to_the_ends_of_the_exponent_range_is_exact(rng: np.random.Generator, n: int) -> None:
  """Scaling by powers of two commutes with every rounding, so matrices whose entries run from about
  1e-300 to 1e300 factor and solve to the unscaled results scaled, bit for bit."""
  d = 2.0 ** np.round(np.linspace(-490, 490, n))  # symmetric scaling: D S D spans 2^-980 .. 2^980
  e = 2.0 ** np.round(np.linspace(-990, 990, n))  # one-sided scaling of columns or rows
  s, g, t, b = _spd(rng, n), _cyclic(rng, n), _triangle_operand(rng, n), rng.standard_normal(n)
  combos = [(lower, trans) for lower in (True, False) for trans in (False, True)]
  for unroll in _forms(n):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      a, m, tt, x, c = sc.sym("a", (n, n)), sc.sym("m", (n, n)), sc.sym("t", (n, n)), sc.sym("b", n), sc.sym("c", n)
      tri = [solve_triangular(tt, x if trans else c, lower=lower, trans=trans) for lower, trans in combos]
      fn = _fn(f"de_scaled{n}_{unroll}", [a, m, tt, x, c], [cholesky(a), ldl(a), lu(m), solve(m, x, assume="gen"), *tri])
    chol, packed, fac, x_gen, *xs = fn._flat_numerical_call(s, g, t, b, b)
    big = [d[:, None] * s * d[None, :], g * e[None, :], e[:, None] * t]
    assert min(np.abs(x).min() for x in big) < 1e-290 and max(np.abs(x).max() for x in big) > 1e290
    chol_s, packed_s, fac_s, x_gen_s, *xs_s = fn._flat_numerical_call(*big, b, e * b)
    np.testing.assert_array_equal(chol_s, d[:, None] * chol)
    expected = packed * np.outer(d, 1.0 / d)
    np.fill_diagonal(expected, np.diag(packed) * d**2)
    np.testing.assert_array_equal(packed_s, expected)
    np.testing.assert_array_equal(fac_s, np.vstack([np.where(np.tri(n, k=-1, dtype=bool), fac[:n], fac[:n] * e), fac[n]]))
    np.testing.assert_array_equal(x_gen_s, x_gen / e)
    for got, ref, (_, trans) in zip(xs_s, xs, combos, strict=True):
      np.testing.assert_array_equal(got, ref / e if trans else ref)


@pytest.mark.parametrize("n", [6, 10])
def test_hilbert_matrices_factor_and_solve_with_a_small_backward_error(rng: np.random.Generator, n: int) -> None:
  h, b = sl.hilbert(n), rng.standard_normal(n)
  norm = np.linalg.norm(h, 2)
  for unroll in _forms(n):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      a, x = sc.sym("a", (n, n)), sc.sym("b", n)
      outs = [cholesky(a), ldl(a), lu(a), solve(a, x), solve(a, x, assume="sym"), solve(a, x, assume="gen")]
    chol, packed, fac, *xs = _fn(f"de_hilbert{n}_{unroll}", [a, x], outs)._flat_numerical_call(h, b)
    perm, unit_l, upper = _unpack_lu(fac)
    for product, target in ((chol @ chol.T, h), (_reconstruct_ldl(packed), h), (unit_l @ upper, h[perm])):
      assert np.linalg.norm(product - target, 2) <= 4 * n * EPS * norm
    for sol in xs:
      assert np.linalg.norm(h @ sol - b) <= 8 * n * EPS * (norm * np.linalg.norm(sol) + np.linalg.norm(b))


# --- non-finite values ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [6, DENSE_UNROLL + 1])
def test_a_nan_entry_poisons_only_what_is_computed_from_it(rng: np.random.Generator, n: int) -> None:
  i, j = n - 2, 1
  s, g, b = _spd(rng, n), _cyclic(rng, n), rng.standard_normal(n)
  bad_s, bad_g, bad_b = s.copy(), g.copy(), b.copy()
  bad_s[i, j] = bad_g[i, j] = bad_b[j] = np.nan
  for unroll in _forms(n):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      a, m, x = sc.sym("a", (n, n)), sc.sym("m", (n, n)), sc.sym("b", n)
      fn = _fn(f"de_nan{n}_{unroll}", [a, m, x], [cholesky(a), ldl(a), solve(m, x, assume="gen")])
    clean_chol, clean_packed, _ = fn._flat_numerical_call(s, g, b)
    chol, packed, x_bad_g = fn._flat_numerical_call(bad_s, bad_g, b)
    for got, clean in ((chol, clean_chol), (packed, clean_packed)):
      np.testing.assert_array_equal(got[:i], clean[:i])  # rows above never read row i
      assert np.isnan(got[i, j]) and np.isnan(got[i:, i]).all()
    assert np.isnan(x_bad_g).all()
    assert np.isnan(fn._flat_numerical_call(s, g, bad_b)[2]).all()


@pytest.mark.parametrize("n", [6, DENSE_UNROLL + 1])
def test_triangular_solves_carry_nonfinite_values_only_to_later_unknowns(rng: np.random.Generator, n: int) -> None:
  """An inf in the right-hand side, a NaN in one column of a matrix right-hand side, or a zero on the
  diagonal leaves every unknown solved before it, and every other column, as without it."""
  i, col = n // 2, 2
  t, b, bm = _triangle_operand(rng, n), rng.standard_normal(n), rng.standard_normal((n, 4))
  b_inf, bm_nan, t_zero = b.copy(), bm.copy(), t.copy()
  b_inf[i], bm_nan[i, col], t_zero[i, i] = np.inf, np.nan, 0.0
  for unroll in _forms(n):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      tt, x, xm = sc.sym("t", (n, n)), sc.sym("b", n), sc.sym("bm", (n, 4))
      outs = [solve_triangular(tt, rhs, lower=lower, trans=trans, unit_diagonal=unit) for lower, trans, unit in FLAGS for rhs in (x, xm)]
    fn = _fn(f"de_inf{n}_{unroll}", [tt, x, xm], outs)
    clean, poisoned, zero = (fn._flat_numerical_call(*args) for args in ((t, b, bm), (t, b_inf, bm_nan), (t_zero, b, bm)))
    for k, (lower, trans, unit) in enumerate(FLAGS):
      before, after = (slice(0, i), slice(i + 1, n)) if lower != trans else (slice(i + 1, n), slice(0, i))
      vec, mat, vec_zero = poisoned[2 * k], poisoned[2 * k + 1], zero[2 * k]
      np.testing.assert_array_equal(vec[before], clean[2 * k][before])
      assert np.isinf(vec[i]) and not np.isfinite(vec[after]).any()
      np.testing.assert_array_equal(np.delete(mat, col, axis=1), np.delete(clean[2 * k + 1], col, axis=1))
      np.testing.assert_array_equal(mat[before, col], clean[2 * k + 1][before, col])
      assert np.isnan(mat[i, col]) and np.isnan(mat[after, col]).all()
      if unit:
        np.testing.assert_array_equal(vec_zero, clean[2 * k])  # the diagonal is never read
      else:
        np.testing.assert_array_equal(vec_zero[before], clean[2 * k][before])
        assert not np.isfinite(vec_zero[i]) and not np.isfinite(vec_zero[after]).any()


# --- quasi-definite ldl --------------------------------------------------------------------------


@pytest.mark.parametrize(("p", "q"), [(0, 1), (1, 1), (2, 3), (5, 4)])
def test_ldl_factors_quasi_definite_matrices_where_cholesky_gives_nan(rng: np.random.Generator, p: int, q: int) -> None:
  n = p + q
  b = rng.standard_normal(n)
  for unroll in _forms(n):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      a, x = sc.sym("a", (n, n)), sc.sym("b", n)
      fn = _fn(f"de_quasi{p}{q}_{unroll}", [a, x], [ldl(a), *ldl_unpack(ldl(a)), solve(a, x, assume="sym"), cholesky(a)])
    for negative_first in (False, True):
      k = _quasi_definite(rng, p, q, negative_first=negative_first)
      packed, unit_l, d, sol, chol = fn._flat_numerical_call(_garbage(k), b)
      signs = [-1.0] * q + [1.0] * p if negative_first else [1.0] * p + [-1.0] * q
      np.testing.assert_array_equal(np.sign(d), signs)  # the inertia of K
      np.testing.assert_array_equal(unit_l, np.tril(packed, -1) + np.eye(n))
      _close(unit_l @ np.diag(d) @ unit_l.T, k)
      _close(sol, np.linalg.solve(k, b), 1e-11)
      first = 0 if negative_first else p
      assert np.isnan(chol[first, first]) and np.isfinite(chol[:first]).all()


# --- float32 -------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, DENSE_UNROLL + 1])
def test_float32_ops_compute_in_single_precision(rng: np.random.Generator, n: int) -> None:
  f32 = np.float32
  s, g, t = _spd(rng, n).astype(f32), _cyclic(rng, n).astype(f32), _triangle_operand(rng, n).astype(f32)
  b, bm = rng.standard_normal(n).astype(f32), rng.standard_normal((n, 2)).astype(f32)
  s64, g64, t64, b64, bm64 = (x.astype(np.float64) for x in (s, g, t, b, bm))
  for unroll in _forms(n):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      a, m, tt = (sc.sym(v, (n, n), dtype="float32") for v in ("a", "m", "t"))
      x, w = sc.sym("b", n, dtype="float32"), sc.sym("w", (n, 2), dtype="float32")
      outs = [
        cholesky(a),
        ldl(a),
        lu(m),
        solve_triangular(tt, x, lower=False, trans=True),
        solve_triangular(tt, w, unit_diagonal=True),
        cho_solve(cholesky(a), w),
        solve(m, x, assume="gen"),
        solve(m, w, assume="gen"),
        lu_solve(lu(m), x, trans=True),
      ]
    assert all(o.type.dtype == dtypes.float32 for o in outs)
    got = _fn(f"de_f32_{n}_{unroll}", [a, m, tt, x, w], outs)._flat_numerical_call(s, g, t, b, bm)
    for out in got:
      np.testing.assert_array_equal(out, out.astype(f32))  # every result is a float32 value
    chol, packed, fac, *xs = got
    _close(chol, np.linalg.cholesky(s64), 1e-5)
    _close(_reconstruct_ldl(packed), s64, 1e-5)
    np.testing.assert_array_equal(fac[n], np.argmax(sl.lu(g64)[0], axis=0))
    refs = [
      sl.solve_triangular(t64, b64, lower=False, trans=1),
      sl.solve_triangular(t64, bm64, unit_diagonal=True, lower=True),
      np.linalg.solve(s64, bm64),
      np.linalg.solve(g64, b64),
      np.linalg.solve(g64, bm64),
      np.linalg.solve(g64.T, b64),
    ]
    for sol, ref in zip(xs, refs, strict=True):
      _close(sol, ref, 1e-5)


def test_float32_ldl_solve() -> None:
  a, b = sc.sym("a", (3, 3), dtype="float32"), sc.sym("b", 3, dtype="float32")
  (x,) = _fn("de_f32_ldl", [a, b], [solve(a, b, assume="sym")])._flat_numerical_call(np.diag([2.0, -1.0, 4.0]), np.ones(3))
  np.testing.assert_allclose(x, [0.5, -1.0, 0.25], rtol=1e-6)


# --- shapes and refusals -------------------------------------------------------------------------


@pytest.mark.parametrize("unroll", [0, DENSE_UNROLL])
def test_empty_orders_and_right_hand_sides_give_empty_results(unroll: int) -> None:
  with sc.options(linalg=dict(dense_unroll=unroll)):
    e, z, t, bz = sc.sym("e", (0, 0)), sc.sym("z", 0), sc.sym("t", (3, 3)), sc.sym("bz", (3, 0))
    empty = [cholesky(e), ldl(e), lu(e), solve_triangular(e, z), cho_solve(cholesky(e), z), lu_solve(lu(e), z), solve(e, z, assume="gen")]
    tri = [solve_triangular(t, bz, lower=lower, trans=trans, unit_diagonal=unit) for lower, trans, unit in FLAGS]
    wide = [cho_solve(cholesky(t), bz), ldl_solve(ldl(t), bz), lu_solve(lu(t), bz), lu_solve(lu(t), bz, trans=True), solve(t, bz, assume="gen")]
  got = _fn(f"de_empty{unroll}", [e, z, t, bz], [*empty, *tri, *wide])._flat_numerical_call(
    np.zeros((0, 0)), np.zeros(0), 3 * np.eye(3), np.zeros((3, 0))
  )
  assert [x.shape for x in got] == [(0, 0), (0, 0), (1, 0), *[(0,)] * 4, *[(3, 0)] * 13]


def _mat(shape, dtype: str = "float64") -> Expr:
  return sc.sym("r", shape, dtype=dtype)


MATRIX_OPS = {
  "cholesky": cholesky,
  "ldl": ldl,
  "lu": lu,
  "solve_triangular": lambda a: solve_triangular(a, sc.sym("b", 2)),
  "solve_gen": lambda a: solve(a, sc.sym("b", 2), assume="gen"),
}
RHS_OPS = {
  "solve_triangular": solve_triangular,
  "cho_solve": cho_solve,
  "solve_gen": lambda a, b: solve(a, b, assume="gen"),
  "solve_sym": lambda a, b: solve(a, b, assume="sym"),
}
REFUSALS = [
  *(
    pytest.param(lambda op=op, shape=shape: op(_mat(shape)), ValueError, "needs a square matrix", id=f"{name}-{shape}")
    for name, op in MATRIX_OPS.items()
    for shape in ((), (2,), (2, 3), (2, 2, 2))
  ),
  *(
    pytest.param(lambda op=op, dtype=dtype: op(_mat((2, 2), dtype)), TypeError, "floating-point matrix", id=f"{name}-{dtype}")
    for name, op in MATRIX_OPS.items()
    for dtype in ("int64", "int32", "bool")
  ),
  *(
    pytest.param(lambda op=op, shape=shape: op(_mat((2, 2)), sc.sym("b", shape)), ValueError, "right-hand side of 2 rows", id=f"{name}-rhs{shape}")
    for name, op in RHS_OPS.items()
    for shape in ((), (3,), (3, 2), (2, 2, 1))
  ),
  pytest.param(lambda: lu_solve(_mat((3, 2)), sc.sym("b", (2, 1, 1))), ValueError, "right-hand side of 2 rows", id="lu_solve-rank3-rhs"),
  pytest.param(lambda: lu_solve(_mat((2, 2)), sc.sym("b", 2)), ValueError, r"shaped \(n \+ 1, n\)", id="lu_solve-square-factor"),
  pytest.param(lambda: lu_solve(_mat((3, 2, 2)), sc.sym("b", 2)), ValueError, r"shaped \(n \+ 1, n\)", id="lu_solve-rank3-factor"),
  pytest.param(lambda: solve_triangular(_mat((2, 2)), sc.sym("b", 2, dtype="int64")), TypeError, "mixed-dtype", id="int-rhs"),
  pytest.param(lambda: solve_triangular(_mat((2, 2), "float32"), sc.sym("b", 2)), TypeError, "mixed-dtype", id="float32-by-float64"),
  pytest.param(
    lambda: solve(_mat((2, 2), "float32"), sc.sym("b", 2), assume="gen"), ValueError, "dtype float64, expected float32", id="gen-float32-by-float64"
  ),
]


@pytest.mark.parametrize(("build", "error", "match"), REFUSALS)
def test_refusals(build, error: type[Exception], match: str) -> None:
  with pytest.raises(error, match=match):
    build()


def test_lu_solve_refuses_a_rank_one_factor_with_its_own_message() -> None:
  with pytest.raises(ValueError, match=r"shaped \(n \+ 1, n\)"):
    lu_solve(sc.sym("f", 6), sc.sym("b", 2))


def test_verify_rejects_malformed_dense_nodes() -> None:
  a, b = sc.sym("a", (3, 3)), sc.sym("b", 3)
  flags = {"lower": True, "trans": False, "unit": False}
  bad = [
    (Expr(CHOLESKY, (a,), TensorType((3, 4))), "needs a square matrix and keeps its shape"),
    (Expr(LDL, (sc.sym("r", (3, 4)),), TensorType((3, 4))), "needs a square matrix and keeps its shape"),
    (Expr(TRISOLVE, (a, b), TensorType((3,)), attrs={"lower": True, "trans": False}), "needs 'lower', 'trans' and 'unit' attrs"),
    (Expr(TRISOLVE, (a, sc.sym("b4", 4)), TensorType((4,)), attrs=flags), "is inconsistent"),
    (Expr(TRISOLVE, (a, b), TensorType((3, 1)), attrs=flags), "is inconsistent"),
  ]
  for expr, match in bad:
    with pytest.raises(VerifyError, match=match):
      verify_expr([expr])


def _trisolve_unroll_flags(expr: Expr) -> set[bool]:
  """The ``unroll`` decision of every triangular solve reachable from ``expr``, through calls."""
  seen, flags, stack = set(), set(), [expr]
  while stack:
    e = stack.pop()
    if e.id in seen:
      continue
    seen.add(e.id)
    if e.op == TRISOLVE:
      flags.add(bool(e.attrs["unroll"]))
    if e.op == ExprOp.CALL:
      stack.extend(e.attrs["callee"].outputs)
    stack.extend(e.args)
  return flags


def test_the_general_solve_follows_dense_unroll() -> None:
  n = 6
  a, b = sc.sym("a", (n, n)), sc.sym("b", n)
  xs = []
  for unroll in (n, n - 1, n):  # whichever form the cache holds, one of these differs from it
    with sc.options(linalg=dict(dense_unroll=unroll)):
      xs.append(solve(a, b, assume="gen"))
      assert _trisolve_unroll_flags(xs[-1]) == {unroll >= n}
  # Both forms in one graph: their Functions are named apart, and agree.
  av = np.eye(n)[::-1] + 0.1 * np.arange(n * n).reshape(n, n) / (n * n)
  got = _fn("de_gen_both_forms", [a, b], xs[:2])((av, np.arange(1.0, n + 1)))
  np.testing.assert_allclose(got[0], got[1], rtol=1e-13)
  np.testing.assert_allclose(got[0], np.linalg.solve(av, np.arange(1.0, n + 1)), rtol=1e-12)


def test_a_linalg_option_leaves_derivative_helpers_alone() -> None:
  """A derivative takes each factorization's choice from the node it differentiates, so the
  ``linalg`` options in force when it is built change neither its helpers nor their names: the
  gradient built under ``dense_unroll=0`` is the default one, node for node."""
  a = sc.sym("a", (3, 3))
  lane = _fn("de_two_unroll_lane", [a], [cholesky(a @ a.T + 3.0 * sc.const(np.eye(3))).sum().reshape((1,))])
  flat = sc.sym("flat", 18)
  cost = sc.vmap(lane, 2, [(flat, 0, 9)]).sum()
  grads = []
  for unroll in (DENSE_UNROLL, 0):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      grads.append(gradient(cost, flat))
  names = {e.attrs["callee"].name for g in grads for e in _nodes(g) if e.op == ExprOp.VMAP}
  assert len(names) == 1 and not any("_o" in name for name in names)
  assert grads[0] is grads[1]
  got = _fn("de_two_unroll", [flat], grads)(np.linspace(-1.0, 1.0, 18))
  np.testing.assert_allclose(got[0], got[1], rtol=1e-13)


def _nodes(e: Expr) -> list[Expr]:
  from scaly.ir.expr import topo

  return topo([e])


# --- derivatives ---------------------------------------------------------------------------------


@pytest.mark.parametrize("unroll", [0, 1])
def test_order_one_derivatives_are_the_scalar_closed_forms(monkeypatch: pytest.MonkeyPatch, unroll: int) -> None:
  """At order 1 each op is scalar arithmetic: ``sqrt(a)``, ``a``, ``b / a`` or ``b``. Forward with
  one seed and with many, reverse, and second order, looped (``dense_unroll=0``) and straight-line."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  av, tv, bv = 2.5, -0.8, -1.5
  with sc.options(linalg=dict(dense_unroll=unroll)):
    a, t, b = sc.sym("a", (1, 1)), sc.sym("t", (1, 1)), sc.sym("b", 1)
    cases = {
      "chol": (cholesky(a), a, [np.sqrt(av), 0.5 / np.sqrt(av), 0.0, -0.25 * av**-1.5]),
      "ldl": (ldl(a), a, [av, 1.0, 0.0, 0.0]),
      **{name: (solve(a, b, assume=name), a, [bv / av, -bv / av**2, 1.0 / av, 2 * bv / av**3]) for name in ("pos", "sym", "gen")},
      **{
        f"tri{int(lower)}{int(trans)}{int(unit)}": (
          solve_triangular(t, b, lower=lower, trans=trans, unit_diagonal=unit),
          t,
          [bv, 0.0, 1.0, 0.0] if unit else [bv / tv, -bv / tv**2, 1.0 / tv, 2 * bv / tv**3],
        )
        for lower, trans, unit in FLAGS
      },
    }
    outs = []
    for out, wrt, _ in cases.values():
      y = out.reshape((1,))
      outs += [y, jacobian(y, wrt), jvp(y, wrt, sc.const(np.ones((1, 1)))), gradient(y.sum(), wrt), jacobian(y, b), hessian(y.sum(), wrt)]
    fn = _fn(f"de_order1_{unroll}", [a, t, b], outs)
  got = fn._flat_numerical_call(np.full((1, 1), av), np.full((1, 1), tv), np.full(1, bv))
  for k, (name, (_, _, (value, d_mat, d_b, d2_mat))) in enumerate(cases.items()):
    y, jac, one, grad, jac_b, hess = (float(np.asarray(v).reshape(-1)[0]) for v in got[6 * k : 6 * k + 6])
    np.testing.assert_allclose([y, jac, one, grad, jac_b, hess], [value, d_mat, d_mat, d_mat, d_b, d2_mat], rtol=1e-13, atol=1e-15, err_msg=name)


@pytest.mark.parametrize("n", [DENSE_UNROLL, DENSE_UNROLL + 1])
def test_jacobians_agree_across_forms_and_with_finite_differences(monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator, n: int) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  s, bv = _spd(rng, n), rng.standard_normal(n)
  weights = rng.standard_normal(2 * n * n + 2 * n)
  got, fns = [], []
  for unroll in _forms(n):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      a, b = sc.sym("a", (n, n)), sc.sym("b", n)
      y = sc.concat(
        [cholesky(a).reshape((n * n,)), ldl(a).reshape((n * n,)), solve_triangular(a, b, lower=False, trans=True), solve(a, b, assume="gen")]
      )
      fn = _fn(f"de_jac{n}_{unroll}", [a, b], [y, jacobian(y, a), jacobian(y, b), gradient((y * sc.const(weights)).sum(), a)])
    fns.append(fn)
    got.append(fn._flat_numerical_call(s, bv))
  for looped, flat in zip(*got, strict=True):
    _close(looped, flat, 1e-11)
  _, jac_a, jac_b, grad_a = got[1]
  fd_a = finite_difference(lambda z: fns[1]._flat_numerical_call(z.reshape(n, n), bv)[0], s)
  fd_b = finite_difference(lambda z: fns[1]._flat_numerical_call(s, z)[0], bv)
  np.testing.assert_allclose(jac_a, fd_a, rtol=1e-5, atol=1e-6 * np.abs(jac_a).max())
  np.testing.assert_allclose(jac_b, fd_b, rtol=1e-5, atol=1e-6 * np.abs(jac_b).max())
  upper = np.flatnonzero(np.triu(np.ones((n, n)), 1))
  assert not jac_a[: 2 * n * n][:, upper].any(), "the factorizations never read the upper triangle"
  _close(grad_a.reshape(-1), weights @ jac_a, 1e-10)


@pytest.mark.parametrize("n", [1, 3, DENSE_UNROLL + 1])
def test_derivatives_read_the_lower_triangle_of_the_direction_as_a_symmetric_matrix(rng: np.random.Generator, n: int) -> None:
  """``cholesky`` and ``ldl`` read the lower triangle of a tangent as the symmetric matrix it stands
  for, and the reverse rule is the adjoint of that: ``<grad, D> = <W, dF[D]>`` for any ``D``, upper
  triangle included. The triangular solves likewise read only their triangle of the tangent."""
  s, t, bv = _spd(rng, n), _triangle_operand(rng, n), rng.standard_normal(n)
  d = rng.standard_normal((n, n))
  sym_d = np.tril(d) + np.tril(d, -1).T
  a, tt, b = sc.sym("a", (n, n)), sc.sym("t", (n, n)), sc.sym("b", n)
  ops = [cholesky(a), ldl(a), *(solve_triangular(tt, b, lower=lower, trans=trans, unit_diagonal=unit) for lower, trans, unit in FLAGS)]
  weights = [rng.standard_normal(op.shape) for op in ops]
  outs = []
  for op, w in zip(ops, weights, strict=True):
    wrt = a if op.op != TRISOLVE else tt
    outs += [jvp(op, wrt, sc.const(d)), jvp(op, wrt, sc.const(np.tril(d))), jvp(op, wrt, sc.const(sym_d)), gradient((op * sc.const(w)).sum(), wrt)]
  got = _fn(f"de_convention{n}", [a, tt, b], outs)._flat_numerical_call(s, t, bv)
  keys = [(True, False)] * 2 + [(lower, unit) for lower, _, unit in FLAGS]
  for k, (w, (lower, unit)) in enumerate(zip(weights, keys, strict=True)):
    along_d, along_low, along_sym, grad = got[4 * k : 4 * k + 4]
    read = np.tril(np.ones((n, n))) if lower else np.triu(np.ones((n, n)))
    if unit:
      np.fill_diagonal(read, 0.0)
    if k < 2:
      np.testing.assert_array_equal(along_d, along_low)
      np.testing.assert_array_equal(along_d, along_sym)
    np.testing.assert_array_equal(grad * (1.0 - read), 0.0)
    _close(np.sum(grad * d), np.sum(w * along_d), 1e-11)
  eps = 1e-6
  chol_fd = (np.linalg.cholesky(s + eps * sym_d) - np.linalg.cholesky(s - eps * sym_d)) / (2 * eps)
  np.testing.assert_allclose(got[0], chol_fd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("n", [1, 5, DENSE_UNROLL + 1])
def test_general_solve_derivatives_with_pivoting_and_several_right_hand_sides(
  monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator, n: int
) -> None:
  """``X = A^{-1} B`` for a matrix that swaps rows at every column and two right-hand sides, which the
  general solve maps over: ``dX = A^{-1} (dB - dA X)``, and the adjoints ``-A^{-T} W X^T`` and ``A^{-T} W``.
  A vector right-hand side takes the multi-seed rule; a mapped one falls back to one seed at a time."""
  g, bm, w = _cyclic(rng, n), rng.standard_normal((n, 2)), rng.standard_normal((n, 2))
  a, b, v = sc.sym("a", (n, n)), sc.sym("b", (n, 2)), sc.sym("v", n)
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  xv_vec = solve(a, v, assume="gen")
  vec = [jacobian(xv_vec, a), jacobian(xv_vec, v)]
  monkeypatch.delenv("SCALY_STRICT_JVP_MANY")
  x = solve(a, b, assume="gen")
  c = (x * sc.const(w)).sum()
  second = [hessian(c, a), jacobian(gradient(c, a).reshape((n * n,)), b)] if n <= 5 else []
  fn = _fn(f"de_gen_d{n}", [a, b, v], [*vec, x, jacobian(x, a), jacobian(x, b), gradient(c, a), gradient(c, b), *second])
  vec_a, vec_v, xv, jac_a, jac_b, grad_a, grad_b, *hess = fn._flat_numerical_call(g, bm, bm[:, 0])
  inv = np.linalg.inv(g)
  _close(vec_a, jac_a[::2], 1e-10)
  _close(vec_v, inv, 1e-10)
  _close(xv, inv @ bm, 1e-11)
  _close(jac_a, np.einsum("ik,lc->ickl", -inv, xv).reshape(2 * n, n * n), 1e-10)
  _close(jac_b, np.einsum("ik,cd->ickd", inv, np.eye(2)).reshape(2 * n, 2 * n), 1e-10)
  _close(grad_a, -(inv.T @ w) @ xv.T, 1e-10)
  _close(grad_b, inv.T @ w, 1e-10)
  if hess:
    grad_fn = _fn(f"de_gen_g{n}", [a, b], [gradient(c, a)])
    fd = finite_difference(lambda z: grad_fn._flat_numerical_call(z.reshape(n, n), bm)[0], g)
    np.testing.assert_allclose(hess[0], fd, rtol=1e-5, atol=1e-6 * np.abs(hess[0]).max())
    fd_b = finite_difference(lambda z: grad_fn._flat_numerical_call(g, z.reshape(n, 2))[0], bm)
    np.testing.assert_allclose(hess[1], fd_b, rtol=1e-5, atol=1e-6 * np.abs(hess[1]).max())


def test_lu_refuses_a_derivative_on_every_path_while_lu_solve_differentiates_in_b(monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  n = 4
  a, b = sc.sym("a", (n, n)), sc.sym("b", n)
  x, xt = lu_solve(lu(a), b), lu_solve(lu(a), b, trans=True)
  fn = _fn("de_lu_in_b", [a, b], [jacobian(x, b), jacobian(xt, b), gradient(sc.sumsqr(x), b)])
  g, bv = _cyclic(rng, n), rng.standard_normal(n)
  jac, jac_t, grad = fn._flat_numerical_call(g, bv)
  inv = np.linalg.inv(g)
  _close(jac, inv, 1e-12)
  _close(jac_t, inv.T, 1e-12)
  _close(grad, 2 * inv.T @ (inv @ bv), 1e-12)
  refused = [lambda: jacobian(x, a), lambda: gradient(xt.sum(), a), lambda: hessian(lu(a).sum(), a), lambda: jvp(x, a, sc.const(np.eye(n)))]
  for build in refused:
    with pytest.raises(NotImplementedError, match="dense LU factorization has no derivative"):
      build()


# --- inside maps and loops -----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 5])
def test_batched_ops_under_vmap_match_single_calls(rng: np.random.Generator, n: int) -> None:
  batch = 3
  data = [(_spd(rng, n), _quasi_definite(rng, n - n // 2, n // 2), _cyclic(rng, n), rng.standard_normal(n)) for _ in range(batch)]
  sizes = [n * n, n * n, n * n, n]
  for unroll in _forms(n):
    with sc.options(linalg=dict(dense_unroll=unroll)):
      s, q, g, b = sc.sym("s", (n, n)), sc.sym("q", (n, n)), sc.sym("g", (n, n)), sc.sym("b", n)
      outs = [
        cholesky(s),
        ldl(q),
        lu(g),
        solve_triangular(s, b, trans=True),
        solve(g, b, assume="gen"),
        cho_solve(cholesky(s), b),
        ldl_solve(ldl(q), b),
      ]
      callee = _fn(f"de_vm_callee{n}_{unroll}", [s, q, g, b], [sc.concat([o.reshape((o.size,)) for o in outs])])
      stacks = [sc.sym(f"all_{v}", batch * size) for v, size in zip("sqgb", sizes, strict=True)]
      mapped = sc.vmap(callee, batch, [(st, 0, size) for st, size in zip(stacks, sizes, strict=True)])
      fn = _fn(f"de_vm{n}_{unroll}", stacks, [mapped])
    (got,) = fn._flat_numerical_call(*(np.concatenate([item[k].reshape(-1) for item in data]) for k in range(4)))
    width = callee.outputs[0].size
    for i, item in enumerate(data):
      (single,) = callee._flat_numerical_call(*item)
      np.testing.assert_array_equal(got[i * width : (i + 1) * width], single)


@pytest.mark.parametrize("n", [3, DENSE_UNROLL + 1])
def test_ops_inside_scan_and_while_loop_bodies_match_the_steps_taken_outside(rng: np.random.Generator, n: int) -> None:
  steps = 4
  ss, gs = [_spd(rng, n) for _ in range(steps)], [_cyclic(rng, n) for _ in range(steps)]
  x, s, g = sc.sym("x", n), sc.sym("s", (n, n)), sc.sym("g", (n, n))
  body = _fn(f"de_scan_body{n}", [x, s, g], [solve(g, cho_solve(cholesky(s), x), assume="gen"), ldl(s)])
  x0, s_all, g_all = sc.sym("x0", n), sc.sym("s_all", steps * n * n), sc.sym("g_all", steps * n * n)
  final, packed = sc.scan(body, x0, [(s_all, 0, n * n), (g_all, 0, n * n)], length=steps)
  scanned = _fn(f"de_scan{n}", [x0, s_all, g_all], [final, packed, gradient(final.sum(), x0)])
  xv = rng.standard_normal(n)
  got_final, got_packed, got_grad = scanned._flat_numerical_call(
    xv, np.concatenate([m.reshape(-1) for m in ss]), np.concatenate([m.reshape(-1) for m in gs])
  )
  ref, chain = xv, np.eye(n)
  for k, (sk, gk) in enumerate(zip(ss, gs, strict=True)):
    ref, packed_k = body._flat_numerical_call(ref, sk, gk)
    np.testing.assert_allclose(got_packed[k * n * n : (k + 1) * n * n], packed_k.reshape(-1), rtol=1e-15, atol=0)
    chain = np.linalg.solve(gk, np.linalg.solve(sk, chain))
  np.testing.assert_allclose(got_final, ref, rtol=1e-15, atol=0)
  _close(got_final, chain @ xv, 1e-11)
  _close(got_grad, chain.T @ np.ones(n), 1e-11)

  # Gauss-Seidel: x += L^{-1} (b - A x), with L the lower triangle of A, until the residual is tiny.
  a, rhs, xw, start = sc.sym("a", (n, n)), sc.sym("rhs", n), sc.sym("xw", n), sc.sym("start", n)
  step = _fn(f"de_gs_step{n}", [xw, a, rhs], [xw + solve_triangular(a, rhs - a @ xw)])
  go = _fn(f"de_gs_go{n}", [xw, a, rhs], [sc.sumsqr(rhs - a @ xw) > 1e-24])
  x_end, iters = sc.while_loop(go, step, start, max_iter=200, params=[a, rhs])
  walked = _fn(f"de_gs{n}", [a, rhs, start], [x_end, iters])
  av, bv = _spd(rng, n) + 2 * n * np.eye(n), rng.standard_normal(n)
  got_x, got_iters = walked._flat_numerical_call(av, bv, np.zeros(n))
  count = int(np.asarray(got_iters).item())
  ref = np.zeros(n)
  for _ in range(count):
    (ref,) = step._flat_numerical_call(ref, av, bv)
  assert 1 < count < 200
  np.testing.assert_allclose(got_x, ref, rtol=1e-15, atol=0)
  _close(got_x, np.linalg.solve(av, bv), 1e-9)


# --- constants -----------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, DENSE_UNROLL + 1])
def test_constant_matrices_agree_with_symbolic_inputs(rng: np.random.Generator, n: int) -> None:
  s, k, g, t = _spd(rng, n), _quasi_definite(rng, n - n // 2, n // 2), _cyclic(rng, n), _triangle_operand(rng, n)
  b = sc.sym("b", n)

  def build(sv, kv, gv, tv):
    solves = [solve_triangular(tv, b, lower=False, trans=True), cho_solve(cholesky(sv), b), solve(kv, b, assume="sym"), solve(gv, b, assume="gen")]
    return [cholesky(sv), ldl(kv), lu(gv), *solves, *(jacobian(x, b) for x in solves)]

  syms, bv = [sc.sym(v, (n, n)) for v in ("s", "k", "g", "t")], rng.standard_normal(n)
  symbolic = _fn(f"de_symbolic{n}", [*syms, b], build(*syms))._flat_numerical_call(s, k, g, t, bv)
  constant = _fn(f"de_constant{n}", [b], build(*(sc.const(m) for m in (s, k, g, t))))._flat_numerical_call(bv)
  for got, ref in zip(constant, symbolic, strict=True):
    _close(got, ref, 1e-14)
  for jac, op in zip(constant[-4:], (np.triu(t).T, s, k, g), strict=True):
    _close(jac, np.linalg.inv(op), 1e-11)
