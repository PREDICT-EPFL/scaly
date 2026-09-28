"""Dense ``lu`` with partial pivoting and the general solve built on it: the factors and the
permutation against SciPy, solves against NumPy, the implicit derivatives against closed forms and
finite differences, and the generated code's two forms."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sl

import scaly as sc
from scaly.ad.forward import jvp
from scaly.ad.sparsity import jacobian_sparsity
from scaly.codegen import render_c_module
from scaly.ir.expr import Expr
from scaly.ir.expr_spec import verify_expr
from scaly.ir.spec import VerifyError
from scaly.ir.types import TensorType
from scaly.linalg import lu, lu_solve, solve
from scaly.linalg.ops import DENSE_UNROLL, LU_NO_DERIVATIVE
from scaly.linalg.ops.dense import LU

RNG = np.random.default_rng(707)
SIZES = [1, 2, 3, DENSE_UNROLL, DENSE_UNROLL + 1, 13, 24, 40]


def _fn(name, inputs, outputs):
  return sc.Function.from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _needs_pivoting(n: int) -> np.ndarray:
  a = RNG.standard_normal((n, n))
  if n > 1:
    a[0, 0] = 0.0  # without a row swap the first pivot is zero
    a[n - 1, n // 2] = 50.0  # a large entry far from the diagonal moves rows around
  return a


def _ill_conditioned(n: int, cond: float) -> np.ndarray:
  u, _ = np.linalg.qr(RNG.standard_normal((n, n)))
  v, _ = np.linalg.qr(RNG.standard_normal((n, n)))
  return u @ np.diag(np.geomspace(1.0, 1.0 / cond, n)) @ v.T


def _factor(n: int):
  a = sc.sym("a", (n, n))
  return _fn(f"lu{n}", [a], [lu(a)])


@pytest.mark.parametrize("n", SIZES)
def test_factors_and_permutation_match_scipy(n: int) -> None:
  for a in (_needs_pivoting(n), RNG.standard_normal((n, n)), np.eye(n)[RNG.permutation(n)]):
    (packed,) = _factor(n)._flat_numerical_call(a)
    perm = packed[n].astype(np.int64)
    assert np.array_equal(packed[n], perm) and sorted(perm) == list(range(n))
    unit_l, upper = np.tril(packed[:n], -1) + np.eye(n), np.triu(packed[:n])
    np.testing.assert_allclose(unit_l @ upper, a[perm], rtol=1e-13, atol=1e-13)
    p, l_ref, u_ref = sl.lu(a)
    np.testing.assert_array_equal(perm, np.argmax(p, axis=0))  # SciPy's a = P L U: the same pivot choices
    np.testing.assert_allclose(upper, u_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(unit_l, l_ref, rtol=1e-12, atol=1e-12)


def test_ties_pick_the_first_row_and_every_multiplier_is_at_most_one() -> None:
  n = 5
  a = np.ones((n, n)) + np.eye(n)  # column 0 ties between rows 1..4 once row 0 is taken as pivot
  a[:, 0] = [1.0, 2.0, 2.0, -2.0, 2.0]
  (packed,) = _factor(n)._flat_numerical_call(a)
  assert packed[n, 0] == 1.0  # the first of the tied rows, as LAPACK's idamax
  assert np.abs(np.tril(packed[:n], -1)).max() <= 1.0


@pytest.mark.parametrize("n", SIZES)
def test_solves_match_numpy_for_vectors_matrices_and_transposes(n: int) -> None:
  a_s, b_s, m_s = sc.sym("a", (n, n)), sc.sym("b", n), sc.sym("m", (n, 3))
  fac = lu(a_s)
  fn = _fn(
    f"solves{n}",
    [a_s, b_s, m_s],
    [solve(a_s, b_s, assume="gen"), solve(a_s, m_s, assume="gen"), lu_solve(fac, b_s, trans=True), lu_solve(fac, m_s, trans=True)],
  )
  a, b, m = _needs_pivoting(n), RNG.standard_normal(n), RNG.standard_normal((n, 3))
  x, xm, xt, xtm = fn._flat_numerical_call(a, b, m)
  np.testing.assert_allclose(x, np.linalg.solve(a, b), rtol=1e-10, atol=1e-11)
  np.testing.assert_allclose(xm, np.linalg.solve(a, m), rtol=1e-10, atol=1e-11)
  np.testing.assert_allclose(xt, np.linalg.solve(a.T, b), rtol=1e-10, atol=1e-11)
  np.testing.assert_allclose(xtm, np.linalg.solve(a.T, m), rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("n", [6, 30])
def test_ill_conditioned_solves_are_backward_stable(n: int) -> None:
  a_s, b_s = sc.sym("a", (n, n)), sc.sym("b", n)
  fn = _fn(f"ill{n}", [a_s, b_s], [solve(a_s, b_s, assume="gen")])
  a, b = _ill_conditioned(n, 1e10), RNG.standard_normal(n)
  (x,) = fn._flat_numerical_call(a, b)
  residual = np.linalg.norm(a @ x - b) / (np.linalg.norm(a, 2) * np.linalg.norm(x) + np.linalg.norm(b))
  assert residual < 1e-14, residual  # partial pivoting: a small relative residual whatever the conditioning


def test_a_singular_matrix_gives_nonfinite_values() -> None:
  a_s, b_s = sc.sym("a", (4, 4)), sc.sym("b", 4)
  (x,) = _fn("singular", [a_s, b_s], [solve(a_s, b_s, assume="gen")])._flat_numerical_call(np.diag([1.0, 1.0, 0.0, 1.0]), np.ones(4))
  assert not np.isfinite(x).all()


@pytest.mark.parametrize("n", [3, DENSE_UNROLL + 2])
def test_implicit_derivatives_are_the_closed_forms(n: int) -> None:
  @sc.function((n, n), n, output="x", name=f"gen{n}")
  def gen(a, b):
    return solve(a, b, assume="gen")

  a, b = _needs_pivoting(n) + n * np.eye(n), RNG.standard_normal(n)
  x, inv = np.linalg.solve(a, b), np.linalg.inv(a)
  np.testing.assert_allclose(sc.jacobian(gen, "b")(a, b), inv, rtol=1e-11, atol=1e-12)
  np.testing.assert_allclose(sc.jacobian(gen, "a")(a, b), -np.kron(inv, x[None, :]), rtol=1e-11, atol=1e-12)
  xbar = RNG.standard_normal(n)
  lam = np.linalg.solve(a.T, xbar)
  abar, bbar = sc.adjoint(gen, "a")(a, b, xbar), sc.adjoint(gen, "b")(a, b, xbar)
  np.testing.assert_allclose(bbar, lam, rtol=1e-11, atol=1e-12)
  np.testing.assert_allclose(abar.reshape(n, n), -np.outer(lam, x), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("n", [3, DENSE_UNROLL + 1])
def test_the_adjoint_differentiates_in_reverse_mode(n: int) -> None:
  """Reverse over reverse: the adjoint solves with the transposed matrix, whose own reverse rule
  (a solve with the matrix and an outer product the other way round) is what this reaches."""

  @sc.function((n, n), n, output="x", name=f"fwd{n}")
  def gen(a, b):
    return solve(a, b, assume="gen")

  adjoint = sc.adjoint(gen, "b")  # (a, b, xbar) -> A^{-T} xbar
  w = RNG.standard_normal(n)

  @sc.function((n, n), n, n, output="c", name=f"weighted{n}")
  def weighted(a, b, xbar):
    return (adjoint(a, b, xbar) * sc.const(w)).sum()

  a, b, xbar = _needs_pivoting(n) + n * np.eye(n), RNG.standard_normal(n), RNG.standard_normal(n)
  lam, mu = np.linalg.solve(a.T, xbar), np.linalg.solve(a, w)  # c = w . lam with A^T lam = xbar: dc/dA = -lam mu^T
  np.testing.assert_allclose(sc.gradient(weighted, "a")(a, b, xbar), -np.outer(lam, mu), rtol=1e-10, atol=1e-11)
  np.testing.assert_allclose(sc.gradient(weighted, "xbar")(a, b, xbar), mu, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("n", [3, DENSE_UNROLL + 1])
def test_second_derivatives_match_finite_differences_of_the_gradient(n: int) -> None:
  @sc.function((n, n), n, output="c", name=f"objective{n}")
  def objective(a, b):
    x = solve(a, b, assume="gen")
    t = solve(a.T, x, assume="gen")  # the transposed matrix goes through a second factorization
    return (x * x).sum() + (t * b).sum() + x[0] * a[n - 1, 0]

  a, b = _needs_pivoting(n) + n * np.eye(n), RNG.standard_normal(n)
  grad, hess = sc.gradient(objective, "a"), sc.hessian(objective, "a")
  eps = 1e-6
  fd = np.zeros((n * n, n * n))
  for k in range(n * n):
    step = np.zeros(n * n)
    step[k] = eps
    fd[:, k] = (grad(a + step.reshape(n, n), b).reshape(-1) - grad(a - step.reshape(n, n), b).reshape(-1)) / (2 * eps)
  np.testing.assert_allclose(hess(a, b), fd, rtol=1e-5, atol=1e-6)
  gb = sc.gradient(objective, "b")(a, b)
  fdb = np.array([(objective(a, b + e) - objective(a, b - e)) / (2 * eps) for e in np.eye(n) * eps])
  np.testing.assert_allclose(gb, fdb, rtol=1e-6, atol=1e-7)


def test_the_factorization_itself_refuses_a_derivative() -> None:
  a = sc.sym("a", (3, 3))
  f = _fn("fac_only", [a], [lu(a)])
  with pytest.raises(NotImplementedError, match="dense LU factorization has no derivative"):
    sc.jacobian(f, "a")
  with pytest.raises(NotImplementedError, match="dense LU factorization has no derivative"):
    sc.gradient(_fn("fac_sum", [a], [lu(a).sum()]), "a")
  with pytest.raises(NotImplementedError, match="dense LU factorization has no derivative"):
    jvp(lu(a), a, sc.sym("da", (3, 3)))  # one seed takes another path than a whole Jacobian
  assert "assume='gen'" in LU_NO_DERIVATIVE
  pattern = jacobian_sparsity(lu(a), a)
  assert pattern.shape == (12, 9) and pattern.nnz == 12 * 9


def test_small_orders_are_straight_line_and_large_ones_loops() -> None:
  srcs = {}
  for n in (DENSE_UNROLL, 24, 48):
    a = sc.sym("a", (n, n))
    srcs[n] = str(render_c_module(_fn(f"lushape{n}", [a], [lu(a)])).body)
  assert "for (" not in srcs[DENSE_UNROLL].split("int lushape")[1]
  lines = {n: len(src.splitlines()) for n, src in srcs.items()}
  assert lines[24] == lines[48], "the loop code does not grow with the order"
  with sc.options(linalg=dict(dense_unroll=0)):
    a_s = sc.sym("a", (DENSE_UNROLL, DENSE_UNROLL))
    looped = _fn("lu_forced_loops", [a_s], [lu(a_s)])
  assert "for (" in str(render_c_module(looped).body).split("int lu_forced_loops")[1]
  a = _needs_pivoting(DENSE_UNROLL)
  np.testing.assert_array_equal(looped._flat_numerical_call(a)[0], _factor(DENSE_UNROLL)._flat_numerical_call(a)[0])


def test_validation() -> None:
  with pytest.raises(ValueError, match="square"):
    lu(sc.sym("r", (2, 3)))
  with pytest.raises(ValueError, match=r"shaped \(n \+ 1, n\)"):
    lu_solve(sc.sym("f", (3, 3)), sc.sym("b", 3))
  with pytest.raises(ValueError, match="right-hand side of 3 rows"):
    lu_solve(sc.sym("f", (4, 3)), sc.sym("b", 4))
  with pytest.raises(ValueError, match="'pos', 'sym' or 'gen'"):
    solve(sc.sym("a", (2, 2)), sc.sym("b", 2), assume="lu")  # ty: ignore[invalid-argument-type]
  a = sc.sym("a", (3, 3))
  bad = Expr(LU, (a,), TensorType((3, 3)))
  with pytest.raises(VerifyError, match="lu needs a square matrix"):
    verify_expr([bad])
  assert sc.linalg.lu is lu and sc.linalg.lu_solve is lu_solve
