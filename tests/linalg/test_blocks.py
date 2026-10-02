"""``linalg.blocks``: the block-tridiagonal Cholesky factorization and its solves against NumPy's on the assembled matrix."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.linalg.blocks import BlockTridiagonalCholesky

SHAPES = [(1, 3, 0), (2, 3, 1), (2, 3, 3), (5, 4, 2), (7, 6, 6), (4, 1, 1), (3, 9, 5), (12, 5, 1)]


def _matrix(rng: np.random.Generator, K: int, B: int, c: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """A random positive definite block-tridiagonal matrix, with its diagonal blocks and the last
  ``c`` columns of the blocks below them."""
  n = K * B
  a = np.zeros((n, n))
  below = rng.normal(size=(max(K - 1, 0), B, c))
  for k in range(K - 1):
    a[(k + 1) * B : (k + 2) * B, (k + 1) * B - c : (k + 1) * B] = below[k]
  a = a + a.T
  diag = rng.normal(size=(K, B, B))
  diag = diag @ np.swapaxes(diag, 1, 2) + 4 * B * np.eye(B)
  for k in range(K):
    a[k * B : (k + 1) * B, k * B : (k + 1) * B] = diag[k]
  return a, diag, below


def _functions(name: str, K: int, B: int, c: int) -> tuple[BlockTridiagonalCholesky, sc.Function]:
  """A factorization over symbols, and the Function of its factor, the factor's diagonal and a solve."""
  d, r = sc.sym("d", (K, B, B)), sc.sym("r", (K * B,))
  e = sc.sym("e", (K - 1, B, c)) if K > 1 else None
  fac = BlockTridiagonalCholesky(d, e, name=name)
  inputs = [d, r] if e is None else [d, e, r]
  fn = sc.Function.from_exprs(f"{name}_fn", inputs, [fac.values, fac.diagonal, fac.solve(r)], [str(i.name) for i in inputs], ["f", "diag", "x"])
  return fac, fn


def _lower_only(diag: np.ndarray) -> np.ndarray:
  """``diag`` with its upper triangles overwritten: the factorization must not read them."""
  return np.tril(diag) + 7.0 * np.triu(np.ones_like(diag), 1)


@pytest.mark.parametrize(("K", "B", "c"), SHAPES)
def test_the_factor_and_a_solve_match_the_dense_cholesky(K: int, B: int, c: int) -> None:
  """The triangular blocks are the dense factor's diagonal blocks, the blocks of ``c`` columns its
  blocks below them (zero left of those columns), and a solve is the dense solve. Only the lower
  triangles of the diagonal blocks are read."""
  rng = np.random.default_rng(K * 100 + B * 10 + c)
  a, diag, below = _matrix(rng, K, B, c)
  fac, fn = _functions(f"blocks_{K}_{B}_{c}", K, B, c)
  rhs = rng.normal(size=K * B)
  args = (_lower_only(diag), rhs) if K == 1 else (_lower_only(diag), below, rhs)
  values, diagonal, x = (np.asarray(v) for v in fn(args))
  dense = np.linalg.cholesky(a)
  assert values.shape == (fac.size,) and fac.n == K * B
  lowers = values[: K * B * B].reshape(K, B, B)
  for k in range(K):
    np.testing.assert_allclose(lowers[k], dense[k * B : (k + 1) * B, k * B : (k + 1) * B], rtol=1e-12, atol=1e-12)
  ws = values[K * B * B :].reshape(max(K - 1, 0), B, c)
  for k in range(K - 1):
    block = dense[(k + 1) * B : (k + 2) * B, k * B : (k + 1) * B]
    np.testing.assert_allclose(ws[k], block[:, B - c :], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(block[:, : B - c], 0.0, atol=1e-12)
  np.testing.assert_allclose(diagonal, np.diag(dense), rtol=1e-12)
  np.testing.assert_allclose(x, np.linalg.solve(a, rhs), rtol=1e-11, atol=1e-12)


def test_a_solve_takes_a_factor_kept_from_another_call() -> None:
  """``solve_with`` reads the factor it is given: a factorization in one Function, a solve in another."""
  K, B, c = 4, 3, 2
  rng = np.random.default_rng(7)
  a, diag, below = _matrix(rng, K, B, c)
  fac, fn = _functions("blocks_kept", K, B, c)
  values = np.asarray(fn((diag, below, np.zeros(K * B)))[0])
  f, r = sc.sym("f", (fac.size,)), sc.sym("r", (K * B,))
  solve = sc.Function.from_exprs("blocks_kept_solve", [f, r], [fac.solve_with(f, r)], ["f", "r"], ["x"])
  rhs = rng.normal(size=K * B)
  np.testing.assert_allclose(solve((values, rhs)), np.linalg.solve(a, rhs), rtol=1e-11, atol=1e-12)
  with pytest.raises(ValueError, match="values must have shape"):
    fac.solve_with(sc.sym("short", (fac.size - 1,)), r)
  with pytest.raises(ValueError, match="right-hand side must have shape"):
    fac.solve_with(f, sc.sym("long", (K * B + 1,)))


def test_a_pivot_that_is_not_positive_shows_on_the_diagonal() -> None:
  """A matrix that is not positive definite in its third block: the factor's diagonal is positive
  through the blocks before it and has an entry that is not, or is not a number, from there on."""
  K, B, c = 5, 3, 2
  rng = np.random.default_rng(3)
  _, diag, below = _matrix(rng, K, B, c)
  diag[2] -= 200.0 * np.eye(B)
  _, fn = _functions("blocks_indefinite", K, B, c)
  diagonal = np.asarray(fn((diag, below, np.zeros(K * B)))[1]).reshape(K, B)
  assert np.all(diagonal[:2] > 0.0)
  assert not np.all(diagonal[2] > 0.0)


def test_the_generated_code_does_not_grow_with_the_number_of_blocks() -> None:
  """The factorization and the solves are loops over the blocks, one step each."""
  from scaly.codegen import render_c_source

  sizes = [len(render_c_source(_functions(f"blocks_grow_{K}", K, 6, 4)[1].concrete)) for K in (4, 40)]
  assert sizes[1] < 1.2 * sizes[0], sizes


def test_a_solve_is_differentiable_in_the_matrix_and_the_right_hand_side() -> None:
  """The factorization is ordinary graph: the gradient of a function of the solution with respect
  to the blocks and the right-hand side is the dense solve's, by central differences."""
  K, B, c = 3, 3, 2
  rng = np.random.default_rng(5)
  _, diag, below = _matrix(rng, K, B, c)
  rhs, weights = rng.normal(size=K * B), rng.normal(size=K * B)
  d, e, r = sc.sym("d", (K, B, B)), sc.sym("e", (K - 1, B, c)), sc.sym("r", (K * B,))
  loss = (BlockTridiagonalCholesky(d, e, name="blocks_grad").solve(r) * weights).sum()
  grad = sc.Function.from_exprs("blocks_grad_fn", [d, e, r], [sc.gradient(loss, e), sc.gradient(loss, r)], ["d", "e", "r"], ["ge", "gr"])
  ge, gr = (np.asarray(v) for v in grad((diag, below, rhs)))

  def value(low: np.ndarray, right: np.ndarray) -> float:
    n = K * B
    a = np.zeros((n, n))
    for k in range(K - 1):
      a[(k + 1) * B : (k + 2) * B, (k + 1) * B - c : (k + 1) * B] = low[k]
    a = a + a.T
    for k in range(K):
      a[k * B : (k + 1) * B, k * B : (k + 1) * B] = diag[k]
    return float(weights @ np.linalg.solve(a, right))

  h = 1e-6
  for index in [(0, 0, 0), (1, 2, 1), (0, 1, 1)]:
    step = np.zeros_like(below)
    step[index] = h
    np.testing.assert_allclose(ge[index], (value(below + step, rhs) - value(below - step, rhs)) / (2 * h), rtol=1e-6, atol=1e-9)
  for index in (0, 4, 8):
    step = np.zeros_like(rhs)
    step[index] = h
    np.testing.assert_allclose(gr[index], (value(below, rhs + step) - value(below, rhs - step)) / (2 * h), rtol=1e-6, atol=1e-9)


def test_the_blocks_are_factored_and_solved_with_running_sums() -> None:
  """The blocks are a few dozen rows, where ``cholesky``'s and ``solve_triangular``'s running sums
  and reciprocal diagonals are the faster form: every factorization and substitution of the
  kernel asks for them, in the step of the factorization and in both passes of a solve."""
  from scaly.ir.expr import topo

  fac, fn = _functions("blocks_sums", 4, 20, 12)
  seen: dict[str, set[str]] = {}

  def walk(function) -> None:
    for node in topo(function.outputs):
      if str(node.op) in ("cholesky", "trisolve"):
        seen.setdefault(str(node.op), set()).add(node.attrs.get("sums", "pairwise"))
      callee = node.attrs.get("callee")
      if callee is not None:
        walk(callee)

  walk(fn.concrete)
  assert seen == {"cholesky": {"running"}, "trisolve": {"running"}}


def test_the_shapes_are_checked() -> None:
  with pytest.raises(ValueError, match="stack of K square blocks"):
    BlockTridiagonalCholesky(sc.sym("d", (3, 4, 5)))
  with pytest.raises(ValueError, match="need the 2 blocks below them"):
    BlockTridiagonalCholesky(sc.sym("d", (3, 4, 4)))
  with pytest.raises(ValueError, match="below must have shape"):
    BlockTridiagonalCholesky(sc.sym("d", (3, 4, 4)), sc.sym("e", (3, 4, 2)))
  with pytest.raises(ValueError, match="below must have shape"):
    BlockTridiagonalCholesky(sc.sym("d", (3, 4, 4)), sc.sym("e", (2, 4, 5)))
