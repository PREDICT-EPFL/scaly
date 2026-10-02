"""``sums="running"`` for ``cholesky`` and ``solve_triangular``: one running sum an entry and the
diagonal's reciprocals, against NumPy and SciPy and against the pairwise form."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_triangular as scipy_solve

import scaly as sc
from scaly.ir.program import ProgramNode, ProgramOp
from scaly.linalg.ops import cholesky, solve_triangular
from scaly.passes.lowering import lower_function

TARGETS = ("apple-m3", "x86-64-v3", "generic")


def _spd(rng: np.random.Generator, n: int) -> np.ndarray:
  m = rng.normal(size=(n, n))
  return m @ m.T + n * np.eye(n)


def _division_depth(fn: sc.Function, target: str = "apple-m3") -> int:
  """The most loops any floating-point division of ``fn``'s lowered program is nested in."""
  deepest = -1

  def walk(node: ProgramNode, depth: int) -> None:
    nonlocal deepest
    if node.op == ProgramOp.DIV and node.dtype.is_floating:
      deepest = max(deepest, depth)
    for arg in node.args:
      walk(arg, depth + (node.op == ProgramOp.FOR))

  walk(lower_function(fn, target=target), 0)
  return deepest


@pytest.mark.parametrize("n", [8, 15, 16, 17, 24, 28, 33, 48, 70])
def test_a_running_cholesky_is_the_factor(n: int) -> None:
  """Every order around the block width, on targets with tiles and on one without: the factor is
  NumPy's, its upper triangle zero, and only the lower triangle of the matrix is read."""
  rng = np.random.default_rng(n)
  a = _spd(rng, n)
  sym = sc.sym("a", (n, n))
  fn = sc.Function.from_exprs(f"run_chol_{n}", [sym], [cholesky(sym, sums="running")], ["a"], ["l"])
  want = np.linalg.cholesky(a)
  for target in TARGETS:
    with sc.target(target):
      got = np.asarray(fn(np.tril(a) + 3.0 * np.triu(np.ones((n, n)), 1)))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    assert np.all(np.triu(got, 1) == 0.0)


@pytest.mark.parametrize(("n", "m"), [(7, 3), (16, 8), (17, 9), (26, 28), (33, 12), (40, 1), (48, 16)])
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("trans", [False, True])
def test_a_running_substitution_is_the_solution(n: int, m: int, lower: bool, trans: bool) -> None:
  rng = np.random.default_rng(n * 100 + m)
  t = rng.normal(size=(n, n)) + n * np.eye(n)
  b = rng.normal(size=(n, m)) if m > 1 else rng.normal(size=n)
  ts, bs = sc.sym("t", (n, n)), sc.sym("b", b.shape)
  tag = f"{n}_{m}_{int(lower)}{int(trans)}"
  outs = [
    solve_triangular(ts, bs, lower=lower, trans=trans, sums="running"),
    solve_triangular(ts, bs, lower=lower, trans=trans, unit_diagonal=True, sums="running"),
  ]
  fn = sc.Function.from_exprs(f"run_tri_{tag}", [ts, bs], outs, ["t", "b"], ["x", "xu"])
  want = scipy_solve(t, b, lower=lower, trans="T" if trans else "N")
  unit = scipy_solve(t, b, lower=lower, trans="T" if trans else "N", unit_diagonal=True)
  for target in TARGETS:
    with sc.target(target):
      got, got_unit = fn((t, b))
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(got_unit, unit, rtol=1e-8, atol=1e-10)


def test_running_sums_divide_once_a_column_and_the_default_is_untouched() -> None:
  """The pairwise forms divide every entry by its diagonal, inside the loop over a block's rows or
  over the right-hand sides; the running forms take a reciprocal a column, or a row of right-hand
  sides, one loop further out, and multiply. A node built without ``sums`` carries no such
  attribute: its graph, and so its key in the JIT's cache, is what it was."""
  a, b = sc.sym("a", (28, 28)), sc.sym("b", (28, 12))
  assert "sums" not in cholesky(a).attrs and "sums" not in solve_triangular(a, b).attrs
  assert cholesky(a, sums="running").attrs["sums"] == solve_triangular(a, b, sums="running").attrs["sums"] == "running"
  for build in (lambda sums: cholesky(a, sums=sums), lambda sums: solve_triangular(a, b, sums=sums)):
    depth = {
      sums: _division_depth(sc.Function.from_exprs(f"run_div_{sums}", [a, b], [build(sums)], ["a", "b"], ["y"])) for sums in ("pairwise", "running")
    }
    assert 0 <= depth["running"] < depth["pairwise"], depth
  with pytest.raises(ValueError, match="sums must be"):
    cholesky(a, sums="kahan")  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match="sums must be"):
    solve_triangular(a, b, sums="kahan")  # ty: ignore[invalid-argument-type]


def _blocked_parts(fn: sc.Function, target: str = "apple-m3") -> int:
  """How many parts the dot products of ``fn``'s blocked substitution run in: one with running
  sums, four (the quarters) without, none where the substitution is not blocked."""
  parts: set[str] = set()

  def walk(node: ProgramNode) -> None:
    name = str(node.attrs.get("name", "")) if node.op == ProgramOp.RANGE else ""
    if name.startswith("tjf"):
      parts.add(name[3])
    for arg in node.args:
      walk(arg)

  walk(lower_function(fn, target=target))
  return len(parts)


@pytest.mark.parametrize(("lower", "trans"), [(True, False), (False, True), (True, True), (False, False)])
def test_a_backward_substitution_in_blocks_is_the_pairwise_one(lower: bool, trans: bool) -> None:
  """From four blocks of the tile's width a substitution that runs forward (``lower`` without
  ``trans``, upper with it) takes the running form, one part where the pairwise form takes four;
  one that runs backward is the pairwise one, which was the faster there. Below four blocks, where
  the pairwise form is not blocked, both directions take the running form in blocks, from two."""
  forward = lower != trans
  for n, blocked in ((32, True), (24, False), (16, False)):  # four, three and two blocks of apple-m3's eight columns
    t, b = sc.sym("t", (n, n)), sc.sym("b", (n, 16))
    parts = {
      sums: _blocked_parts(
        sc.Function.from_exprs(
          f"run_back_{n}_{int(lower)}{int(trans)}_{sums}", [t, b], [solve_triangular(t, b, lower=lower, trans=trans, sums=sums)], ["t", "b"], ["x"]
        )
      )
      for sums in ("pairwise", "running")
    }
    assert parts == {"pairwise": 4 if blocked else 0, "running": 4 if blocked and not forward else 1}, (n, parts)


def test_a_running_factorization_differentiates_as_the_pairwise_one() -> None:
  n = 20
  rng = np.random.default_rng(1)
  a, da = _spd(rng, n), rng.normal(size=(n, n))
  da = da + da.T
  sym, tangent = sc.sym("a", (n, n)), sc.sym("da", (n, n))
  outs = [sc.jvp(cholesky(sym, sums=sums), sym, tangent) for sums in ("running", "pairwise")]
  fn = sc.Function.from_exprs("run_chol_jvp", [sym, tangent], outs, ["a", "da"], ["running", "pairwise"])
  got, want = fn((a, da))
  np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)
