"""A product of a matrix with its own transpose in register tiles: one triangle computed, the other copied."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.ir.program import ProgramNode, ProgramOp
from scaly.passes.lowering import lower_function


def _mirrored(fn: sc.Function, target: str) -> bool:
  """Whether ``fn``'s program, as lowering leaves it and before any pass, copies a product's lower
  triangle across its diagonal."""
  stages: dict[str, ProgramNode] = {}
  lower_function(fn, target=target, observe=lambda name, prog: stages.__setitem__(name, prog))
  stack: list[ProgramNode] = [stages["lowered"]]
  while stack:
    node = stack.pop()
    if node.op == ProgramOp.RANGE and str(node.attrs.get("name", "")).startswith("si_"):
      return True
    stack.extend(node.args)
  return False


SHAPES = [(28, 26), (28, 28), (33, 27), (42, 39), (64, 64), (65, 30), (96, 20), (25, 4)]


@pytest.mark.parametrize(("m", "k"), SHAPES)
@pytest.mark.parametrize("left", [True, False], ids=["a a'", "a' a"])
def test_a_matrix_times_its_transpose_computes_one_triangle_and_is_the_full_product_to_the_bit(m: int, k: int, left: bool) -> None:
  """``a @ a.T`` and ``a.T @ a`` against the same product with the transpose handed in as an
  input of its own, which lowering cannot know is symmetric: entry ``(i, j)`` and entry ``(j, i)``
  are the same products added in the same order, so the two agree in every bit, and the result is
  exactly symmetric. Tiles of columns the lower triangle does not reach are not computed."""
  rng = np.random.default_rng(m * 100 + k)
  av = rng.standard_normal((m, k))
  a, other = sc.sym("a", (m, k)), sc.sym("o", (k, m))
  tag = f"{m}_{k}_{int(left)}"
  symmetric = sc.Function.from_exprs(f"sym_{tag}", [a], [a @ a.T if left else a.T @ a], ["a"], ["y"])
  full = sc.Function.from_exprs(f"full_{tag}", [a, other], [a @ other if left else other @ a], ["a", "o"], ["y"])
  order = m if left else k
  for target, tile in (("apple-m3", 8), ("x86-64-v3", 8), ("armv8-a", 4)):
    taken = order >= 3 * tile and (k if left else m) >= 4
    assert _mirrored(symmetric, target) == taken, (target, order)
    assert not _mirrored(full, target)
    with sc.target(target):
      got, want = np.asarray(symmetric(av)), np.asarray(full((av, av.T.copy())))
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got, got.T)
  np.testing.assert_allclose(got, av @ av.T if left else av.T @ av, rtol=1e-12, atol=1e-12)


def test_products_that_are_not_symmetric_or_may_be_expanded_are_left_whole() -> None:
  """A product with another matrix's transpose, one weighted between the two, one under three
  tiles of columns wide, one of fewer than four terms, one in a procedure asked to be scalar code,
  and any product on a target without tiles: each is computed entry by entry as before."""
  a, b, w = sc.sym("a", (30, 12)), sc.sym("b", (30, 12)), sc.sym("w", (12,))
  cases = {
    "other": a @ b.T,
    "weighted": (a * w) @ a.T,
    "narrow": a[:20] @ a[:20].T,
    "few_terms": a[:, :3] @ a[:, :3].T,
  }
  for tag, expr in cases.items():
    fn = sc.Function.from_exprs(f"whole_{tag}", [a, b, w], [expr], ["a", "b", "w"], ["y"])
    assert not _mirrored(fn, "apple-m3"), tag
  taken = sc.Function.from_exprs("whole_taken", [a], [a @ a.T], ["a"], ["y"])
  assert _mirrored(taken, "apple-m3") and not _mirrored(taken, "generic")
  scalar = sc.Function.from_exprs("whole_scalar", [a], [(a @ a.T).scalar()], ["a"], ["y"])
  assert not _mirrored(scalar, "apple-m3")
  av = np.random.default_rng(0).standard_normal((30, 12))
  np.testing.assert_allclose(scalar(av), av @ av.T, rtol=1e-12, atol=1e-12)
