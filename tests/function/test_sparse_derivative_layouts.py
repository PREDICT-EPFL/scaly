"""Sparse derivative outputs are sparse leaves whose values follow the declared layout, checked against SciPy."""

from typing import Literal

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ad import finite_difference
from scaly.ir.layout import SparseLayout

N = 4
POINT = np.linspace(-0.8, 0.9, 2 * N)


@sc.function(sc.arg("z", 2), sc.arg("znext", 2), outputs=sc.arg("d", 2))
def defect(z: sc.Expr, znext: sc.Expr) -> sc.Expr:
  return znext - sc.stack([z[0] + 0.1 * z[1].sin(), z[1] * z[0]])


@sc.function(sc.arg("zs", 2 * N), outputs=sc.arg("ds"))
def shooting(zs: sc.Expr) -> sc.Expr:
  return sc.vmap(defect, N - 1)(sc.window(zs, 0, 2), sc.window(zs, 2, 2)).vec()


@sc.function(sc.arg("zs", 2 * N), outputs=sc.arg("cost"))
def coupled(zs: sc.Expr) -> sc.Expr:
  return (zs[:-1] * zs[1:]).sin().sum() + (zs**3).sum()


@sc.function(sc.arg("z", 2), outputs=sc.arg("cost"))
def stage_cost(z: sc.Expr) -> sc.Expr:
  return (z[0] * z[1]).sin() + z[1] ** 3


def _coordinates(pattern) -> sc.SparsityPattern:
  shuffled = np.random.default_rng(48).permutation(pattern.nnz)
  return sc.SparsityPattern(pattern.shape, tuple(np.asarray(pattern.rows)[shuffled]), tuple(np.asarray(pattern.cols)[shuffled]))


@pytest.mark.parametrize(("layout", "triangle"), [("csc", "full"), ("csr", "upper"), ("coordinate", "lower")])
def test_values_follow_the_declared_layout(layout: str, triangle: Literal["full", "lower", "upper"]) -> None:
  for kind in ("jacobian", "hessian"):

    def build(declared):
      return sc.sparse_jacobian(shooting, layout=declared) if kind == "jacobian" else sc.sparse_hessian(coupled, triangle=triangle, layout=declared)

    pattern = build("csc").sparsity()
    assert pattern is not None
    declared = _coordinates(pattern) if layout == "coordinate" else layout
    dense = finite_difference(shooting, POINT) if kind == "jacobian" else finite_difference(sc.gradient(coupled), POINT)
    if kind == "hessian":
      dense = {"full": dense, "upper": np.triu(dense), "lower": np.tril(dense)}[triangle]
    stored = sparse.coo_array((dense[pattern.rows, pattern.cols], (pattern.rows, pattern.cols)), shape=pattern.shape)
    if isinstance(declared, sc.SparsityPattern):
      expected = dense[np.asarray(declared.rows), np.asarray(declared.cols)]
    else:
      expected = stored.tocsc().data if layout == "csc" else stored.tocsr().data
    result = build(declared)(POINT)
    assert isinstance(result, sparse.csc_array | sparse.csr_array | sparse.coo_array)
    assert result.format == ("coo" if layout == "coordinate" else layout)
    np.testing.assert_allclose(result.data, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.toarray(), dense, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(("layout", "triangle"), [("csc", "full"), ("csr", "upper"), ("coordinate", "lower")])
def test_a_map_of_independent_stages_keeps_one_batched_pattern(layout: str, triangle: Literal["full", "lower", "upper"]) -> None:
  pattern = sc.sparse_hessian(stage_cost, triangle=triangle).sparsity()
  assert pattern is not None
  declared = _coordinates(pattern) if layout == "coordinate" else layout
  derivative = sc.sparse_hessian(stage_cost, triangle=triangle, layout=declared)
  local = derivative.sparsity()
  assert local is not None
  batched = sc.vmap(derivative, N)
  assert batched.sparsity() is local
  points = POINT.reshape((N, 2))
  result = batched(points)
  assert isinstance(result, sparse.coo_array) and result.shape == (N, *local.shape)
  dense = np.stack([finite_difference(sc.gradient(stage_cost), point) for point in points])
  dense = {"full": dense, "upper": np.triu(dense), "lower": np.tril(dense)}[triangle]
  leaf_layout = derivative.instantiate().output_tree.layouts[0]
  assert isinstance(leaf_layout, SparseLayout)
  expected = dense[:, leaf_layout.rows, leaf_layout.cols].reshape(-1)
  np.testing.assert_allclose(result.data, expected, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(result.toarray(), dense, rtol=1e-6, atol=1e-6)


def test_a_transposed_sparse_output_keeps_its_pattern_through_a_call() -> None:
  derivative = sc.sparse_jacobian(defect, "d", "z")

  @sc.function(sc.arg("z", 2), sc.arg("znext", 2))
  def transposed(z: sc.Expr, znext: sc.Expr) -> sc.Expr:
    return derivative(z, znext).T

  pattern, local = transposed.sparsity(), derivative.sparsity()
  assert pattern is not None and local is not None
  assert set(zip(pattern.rows, pattern.cols)) == set(zip(local.cols, local.rows))
  point, next_point = POINT[:2], POINT[2:4]
  result = transposed(point, next_point)
  assert isinstance(result, sparse.csc_array)
  np.testing.assert_allclose(result.toarray(), finite_difference(lambda z: defect(z, next_point), point).T, atol=1e-6)


def test_a_coordinate_layout_must_list_the_derivative_pattern() -> None:
  with pytest.raises(ValueError, match="different stored pattern"):
    sc.sparse_jacobian(defect, "d", "z", layout=sc.SparsityPattern((2, 2), (0,), (0,)))


@pytest.mark.parametrize("layout", ["csr", "csc"])
def test_declared_sparse_output_preserves_gather_fill(layout):
  from scaly.ir.types import CanonicalPattern

  pattern = CanonicalPattern((2, 2), (0, 1), (1, 0))

  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", sc.TensorType(pattern.shape, pattern=pattern), layout=layout))
  def filled(x):
    return sc.pack(sc.gather(x, [9, 0], mode="fill", fill_value=-7), pattern)

  np.testing.assert_array_equal(filled(np.array([2.0, 3.0])).toarray(), [[0.0, -7.0], [2.0, 0.0]])
