"""Sparse derivative outputs are sparse leaves whose values follow the declared layout, checked against SciPy."""

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ad import finite_difference

N = 4
POINT = np.linspace(-0.8, 0.9, 2 * N)


@sc.function(sc.arg("z", 2), outputs=sc.arg("r", 3))
def stage(z: sc.Expr) -> sc.Expr:
  return sc.stack([z[0].sin() * z[1], z[1] ** 2, z[0] * z[0]])


@sc.function(sc.arg("z", 2), outputs=sc.arg("cost"))
def stage_cost(z: sc.Expr) -> sc.Expr:
  return (z[0] * z[1]).sin() + z[1] ** 3


@sc.function(sc.arg("x", 3), outputs=sc.arg("y", 3))
def plain(x: sc.Expr) -> sc.Expr:
  return sc.stack([x[0] * x[2], x[1].exp(), x[2] ** 2])


@sc.function(sc.arg("zs", 2 * N), outputs=sc.arg("rs"))
def mapped(zs: sc.Expr) -> sc.Expr:
  return sc.vmap(stage, N)(zs.reshape((N, 2))).vec()


@sc.function(sc.arg("z", 2), sc.arg("znext", 2), outputs=sc.arg("d", 2))
def defect(z: sc.Expr, znext: sc.Expr) -> sc.Expr:
  return znext - sc.stack([z[0] + 0.1 * z[1].sin(), z[1] * z[0]])


@sc.function(sc.arg("zs", 2 * N), outputs=sc.arg("ds"))
def shooting(zs: sc.Expr) -> sc.Expr:
  return sc.vmap(defect, N - 1)(sc.window(zs, 0, 2), sc.window(zs, 2, 2)).vec()


@sc.function(sc.arg("zs", 2 * N), outputs=sc.arg("cost"))
def coupled(zs: sc.Expr) -> sc.Expr:
  return (zs[:-1] * zs[1:]).sin().sum() + (zs**3).sum()


def _reference(fn: sc.Function, x: np.ndarray, hessian: bool) -> np.ndarray:
  return finite_difference(sc.gradient(fn), x) if hessian else finite_difference(fn, x).reshape((-1, x.size))


def _layouts(pattern) -> list:
  shuffled = np.random.default_rng(48).permutation(pattern.nnz)
  coordinate = sc.SparsityPattern(pattern.shape, tuple(np.asarray(pattern.rows)[shuffled]), tuple(np.asarray(pattern.cols)[shuffled]))
  return ["csc", "csr", coordinate]


def _expected_values(dense: np.ndarray, pattern, layout) -> np.ndarray:
  stored = sparse.coo_array((dense[pattern.rows, pattern.cols], (pattern.rows, pattern.cols)), shape=pattern.shape)
  if layout == "csc":
    return stored.tocsc().data
  if layout == "csr":
    return stored.tocsr().data
  return dense[np.asarray(layout.rows), np.asarray(layout.cols)]


@pytest.mark.parametrize(
  ("source", "kind", "triangle"),
  [
    ("plain", "jacobian", "full"),
    ("mapped", "jacobian", "full"),
    ("shooting", "jacobian", "full"),
    ("plain", "hessian", "full"),
    ("plain", "hessian", "upper"),
    ("plain", "hessian", "lower"),
  ],
)
def test_values_follow_the_declared_layout(source: str, kind: str, triangle: str) -> None:
  fn = coupled if kind == "hessian" else {"plain": plain, "mapped": mapped, "shooting": shooting}[source]
  x = POINT[:3] if fn is plain else POINT

  def build(layout):
    if kind == "hessian":
      return sc.sparse_hessian(fn, "cost", "zs", triangle=triangle, layout=layout)
    return sc.sparse_jacobian(fn, layout=layout)

  pattern = build("csc").sparsity()
  assert pattern is not None
  dense = _reference(fn, x, kind == "hessian")
  if triangle != "full":
    dense = np.triu(dense) if triangle == "upper" else np.tril(dense)
  for layout in _layouts(pattern):
    result = build(layout)(x)
    assert isinstance(result, sparse.csc_array | sparse.csr_array | sparse.coo_array)
    assert result.shape == pattern.shape
    np.testing.assert_allclose(result.data, _expected_values(dense, pattern, layout), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.toarray(), dense, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kind", ["jacobian", "hessian"])
def test_a_map_of_independent_stages_keeps_one_batched_pattern(kind: str) -> None:
  derivative = sc.sparse_jacobian(stage) if kind == "jacobian" else sc.sparse_hessian(stage_cost, triangle="lower", layout="csr")
  local = derivative.sparsity()
  assert local is not None
  batched = sc.vmap(derivative, N)
  assert batched.sparsity() is local
  points = POINT.reshape((N, 2))
  result = batched(points)
  assert isinstance(result, sparse.coo_array) and result.shape == (N, *local.shape)
  source = stage if kind == "jacobian" else stage_cost
  dense = np.stack([_reference(source, point, kind == "hessian") for point in points])
  if kind == "hessian":
    dense = np.tril(dense)
  np.testing.assert_allclose(result.toarray(), dense, rtol=1e-6, atol=1e-6)
  for k, point in enumerate(points):
    single = derivative(point)
    np.testing.assert_allclose(single.data, _expected_values(dense[k], local, "csr" if kind == "hessian" else "csc"), rtol=1e-6, atol=1e-6)


def test_a_transposed_sparse_output_keeps_its_pattern_through_a_call() -> None:
  derivative = sc.sparse_jacobian(stage)

  @sc.function(sc.arg("z", 2))
  def transposed(z: sc.Expr) -> sc.Expr:
    return derivative(z).T

  pattern = transposed.sparsity()
  local = derivative.sparsity()
  assert pattern is not None and local is not None
  assert set(zip(pattern.rows, pattern.cols)) == set(zip(local.cols, local.rows))
  np.testing.assert_allclose(transposed(POINT[:2]).toarray(), derivative(POINT[:2]).toarray().T)


def test_a_coordinate_layout_must_list_the_derivative_pattern() -> None:
  with pytest.raises(ValueError, match="different stored pattern"):
    sc.sparse_jacobian(stage, layout=sc.SparsityPattern((3, 2), (0,), (0,))).instantiate()
