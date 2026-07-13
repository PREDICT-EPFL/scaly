import numpy as np
import pytest

import alloy as al
from alloy.expr import topo
from benchmarks.problems.tracking_nmpc import NX, NZ, ca_tracking_eq_jac, tracking_eq_function

pytest.importorskip("casadi")


@pytest.mark.parametrize("horizon", [1, 2])
def test_tracking_eq_jacobian_matches_casadi(horizon: int) -> None:
  fn = tracking_eq_function(horizon)
  jf = al.jacobian(fn, "z", "eq")
  ca_jf = ca_tracking_eq_jac(horizon)

  rng = np.random.default_rng(0)
  zv = rng.normal(size=NZ * (horizon + 1))
  pv = rng.normal(size=NX * (horizon + 1))

  np.testing.assert_allclose(jf(zv), np.array(ca_jf(zv, pv)), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("horizon", [1, 2])
def test_tracking_eq_sparse_jacobian_metadata_matches_dense(horizon: int) -> None:
  fn = tracking_eq_function(horizon)
  spjf = al.spjacobian(fn, "z", "eq")
  jf = al.jacobian(fn, "z", "eq")

  rng = np.random.default_rng(1)
  zv = rng.normal(size=NZ * (horizon + 1))

  sparsity = spjf.output_sparsities[0]
  assert sparsity is not None
  dense = jf(zv)
  compact = spjf(zv)
  assert isinstance(dense, np.ndarray)
  assert isinstance(compact, np.ndarray)
  flat = np.asarray(sparsity.rows) * dense.shape[1] + np.asarray(sparsity.cols)
  np.testing.assert_allclose(compact, np.ravel(dense)[flat])
  assert sparsity.nnz < dense.size


@pytest.mark.parametrize("horizon", [1, 2])
def test_tracking_eq_colored_sparse_jacobian_matches_reference_path(horizon: int) -> None:
  fn = tracking_eq_function(horizon)
  colored = al.sparse_jacobian_colored(fn.outputs[0], fn.inputs[0])
  reference = al.sparse_jacobian_reference(fn.outputs[0], fn.inputs[0])
  colored_fn = al.Function("tracking_spjac_colored", [fn.inputs[0]], [colored.values], ["z"], ["colored"])
  reference_fn = al.Function("tracking_spjac_reference", [fn.inputs[0]], [reference.values], ["z"], ["reference"])
  compare = al.Function("tracking_spjac_compare", [fn.inputs[0]], [colored.values, reference.values], ["z"], ["colored", "reference"])

  rng = np.random.default_rng(2)
  zv = rng.normal(size=NZ * (horizon + 1))

  assert colored.sparsity == reference.sparsity
  assert len(topo(colored_fn.outputs)) < len(topo(reference_fn.outputs))
  colored_values, reference_values = compare(zv)
  np.testing.assert_allclose(colored_values, reference_values, rtol=1e-10, atol=1e-10)
