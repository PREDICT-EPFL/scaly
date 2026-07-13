import numpy as np
import pytest

import alloy as al
from alloy.expr import topo
from benchmarks.problems.unbumpercars import (
  ca_unbumpercars_ineq_jac,
  dynamics_fn,
  sample_inputs,
  unbumpercars_ineq_function,
)

casadi = pytest.importorskip("casadi")


def _sample_inputs_or_skip(ncars: int) -> tuple[np.ndarray, np.ndarray]:
  try:
    return sample_inputs(ncars)
  except FileNotFoundError as exc:
    pytest.skip(str(exc))


def test_unbumpercars_reduced_ineq_jacobian_matches_casadi_mx() -> None:
  ncars = 2
  fn = unbumpercars_ineq_function(ncars)
  jf = fn.factory("unbumpercars_reduced_ineq_jac_N2", ["u", "p"], ["jac:ineq:u"])
  ca_jf = ca_unbumpercars_ineq_jac(ncars, casadi.MX)
  uv, pv = _sample_inputs_or_skip(ncars)

  np.testing.assert_allclose(jf(uv, pv), np.asarray(ca_jf(uv, pv)), rtol=1e-9, atol=1e-9)


def test_unbumpercars_reduced_colored_sparse_jacobian_matches_dense_and_casadi_structure() -> None:
  ncars = 2
  fn = unbumpercars_ineq_function(ncars)
  spjf = fn.factory("unbumpercars_reduced_ineq_spjac_N2", ["u", "p"], ["spjac:ineq:u"])
  jf = fn.factory("unbumpercars_reduced_ineq_jac_N2", ["u", "p"], ["jac:ineq:u"])
  ca_jf = ca_unbumpercars_ineq_jac(ncars, casadi.MX)
  uv, pv = _sample_inputs_or_skip(ncars)

  sparsity = spjf.output_sparsities[0]
  assert sparsity is not None
  dense = jf(uv, pv)
  compact = spjf(uv, pv)
  assert isinstance(dense, np.ndarray)
  assert isinstance(compact, np.ndarray)
  flat = np.asarray(sparsity.rows) * dense.shape[1] + np.asarray(sparsity.cols)
  np.testing.assert_allclose(compact, np.ravel(dense)[flat], rtol=1e-9, atol=1e-9)

  ca_mask = np.zeros(dense.shape, dtype=bool)
  ca_rows, ca_cols = ca_jf.sparsity_out(0).get_triplet()
  ca_mask[np.asarray(ca_rows, dtype=np.int64), np.asarray(ca_cols, dtype=np.int64)] = True
  np.testing.assert_array_equal(sparsity.to_mask(), ca_mask)


def test_unbumpercars_reduced_fixture_marks_dense_mlp_and_sparse_constraints() -> None:
  assert any(node.op == al.Ops.MATMUL and node.lowering == "block" for node in topo(dynamics_fn.outputs))
  fn = unbumpercars_ineq_function(2)
  assert fn.outputs[0].lowering == "scalar"
