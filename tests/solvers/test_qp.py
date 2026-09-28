from __future__ import annotations

import numpy as np

import scaly as sc
from scaly.solvers.qp import _qp_matrix_sparsity


def test_sparse_qp_dependency_mask_keeps_entries_that_probe_to_zero() -> None:
  """A parameter-dependent entry whose value happens to be zero at the probe
  draw must stay in the pattern (the dependency mask, not the probe, keeps it)."""
  t = sc.sym("t", 1)
  zero = sc.const(0.0)
  P = sc.stack([sc.stack([sc.const(2.0), t[0] - t[0]]), sc.stack([zero, sc.const(2.0)])], axis=0)
  sparsity = _qp_matrix_sparsity(P, (t,), np.diag([2.0, 2.0]), triu=True)
  assert set(zip(sparsity.rows, sparsity.cols)) == {(0, 0), (0, 1), (1, 1)}
