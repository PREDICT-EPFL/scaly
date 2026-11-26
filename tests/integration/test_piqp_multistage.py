"""The arrow structure of ``examples/piqp_multistage`` is PIQP 0.7.0's, and the factor of ``Psi`` stays inside it."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _load(name: str):
  spec = importlib.util.spec_from_file_location(name, EXAMPLES / name / "__init__.py", submodule_search_locations=[str(EXAMPLES / name)])
  assert spec is not None and spec.loader is not None
  if name not in sys.modules:
    sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[name])
  return sys.modules[name]


_load("piqp_sparse")
piqp_multistage = _load("piqp_multistage")
problems = importlib.import_module("piqp_multistage.problems")
mpc_qp, banded_qp = problems.mpc_qp, problems.banded_qp
Structure = importlib.import_module("piqp_sparse.data").Structure


def _kkt(qp):
  structure, *_ = Structure.from_matrices(qp.P, qp.A, qp.G)
  return piqp_multistage.MultistageKKT(structure)


# PIQP's MultistageKKT::block_info for these problems, printed by
# internal/notes/piqp_multistage_arrow_2026-11-26/block_info.cpp
PIQP_BLOCK_INFO = {
  (mpc_qp, 5, 3, 2, 0, 0): [(0, 5, 3), (5, 5, 3), (10, 5, 3), (15, 5, 3), (20, 8, 0), (28, 0, 0)],
  (mpc_qp, 8, 4, 2, 2, 1): [(0, 6, 4), (6, 6, 4), (12, 6, 4), (18, 6, 4), (24, 6, 4), (30, 6, 4), (36, 6, 4), (42, 10, 0), (52, 2, 0)],
  (mpc_qp, 10, 2, 1, 0, 3): [
    (0, 3, 2),
    (3, 3, 2),
    (6, 3, 2),
    (9, 3, 2),
    (12, 3, 2),
    (15, 3, 2),
    (18, 3, 2),
    (21, 3, 2),
    (24, 3, 2),
    (27, 5, 0),
    (32, 0, 0),
  ],
  (mpc_qp, 20, 6, 3, 3, 2): [*((9 * k, 9, 6) for k in range(19)), (171, 14, 0), (185, 4, 0)],
  # irregular blocks, which reach the ratio that closes a block and the rounding of a block's halves
  (banded_qp, 16, 4, 0.3, 2, 3): [(0, 6, 3), (6, 8, 0), (14, 2, 0)],
  (banded_qp, 16, 2, 0.3, 0, 2): [(0, 1, 0), (1, 2, 1), (3, 2, 0), (5, 1, 0), (6, 1, 0), (7, 5, 0), (12, 4, 0), (16, 0, 0)],
}


@pytest.mark.parametrize("problem", PIQP_BLOCK_INFO, ids=lambda key: f"{key[0].__name__}{key[1:]}")
def test_block_info_is_piqps(problem):
  make, *args, seed = problem
  kkt = _kkt(make(*args, seed=seed))
  assert [tuple(b) for b in kkt.block_info] == PIQP_BLOCK_INFO[problem]


def test_factor_stays_inside_the_blocks():
  qp = mpc_qp(horizon=6, nx=4, nu=2, n_global=2, seed=5)
  kkt = _kkt(qp)
  plot = importlib.import_module("piqp_multistage.plot_arrow_structure")
  Psi, L = plot.psi(qp), plot.factor(kkt, qp)
  np.testing.assert_allclose((L @ L.T).toarray(), Psi.toarray(), atol=1e-10 * abs(Psi).max())
  mask = kkt.block_mask()
  for matrix in (Psi, L):
    lower = sparse.tril(matrix).tocoo()
    assert mask[lower.row, lower.col].all()
  # the arrow holds the shared parameters, and L fills every coupling block
  assert kkt.arrow_width >= 2
  for block in kkt.blocks:
    if block.kind == "B":
      assert (L.toarray()[block.rows.start : block.rows.stop, block.cols.start : block.cols.stop] != 0).any()
