"""The PIQP trace harness: vendored PIQP 0.6.2 run through the C driver, its output parsed."""

from __future__ import annotations

import numpy as np
import pytest

from tests.opt.ipm import piqp_trace
from tests.opt.ipm.problems import infeasible_problems, kkt_residuals, maros_meszaros, maros_meszaros_names, mpc_qp

NAMES = maros_meszaros_names()


@pytest.mark.solver("piqp")
def test_hs21_matches_its_known_optimum() -> None:
  qp = maros_meszaros("HS21")
  trace = piqp_trace.run(qp)
  assert trace.status == 1 and trace.info["iter"] == 9
  assert trace.table.shape == (10, len(piqp_trace.COLUMNS))  # iteration 0 is the initial point
  np.testing.assert_array_equal(trace.column("iter"), np.arange(10))
  np.testing.assert_allclose(qp.objective(trace.vectors["x"]), -99.96, rtol=1e-9)
  np.testing.assert_allclose(trace.column("primal_obj")[-1], trace.info["primal_obj"], rtol=1e-5)


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("name", NAMES)
def test_every_problem_solves_the_same_with_both_backends(name: str) -> None:
  qp = maros_meszaros(name)
  sparse_run, dense_run = piqp_trace.run(qp), piqp_trace.run(qp, dense=True)
  assert sparse_run.status == 1 and dense_run.status == 1
  assert (sparse_run.backend, dense_run.backend) == ("sparse", "dense")
  scale = 1.0 + np.abs(sparse_run.info["primal_obj"])
  assert abs(sparse_run.info["primal_obj"] - dense_run.info["primal_obj"]) <= 1e-5 * scale
  primal, dual = kkt_residuals(qp, **sparse_run.vectors)
  assert primal <= 1e-6 * (1.0 + np.abs(sparse_run.vectors["x"]).max()) and dual <= 1e-5 * scale


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("name", ["HS21", "QAFIRO", "DUAL1", "CVXQP1_S", "QPCBLEND"])
def test_a_truncated_run_is_a_prefix_of_the_full_run(name: str) -> None:
  """``max_iter = k`` stops after iteration ``k`` of the full run: its full-precision result matches
  row ``k`` of the printed table to the table's precision."""
  qp = maros_meszaros(name)
  full = piqp_trace.run(qp)
  exact = piqp_trace.exact_trace(qp)
  assert exact.shape == (full.info["iter"], 6)
  table = full.table[1:]
  for col, name_ in enumerate(("rho", "delta", "mu")):
    np.testing.assert_allclose(exact[:, col], table[:, piqp_trace.COLUMNS.index(name_)], rtol=6e-4)
  for col, name_ in ((4, "primal_step"), (5, "dual_step")):
    np.testing.assert_allclose(exact[:, col], table[:, piqp_trace.COLUMNS.index(name_)], atol=6e-5)
  assert np.all((0.0 < exact[:, 3]) & (exact[:, 3] <= 1.0))  # sigma, which the table leaves out


def test_the_parser_reads_every_part() -> None:
  out = (
    "dense backend (dense_cholesky)\n"
    "iter  prim_obj ...\n"
    "  0    1.00000e+00   -2.00000e+00   3.00000e+00   4.00000e-01   5.00000e-01   1.000e-06   1.000e-04   6.000e+00   0.0000   0.0000\n"
    " 12   -1.00000e+00    2.00000e+00   3.00000e+00   4.00000e-01   5.00000e-01   1.000e-06   1.000e-04   6.000e+00   0.9900   0.9500\n"
    "10 more lines that are not iterations\n"
    "status: solved\nSCALY_TRACE_RESULT\ninfo status 1\ninfo iter 12\nvec x 1 -2.5 3e-3\nvec y\n"
  )
  trace = piqp_trace.parse(out)
  assert trace.status == 1 and trace.info["iter"] == 12 and trace.backend == "dense"
  np.testing.assert_array_equal(trace.column("iter"), [0.0, 12.0])
  np.testing.assert_array_equal(trace.column("dual_step"), [0.0, 0.95])
  np.testing.assert_array_equal(trace.vectors["x"], [1.0, -2.5, 3e-3])
  assert trace.vectors["y"].size == 0


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("name", sorted(piqp_trace.INFEASIBLE_STATUS))
def test_piqp_statuses_on_the_infeasible_set(name: str) -> None:
  qp, _ = infeasible_problems()[name]
  for dense in (False, True):
    assert piqp_trace.run(qp, dense=dense).status == piqp_trace.INFEASIBLE_STATUS[name]


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("args", [(4, 2, 10), (12, 4, 20)])
def test_mpc_problems_solve(args) -> None:
  qp = mpc_qp(*args)
  trace = piqp_trace.run(qp)
  assert trace.status == 1 and trace.info["iter"] < 20
  primal, _ = kkt_residuals(qp, **trace.vectors)
  assert primal < 1e-7
  assert trace.info["solve_time"] > 0.0
