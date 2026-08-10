from __future__ import annotations


import numpy as np
import pytest

import alloy as al


def _problem(*, max_iter: int | None = None) -> al.SolverFunction:
  options: dict[str, float | int] = {} if max_iter is None else {"max_iter": max_iter}
  return al.qp(
    P=np.array([[4.0, 1.0], [1.0, 2.0]]),
    c=np.array([-1.0, -1.0]),
    G_ineq=np.array([[1.0, 2.0], [-1.0, 2.0]]),
    l_ineq=np.array([1.0, -2.0]),
    u_ineq=np.array([3.0, 2.0]),
    x_lb=np.array([-1.0, -1.0]),
    x_ub=np.array([2.0, 2.0]),
    options=options,
    name=f"stats_qp_{max_iter}",
  )


@pytest.mark.solver("piqp")
def test_qp_stats_success_and_timing_split() -> None:
  qp = _problem()
  out = qp(np.zeros(2), np.zeros(0), np.zeros(2))
  stats = qp.last_stats
  assert stats is not None
  assert stats.version == al.ALLOY_SOLVER_STATS_VERSION
  assert stats.status == al.AlloySolveStatus.OK
  assert stats.iter > 0
  assert stats.obj == pytest.approx(float(out["cost"]), rel=1e-12, abs=1e-12)
  assert stats.n_eval_f == 1
  assert all(value >= 0.0 for value in (stats.t_total, stats.t_fe, stats.t_solver, stats.t_glue))
  assert stats.t_total == pytest.approx(stats.t_fe + stats.t_solver + stats.t_glue, rel=0.1, abs=1e-12)


@pytest.mark.solver("piqp")
def test_qp_stats_maps_max_iter_status() -> None:
  qp = _problem(max_iter=1)
  qp(np.zeros(2), np.zeros(0), np.zeros(2))
  assert qp.last_stats is not None
  assert qp.last_stats.status == al.AlloySolveStatus.MAX_ITER
  assert qp.last_stats.native_status == -1
  assert qp.last_stats.iter == 1
  assert qp.last_status is not None and not qp.last_status.ok


@pytest.mark.solver("piqp")
def test_qp_reserved_name_compiles_solves_and_exposes_stats() -> None:
  qp = al.qp(P=np.eye(2), c=np.array([-0.25, 0.5]), name="w")
  out = qp(np.zeros(2), np.zeros(0), np.zeros(0))
  np.testing.assert_allclose(out["x"], [0.25, -0.5], atol=1e-8)
  assert qp.solver_stats("w").status == al.AlloySolveStatus.OK
