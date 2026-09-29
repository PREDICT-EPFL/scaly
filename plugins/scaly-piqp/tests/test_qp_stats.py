from __future__ import annotations


import numpy as np
import pytest

import scaly as sc
from scaly.testing.helpers import build_qp, solve_qp


def _problem(*, max_iter: int | None = None) -> sc.Function:
  options: dict[str, float | int] = {} if max_iter is None else {"max_iter": max_iter}
  return build_qp(
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


@pytest.mark.method("opt.piqp")
def test_qp_stats_success_and_timing_split() -> None:
  qp = _problem()
  out = solve_qp(qp, np.zeros(2), np.zeros(0), np.zeros(2))
  stats = sc.opt.solver_stats(qp)
  assert stats is not None
  assert stats.version == sc.opt.SCALY_SOLVER_STATS_VERSION
  assert stats.status == sc.Status.OK
  assert stats.iter > 0
  assert stats.obj == pytest.approx(float(out["cost"]), rel=1e-12, abs=1e-12)
  assert stats.n_eval_f == 1
  assert all(value >= 0.0 for value in (stats.t_total, stats.t_fe, stats.t_solver, stats.t_qp, stats.t_globalization, stats.t_glue))
  assert stats.t_solver == 0.0 and stats.t_qp > 0.0 and stats.t_globalization == 0.0
  assert stats.t_total == pytest.approx(stats.t_fe + stats.t_solver + stats.t_qp + stats.t_globalization + stats.t_glue, rel=0.1, abs=1e-12)
  # v3 diagnostics: native iteration count and primal residual; the
  # line-search/merit fields do not apply to a direct QP solve.
  assert stats.qp_iter == stats.iter > 0
  assert 0.0 <= stats.primal_viol < 1e-6
  assert stats.step_inf == 0.0 and stats.alpha == 0.0 and stats.merit_penalty == 0.0 and stats.backtracks == 0


@pytest.mark.method("opt.piqp")
def test_qp_stats_maps_max_iter_status() -> None:
  qp = _problem(max_iter=1)
  solve_qp(qp, np.zeros(2), np.zeros(0), np.zeros(2))
  assert sc.opt.solver_stats(qp) is not None
  assert sc.opt.solver_stats(qp).status == sc.Status.MAX_ITER
  assert sc.opt.solver_stats(qp).native_status == -1
  assert sc.opt.solver_stats(qp).iter == 1
  assert sc.opt.solver_stats(qp).to_solver_status() is not None and not sc.opt.solver_stats(qp).to_solver_status().ok


@pytest.mark.method("opt.piqp")
def test_qp_reserved_name_compiles_solves_and_exposes_stats() -> None:
  qp = build_qp(P=np.eye(2), c=np.array([-0.25, 0.5]), name="w")
  out = solve_qp(qp, np.zeros(2), np.zeros(0), np.zeros(0))
  np.testing.assert_allclose(out["x"], [0.25, -0.5], atol=1e-8)
  assert sc.opt.solver_stats(qp, "w").status == sc.Status.OK
