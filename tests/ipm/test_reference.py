"""The NumPy reference of PIQP 0.6.2 against vendored PIQP, and on its own.

The gate (plan, Tier 3): the reference reproduces PIQP's decision traces on at least 90% of the
stored Maros–Mészáros subset. A decision trace matches when the status and the iteration count are
equal and every iteration's proximal parameters rho and delta, which PIQP's update rules decide,
agree to 1e-3 relative (the table prints four digits). The comparison is with PIQP's sparse
backend, whose full KKT system the reference solves; its dense backend condenses the system and
rounds differently, which changes the path on a few rounding-sensitive problems.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from tests.ipm import piqp_trace
from tests.ipm import reference as ref
from tests.ipm.problems import _qp, gate_problems, infeasible_problems, kkt_residuals, maros_meszaros, maros_meszaros_names, mpc_qp, random_qp

NAMES = maros_meszaros_names()
GATE = 0.9


def _decisions_match(pq: piqp_trace.Trace, r: ref.Result) -> bool:
  column = r.trace[:, ref.TRACE_COLUMNS.index("rho")], r.trace[:, ref.TRACE_COLUMNS.index("delta")]
  return piqp_trace.same_decisions(pq, r.status, r.info.iter, *column)


def _exact_rows(r: ref.Result, fields: tuple[str, ...]) -> np.ndarray:
  return r.trace[1:, [ref.TRACE_COLUMNS.index(f) for f in fields]]


@pytest.mark.solver("piqp")
def test_the_decision_trace_gate() -> None:
  misses = [name for name in NAMES if not _decisions_match(piqp_trace.run(maros_meszaros(name)), ref.solve(maros_meszaros(name)))]
  assert len(NAMES) - len(misses) >= GATE * len(NAMES), f"traces differ on {misses}"


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("name, rtol", [("HS21", 1e-11), ("DUAL1", 1e-9), ("HS118", 1e-8), ("TAME", 1e-6)])
def test_full_precision_on_well_conditioned_problems(name: str, rtol: float) -> None:
  """Every iteration at full precision: rho, delta, mu, sigma and both step lengths. TAME's duals
  fall below machine epsilon, so PIQP shifts them off the boundary; without the shift mu is 61% off."""
  qp = maros_meszaros(name)
  fields = ("rho", "delta", "mu", "sigma", "primal_step", "dual_step")
  exact = piqp_trace.exact_trace(qp, fields=fields)
  np.testing.assert_allclose(_exact_rows(ref.solve(qp), fields), exact, rtol=rtol)


@pytest.mark.solver("piqp")
def test_early_iterations_agree_to_rounding_everywhere() -> None:
  """After one iteration the reference and PIQP differ only by rounding on every problem."""
  for name in NAMES:
    qp = maros_meszaros(name)
    one = piqp_trace.run(qp, max_iter=1)
    r = ref.solve(qp, ref.Settings(max_iter=1))
    for f in ("rho", "delta", "mu", "sigma", "primal_step", "dual_step"):
      np.testing.assert_allclose(getattr(r.info, f), one.info[f], rtol=1e-7, err_msg=f"{name} {f}")
    np.testing.assert_allclose(r.x, one.vectors["x"], rtol=1e-7, atol=1e-9 * (1 + np.abs(r.x).max()), err_msg=name)


@pytest.mark.solver("piqp")
def test_iterative_refinement_always_on() -> None:
  """With refinement from the first factorization (PIQP turns it on after a failed one)."""
  settings = ref.Settings(iterative_refinement_always_enabled=True)
  misses = [n for n in NAMES if not _decisions_match(piqp_trace.run(maros_meszaros(n), refine_always=True), ref.solve(maros_meszaros(n), settings))]
  assert len(NAMES) - len(misses) >= GATE * len(NAMES), f"traces differ on {misses}"


@pytest.mark.solver("piqp")
def test_cost_scaling_including_its_aliased_stopping_test() -> None:
  """PIQP computes the cost scaling in the memory the Ruiz loop's stopping test reads for the box
  scaling. On an LP whose constraint scaling converges at once, that keeps Ruiz iterating; without
  the aliasing the reference stops early and differs from PIQP by 7e-5 after one iteration."""
  qp = _qp("near_unit_lp", sparse.csc_array((2, 2)), [0.5, 0.3], G=np.array([[1.0001, 0.9999]]), h_l=[1.0], x_l=[0.0, 0.0])
  for k in (1, 2):
    pq = piqp_trace.run(qp, scale_cost=True, max_iter=k)
    r = ref.solve(qp, ref.Settings(preconditioner_scale_cost=True, max_iter=k))
    for f in ("rho", "delta", "mu", "sigma", "primal_step", "dual_step"):
      np.testing.assert_allclose(getattr(r.info, f), pq.info[f], rtol=1e-12, err_msg=f)
  for name in ("QAFIRO", "CVXQP1_S", "DUALC1", "QPCBLEND"):
    qp = maros_meszaros(name)
    pq = piqp_trace.run(qp, scale_cost=True, max_iter=2)
    r = ref.solve(qp, ref.Settings(preconditioner_scale_cost=True, max_iter=2))
    np.testing.assert_allclose(r.info.mu, pq.info["mu"], rtol=1e-8, err_msg=name)
    np.testing.assert_allclose(r.info.delta, pq.info["delta"], rtol=1e-8, err_msg=name)


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("args", [(4, 2, 10), (12, 4, 20)])
def test_mpc_traces(args) -> None:
  qp = mpc_qp(*args)
  assert _decisions_match(piqp_trace.run(qp), ref.solve(qp))


@pytest.mark.parametrize("name", sorted(piqp_trace.INFEASIBLE_STATUS))
def test_infeasible_problems_end_as_piqp_ends_them(name: str) -> None:
  qp, _ = infeasible_problems()[name]
  assert ref.solve(qp).status == piqp_trace.INFEASIBLE_STATUS[name]


@pytest.mark.parametrize("name", NAMES)
def test_reference_solutions_satisfy_the_kkt_conditions(name: str) -> None:
  qp = maros_meszaros(name)
  r = ref.solve(qp)
  assert r.status == ref.SOLVED
  primal, dual = kkt_residuals(qp, x=r.x, y=r.y, z_l=r.z_l, z_u=r.z_u, z_bl=r.z_bl, z_bu=r.z_bu)
  scale = 1.0 + abs(qp.objective(r.x))
  assert primal <= 1e-6 * (1 + np.abs(r.x).max()) and dual <= 1e-5 * scale
  # Complementarity: an active bound has a positive multiplier, an inactive one none.
  finite_l = np.isfinite(qp.x_l)
  gap = np.abs(r.z_bl[finite_l] * (r.x[finite_l] - qp.x_l[finite_l]))
  assert gap.max(initial=0.0) <= 1e-5 * scale


def test_rows_infinite_on_both_sides_stay_as_zero_rows() -> None:
  qp = _qp("free_row", np.eye(2), [1.0, 1.0], G=np.array([[1.0, 2.0], [3.0, 4.0]]), h_l=[-np.inf, 0.0], h_u=[np.inf, 5.0])
  d = ref.setup_data(qp)
  assert np.all(d.G.toarray()[0] == 0.0) and np.all(d.G.toarray()[1] == [3.0, 4.0])
  assert (d.h_l[0], d.h_u[0]) == (-1.0, 1.0)
  np.testing.assert_array_equal(d.h_l_idx, [0, 1])
  np.testing.assert_array_equal(d.h_u_idx, [0, 1])
  assert ref.solve(qp).status == ref.SOLVED


def test_box_terms_enter_the_residual_signed() -> None:
  """PIQP takes max(inf, value) for the box terms, so a negative box residual never counts."""
  assert ref._signed_max(0.5, np.array([-3.0, 0.2]), np.array([])) == 0.5
  assert ref._signed_max(0.5, np.array([-3.0, 0.7])) == 0.7
  assert ref._inf_norm(np.array([-3.0, 0.7])) == 3.0


def test_the_no_inequality_branch_takes_full_steps() -> None:
  r = ref.solve(maros_meszaros("GENHS28"))
  assert r.status == ref.SOLVED
  np.testing.assert_array_equal(r.trace[1:, ref.TRACE_COLUMNS.index("primal_step")], 1.0)


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("name", sorted(gate_problems()))
def test_decisions_match_wherever_piqps_backends_agree(name: str) -> None:
  """Where PIQP's own two backends follow the same path, the reference has to take the same decisions."""
  qp = gate_problems()[name]
  sparse_run = piqp_trace.run(qp)
  if not piqp_trace.backends_agree(sparse_run, piqp_trace.run(qp, dense=True)):
    pytest.skip("PIQP's backends take different paths: the problem is sensitive to rounding")
  assert _decisions_match(sparse_run, ref.solve(qp))


SCALING_CASES = {
  # Ruiz converges at once: the stopping test ends it after one pass.
  "near_unit_lp": _qp("near_unit_lp", sparse.csc_array((2, 2)), [0.5, 0.3], G=np.array([[1.0001, 0.9999]]), h_l=[1.0], x_l=[0.0, 0.0]),
  # Column maxima below 1e-4 (left unscaled) and above 1e4 (capped).
  "tiny_and_huge": _qp(
    "tiny_and_huge",
    np.diag([3e-6, 2e5, 1.0]),
    [1.0, -1.0, 0.5],
    G=np.array([[5e-5, 1.0, 1.0], [1.0, 3e5, 0.0]]),
    h_l=[-1.0, -2.0],
    h_u=[1.0, 2.0],
    x_l=[-1.0, -1.0, -1.0],
    x_u=[1.0, 1.0, 1.0],
  ),
  # Small entries in a column: the box scaling (1) still dominates its maximum.
  "small_columns": _qp("small_columns", np.diag([2e-5, 1.0]), [1.0, 1.0], A=np.array([[3e-5, 1.0]]), b=[0.5], x_l=[-10.0, -10.0]),
  # Row maxima between 1e-6 and 1e-4: below Ruiz's lower limit, so those rows stay unscaled.
  "small_rows": _qp("small_rows", np.eye(2), [1.0, -1.0], A=np.array([[3e-5, 2e-5]]), b=[1e-5], G=np.array([[5e-6, -4e-5]]), h_u=[1e-5]),
}


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("name", sorted(SCALING_CASES))
def test_scaling_edge_cases_at_full_precision(name: str) -> None:
  qp = SCALING_CASES[name]
  for k in (1, 2, 3):
    pq = piqp_trace.run(qp, max_iter=k)
    r = ref.solve(qp, ref.Settings(max_iter=k))
    for f in ("rho", "delta", "mu", "sigma", "primal_step", "dual_step"):
      np.testing.assert_allclose(getattr(r.info, f), pq.info[f], rtol=1e-8, err_msg=f"{name} after {k}: {f}")


@pytest.mark.solver("piqp")
def test_the_inequality_dual_shift() -> None:
  """An LP whose inequality duals reach machine epsilon by iteration 11: PIQP shifts them by eps,
  and without that shift the reference is 20-40% off in the next two iterations."""
  qp = random_qp(20, 15, 5, seed=7, lp=True)
  fields = ("rho", "delta", "mu", "sigma", "primal_step", "dual_step")
  for k in range(1, 13):
    pq = piqp_trace.run(qp, max_iter=k)
    r = ref.solve(qp, ref.Settings(max_iter=k))
    np.testing.assert_allclose([getattr(r.info, f) for f in fields], [pq.info[f] for f in fields], rtol=1e-9, err_msg=f"after {k}")
