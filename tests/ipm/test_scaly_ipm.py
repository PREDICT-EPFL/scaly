"""The generated interior-point solver against vendored PIQP and the NumPy reference.

The tier gate (plan, Tier 3): the generated solver takes PIQP's decisions. Each backend is compared
with PIQP's run of the same backend, by decision trace (``piqp_trace.same_decisions``), wherever
PIQP's two backends follow the same path (``piqp_trace.backends_agree``); elsewhere the path is
sensitive to rounding and only the status has to agree. The dense backend's condensed Cholesky
loses up to all digits of its late solves, in PIQP as here (LAPACK's does too), so where a pivot's
sign or a late step is rounding noise it may leave PIQP's path: ``DENSE_ROUNDING`` names those
problems, and there only the status and the iteration count to within a few are held to PIQP's.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

import scaly as sc
from scaly.solvers.ipm import MAX_ITER_REACHED, NUMERICS, SOLVED, TRACE_FIELDS, Backend, QPValues, Settings, Solver
from tests.ipm import piqp_trace
from tests.ipm import reference as ref
from tests.ipm.problems import QP, _qp, gate_problems, ipm_inputs, kkt_residuals, maros_meszaros

ORDER = ("P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u")
RESULT = ("x", "y", "z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu", "status", "iter", "trace", "trace_rows", "info")
RHO, DELTA = TRACE_FIELDS.index("rho"), TRACE_FIELDS.index("delta")
# Where the dense backend's condensed Cholesky rounds off PIQP's path (see the module docstring).
DENSE_ROUNDING = {"QADLITTL"}  # its last factorization, with refinement on, has a negative noise pivot here
NONCONVEX = {
  # The condensed matrix turns indefinite mid-run: PIQP's dense backend retries with rho and delta
  # scaled by 100 (sparse LDL^T fails only on a zero pivot, so that backend never retries).
  "nc_box": _qp("nc_box", np.diag([-1.0, 1.0]), [0.1, -0.5], G=np.array([[1.0, 1.0]]), h_u=[5.0], x_l=[-10.0, -10.0], x_u=[10.0, 10.0]),
  "nc_box3": _qp("nc_box3", np.diag([-2.0, 1.0, 0.5]), [0.3, -0.5, 0.2], A=np.array([[1.0, 1.0, 1.0]]), b=[1.0], x_l=[-20.0] * 3, x_u=[20.0] * 3),
}
# A variable in no constraint and no cost term: with rho = 0 the KKT matrix is singular.
LOOSE = _qp("loose_variable", np.diag([1.0, 2.0, 0.0]), [1.0, -1.0, 0.0], G=np.array([[1.0, 1.0, 0.0]]), h_u=[1.0], x_l=[-5.0, -5.0, -np.inf])


def _problem(name: str) -> QP:
  return {**gate_problems(), **NONCONVEX, LOOSE.name: LOOSE}[name]


@cache
def _solver(name: str, backend: Backend, settings: Settings) -> sc.Function:
  qp = _problem(name)
  s, values = ipm_inputs(qp)
  syms = {k: sc.sym(k, np.shape(values[k])) for k in ORDER}
  out = Solver(s, backend, settings, name=f"ipm_{qp.name}").solve(QPValues.preprocess(s, **syms), trace=True)
  tag = abs(hash(settings)) % 10**8
  return sc.Function._from_exprs(f"ipm_{qp.name}_{backend}_{tag}", [syms[k] for k in ORDER], [out[k] for k in RESULT], list(ORDER), list(RESULT))


def solve(qp: QP, backend: Backend, settings: Settings | None = None) -> dict[str, np.ndarray]:
  _, values = ipm_inputs(qp)
  got = dict(zip(RESULT, _solver(qp.name, backend, settings or Settings())(tuple(values[k] for k in ORDER)), strict=True))
  got["trace"] = got["trace"][: int(got["trace_rows"])]
  return got


def _matches(pq: piqp_trace.Trace, got: dict[str, np.ndarray]) -> bool:
  return piqp_trace.same_decisions(pq, int(got["status"]), int(got["iter"]), got["trace"][:, RHO], got["trace"][:, DELTA])


# --- the gate -----------------------------------------------------------------------------------


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("backend", ["sparse", "dense"])
@pytest.mark.parametrize("name", sorted(gate_problems()))
def test_decisions_match_piqp(name: str, backend: Backend) -> None:
  qp = gate_problems()[name]
  mine = piqp_trace.run(qp, dense=backend == "dense")
  other = piqp_trace.run(qp, dense=backend != "dense")
  got = solve(qp, backend)
  assert int(got["status"]) == mine.status
  if not piqp_trace.backends_agree(mine, other):
    pytest.skip("PIQP's backends take different paths: the problem is sensitive to rounding")
  if backend == "dense" and name in DENSE_ROUNDING:
    assert abs(int(got["iter"]) - int(mine.info["iter"])) <= 3
    return
  assert _matches(mine, got)


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("backend", ["sparse", "dense"])
@pytest.mark.parametrize("name", sorted(NONCONVEX))
def test_factorization_retries_follow_piqp(name: str, backend: Backend) -> None:
  """PIQP's dense Cholesky fails on the indefinite matrix, turns refinement on, then scales rho and
  delta by 100 until it factors; the generated solver takes the same steps, and the same none
  with the sparse backend."""
  qp = NONCONVEX[name]
  pq = piqp_trace.run(qp, dense=backend == "dense")
  got = solve(qp, backend)
  assert _matches(pq, got)
  rho = got["trace"][:, RHO]
  assert (backend == "dense") == bool(np.any(rho[1:] > 10 * rho[:-1])) == bool(got["info"][-1])


@pytest.mark.solver("piqp")
@pytest.mark.parametrize(
  "name, backend",
  [(name, "sparse") for name in ("HS21", "QAFIRO", "DUAL1", "CVXQP1_S", "QPCBLEND", "HS118")]
  + [(name, "dense") for name in ("HS21", "DUAL1", "CVXQP1_S", "HS118")],
)
def test_refinement_always_on(name: str, backend: Backend) -> None:
  """With refinement from the first factorization. The dense backend's late pivots on QAFIRO and
  QPCBLEND are rounding noise even with the static regularization, so only well-conditioned
  problems are held to PIQP's path there."""
  qp = maros_meszaros(name)
  got = solve(qp, backend, Settings(iterative_refinement_always_enabled=True))
  assert got["info"][-1] == 1.0
  assert _matches(piqp_trace.run(qp, dense=backend == "dense", refine_always=True), got)


# --- against the reference, step by step --------------------------------------------------------


@pytest.mark.parametrize(
  "name, rtol",
  [
    ("HS21", 1e-12),
    ("DUAL1", 1e-10),
    ("HS118", 1e-9),
    ("TAME", 1e-7),
    ("GENHS28", 1e-12),
    ("QPCBLEND", 1e-5),
    ("mpc_4x2_N10", 1e-11),
    ("empty_slab", 1e-10),
  ],
)
def test_every_iteration_matches_the_reference(name: str, rtol: float) -> None:
  """Every iteration at full precision with the sparse backend, which solves the same full KKT
  system as the reference: rho, delta, mu, sigma and the step lengths to ``rtol``, and every field
  (objectives and residuals too, which fall by orders of magnitude) to ``rtol`` of its largest
  value or of one. GENHS28 has no inequalities (PIQP's other branch), empty_slab a row with no bounds and no
  solution, TAME duals that PIQP shifts off the boundary."""
  qp = gate_problems()[name]
  got, r = solve(qp, "sparse"), ref.solve(qp)
  assert (int(got["status"]), int(got["iter"])) == (r.status, r.info.iter)
  decisions = [TRACE_FIELDS.index(f) for f in ("rho", "delta", "mu", "sigma", "primal_step", "dual_step")]
  np.testing.assert_allclose(got["trace"][:, decisions], r.trace[:, decisions], rtol=rtol)
  for i, field in enumerate(TRACE_FIELDS):
    np.testing.assert_allclose(got["trace"][:, i], r.trace[:, i], rtol=0, atol=rtol * max(np.abs(r.trace[:, i]).max(), 1.0), err_msg=field)


@pytest.mark.parametrize("backend", ["sparse", "dense"])
@pytest.mark.parametrize("name", ["HS21", "QAFIRO", "DUALC1", "CVXQP1_S", "PRIMALC1", "QPCBLEND", "HS76", "mpc_4x2_N10", "unbounded_lp"])
def test_the_result_matches_the_reference(name: str, backend: Backend) -> None:
  """The unscaled solution in PIQP's result layout: box values on their variables, and absent
  bounds with zero duals and slacks of PIQP_INF. The sparse backend follows the reference's path
  and ends where it ends; the dense one, whose late solves are rounded differently, may end
  elsewhere on a degenerate problem's optimal face (QAFIRO's), so it is held to the objective."""
  qp = gate_problems()[name]
  got, r = solve(qp, backend), ref.solve(qp)
  assert int(got["status"]) == r.status
  for key in ("x", "y", "z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu"):
    want = getattr(r, key)
    finite = want < 1e29
    np.testing.assert_array_equal(got[key] >= 1e29, ~finite, err_msg=key)
    size = 1.0 + np.abs(want[finite]).max(initial=0.0)
    if backend == "sparse":
      np.testing.assert_allclose(got[key][finite], want[finite], rtol=0, atol=1e-6 * size, err_msg=key)
  if r.status == ref.SOLVED:
    np.testing.assert_allclose(qp.objective(got["x"]), qp.objective(r.x), rtol=1e-6, atol=1e-8)
    primal, dual = kkt_residuals(qp, **{k: got[k] for k in ("x", "y", "z_l", "z_u", "z_bl", "z_bu")})
    assert primal <= 1e-6 * (1 + np.abs(got["x"]).max()) and dual <= 1e-5 * (1 + abs(qp.objective(got["x"])))


# --- retries and refinement against the reference ------------------------------------------------


@pytest.mark.parametrize("backend", ["sparse", "dense"])
def test_a_zero_pivot_turns_refinement_on(backend: Backend) -> None:
  """With rho starting at zero, the loose variable's pivot is exactly zero: the first factorization
  fails, refinement turns on and the static regularization lets it factor; the reference, whose
  SuperLU fails on the exactly singular matrix, does the same."""
  settings = Settings(rho_init=0.0)
  got = solve(LOOSE, backend, settings)
  r = ref.solve(LOOSE, ref.Settings(rho_init=0.0))
  assert got["info"][-1] == 1.0 and r.status == ref.SOLVED
  assert (int(got["status"]), int(got["iter"])) == (r.status, r.info.iter)
  np.testing.assert_allclose(got["trace"][:, [RHO, DELTA]], r.trace[:, [RHO, DELTA]], rtol=1e-6)
  np.testing.assert_allclose(got["x"], r.x, atol=1e-6)


@pytest.mark.parametrize("always", [False, True])
@pytest.mark.parametrize("backend", ["sparse", "dense"])
def test_a_factorization_that_never_succeeds(backend: Backend, always: bool) -> None:
  """Without static regularization and with rho at zero, no retry helps: after refinement (on
  from the start, or turned on by the first failure) and ten scalings of delta by 100 the solve
  ends as NUMERICS before its first iteration, with the regularization floor raised to eps_abs,
  as the reference ends it."""
  eps, rel = 0.0, 0.0  # no static regularization
  settings = Settings(
    rho_init=0.0,
    iterative_refinement_always_enabled=always,
    iterative_refinement_static_regularization_eps=eps,
    iterative_refinement_static_regularization_rel=rel,
  )
  got = solve(LOOSE, backend, settings)
  r = ref.solve(
    LOOSE,
    ref.Settings(
      rho_init=0.0,
      iterative_refinement_always_enabled=always,
      iterative_refinement_static_regularization_eps=eps,
      iterative_refinement_static_regularization_rel=rel,
    ),
  )
  assert int(got["status"]) == NUMERICS == r.status
  assert int(got["iter"]) == 0 == r.info.iter and int(got["trace_rows"]) == 0
  info = dict(zip(("rho", "delta", "reg_limit"), got["info"][[2, 3, 24]], strict=True))
  np.testing.assert_allclose([info["rho"], info["delta"], info["reg_limit"]], [0.0, 1e-4 * 100.0**10, 1e-8], rtol=1e-12)
  np.testing.assert_allclose([r.info.rho, r.info.delta, r.info.reg_limit], [0.0, 1e-4 * 100.0**10, 1e-8], rtol=1e-12)


def test_a_failure_mid_run_ends_the_solve_after_the_count() -> None:
  """With no retries allowed, the dense backend's first failure with refinement on ends the solve
  as NUMERICS: the pass counts as an iteration, its row is not printed, and the rows before are
  those of the run with retries."""
  qp = NONCONVEX["nc_box"]
  full, cut = solve(qp, "dense"), solve(qp, "dense", Settings(max_factor_retires=0))
  assert int(cut["status"]) == NUMERICS and int(cut["iter"]) == int(cut["trace_rows"]) >= 1
  np.testing.assert_array_equal(cut["trace"], full["trace"][: int(cut["trace_rows"])])


# --- the trace ------------------------------------------------------------------------------------


def test_trace_rows_are_the_rows_piqp_prints() -> None:
  """A row per pass that reaches the convergence test: iter + 1 of them for a solved problem, iter
  for one that runs out of iterations, whose last pass ends at the limit instead."""
  solved = solve(maros_meszaros("HS21"), "sparse")
  assert int(solved["status"]) == SOLVED and int(solved["trace_rows"]) == int(solved["iter"]) + 1
  np.testing.assert_array_equal(solved["trace"][:, 0], np.arange(int(solved["iter"]) + 1))
  capped = solve(maros_meszaros("HS21"), "sparse", Settings(max_iter=4))
  assert int(capped["status"]) == MAX_ITER_REACHED and int(capped["iter"]) == 4 and int(capped["trace_rows"]) == 4
  np.testing.assert_array_equal(capped["trace"], solved["trace"][:4])
