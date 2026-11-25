"""The PIQP example in ``examples/piqp_sparse`` takes PIQP 0.7.0's decisions: its status, iterations, ``rho`` and ``delta`` at every iteration, and its solution."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE = EXAMPLES / "piqp_sparse"


def _load_example():
  spec = importlib.util.spec_from_file_location("piqp_sparse", EXAMPLE / "__init__.py", submodule_search_locations=[str(EXAMPLE)])
  assert spec is not None and spec.loader is not None
  module = sys.modules.setdefault("piqp_sparse", importlib.util.module_from_spec(spec))
  if not hasattr(module, "SparseSolver"):
    spec.loader.exec_module(module)
  return module


piqp_sparse = _load_example()
_mm_spec = importlib.util.spec_from_file_location("maros_meszaros", EXAMPLES / "case_studies" / "maros_meszaros" / "maros_meszaros.py")
assert _mm_spec is not None and _mm_spec.loader is not None
maros_meszaros = importlib.util.module_from_spec(_mm_spec)
_mm_spec.loader.exec_module(maros_meszaros)
reference = importlib.import_module("piqp_sparse.reference")

Settings, SparseSolver, Status = piqp_sparse.Settings, piqp_sparse.SparseSolver, piqp_sparse.Status


def _qp(seed, n=14, p=2, m=9):
  """A feasible QP with every kind of constraint: equalities, one- and two-sided inequalities, a row with both bounds infinite, and finite and infinite box bounds."""
  rng = np.random.default_rng(seed)
  M = sparse.random_array((n, n), density=0.25, rng=rng)
  A = sparse.random_array((p, n), density=0.5, rng=rng)
  G = sparse.random_array((m, n), density=0.35, rng=rng)
  x0 = rng.standard_normal(n)
  h_l, h_u = G @ x0 - rng.uniform(0.1, 1.0, m), G @ x0 + rng.uniform(0.1, 1.0, m)
  h_l[[1, 4]], h_u[[2, 4]] = -np.inf, np.inf
  x_l, x_u = x0 - rng.uniform(0.1, 2.0, n), x0 + rng.uniform(0.1, 2.0, n)
  x_l[::3], x_u[1::4] = -np.inf, np.inf
  P = (M @ M.T + 1e-2 * sparse.eye_array(n)).tocsc()
  return {"P": P, "c": rng.standard_normal(n), "A": A.tocsc(), "b": A @ x0, "G": G.tocsc(), "h_l": h_l, "h_u": h_u, "x_l": x_l, "x_u": x_u}


def _lp():
  rng = np.random.default_rng(3)
  n, m = 10, 16
  G = sparse.random_array((m, n), density=0.4, rng=rng).tocsc()
  x0 = rng.standard_normal(n)
  return {"P": sparse.csc_array((n, n)), "c": rng.standard_normal(n), "G": G, "h_l": G @ x0 - 1, "h_u": G @ x0 + 1, "x_l": x0 - 3, "x_u": x0 + 3}


def _equalities_only():
  rng = np.random.default_rng(4)
  n, p = 8, 3
  M = sparse.random_array((n, n), density=0.3, rng=rng)
  return {
    "P": (M @ M.T + sparse.eye_array(n)).tocsc(),
    "c": rng.standard_normal(n),
    "A": sparse.random_array((p, n), density=0.6, rng=rng).tocsc(),
    "b": rng.standard_normal(p),
  }


def _bounds_only():
  rng = np.random.default_rng(5)
  n = 8
  M = sparse.random_array((n, n), density=0.3, rng=rng)
  return {"P": (M @ M.T + sparse.eye_array(n)).tocsc(), "c": 3 * rng.standard_normal(n), "x_l": -np.ones(n), "x_u": np.ones(n)}


def _primal_infeasible():
  G = sparse.csc_array(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]))
  return {
    "P": sparse.csc_array((3, 3)),
    "c": np.ones(3),
    "G": G,
    "h_l": np.array([1.0, -np.inf, -1.0]),
    "h_u": np.array([np.inf, -1.0, 1.0]),
    "x_l": -5 * np.ones(3),
  }


def _dual_infeasible():
  G = sparse.csc_array(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
  return {"P": sparse.csc_array((3, 3)), "c": np.array([-1.0, 0.0, 0.0]), "G": G, "h_l": np.array([0.0, -1.0]), "h_u": np.array([np.inf, 1.0])}


def _solve(problem, settings):
  solver = SparseSolver(settings)
  solver.setup(**problem)
  return solver, solver.solve()


def _assert_same_result(result, ref, rtol=1e-7):
  """Every variable as PIQP returns it, infinite slacks included, to ``rtol`` of the largest entry of its kind."""
  for name, expected in ref.variables.items():
    actual = getattr(result, name)
    assert np.array_equal(np.isinf(actual), np.isinf(expected)), name
    finite = np.isfinite(expected)
    scale = max(1.0, np.abs(expected[finite]).max(initial=0.0))
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=0.0, atol=rtol * scale, err_msg=name)


def _log(text):
  """The generated solver's iteration table, printed with ``Settings(verbose=True)``."""
  rows = []
  for line in text.splitlines():
    if line.startswith("piqp_sparse "):
      rows.append({key: float(value) for key, value in (field.split("=") for field in line.split()[1:])})
  return rows


@pytest.mark.solver("piqp")
@pytest.mark.skipif(sys.platform == "win32", reason="the trace is read from the C runtime's standard output, which only POSIX flushes on demand")
def test_iterations_follow_piqp():
  problem, settings = _qp(0), Settings(verbose=True)
  with reference._captured_stdout() as captured:
    solver, status = _solve(problem, settings)
  log = _log(captured.read())
  ref = reference.solve(**problem, settings=settings)
  assert status == ref.status == Status.SOLVED
  assert solver.result.info.iter == ref.info["iter"]
  assert len(log) == len(ref.log) == ref.info["iter"] + 1
  for ours, theirs in zip(log, ref.log, strict=True):
    assert ours["iter"] == theirs["iter"]
    # PIQP prints rho and delta to four significant digits.
    for key in ("rho", "delta"):
      assert ours[key] == pytest.approx(theirs[key], rel=1e-3), (theirs["iter"], key)
  _assert_same_result(solver.result, ref)


@pytest.mark.solver("piqp")
def test_update_follows_piqp():
  problem = _qp(1)
  solver, status = _solve(problem, Settings())
  assert status == reference.solve(**problem).status == Status.SOLVED

  rng = np.random.default_rng(11)
  P = problem["P"].copy()
  P.data *= 1.5
  changed = problem | {"P": P, "c": rng.standard_normal(len(problem["c"]))}
  solver.update(P=changed["P"], c=changed["c"])
  ref = reference.solve(**changed)
  assert solver.solve() == ref.status == Status.SOLVED
  assert solver.result.info.iter == ref.info["iter"]
  _assert_same_result(solver.result, ref)

  # Only vectors: PIQP reuses the scaling, and so does the generated update.
  wider = changed | {"h_u": changed["h_u"] + 0.5}
  solver.update(h_u=wider["h_u"])
  ref = reference.solve(**wider)
  assert solver.solve() == ref.status
  assert solver.result.info.iter == ref.info["iter"]
  _assert_same_result(solver.result, ref)


def test_update_refuses_another_pattern():
  problem = _qp(2)
  solver = SparseSolver()
  solver.setup(**problem)
  denser = problem["G"].toarray()
  denser[denser == 0] = 1.0
  with pytest.raises(ValueError, match="G has a different sparsity pattern"):
    solver.update(G=sparse.csc_array(denser))


def _ruiz(problem, max_iter=10, epsilon=1e-3):
  """PIQP's Ruiz loop in NumPy, on dense matrices: the scalings ``(c, delta_x, delta_y, delta_z, delta_b)`` and the number of passes."""
  P, G = np.triu(problem["P"].toarray()), problem["G"].toarray()
  G[np.isinf(problem["h_l"]) & np.isinf(problem["h_u"])] = 0.0
  A, n, p = problem["A"].toarray(), P.shape[0], problem["A"].shape[0]
  x_b, delta, delta_b = np.ones(n), np.ones(n + p + G.shape[0]), np.ones(n)
  delta_iter, delta_iter_b = np.zeros_like(delta), np.zeros(n)
  limit = lambda d: np.where(d < 1e-4, 1.0, np.minimum(d, 1e4))
  passes = 0
  while passes < max_iter and max(np.abs(1 - delta_iter).max(), np.abs(1 - delta_iter_b).max()) > epsilon:
    full = np.abs(np.block([[P + np.triu(P, 1).T, A.T, G.T], [A, np.zeros((p, p + G.shape[0]))], [G, np.zeros((G.shape[0], p + G.shape[0]))]]))
    norms = full.max(axis=0)
    norms[:n] = np.maximum(norms[:n], x_b)
    delta_iter, delta_iter_b = 1 / np.sqrt(limit(norms)), 1 / np.sqrt(limit(x_b))
    d_x, d_y, d_z = delta_iter[:n], delta_iter[n : n + p], delta_iter[n + p :]
    P, A, G = d_x[:, None] * P * d_x, d_y[:, None] * A * d_x, d_z[:, None] * G * d_x
    x_b = x_b * delta_iter_b * d_x
    delta, delta_b, passes = delta * delta_iter, delta_b * delta_iter_b, passes + 1
  return (1.0, delta[:n], delta[n : n + p], delta[n + p :], delta_b), passes


@pytest.mark.parametrize("max_iter", [5, 30])
def test_equilibration_follows_piqp(max_iter):
  """The scaling after setup is PIQP's: at most ``max_iter`` passes, and none after the first whose scalings are all near one.

  The generated loop runs every pass and skips the ones after the stop; with 30 passes allowed,
  the problem stops after 11.
  """
  problem, settings = _qp(0), Settings(preconditioner_iter=max_iter)
  expected, passes = _ruiz(problem, max_iter)
  assert passes == min(max_iter, 11)
  solver = SparseSolver(settings)
  solver.setup(**problem)
  actual = [np.asarray(leaf) for leaf in piqp_sparse.solver._leaves(solver._scaled[1])]
  for name, ours, theirs in zip(("c", "delta_x", "delta_y", "delta_z", "delta_b"), actual, expected, strict=True):
    np.testing.assert_allclose(ours, theirs, rtol=1e-13, err_msg=name)


CASES = {
  "lp": (_lp, Settings()),
  "equalities_only": (_equalities_only, Settings()),
  "bounds_only": (_bounds_only, Settings()),
  "primal_infeasible": (_primal_infeasible, Settings()),
  "dual_infeasible": (_dual_infeasible, Settings()),
  "refinement": (lambda: _qp(3), Settings(iterative_refinement_always_enabled=True)),
  "cost_scaling": (lambda: _qp(4), Settings(preconditioner_scale_cost=True)),
  "no_equilibration": (lambda: _qp(5), Settings(preconditioner_iter=0)),
}


@pytest.mark.slow
@pytest.mark.solver("piqp")
@pytest.mark.parametrize("case", sorted(CASES))
def test_cases_follow_piqp(case):
  make, settings = CASES[case]
  problem = make()
  solver, status = _solve(problem, settings)
  ref = reference.solve(**problem, settings=settings)
  assert status == ref.status
  assert solver.result.info.iter == ref.info["iter"]
  _assert_same_result(solver.result, ref)


# Small enough to generate in seconds; the rest of the 48 run with the slow tests.
FAST_PROBLEMS = ("HS21", "HS35", "HS118", "QAFIRO", "DUALC1")


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("name", [pytest.param(name, marks=() if name in FAST_PROBLEMS else pytest.mark.slow) for name in maros_meszaros.problems()])
def test_maros_meszaros_follows_piqp(name):
  """Same status, iterations and objective, unless the problem is sensitive to rounding.

  A problem is sensitive when PIQP's own full and condensed KKT systems disagree on it, in status or
  iterations: late in a solve, with ``rho`` and ``delta`` at ``1e-10``, the condensed matrix is so
  ill-conditioned that the elimination order alone decides whether a step makes progress. On those
  problems the objectives must agree when both solvers solve. Several problems are linear programs
  whose optimum is not unique, so the solutions may differ where the objectives agree.
  """
  problem = maros_meszaros.load(name)
  solver, status = _solve(problem, Settings())
  ref = reference.solve(**problem)
  full = reference.solve(**problem, kkt_solver=reference.SPARSE_LDLT)
  if (full.status, full.info["iter"]) == (ref.status, ref.info["iter"]):
    assert status == ref.status
    assert solver.result.info.iter == ref.info["iter"]
  if status == ref.status == Status.SOLVED:
    assert solver.result.info.primal_obj == pytest.approx(ref.info["primal_obj"], rel=1e-6, abs=1e-6)
