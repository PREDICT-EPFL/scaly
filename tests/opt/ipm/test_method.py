"""``IPM``, the method: its solver in one Function, and as PIQP's setup and solve (``IPM.split``)."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc

N, P_EQ, M_INEQ = 5, 1, 3


def _problem(name: str):
  """A QP whose matrices are parameters: ``P = M M^T + I``, ``A x = b``, ``-h <= G x <= h`` and
  ``-2 <= x <= 2`` (the box bounds have scalings of their own)."""
  params = sc.G(sc.L("M", (N, N)), sc.L("c", N), sc.L("A", (P_EQ, N)), sc.L("b", P_EQ), sc.L("G", (M_INEQ, N)), sc.L("h", M_INEQ))

  @sc.opt.problem(vars=sc.L("x", N), params=params, name=name)
  def problem(x, p):
    M, c, A, b, G, h = p
    return sc.opt.ProblemSpec(
      minimize=0.5 * sc.sumsqr(M.T @ x) + 0.5 * sc.sumsqr(x) + c @ x,
      eq=(A @ x - b,),
      ineq=(sc.opt.bounded(G @ x, -h, h),),
      lb=sc.const(-2.0 * np.ones(N)),
      ub=sc.const(2.0 * np.ones(N)),
    )

  return problem


def _values(seed: int, *, matrices_seed: int = 0) -> tuple:
  rng, mats = np.random.default_rng(seed), np.random.default_rng(matrices_seed)
  M, A, G = mats.standard_normal((N, N)), mats.standard_normal((P_EQ, N)), mats.standard_normal((M_INEQ, N))
  return M, rng.standard_normal(N), A, rng.standard_normal(P_EQ), G, 0.2 + rng.random(M_INEQ)


ARGS = (np.zeros(N), np.zeros(N), np.zeros(P_EQ), np.zeros(M_INEQ))


def _same(a, b) -> None:
  """Every result the same: the solution, the multipliers and the solve's info."""
  for x, y in zip(a[:4], b[:4], strict=True):
    np.testing.assert_array_equal(x, y)
  assert (a[4].status, a[4].iter, a[4].objective, a[4].primal_residual) == (b[4].status, b[4].iter, b[4].objective, b[4].primal_residual)


@pytest.mark.parametrize("sparse", [True, False])
def test_setup_then_solve_is_the_solver_bit_for_bit(sparse: bool) -> None:
  method = sc.opt.IPM(sparse=sparse)
  problem = _problem(f"ipm_split_{sparse}")
  full = sc.opt.solver(problem, method, name=f"ipm_split_full_{sparse}")
  setup, solve = method.split(problem, name=f"ipm_split_{sparse}")
  assert setup.name == f"ipm_split_{sparse}_setup"
  for seed in range(3):
    values = _values(seed, matrices_seed=seed)
    got = solve.numerical_call(*ARGS, values, setup.numerical_call(values))
    _same(got, full.numerical_call(*ARGS, values))
    assert got[4].status == sc.Status.OK


@pytest.mark.parametrize("sparse", [True, False])
def test_the_setup_reads_only_the_matrices_and_serves_every_solve(sparse: bool) -> None:
  """With the matrices fixed, as in MPC, one setup serves every solve, as PIQP's ``update`` keeps its
  scaling: the equilibration does not read ``c``, ``b`` or the bounds."""
  method = sc.opt.IPM(sparse=sparse)
  problem = _problem(f"ipm_split_reuse_{sparse}")
  full = sc.opt.solver(problem, method, name=f"ipm_split_reuse_full_{sparse}")
  setup, solve = method.split(problem, name=f"ipm_split_reuse_{sparse}")
  scaling = setup.numerical_call(_values(0))
  for seed in (1, 2, 3):
    values = _values(seed)  # other vectors, the same matrices
    np.testing.assert_array_equal(setup.numerical_call(values), scaling)
    _same(solve.numerical_call(*ARGS, values, scaling), full.numerical_call(*ARGS, values))
  other = _values(0, matrices_seed=7)
  assert not np.array_equal(setup.numerical_call(other), scaling)
  # The solve runs on the scaling it is given: another problem's still solves this one, on another path.
  values = _values(1)
  foreign = solve.numerical_call(*ARGS, values, setup.numerical_call(other))
  reference = full.numerical_call(*ARGS, values)
  assert foreign[4].status == sc.Status.OK
  np.testing.assert_allclose(foreign[0], reference[0], atol=1e-6)
  assert not np.array_equal(foreign[0], reference[0])


@pytest.mark.parametrize("sparse", [True, False])
def test_a_scaled_cost_makes_the_setup_read_c(sparse: bool) -> None:
  """The sparse backend's setup also keeps the cost maxima where its stopping test reads them, as
  PIQP's sparse interface does (``alias_cost``)."""
  method = sc.opt.IPM(sparse=sparse, options={"preconditioner_scale_cost": True})
  problem = _problem(f"ipm_split_cost_{sparse}")
  setup, solve = method.split(problem, name=f"ipm_split_cost_{sparse}")
  full = sc.opt.solver(problem, method, name=f"ipm_split_cost_full_{sparse}")
  a = _values(0)
  b = (a[0], 100.0 * a[1], *a[2:])  # a cost large enough to set the cost scale over P's
  assert not np.array_equal(setup.numerical_call(a), setup.numerical_call(b))
  _same(solve.numerical_call(*ARGS, b, setup.numerical_call(b)), full.numerical_call(*ARGS, b))


def test_the_solver_takes_the_five_arguments_every_solver_takes() -> None:
  full = sc.opt.solver(_problem("ipm_signature"), sc.opt.IPM(), name="ipm_signature")
  with pytest.raises(TypeError, match="takes 5 arguments"):
    full.numerical_call(*ARGS)
