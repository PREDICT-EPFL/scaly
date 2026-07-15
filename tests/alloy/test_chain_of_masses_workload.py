from __future__ import annotations

import numpy as np
import pytest

from alloy.toolchain import solver_diagnostic, solver_loadable
from benchmarks.problems.chain_of_masses import (
  NU,
  ChainParams,
  ca_chain_eq_jac,
  ca_chain_nlpsol,
  chain_eq_function,
  chain_eq_jac_dense_reference,
  chain_nlp,
  chain_step_fn,
  initial_state,
  n_dec,
  n_state,
  rk4_step_np,
  sample_inputs,
)

pytest.importorskip("casadi")

need_ipopt = pytest.mark.skipif(not solver_loadable("ipopt"), reason=solver_diagnostic("ipopt"))


def test_chain_dims_and_rk4_match_numpy() -> None:
  assert n_state(5) == 21
  params = ChainParams()
  rng = np.random.default_rng(3)
  for n_masses in (3, 5):
    x = initial_state(n_masses) + rng.normal(scale=0.02, size=n_state(n_masses))
    u = rng.normal(scale=0.1, size=NU)
    actual = chain_step_fn(n_masses)(x, u, *[np.array([value]) for value in params.array()])
    np.testing.assert_allclose(actual, rk4_step_np(x, u, params), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(("n_masses", "horizon"), [(3, 2), (5, 3)])
def test_chain_eq_jacobian_and_sparse_values_match(n_masses: int, horizon: int) -> None:
  fn = chain_eq_function(n_masses, horizon)
  dense = fn.factory(f"chain_dense_M{n_masses}_N{horizon}", ["z", "p"], ["jac:eq:z"])
  sparse = fn.factory(f"chain_sparse_M{n_masses}_N{horizon}", ["z", "p"], ["spjac:eq:z"])
  ca_dense = ca_chain_eq_jac(n_masses, horizon)
  zv, pv = sample_inputs(n_masses, horizon, seed=11)

  actual = np.asarray(dense(zv, pv))
  np.testing.assert_allclose(actual, np.asarray(ca_dense(zv, pv)), rtol=1e-9, atol=1e-9)
  np.testing.assert_allclose(chain_eq_jac_dense_reference(n_masses, horizon, zv, pv), actual, rtol=1e-10, atol=1e-10)

  sparsity = sparse.output_sparsities[0]
  assert sparsity is not None
  compact = np.asarray(sparse(zv, pv)).reshape(-1)
  flat = np.asarray(sparsity.rows) * actual.shape[1] + np.asarray(sparsity.cols)
  np.testing.assert_allclose(compact, actual.ravel()[flat], rtol=1e-10, atol=1e-10)
  assert sparsity.nnz < actual.size


@need_ipopt
def test_chain_nlp_matches_casadi_objective() -> None:
  n_masses, horizon = 3, 10
  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  zv, pv = sample_inputs(n_masses, horizon, seed=0)
  base = initial_state(n_masses)
  for i in range(horizon):
    zv[i * nz : i * nz + nx] = base
    zv[i * nz + nx : (i + 1) * nz] = 0.0
  zv[horizon * nz :] = base

  generated = chain_nlp(n_masses, horizon)
  alloy_out = generated(zv, np.zeros(nx * (horizon + 1)), np.zeros(0), np.zeros(n_dec(n_masses, horizon)), pv)
  assert generated.last_status is not None and generated.last_status.ok
  assert generated.last_stats is not None and generated.last_stats.iter > 0

  lb, ub = np.full(n_dec(n_masses, horizon), -np.inf), np.full(n_dec(n_masses, horizon), np.inf)
  for i in range(horizon):
    lb[i * nz + nx : (i + 1) * nz] = -1.0
    ub[i * nz + nx : (i + 1) * nz] = 1.0
  ca_solver = ca_chain_nlpsol(n_masses, horizon)
  ca_out = ca_solver(x0=zv, p=pv, lbg=0.0, ubg=0.0, lbx=lb, ubx=ub)
  assert ca_solver.stats()["success"]
  np.testing.assert_allclose(float(alloy_out["f"]), float(ca_out["f"]), rtol=1e-6)
