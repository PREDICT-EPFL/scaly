"""The stored Maros–Mészáros subset and its mapping to PIQP's form."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from tests.opt.ipm import reference as ref
from tests.opt.ipm.problems import INFINITE, gate_problems, infeasible_problems, maros_meszaros, maros_meszaros_names, mpc_qp, random_qp, raw

NAMES = maros_meszaros_names()


def test_the_subset_is_complete() -> None:
  assert len(NAMES) == 48
  assert {"HS21", "QAFIRO", "CVXQP1_S", "DUAL1", "QCAPRI"} <= set(NAMES)
  for name in NAMES:
    p = raw(name)
    assert p.P.shape[0] + p.A.shape[0] <= 1000


@pytest.mark.parametrize("name", NAMES)
def test_the_split_matches_a_vectorized_reading(name: str) -> None:
  """Each row of ``A`` lands in exactly one of the box, equality and inequality parts."""
  p, qp = raw(name), maros_meszaros(name)
  a = sparse.csr_array(p.A)
  lo = np.where(p.l <= -INFINITE, -np.inf, p.l)
  hi = np.where(p.u >= INFINITE, np.inf, p.u)
  first = a.data[np.minimum(a.indptr[:-1], a.data.size - 1)] if a.nnz else np.zeros(a.shape[0])
  box = (np.diff(a.indptr) == 1) & (first == 1.0) & (lo != hi)
  eq = ~box & (lo == hi)
  ineq = ~box & ~eq
  np.testing.assert_array_equal(qp.A.toarray(), a[np.flatnonzero(eq)].toarray())
  np.testing.assert_array_equal(qp.b, hi[eq])
  np.testing.assert_array_equal(qp.G.toarray(), a[np.flatnonzero(ineq)].toarray())
  np.testing.assert_array_equal(qp.h_l, lo[ineq])
  np.testing.assert_array_equal(qp.h_u, hi[ineq])
  cols = a.indices[a.indptr[:-1][box]]
  want_l, want_u = np.full(qp.n, -np.inf), np.full(qp.n, np.inf)
  np.maximum.at(want_l, cols, lo[box])
  np.minimum.at(want_u, cols, hi[box])
  np.testing.assert_array_equal(qp.x_l, want_l)
  np.testing.assert_array_equal(qp.x_u, want_u)
  for v in (qp.b, qp.h_l, qp.h_u, qp.x_l, qp.x_u):
    assert not np.any(np.isfinite(v) & (np.abs(v) >= INFINITE))
  x = np.random.default_rng(0).standard_normal(qp.n)
  np.testing.assert_allclose(qp.objective(x), 0.5 * x @ (p.P @ x) + p.q @ x + p.r, rtol=1e-14)
  assert abs(qp.P - qp.P.T).max() == 0.0 if qp.P.nnz else True


def test_a_rounded_sentinel_is_still_infinite() -> None:
  """QISRAEL stores some absent bounds as -9.999999999999998e19, just short of -1e20."""
  p = raw("QISRAEL")
  assert np.any((p.l > -1e20) & (p.l < -9e19))
  qp = maros_meszaros("QISRAEL")
  assert np.all(np.isinf(qp.h_l) | (np.abs(qp.h_l) < 1e7))


def test_known_shapes() -> None:
  """Spot checks: HS21's two bounds and one inequality; DPKLO1's bound rows are infinite on both sides."""
  hs21 = maros_meszaros("HS21")
  assert (hs21.n, hs21.A.shape[0], hs21.G.shape[0]) == (2, 0, 1)
  np.testing.assert_array_equal(hs21.x_l, [2.0, -50.0])
  np.testing.assert_array_equal(hs21.x_u, [50.0, 50.0])
  dpklo1 = maros_meszaros("DPKLO1")
  assert dpklo1.A.shape[0] == 77 and dpklo1.G.shape[0] == 0
  assert np.all(np.isinf(dpklo1.x_l)) and np.all(np.isinf(dpklo1.x_u))


def test_mpc_problems_have_the_simultaneous_form() -> None:
  qp = mpc_qp(4, 2, 10, seed=3)
  nz, nv = 11 * 4, 10 * 2
  assert qp.n == nz + nv and qp.A.shape == (4 + 10 * 4, nz + nv) and qp.G.shape[0] == 0
  # A trajectory of the dynamics from the fixed initial state satisfies the equalities exactly.
  # Row block k of the dynamics reads x_{k+1} - A x_k - B u_k = 0.
  a_blocks, b_mat = -qp.A[4:8, :4].toarray(), -qp.A[4:8, nz : nz + 2].toarray()
  x0 = qp.b[:4]
  u = np.random.default_rng(1).uniform(-0.5, 0.5, (10, 2))
  xs = [x0]
  for k in range(10):
    xs.append(a_blocks @ xs[-1] + b_mat @ u[k])
  w = np.r_[np.concatenate(xs), u.reshape(-1)]
  np.testing.assert_allclose(qp.A @ w, qp.b, atol=1e-12)
  assert np.all(qp.x_l < qp.x_u) and qp.P.nnz == qp.n
  # A zero initial state would make the optimum trivially zero; it is random and seeded.
  assert np.abs(x0).min() > 0 and not np.array_equal(x0, mpc_qp(4, 2, 10, seed=4).b[:4])


def test_infeasible_problems_are_labelled() -> None:
  problems = infeasible_problems()
  assert {why for _, why in problems.values()} == {"primal", "dual"}
  qp, _ = problems["unbounded_ray"]
  x = np.array([0.0, 1e6])  # feasible, and the objective falls without bound along x_2
  assert np.all(x >= qp.x_l) and qp.objective(x) < -1e5


@pytest.mark.parametrize("args", [(40, 30, 10, 1), (20, 15, 5, 35), (40, 30, 10, 37)])
def test_the_gate_lps_exercise_the_proximal_threshold(args: tuple[int, int, int, int]) -> None:
  """``gate_problems`` includes these LPs because their primal residual falls by between 5% and
  10% in some iteration, close to the 5% PIQP's proximal update asks for. A new random stream
  must keep that, or the gate loses the case."""
  n, m, p, seed = args
  qp = random_qp(n, m, p, seed=seed, lp=True)
  assert qp.name in gate_problems()
  res = ref.solve(qp).trace[:, ref.TRACE_COLUMNS.index("primal_res")]
  ratios = res[1:] / res[:-1]
  assert np.any((0.90 <= ratios) & (ratios < 0.95)), ratios
