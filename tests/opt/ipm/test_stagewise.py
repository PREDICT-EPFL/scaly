"""The stagewise KKT backend: the condensed matrix factored block by block in the order ``stages``
finds, against the reference's solves and against the dense backend, which factors the same
matrix whole."""

from __future__ import annotations

from functools import cache
from unittest import mock

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.opt.ipm import INFO_FIELDS, Backend, Kernels, QPValues, Settings, Solver
from scaly.opt.ipm import method as method_module
from scaly.opt.ipm.stages import dense_blocks, stages
from scaly.testing.qp import QP, kkt_residuals, make_qp, maros_meszaros, mpc_qp
from tests.opt.ipm.problems import ipm_inputs

ORDER = ("P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u")
RESULT = ("x", "status", "iter", "info")


def staged_qp(seed: int) -> tuple[QP, tuple[int, int, int]]:
  """A multistage problem with everything a stage can hold: dense dynamics, a cost that couples a
  stage's states and inputs, boxes, rows of path inequalities over a stage's states and inputs,
  one of them two-sided; and its variables in a random order."""
  rng = np.random.default_rng(seed)
  nx, nu, horizon = int(rng.integers(2, 6)), int(rng.integers(1, 4)), int(rng.integers(3, 8))
  base = mpc_qp(nx, nu, horizon, seed=seed)
  n = base.n
  state = lambda k: k * nx + np.arange(nx)  # noqa: E731
  control = lambda k: (horizon + 1) * nx + k * nu + np.arange(nu)  # noqa: E731
  P = sparse.lil_array(base.P)
  rows = []
  for k in range(horizon):
    stage = np.r_[state(k), control(k)]
    half = 0.3 * rng.standard_normal((stage.size, stage.size))
    P[np.ix_(stage, stage)] = P[np.ix_(stage, stage)].toarray() + half @ half.T
    for _ in range(int(rng.integers(0, 3))):
      row = np.zeros(n)
      row[stage] = rng.standard_normal(stage.size)
      rows.append(row)
  G = np.array(rows).reshape(len(rows), n)
  reach = np.abs(G) @ np.r_[np.full((horizon + 1) * nx, 2.0), np.full(horizon * nu, 0.5)]
  h_u = 0.4 * reach
  h_l = np.where(np.arange(len(rows)) % 2 == 0, -0.6 * reach, -np.inf)
  order = rng.permutation(n)
  qp = make_qp(
    f"staged{seed}",
    sparse.csc_array(P)[order][:, order],
    rng.standard_normal(n)[order],
    A=sparse.csc_array(base.A)[:, order],
    b=base.b,
    G=sparse.csc_array(G)[:, order],
    h_l=h_l,
    h_u=h_u,
    x_l=base.x_l[order],
    x_u=base.x_u[order],
  )
  return qp, (nx, nu, horizon)


@cache
def _solver(qp_name: str, backend: Backend, problem) -> sc.Function:
  qp = problem()
  s, values = ipm_inputs(qp)
  syms = {k: sc.sym(k, np.shape(values[k])) for k in ORDER}
  out = Solver(s, backend, Settings(), name=f"sw_{qp_name}").solve(QPValues.preprocess(s, **syms))
  return sc.Function.from_exprs(f"sw_{qp_name}_{backend}", [syms[k] for k in ORDER], [out[k] for k in RESULT], list(ORDER), list(RESULT))


def solve(qp: QP, backend: Backend) -> dict[str, np.ndarray]:
  _, values = ipm_inputs(qp)
  return dict(zip(RESULT, _solver(qp.name, backend, lambda: qp)(tuple(values[k] for k in ORDER)), strict=True))


@pytest.mark.parametrize("seed", range(12))
def test_random_multistage_problems_take_the_dense_backends_path(seed: int) -> None:
  """The two backends factor one matrix, so they take the same iterations to the same solution:
  on multistage problems with path inequalities and a coupled stage cost, in a random variable
  order, which the partition undoes."""
  qp, (nx, nu, horizon) = staged_qp(seed)
  s, _ = ipm_inputs(qp)
  st = stages(s)
  assert (st.K, st.B) == (horizon + 1, nx + nu) and st.c <= nx + nu
  got, want = solve(qp, "stagewise"), solve(qp, "dense")
  assert (int(got["status"]), int(got["iter"])) == (int(want["status"]), int(want["iter"]))
  np.testing.assert_allclose(got["x"], want["x"], rtol=1e-6, atol=1e-7)
  if int(got["status"]) == 1:
    assert np.all(qp.G @ got["x"] <= qp.h_u + 1e-6) and np.all(qp.A @ got["x"] - qp.b <= 1e-6)


@pytest.mark.parametrize("name", ["HS21", "HS118", "CVXQP1_S", "DUAL1", "QAFIRO", "GENHS28", "mpc_4x2_N10", "mpc_12x4_N20"])
def test_problems_without_stages_are_solved_as_the_dense_backend_solves_them(name: str) -> None:
  """A problem with no stages is a few large blocks, or one: the backend still solves it, in the
  dense backend's iterations where the path does not hang on the last bits, and to its objective."""
  qp = {q.name: q for q in (mpc_qp(4, 2, 10), mpc_qp(12, 4, 20))}.get(name) or maros_meszaros(name)
  got, want = solve(qp, "stagewise"), solve(qp, "dense")
  assert int(got["status"]) == int(want["status"]) == 1
  assert abs(int(got["iter"]) - int(want["iter"])) <= 1
  np.testing.assert_allclose(qp.objective(got["x"]), qp.objective(want["x"]), rtol=1e-6, atol=1e-8)


def test_the_dense_arrays_are_taken_from_eight_columns_on() -> None:
  """The stage rows' products run as dense arrays where ``dense_rows``'s rule says they pay:
  eight columns and 4 096 products. A stage of six states and inputs keeps its index tables; one
  of ten takes arrays for ``A``, and its products with a vector go through them."""
  small, _ = ipm_inputs(mpc_qp(4, 2, 10))
  assert Kernels(small, "stagewise").stage_arrays == {}
  # Each floor alone: seven columns with products enough, and eight columns with too few of them.
  narrow, _ = ipm_inputs(mpc_qp(5, 2, 30))
  few, _ = ipm_inputs(mpc_qp(6, 2, 3))
  for s, columns, products in ((narrow, 7, 150 * 49), (few, 8, 18 * 64)):
    found = dense_blocks(stages(s), s.A_rows, s.A_cols, s.p)
    assert found is not None and (found.entries.shape[2], found.products) == (columns, products)
    assert Kernels(s, "stagewise").stage_arrays == {}
  large, _ = ipm_inputs(mpc_qp(8, 2, 10))
  kernels = Kernels(large, "stagewise", name="arrays")
  assert sorted(kernels.stage_arrays) == ["A"]
  assert kernels.stage_arrays["A"].found.entries.shape == (11, 8, 10)
  assert Kernels(large, "dense").stage_arrays == {} and Kernels(large, "sparse").stage_arrays == {}
  # The products with a vector, through the arrays, are the matrix's.
  rng = np.random.default_rng(2)
  a = sc.sym("A", (large.A_rows.size,))
  x, y = sc.sym("x", (large.n,)), sc.sym("y", (large.p,))
  mats = kernels.matrices_of(sc.sym("P", (large.P_rows.size,)), a, sc.sym("G", (0,)))
  fn = sc.Function.from_exprs("arrays_products", [a, x, y], [mats.A_times(x), mats.At_times(y)], ["A", "x", "y"], ["Ax", "Aty"])
  av, xv, yv = rng.standard_normal(large.A_rows.size), rng.standard_normal(large.n), rng.standard_normal(large.p)
  matrix = sparse.csr_array((av, (large.A_rows, large.A_cols)), shape=(large.p, large.n))
  ax, aty = fn((av, xv, yv))
  np.testing.assert_allclose(ax, matrix @ xv, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(aty, matrix.T @ yv, rtol=1e-12, atol=1e-12)


def test_a_variable_in_two_blocks_arrays_takes_both_shares() -> None:
  """Dynamics ``E x_{k+1} = A x_k + B u_k`` with a dense ``E``: a state is in the array of its own
  stage's rows and in the next stage's, so a product with a vector from the left adds two shares
  for it, and the block storage two products. The products are the matrix's, and the solve the
  dense backend's."""
  nx, nu, horizon = 8, 2, 6
  base = mpc_qp(nx, nu, horizon, seed=5)
  rng = np.random.default_rng(5)
  A = sparse.lil_array(base.A)
  for k in range(horizon):  # the rows of stage k + 1 read x_{k+1} through the identity: make it dense
    rows = nx + k * nx + np.arange(nx)
    A[np.ix_(rows, (k + 1) * nx + np.arange(nx))] = np.eye(nx) + 0.1 * rng.standard_normal((nx, nx))
  qp = make_qp("dense_e", base.P, base.c, A=sparse.csc_array(A), b=base.b, x_l=base.x_l, x_u=base.x_u)
  s, values = ipm_inputs(qp)
  kernels = Kernels(s, "stagewise", name="dense_e")
  arrays = kernels.stage_arrays["A"]
  assert arrays.found.entries.shape == (horizon + 1, nx, 2 * nx + nu)
  assert np.bincount(arrays.variable[arrays.variable >= 0]).max() == 2
  a, x, y = sc.sym("A", (s.A_rows.size,)), sc.sym("x", (s.n,)), sc.sym("y", (s.p,))
  mats = kernels.matrices_of(sc.sym("P", (s.P_rows.size,)), a, sc.sym("G", (0,)))
  fn = sc.Function.from_exprs("dense_e_products", [a, x, y], [mats.A_times(x), mats.At_times(y)], ["A", "x", "y"], ["Ax", "Aty"])
  xv, yv = rng.standard_normal(s.n), rng.standard_normal(s.p)
  matrix = sparse.csr_array((values["A"], (s.A_rows, s.A_cols)), shape=(s.p, s.n))
  ax, aty = fn((values["A"], xv, yv))
  np.testing.assert_allclose(ax, matrix @ xv, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(aty, matrix.T @ yv, rtol=1e-12, atol=1e-12)
  got, want = solve(qp, "stagewise"), solve(qp, "dense")
  assert (int(got["status"]), int(got["iter"])) == (int(want["status"]), int(want["iter"])) and int(got["status"]) == 1
  np.testing.assert_allclose(got["x"], want["x"], rtol=1e-6, atol=1e-7)


def test_weighted_rows_of_inequalities_go_through_dense_arrays_too() -> None:
  """Path inequalities dense over a stage: ``G^T W G`` is one weighted product a block, and the
  solve agrees with the dense backend's."""
  nx, nu, horizon = 8, 3, 6
  base = mpc_qp(nx, nu, horizon, seed=3)
  rng = np.random.default_rng(3)
  rows = []
  for k in range(horizon):
    for _ in range(nx + nu):
      row = np.zeros(base.n)
      row[np.r_[k * nx + np.arange(nx), (horizon + 1) * nx + k * nu + np.arange(nu)]] = rng.standard_normal(nx + nu)
      rows.append(row)
  G = np.array(rows)
  reach = np.abs(G) @ np.r_[np.full((horizon + 1) * nx, 2.0), np.full(horizon * nu, 0.5)]
  qp = make_qp("dense_paths", base.P, base.c, A=base.A, b=base.b, G=G, h_l=-0.5 * reach, h_u=0.5 * reach, x_l=base.x_l, x_u=base.x_u)
  s, _ = ipm_inputs(qp)
  assert sorted(Kernels(s, "stagewise").stage_arrays) == ["A", "G"]
  got, want = solve(qp, "stagewise"), solve(qp, "dense")
  assert (int(got["status"]), int(got["iter"])) == (int(want["status"]), int(want["iter"])) and int(got["status"]) == 1
  np.testing.assert_allclose(got["x"], want["x"], rtol=1e-6, atol=1e-7)
  primal, dual = kkt_residuals(qp, **_duals(qp))
  assert primal <= 1e-6 and dual <= 1e-5


def _duals(qp: QP) -> dict[str, np.ndarray]:
  s, values = ipm_inputs(qp)
  keys = ("x", "y", "z_l", "z_u", "z_bl", "z_bu")
  syms = {k: sc.sym(k, np.shape(values[k])) for k in ORDER}
  out = Solver(s, "stagewise", Settings(), name="sw_duals").solve(QPValues.preprocess(s, **syms))
  fn = sc.Function.from_exprs("sw_duals_fn", [syms[k] for k in ORDER], [out[k] for k in keys], list(ORDER), list(keys))
  return dict(zip(keys, fn(tuple(values[k] for k in ORDER)), strict=True))


def test_a_matrix_that_is_not_positive_definite_fails_as_the_dense_backends_does() -> None:
  """An indefinite cost: the Cholesky of a block fails, refinement turns on and the retry scales
  the regularization, as with the dense backend."""
  qp = make_qp("sw_nc", np.diag([-1.0, 1.0]), [0.1, -0.5], G=np.array([[1.0, 1.0]]), h_u=[5.0], x_l=[-10.0, -10.0], x_u=[10.0, 10.0])
  got, want = solve(qp, "stagewise"), solve(qp, "dense")
  ir = INFO_FIELDS.index("ir")
  assert got["info"][ir] == want["info"][ir] == 1.0
  assert (int(got["status"]), int(got["iter"])) == (int(want["status"]), int(want["iter"]))
  np.testing.assert_allclose(got["x"], want["x"], rtol=1e-6, atol=1e-7)


def test_the_method_takes_the_backend_by_name() -> None:
  """``IPM(backend="stagewise")`` builds the solver on the stagewise backend, and its solution is
  the dense backend's; ``backend`` and ``sparse`` are two ways to say one thing, so not both."""
  n, p_eq = 5, 1
  params = sc.G(sc.L("M", (n, n)), sc.L("c", n), sc.L("A", (p_eq, n)), sc.L("b", p_eq))

  def problem(name: str):
    @sc.opt.problem(vars=sc.L("x", n), params=params, name=name)
    def build(x, p):
      M, c, A, b = p
      return sc.opt.ProblemSpec(
        minimize=0.5 * sc.sumsqr(M.T @ x) + 0.5 * sc.sumsqr(x) + c @ x,
        eq=(A @ x - b,),
        lb=sc.const(-2.0 * np.ones(n)),
        ub=sc.const(2.0 * np.ones(n)),
      )

    return build

  rng = np.random.default_rng(4)
  values = (rng.standard_normal((n, n)), rng.standard_normal(n), rng.standard_normal((p_eq, n)), rng.standard_normal(p_eq))
  args = (np.zeros(n), np.zeros(n), np.zeros(p_eq), np.zeros(0))
  results = {}
  for backend in ("stagewise", "dense"):
    with mock.patch.object(method_module, "Solver", wraps=method_module.Solver) as made:
      fn = sc.opt.solver(problem(f"ipm_named_{backend}"), sc.opt.IPM(backend=backend), name=f"ipm_named_{backend}_solver")
    assert made.call_args.args[1] == backend
    results[backend] = fn.numerical_call(*args, values)
    assert results[backend][4].status == sc.Status.OK
  np.testing.assert_allclose(results["stagewise"][0], results["dense"][0], rtol=1e-7, atol=1e-8)
  assert results["stagewise"][4].iter == results["dense"][4].iter
  with pytest.raises(TypeError, match="not both"):
    sc.opt.IPM(sparse=True, backend="dense")
  with pytest.raises(ValueError, match="backend must be"):
    sc.opt.IPM(backend="banded")
