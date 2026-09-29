"""The SymForce case study's paths, small: factor Jacobians in a `vmap`, a Hessian assembled by summed
coordinates, and Levenberg-Marquardt with the generated sparse `L D L'` inside a `while_loop`.

`examples/case_studies/symforce` linearizes each factor type in a `vmap` body with `sc.jacobian` taken in a
zero tangent offset passed as an input, assembles `J'J` with `SparseMatrix.from_coo` (repeated coordinates
summed), and runs SymForce's LM in a `while_loop` whose body damps the Hessian (`with_values`,
`add_diagonal`) and factors it with `SparseLDL`. On a planar pose graph (3 poses, 4 landmarks) this
checks the assembled gradient and Hessian against finite differences of the stacked residual, the LM
loop's first damped step against NumPy's solve and its optimum against SciPy's `least_squares`; and the
SO(3) exponential and logarithm against SymForce's conventions.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

import scaly as sc
from scaly.geometry import quaternion as quat
from scaly import linalg

N, M = 3, 4
RNG = np.random.default_rng(5)
LANDMARKS = RNG.uniform(0, 5, (M, 2))
TRUTH = np.array([[0.1, 0.0, 0.0], [0.5, 1.0, 0.3], [0.9, 2.0, 0.4]])  # (theta, x, y)


def rot(th):
  return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


MEAS = np.array([[rot(p[0]).T @ (lm - p[1:]) for lm in LANDMARKS] for p in TRUTH]) + RNG.normal(0, 0.05, (N, M, 2))
ODOM = np.array([[TRUTH[k + 1, 0] - TRUTH[k, 0], *(rot(TRUTH[k, 0]).T @ (TRUTH[k + 1, 1:] - TRUTH[k, 1:]))] for k in range(N - 1)]) + RNG.normal(
  0, 0.02, (N - 1, 3)
)


def matching(pose, landmark, measured):
  c, s = pose[0].cos(), pose[0].sin()
  d = landmark - pose[1:]
  return sc.stack([c * d[0] + s * d[1], -s * d[0] + c * d[1]]) - measured


def odometry(a, b, measured):
  c, s = a[0].cos(), a[0].sin()
  d = b[1:] - a[1:]
  return sc.stack([b[0] - a[0], c * d[0] + s * d[1], -s * d[0] + c * d[1]]) - measured


LOW3, LOW6 = np.tril_indices(3), np.tril_indices(6)


def blocks(r, J, low):
  JtJ = J.T @ J
  n = JtJ.shape[0]
  return sc.concat([(0.5 * (r * r).sum()).reshape((1,)), J.T @ r, sc.gather(JtJ.reshape((n * n,)), low[0] * n + low[1])])


@sc.function(sc.L("pose", 3), sc.L("delta", 3), sc.L("landmark", 2), sc.L("measured", 2))
def match_factor(pose, delta, landmark, measured):
  r = matching(pose + delta, landmark, measured)
  return blocks(r, sc.jacobian(r, delta), LOW3)


@sc.function(sc.L("a", 3), sc.L("b", 3), sc.L("delta", 6), sc.L("measured", 3))
def odom_factor(a, b, delta, measured):
  r = odometry(a + delta[:3], b + delta[3:], measured)
  return blocks(r, sc.jacobian(r, delta), LOW6)


ROWS = np.concatenate([3 * i + LOW3[0] for i in range(N) for _ in range(M)] + [np.r_[3 * k + np.arange(6)][LOW6[0]] for k in range(N - 1)])
COLS = np.concatenate([3 * i + LOW3[1] for i in range(N) for _ in range(M)] + [np.r_[3 * k + np.arange(6)][LOW6[1]] for k in range(N - 1)])
GRAD = np.concatenate([3 * i + np.arange(3) for i in range(N) for _ in range(M)] + [3 * k + np.arange(6) for k in range(N - 1)])


def linearize(X):
  pairs = sc.gather(X, np.repeat(np.arange(N), M)[:, None] * 3 + np.arange(3)[None, :]).reshape((N * M * 3,))
  mf = sc.vmap(
    match_factor,
    N * M,
    [(pairs, 0, 3), (sc.const(np.zeros(3)), 0, 0), (sc.const(np.tile(LANDMARKS.reshape(-1), N)), 0, 2), (sc.const(MEAS.reshape(-1)), 0, 2)],
  )
  of = sc.vmap(odom_factor, N - 1, [(X, 0, 3), (X, 3, 3), (sc.const(np.zeros(6)), 0, 0), (sc.const(ODOM.reshape(-1)), 0, 3)])
  mf, of = mf.reshape((N * M, 10)), of.reshape((N - 1, 28))
  error = mf[:, 0].sum() + of[:, 0].sum()
  grad = sc.segment_sum(sc.concat([mf[:, 1:4].reshape((N * M * 3,)), of[:, 1:7].reshape(((N - 1) * 6,))]), GRAD, 3 * N)
  hess = linalg.SparseMatrix.from_coo(ROWS, COLS, sc.concat([mf[:, 4:].reshape((N * M * 6,)), of[:, 7:].reshape(((N - 1) * 21,))]), (3 * N, 3 * N))
  return error, grad, hess


@sc.function(sc.L("X", 3 * N), output=sc.G("error", "grad", "hess"))
def lin(X):
  e, g, H = linearize(X)
  return e, g, H.to_dense()


def np_residual(X):
  P = X.reshape(N, 3)
  out = [rot(P[i, 0]).T @ (LANDMARKS[j] - P[i, 1:]) - MEAS[i, j] for i in range(N) for j in range(M)]
  out += [np.r_[P[k + 1, 0] - P[k, 0], rot(P[k, 0]).T @ (P[k + 1, 1:] - P[k, 1:])] - ODOM[k] for k in range(N - 1)]
  return np.concatenate(out)


def test_assembled_gradient_and_hessian_match_finite_differences() -> None:
  X = RNG.normal(0, 0.3, 3 * N)
  e, g, H = (np.asarray(a) for a in lin(X))
  r = np_residual(X)
  J = np.stack([(np_residual(X + 1e-6 * u) - np_residual(X - 1e-6 * u)) / 2e-6 for u in np.eye(3 * N)], axis=1)
  assert abs(float(e) - 0.5 * r @ r) < 1e-12 * (1 + r @ r)
  np.testing.assert_allclose(g, J.T @ r, atol=1e-7)
  np.testing.assert_allclose(np.tril(H), np.tril(J.T @ J), atol=1e-7)


def test_lm_with_the_sparse_factor_in_the_loop_reaches_the_least_squares_optimum() -> None:
  _, _, template = linearize(sc.const(np.zeros(3 * N)))
  nnz = template.nnz
  sizes = [3 * N, 1, 3 * N, nnz, 1, 1]
  offs = np.cumsum([0, *sizes])

  @sc.function
  def step(carry):
    X, e, g, h, lam = (carry[offs[k] : offs[k + 1]] for k in range(5))
    upd = -linalg.SparseLDL(template.with_values(h).add_diagonal(lam[0])).solve(g)
    e_new, g_new, H_new = linearize(X + upd)
    accept = sc.less(e_new, e[0])
    small = sc.less((e[0] - e_new).abs(), 1e-14 * e[0])
    return sc.concat(
      [
        sc.where(accept, X + upd, X),
        sc.where(accept, e_new, e[0]).reshape((1,)),
        sc.where(accept, g_new, g),
        sc.where(accept, H_new.values, h),
        sc.where(accept, 0.5 * lam[0], 4.0 * lam[0]).reshape((1,)),
        sc.cast(small, "float64").reshape((1,)),
      ]
    )

  @sc.function
  def go(carry):
    return sc.less(carry[int(offs[5])], 0.5)

  def lm_function(max_iter: int) -> sc.Function:
    @sc.function(sc.L("X0", 3 * N), output=sc.G("X", "error"), name=f"lm_{max_iter}")
    def lm(X0):
      e, g, H = linearize(X0)
      carry, _ = sc.while_loop(go, step, sc.concat([X0, e.reshape((1,)), g, H.values, sc.const(np.array([1.0, 0.0]))]), max_iter=max_iter)
      return carry[: 3 * N], carry[3 * N]

    return lm

  # The first iteration is the damped step (lambda = 1) from NumPy's solve, on the linearization checked above.
  X0 = np.zeros(3 * N)
  _, g0, H0 = (np.asarray(a) for a in lin(X0))
  H0 = np.tril(H0) + np.tril(H0, -1).T
  X1 = X0 - np.linalg.solve(H0 + np.eye(3 * N), g0)
  assert 0.5 * np_residual(X1) @ np_residual(X1) < 0.5 * np_residual(X0) @ np_residual(X0)  # accepted
  np.testing.assert_allclose(np.asarray(lm_function(1)(X0)[0]), X1, atol=1e-12)

  X, err = lm_function(200)(X0)
  ref = optimize.least_squares(np_residual, X0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
  assert abs(float(err) - ref.cost) < 1e-10 * ref.cost
  np.testing.assert_allclose(X, ref.x, atol=1e-7)


@sc.function(sc.L("v", 3), sc.L("q", 4), output=sc.G("exp", "log_exp", "log_q"))
def so3(v, q):
  # The study's SO(3) is scaly.geometry's quaternion module.
  return quat.exp(v), quat.log(quat.exp(v)), quat.log(q)


def test_so3_exp_and_log_follow_symforce_conventions() -> None:
  v = np.array([0.3, -1.2, 2.0])
  q = np.array([0.1, 0.2, -0.3, -0.9])
  q = q / np.linalg.norm(q)
  e, le, lq = (np.asarray(a) for a in so3(v, q))
  th = np.linalg.norm(v)
  np.testing.assert_allclose(e, np.r_[np.sin(th / 2) / th * v, np.cos(th / 2)], atol=1e-15)
  np.testing.assert_allclose(le, v, atol=1e-12)
  # w < 0: the logarithm of q is that of -q, the rotation by less than pi
  qn = -q
  angle = 2 * np.arccos(qn[3])
  np.testing.assert_allclose(lq, angle * qn[:3] / np.linalg.norm(qn[:3]), atol=1e-12)
