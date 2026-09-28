"""The SymForce study in Scaly: robot 3D localization, and SymForce's Levenberg-Marquardt, as one C function.

SymForce's example (`symforce/examples/robot_3d_localization`, the problem of its paper's Table IV):
`n` poses `world_T_body` in SO(3) x R^3 are estimated from

- matching factors: each pose sees each of `m` known landmarks, residual
  `(world_T_body^-1 * world_t_landmark - body_t_landmark) / sigma`, `sigma = 0.1`;
- odometry factors between consecutive poses, residual
  `diag(sigmas)^-1 local_coordinates(world_T_a^-1 * world_T_b, a_T_b)` with
  `local_coordinates(a, b) = [Log(a.R^-1 b.R); b.t - a.t]`, sigmas 0.05 (rotation) and 0.2 (translation).

The conventions are SymForce's, so the iterates can be compared: quaternions stored `[x, y, z, w]`;
the tangent of a pose is `[rotation, translation]` and its retraction `R <- R Exp(dR)`, `t <- t + dt`;
`Exp(v)` with `theta = sqrt(|v|^2 + eps^2)`; `Log` with `w` clamped to `1 - eps`; `eps` SymForce's
default epsilon, ten times the float64 machine epsilon. Each factor's residual is a Scaly expression of
its poses retracted by tangent offsets, and its Jacobian is `sc.jacobian` in those offsets at zero, per
factor inside a `vmap`, so the generated code is one body per factor type whatever the problem size.

`lm_function` is SymForce's `LevenbergMarquardtSolver` as a `while_loop` (`levenberg_marquardt_solver.tcc`
with the example's parameters): the Gauss-Newton Hessian `J'J` and gradient `J'r` assembled from the
factors' blocks, unit damping `H + lambda I`, the generated sparse `L D L'` on the Hessian's fixed
pattern, the step retracted, the new point relinearized, the step accepted on any decrease (`lambda`
halved) or refused (`lambda` quadrupled, clamped at 1e6), and the loop stopped when the relative
reduction lies in `(-1e-7, 1e-6)` or after 50 iterations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import scaly as sc
from scaly import linalg

EPS = 10 * np.finfo(np.float64).eps
MATCH_SIGMA = 0.1
ODOM_SIGMAS = np.array([0.05, 0.05, 0.05, 0.2, 0.2, 0.2])


@dataclass
class Problem:
  """Poses are optimized; the landmarks and measurements are data."""

  landmarks: np.ndarray  # (m, 3)
  matches: np.ndarray  # (n, m, 3): body_t_landmark measurements
  odometry: np.ndarray  # (n - 1, 7): a_T_b as [qx, qy, qz, qw, tx, ty, tz]

  @property
  def n_poses(self) -> int:
    return self.matches.shape[0]

  @property
  def n_landmarks(self) -> int:
    return self.landmarks.shape[0]


def parse_measurements(path: Path) -> Problem:
  """SymForce's checked-in `gen/measurements.cc` (12 decimals), which its C++ benchmark reads."""
  text = Path(path).read_text()

  def block(name: str) -> str:
    start = text.index(name)
    return text[start : text.index("};", start)]

  num = r"-?\d+\.\d+(?:e-?\d+)?"
  matches = [[float(x) for x in re.findall(num, v)] for v in re.findall(r"Vector3d\(([^)]*)\)", block("body_t_landmark_measurements"))]
  poses = [[float(x) for x in re.findall(num, v)] for v in re.findall(r"<<([^;]*?)\)\s*\.finished", block("odometry_relative_pose_measurements"))]
  landmarks = [[float(x) for x in re.findall(num, v)] for v in re.findall(r"Vector3d\(([^)]*)\)", block("landmark_positions"))]
  m = len(landmarks)
  return Problem(np.array(landmarks), np.array(matches).reshape(-1, m, 3), np.array(poses))


def save(p: Problem, path: Path) -> None:
  Path(path).write_text(json.dumps({"landmarks": p.landmarks.tolist(), "matches": p.matches.tolist(), "odometry": p.odometry.tolist()}))


def load(path: Path) -> Problem:
  d = json.loads(Path(path).read_text())
  return Problem(np.array(d["landmarks"]), np.array(d["matches"]), np.array(d["odometry"]))


def synthetic(n_poses: int, n_landmarks: int = 20, seed: int = 0) -> Problem:
  """A larger problem of the same kind for the scaling sweep: a smooth path, landmarks in a box, every
  pose seeing every landmark, measurement noise as SymForce's generator draws it."""
  rng = np.random.default_rng(seed)
  t = np.linspace(0, 1, n_poses)
  rotvec = np.stack([0.3 * np.sin(2 * np.pi * t), 0.2 * np.cos(2 * np.pi * t), 2 * np.pi * t], axis=1)
  trans = np.stack([5 + 3 * np.cos(2 * np.pi * t), 5 + 3 * np.sin(2 * np.pi * t), 1 + t], axis=1)
  quats = np.array([np_exp(v) for v in rotvec])
  landmarks = rng.uniform(0, 10, (n_landmarks, 3))
  matches = np.array([[np_rotate(np_conj(q), lm - tr) for lm in landmarks] for q, tr in zip(quats, trans, strict=True)])
  matches += rng.normal(0, MATCH_SIGMA, matches.shape)
  odom = []
  for k in range(n_poses - 1):
    qa, qb = quats[k], quats[k + 1]
    rel_q = np_mul(np_conj(qa), qb)
    rel_t = np_rotate(np_conj(qa), trans[k + 1] - trans[k])
    noise = rng.normal(0, ODOM_SIGMAS)
    odom.append(np.r_[np_mul(rel_q, np_exp(noise[:3])), rel_t + noise[3:]])
  return Problem(landmarks, matches, np.array(odom))


# --- quaternion helpers in NumPy, for the data and the checks ---


def np_mul(a, b):
  ax, ay, az, aw = a
  bx, by, bz, bw = b
  return np.array(
    [
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
    ]
  )


def np_conj(q):
  return np.array([-q[0], -q[1], -q[2], q[3]])


def np_rotate(q, v):
  return np_mul(np_mul(q, np.r_[v, 0.0]), np_conj(q))[:3]


def np_exp(v):
  theta = np.sqrt(v @ v + EPS**2)
  return np.r_[np.sin(theta / 2) / theta * v, np.cos(theta / 2)]


# --- the same in Scaly expressions ---


def q_mul(a, b):
  ax, ay, az, aw = a[0], a[1], a[2], a[3]
  bx, by, bz, bw = b[0], b[1], b[2], b[3]
  return sc.stack(
    [
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
    ]
  )


def q_conj(q):
  return sc.stack([-q[0], -q[1], -q[2], q[3]])


def q_rotate(q, v):
  """`R(q) v` through the rotation matrix, as SymForce's generated code computes it."""
  x, y, z, w = q[0], q[1], q[2], q[3]
  r = [
    [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
    [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
    [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
  ]
  return sc.stack([r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2] for i in range(3)])


def q_exp(v):
  theta = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + EPS * EPS).sqrt()
  s = (0.5 * theta).sin() / theta
  return sc.stack([s * v[0], s * v[1], s * v[2], (0.5 * theta).cos()])


def q_log(q):
  """SymForce's `Rot3.to_tangent`: `w` clamped to `1 - eps`, the sign of `w` taken with `sign(0) = 1`."""
  w = q[3]
  sign = sc.where(sc.less(w, 0.0), -1.0, 1.0)
  ws = sc.minimum(w.abs(), 1.0 - EPS)
  scale = sign * 2.0 * ws.acos() / (1.0 - ws * ws).sqrt()
  return sc.stack([scale * q[0], scale * q[1], scale * q[2]])


def retract(pose, delta):
  """SymForce's `Pose3.retract`: rotation on the right by `Exp`, translation added."""
  return sc.concat([q_mul(pose[:4], q_exp(delta[:3])), pose[4:] + delta[3:]])


def matching_residual(pose, landmark, measured):
  return (q_rotate(q_conj(pose[:4]), landmark - pose[4:]) - measured) / MATCH_SIGMA


def odometry_residual(pose_a, pose_b, measured):
  qa_inv = q_conj(pose_a[:4])
  pred_q = q_mul(qa_inv, pose_b[:4])
  pred_t = q_rotate(qa_inv, pose_b[4:] - pose_a[4:])
  # local_coordinates(pred, measured) = [Log(pred.R^-1 measured.R); measured.t - pred.t]
  rot = q_log(q_mul(q_conj(pred_q), measured[:4]))
  return sc.concat([rot, measured[4:] - pred_t]) / sc.const(ODOM_SIGMAS)


LOW6 = np.tril_indices(6)
LOW12 = np.tril_indices(12)


def gauss_newton(r, J, low):
  """A factor's contribution: its error `|r|^2 / 2`, gradient `J'r` and the lower triangle of `J'J`."""
  JtJ = J.T @ J
  n = JtJ.shape[0]
  lower = sc.gather(JtJ.reshape((n * n,)), low[0] * n + low[1])
  return sc.concat([(0.5 * (r * r).sum()).reshape((1,)), J.T @ r, lower])


# One matching factor at pose `pose` (the tangent offset `delta` is evaluated at zero).
@sc.function(sc.L("pose", 7), sc.L("delta", 6), sc.L("landmark", 3), sc.L("measured", 3), name="symforce_matching_factor")
def matching_factor(pose, delta, landmark, measured):
  r = matching_residual(retract(pose, delta), landmark, measured)
  return gauss_newton(r, sc.jacobian(r, delta), LOW6)


# One odometry factor between two poses.
@sc.function(sc.L("pose_a", 7), sc.L("pose_b", 7), sc.L("delta", 12), sc.L("measured", 7), name="symforce_odometry_factor")
def odometry_factor(pose_a, pose_b, delta, measured):
  r = odometry_residual(retract(pose_a, delta[:6]), retract(pose_b, delta[6:]), measured)
  return gauss_newton(r, sc.jacobian(r, delta), LOW12)


def pose_matching_function(m: int) -> sc.Function:
  """All `m` matching factors of one pose in one body: the pose's rotation and its derivative are formed
  once and shared by the pose's landmarks, as SymForce's flattened code shares them by elimination."""

  @sc.function(sc.L("pose", 7), sc.L("delta", 6), sc.L("landmarks", 3 * m), sc.L("measured", 3 * m), name=f"symforce_pose_matching_{m}")
  def pose_matching(pose, delta, landmarks, measured):
    moved = retract(pose, delta)
    q_inv, t = q_conj(moved[:4]), moved[4:]
    rows = []
    for j in range(m):
      rows.append((q_rotate(q_inv, landmarks[3 * j : 3 * j + 3] - t) - measured[3 * j : 3 * j + 3]) / MATCH_SIGMA)
    r = sc.concat(rows)
    return gauss_newton(r, sc.jacobian(r, delta), LOW6)

  return pose_matching


@sc.function(sc.L("pose", 7), sc.L("delta", 6), name="symforce_retract")
def retract_step(pose, delta):
  return retract(pose, delta)


class Layout:
  """Where every factor's contributions land in the 6n gradient and the Hessian's lower triangle."""

  def __init__(self, p: Problem, grouped: bool = True):
    n, m = p.n_poses, p.n_landmarks
    self.n, self.m, self.grouped = n, m, grouped
    rows, cols = [], []
    for i in range(n):  # matching factors, pose-major: one block per pose, or one per (pose, landmark)
      for _ in range(1 if grouped else m):
        rows.append(6 * i + LOW6[0])
        cols.append(6 * i + LOW6[1])
    for i in range(n - 1):  # odometry factors: local index k < 6 is pose i, k >= 6 pose i + 1
      g = np.r_[6 * i + np.arange(6), 6 * (i + 1) + np.arange(6)]
      rows.append(g[LOW12[0]])
      cols.append(g[LOW12[1]])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    # from_coo sums repeated coordinates, which assembles J'J from the factors' blocks.
    self.rows, self.cols = rows, cols
    per_pose = 1 if grouped else m
    self.grad_index = np.r_[
      np.concatenate([6 * i + np.arange(6) for i in range(n) for _ in range(per_pose)]),
      np.concatenate([6 * i + np.arange(12) for i in range(n - 1)]) if n > 1 else np.zeros(0, dtype=int),
    ]
    self.landmarks = np.tile(p.landmarks.reshape(-1), n)
    self.matches = p.matches.reshape(-1)
    self.odometry = p.odometry.reshape(-1)


def linearize(p: Problem, lay: Layout, X):
  """`(error, gradient J'r, Hessian J'J as a SparseMatrix)` of all factors at the poses `X` (n x 7, flat)."""
  n, m = lay.n, lay.m
  if lay.grouped:
    mf = sc.vmap(
      pose_matching_function(m),
      n,
      [(X, 0, 7), (sc.const(np.zeros(6)), 0, 0), (sc.const(lay.landmarks[: 3 * m]), 0, 0), (sc.const(lay.matches), 0, 3 * m)],
    )
    k = n
  else:
    X_pairs = sc.gather(X, np.repeat(np.arange(n), m)[:, None] * 7 + np.arange(7)[None, :])  # pose i for each (i, j)
    mf = sc.vmap(
      matching_factor,
      n * m,
      [(X_pairs.reshape((n * m * 7,)), 0, 7), (sc.const(np.zeros(6)), 0, 0), (sc.const(lay.landmarks), 0, 3), (sc.const(lay.matches), 0, 3)],
    )
    k = n * m
  mf = mf.reshape((k, 1 + 6 + 21))
  parts_err, parts_grad, parts_hess = [mf[:, 0]], [mf[:, 1:7].reshape((k * 6,))], [mf[:, 7:].reshape((k * 21,))]
  if n > 1:
    of = sc.vmap(odometry_factor, n - 1, [(X, 0, 7), (X, 7, 7), (sc.const(np.zeros(12)), 0, 0), (sc.const(lay.odometry), 0, 7)])
    of = of.reshape((n - 1, 1 + 12 + 78))
    parts_err.append(of[:, 0])
    parts_grad.append(of[:, 1:13].reshape(((n - 1) * 12,)))
    parts_hess.append(of[:, 13:].reshape(((n - 1) * 78,)))
  error = sc.concat(parts_err).sum()
  grad = sc.segment_sum(sc.concat(parts_grad), lay.grad_index, 6 * n)
  hess = linalg.SparseMatrix.from_coo(lay.rows, lay.cols, sc.concat(parts_hess), (6 * n, 6 * n))
  return error, grad, hess


INITIAL_LAMBDA, LAMBDA_DOWN, LAMBDA_UP, LAMBDA_MAX = 1e4, 0.5, 4.0, 1e6
EARLY_EXIT, MAX_ITERATIONS = 1e-6, 50


def identity_poses(n: int) -> np.ndarray:
  """The example's initial values: every pose at the identity."""
  return np.tile(np.r_[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], n)


def lm_function(p: Problem, max_iterations: int = MAX_ITERATIONS, name: str = "robot_3d_localization", grouped: bool = True) -> sc.Function:
  """`lm(X0) -> (X, error, iterations, trace_error, trace_lambda, trace_accepted)`, SymForce's solver.

  The traces hold, per iteration, the error after the step (accepted or not), the lambda it used and
  whether it was accepted, as SymForce's `optimization_iteration_t` records them. `grouped` evaluates a
  pose's matching factors in one body (the default) instead of one body per factor."""
  lay = Layout(p, grouped)
  n = p.n_poses
  _, _, template = linearize(p, lay, sc.const(identity_poses(n)))
  nnz = template.nnz
  K = max_iterations
  sizes = [7 * n, 1, 6 * n, nnz, 1, 1, K, K, K]
  offs = np.cumsum([0, *sizes])

  def part(c, k):
    return c[offs[k] : offs[k + 1]]

  @sc.function(name=f"{name}_lm_iteration")
  def iteration(carry, index):
    X, e, grad, hvals, lam = part(carry, 0), part(carry, 1)[0], part(carry, 2), part(carry, 3), part(carry, 4)[0]
    damped = template.with_values(hvals).add_diagonal(lam)
    update = -linalg.SparseLDL(damped, name=f"{name}_hessian").solve(grad)
    X_new = sc.vmap(retract_step, n, [(X, 0, 7), (update, 0, 6)])
    e_new, grad_new, H_new = linearize(p, lay, X_new)
    rr = (e - e_new) / (e + EPS)
    success = sc.logical_and(sc.greater(rr, -EARLY_EXIT / 10), sc.less(rr, EARLY_EXIT))
    accept = sc.greater(rr, 0.0)
    failed = sc.logical_and(sc.logical_not(accept), sc.greater_equal(lam, LAMBDA_MAX))
    status = sc.where(failed, 2.0, sc.where(success, 1.0, 0.0))
    lam_next = sc.minimum(sc.maximum(sc.where(accept, lam * LAMBDA_DOWN, lam * LAMBDA_UP), 0.0), LAMBDA_MAX)
    at = sc.cast(sc.equal(sc.const(np.arange(K, dtype=np.float64)), index.cast("float64")), "float64")
    return sc.concat(
      [
        sc.where(accept, X_new, X),
        sc.where(accept, e_new, e).reshape((1,)),
        sc.where(accept, grad_new, grad),
        sc.where(accept, H_new.values, hvals),
        lam_next.reshape((1,)),
        status.reshape((1,)),
        part(carry, 6) + at * e_new,
        part(carry, 7) + at * lam,
        part(carry, 8) + at * sc.cast(accept, "float64"),
      ]
    )

  @sc.function(name=f"{name}_lm_running")
  def running(carry):
    return sc.less(part(carry, 5)[0], 0.5)

  @sc.function(sc.L("X0", 7 * n), output=sc.G("X", "error", "iterations", "trace_error", "trace_lambda", "trace_accepted"), name=f"{name}_lm")
  def lm(X0):
    e0, g0, H0 = linearize(p, lay, X0)
    start = sc.concat([X0, e0.reshape((1,)), g0, H0.values, sc.const(np.array([INITIAL_LAMBDA, 0.0])), sc.const(np.zeros(3 * K))])
    carry, count = sc.while_loop(running, iteration, start, max_iter=K, index=True)
    return part(carry, 0), part(carry, 1)[0], count, part(carry, 6), part(carry, 7), part(carry, 8)

  return lm


def linearize_function(p: Problem, name: str = "robot_3d_localization", grouped: bool = True) -> sc.Function:
  """`linearize(X) -> (error, gradient, Hessian values)`: what SymForce's `Relinearize` computes."""
  lay = Layout(p, grouped)
  n = p.n_poses

  @sc.function(sc.L("X", 7 * n), output=sc.G("error", "gradient", "hessian"), name=f"{name}_linearize")
  def lin(X):
    e, g, H = linearize(p, lay, X)
    return e, g, H.values

  return lin
