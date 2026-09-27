"""Kinematics of a 7-joint arm: forward kinematics as a scan, its Jacobian, and batched inverse kinematics.

The arm is described by modified Denavit-Hartenberg parameters (the values of a Panda-like arm);
link ``i`` contributes

    T_i(q_i) = RotX(alpha_i) TransX(a_i) RotZ(q_i) TransZ(d_i),

and the flange pose is the product ``T_1 ... T_7 T_flange``. That product is a ``sc.scan`` over the
links whose carry is the ``3 x 4`` pose, reading one row ``(q_i, a_i, d_i, alpha_i)`` of the
table per step. The position Jacobian ``dp/dq`` (``3 x 7``) is ``sc.jacobian`` through the scan:
forward mode carries the seven tangents alongside the pose in the same loop.

Inverse kinematics is damped least squares,

    q <- clip(q + J^T (J J^T + lambda^2 I)^{-1} (p* - p(q)), q_min, q_max),

in a ``sc.while_loop`` that stops when the position error is below 1e-10 m, with the ``3 x 3``
solve done by the generated ``cholesky`` / ``cho_solve``. One IK solve is a ``Function``;
``sc.vmap`` runs it for a whole batch of targets as one C loop, the use case of a planner that
checks the reachability of many grasps.

The generated C lands in ``examples/generated/robot_arm_ik/``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import scaly as sc
from scaly import linalg
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "robot_arm_ik"
NJ = 7
# Modified DH parameters (a_{i-1}, d_i, alpha_{i-1}) of a Panda-like arm, and the flange offset.
A = np.array([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088])
D = np.array([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0])
ALPHA = np.array([0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2])
D_FLANGE = 0.107
Q_MIN = np.array([-2.9, -1.76, -2.9, -3.07, -2.9, -0.02, -2.9])
Q_MAX = np.array([2.9, 1.76, 2.9, -0.07, 2.9, 3.75, 2.9])
Q_REST = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8])
DAMPING, TOL, MAX_IK = 1e-3, 1e-10, 200
BATCH = 500


@sc.function((3, 4), 4)
def link_step(pose: sc.Expr, link: sc.Expr) -> sc.Expr:
  q, a, d, alpha = link[0], link[1], link[2], link[3]
  cq, sq, ca, sa = q.cos(), q.sin(), alpha.cos(), alpha.sin()
  zero, one = sc.const(0.0), sc.const(1.0)
  local = sc.stack(
    [
      sc.stack([cq, -sq, zero, a]),
      sc.stack([sq * ca, cq * ca, -sa, -d * sa]),
      sc.stack([sq * sa, cq * sa, ca, d * ca]),
      sc.stack([zero, zero, zero, one]),
    ]
  )
  return pose @ local


def flange_pose(q: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  """The flange position and orientation."""
  table = sc.stack([q, sc.const(A), sc.const(D), sc.const(ALPHA)], axis=1).reshape((4 * NJ,))
  pose = sc.scan(link_step, sc.const(np.eye(4)[:3]), [(table, 0, 4)], length=NJ)[0]
  return pose[:, 3] + D_FLANGE * pose[:, 2], pose[:, :3]


@sc.function(NJ, output=sc.G("p", "R", "J"))
def kinematics(q: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  p, rot = flange_pose(q)
  return p, rot, sc.jacobian(p, q)


@sc.function
def dls_step(q: sc.Expr, target: sc.Expr) -> sc.Expr:
  p, _, jac = kinematics(q)
  gram = jac @ jac.T + sc.const(DAMPING**2 * np.eye(3))
  dq = jac.T @ linalg.cho_solve(linalg.cholesky(gram), target - p)
  return sc.minimum(sc.maximum(q + dq, sc.const(Q_MIN)), sc.const(Q_MAX))


@sc.function
def far(q: sc.Expr, target: sc.Expr) -> sc.Expr:
  p, _, _ = kinematics(q)
  return sc.greater(sc.norm_inf(target - p), TOL)


@sc.function(3, output=sc.G("q", "error", "iterations"))
def inverse_kinematics(target: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  q, n_iter = sc.while_loop(far, dls_step, sc.const(Q_REST), max_iter=MAX_IK, params=(target,))
  p, _, _ = kinematics(q)
  return q, sc.norm_2(target - p).reshape((1,)), n_iter.reshape((1,))


@sc.function(3 * BATCH, output=sc.G("q", "error", "iterations"))
def batch_ik(targets: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  q, error, iterations = (sc.vmap(inverse_kinematics, BATCH, [(targets, 0, 3)], output=k) for k in range(3))
  return q, error, iterations


def fk_numpy(q: np.ndarray) -> np.ndarray:
  pose = np.eye(4)
  for qi, a, d, al in zip(q, A, D, ALPHA, strict=True):
    ca, sa, cq, sq = np.cos(al), np.sin(al), np.cos(qi), np.sin(qi)
    pose = pose @ np.array([[cq, -sq, 0, a], [sq * ca, cq * ca, -sa, -d * sa], [sq * sa, cq * sa, ca, d * ca], [0, 0, 0, 1]])
  return pose[:3, 3] + D_FLANGE * pose[:3, 2]


def main(seed: int = 0) -> dict:
  rng = np.random.default_rng(seed)
  q = rng.uniform(Q_MIN, Q_MAX)
  p, rot, jac = kinematics(q)
  eps = 1e-7
  jac_fd = np.stack([(fk_numpy(q + eps * e) - fk_numpy(q - eps * e)) / (2 * eps) for e in np.eye(NJ)], axis=1)

  # Reachable targets: the flange positions of random configurations near the rest pose.
  q_samples = np.clip(Q_REST + rng.uniform(-0.8, 0.8, (BATCH, NJ)), Q_MIN, Q_MAX)
  targets = np.stack([fk_numpy(qs) for qs in q_samples])
  q_ik, error, iterations = batch_ik(targets.reshape(-1))  # the first call compiles
  start = time.perf_counter()
  batch_ik(targets.reshape(-1))
  elapsed = time.perf_counter() - start
  q_ik = q_ik.reshape(BATCH, NJ)
  reached = np.stack([fk_numpy(qi) for qi in q_ik])
  return {
    "fk_error": np.abs(p - fk_numpy(q)).max(),
    "rotation_orthogonality": np.abs(rot @ rot.T - np.eye(3)).max(),
    "jacobian_error": np.abs(jac - jac_fd).max(),
    "ik_error": error,
    "ik_error_numpy": np.linalg.norm(reached - targets, axis=1),
    "iterations": iterations,
    "within_limits": bool(np.all((q_ik >= Q_MIN - 1e-12) & (q_ik <= Q_MAX + 1e-12))),
    "seconds": elapsed,
  }


if __name__ == "__main__":
  out = main()
  print(
    f"forward kinematics vs NumPy {out['fk_error']:.1e}, R orthogonal to {out['rotation_orthogonality']:.1e}, Jacobian vs central differences {out['jacobian_error']:.1e}"
  )
  solved = out["ik_error_numpy"] < 1e-8
  print(
    f"batched IK: {solved.sum()} of {BATCH} targets reached to 1e-8 m (median {np.median(out['iterations']):.0f} iterations, max {int(out['iterations'].max())})"
  )
  print(
    f"the whole batch takes {1e3 * out['seconds']:.2f} ms ({1e6 * out['seconds'] / BATCH:.1f} us per IK solve); joint limits respected: {out['within_limits']}"
  )
  for fn in (kinematics, inverse_kinematics, batch_ik):
    write_module(fn, GENERATED)
  print(f"generated C in {GENERATED}")
