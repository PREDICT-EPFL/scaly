from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import alloy as al
from alloy.expr import topo
from alloy.utils import load_torch_state_dict

casadi = pytest.importorskip("casadi")
if TYPE_CHECKING:
  import casadi

LF = 0.54
LR = 0.33
DT = 0.1
STEERING_RANGE = 4.024
TL_STEERING = 0.1550
ARENA_X_MIN = 0.0
ARENA_X_MAX = 10.0
ARENA_Y_MIN = 0.0
ARENA_Y_MAX = 10.0
SAFETY_RADIUS = 1.7
WALL_SAFETY_RADIUS = 1.0
VEL_DAMPING = 10 * DT
GAMMA = 0.5
VF_NORM = 1.9
AF_NORM = 0.6
AR_NORM = 0.1
DELTA_NORM = 2.012

NSTATE = 7
NCTRL = 2
W0_SHAPE = (256, 6)
B0_SHAPE = (256,)
W1_SHAPE = (128, 256)
B1_SHAPE = (128,)
W2_SHAPE = (3, 128)
B2_SHAPE = (3,)
W0_SIZE = W0_SHAPE[0] * W0_SHAPE[1]
B0_SIZE = B0_SHAPE[0]
W1_SIZE = W1_SHAPE[0] * W1_SHAPE[1]
B1_SIZE = B1_SHAPE[0]
W2_SIZE = W2_SHAPE[0] * W2_SHAPE[1]
B2_SIZE = B2_SHAPE[0]
OFFSETS = tuple(int(x) for x in np.cumsum([0, W0_SIZE, B0_SIZE, W1_SIZE, B1_SIZE, W2_SIZE, B2_SIZE]))
N_WEIGHTS = OFFSETS[-1]
MODEL_PATH = Path(__file__).resolve().parents[2] / "examples" / "unbumpercars" / "data" / "model_kinematic_mlp.pth"


def official_weights() -> np.ndarray:
  if not MODEL_PATH.exists():
    pytest.skip(f"unbumpercars MLP checkpoint not found at {MODEL_PATH}")
  weights = load_torch_state_dict(MODEL_PATH)
  parts = [
    weights["model.0.weight"].astype(np.float64).reshape(-1),
    weights["model.0.bias"].astype(np.float64).reshape(-1),
    weights["model.2.weight"].astype(np.float64).reshape(-1),
    weights["model.2.bias"].astype(np.float64).reshape(-1),
    weights["model.4.weight"].astype(np.float64).reshape(-1),
    weights["model.4.bias"].astype(np.float64).reshape(-1),
  ]
  return np.concatenate(parts)


def n_dec(ncars: int) -> int:
  return NCTRL * ncars + 1


def n_desired(ncars: int) -> int:
  return NCTRL * ncars


def n_param(ncars: int) -> int:
  return N_WEIGHTS + NSTATE * ncars + n_desired(ncars)


def n_ineq(ncars: int) -> int:
  return ncars * (ncars - 1) // 2 + 4 * ncars


def _softplus(x):
  return (1 + x.exp()).log()


def _unpack_weights(pw):
  return (
    pw[OFFSETS[0] : OFFSETS[1]].reshape(W0_SHAPE).block(),
    pw[OFFSETS[1] : OFFSETS[2]].block(),
    pw[OFFSETS[2] : OFFSETS[3]].reshape(W1_SHAPE).block(),
    pw[OFFSETS[3] : OFFSETS[4]].block(),
    pw[OFFSETS[4] : OFFSETS[5]].reshape(W2_SHAPE).block(),
    pw[OFFSETS[5] : OFFSETS[6]].block(),
  )


def _mlp_forward(mlp_in, pw):
  w0, b0, w1, b1, w2, b2 = _unpack_weights(pw)
  h = _softplus((w0 @ mlp_in.block() + b0).block()).block()
  h = _softplus((w1 @ h + b1).block()).block()
  return (w2 @ h + b2).block()


def _pose_continuous(pose, vf, beta_f, beta_r):
  theta = pose[2]
  omega = vf * (beta_f - beta_r).sin() / ((LF + LR) * beta_r.cos())
  vx_b = vf * beta_f.cos()
  vy_b = vf * beta_f.sin() - LF * omega
  return al.stack(
    [
      vx_b * theta.cos() - vy_b * theta.sin(),
      vx_b * theta.sin() + vy_b * theta.cos(),
      omega,
    ]
  )


def _rk4_pose(pose, vf, beta_f, beta_r):
  k1 = _pose_continuous(pose, vf, beta_f, beta_r)
  k2 = _pose_continuous(pose + DT / 2 * k1, vf, beta_f, beta_r)
  k3 = _pose_continuous(pose + DT / 2 * k2, vf, beta_f, beta_r)
  k4 = _pose_continuous(pose + DT * k3, vf, beta_f, beta_r)
  return pose + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@al.function("unbumpercar_official_mlp_dynamics", {"state": NSTATE, "u": NCTRL, "pw": N_WEIGHTS})
def dynamics_fn(state, u, pw):
  vf, alpha_f, alpha_r, delta = state[3], state[4], state[5], state[6]
  throttle, steering = u[0], u[1]
  beta_f = alpha_f + delta
  beta_r = alpha_r
  pose_next = _rk4_pose(al.stack([state[0], state[1], state[2]]), vf, beta_f, beta_r)
  mlp_in = al.stack([vf / VF_NORM, alpha_f / AF_NORM, alpha_r / AR_NORM, delta / DELTA_NORM, steering, throttle]).block()
  vel_next = _mlp_forward(mlp_in, pw)
  delta_ref = steering * (STEERING_RANGE / 2)
  delta_next = delta + (delta_ref - delta) / TL_STEERING * DT
  return {"next": al.stack([pose_next[0], pose_next[1], pose_next[2], _softplus(vel_next[0]), vel_next[1], vel_next[2], delta_next])}


def _global_velocity(state):
  th, v, bf, br = state[2], state[3], state[4], state[5]
  omega = v * (bf - br).sin() / ((LF + LR) * br.cos())
  vx_b = v * bf.cos()
  vy_b = v * bf.sin() - LF * omega
  return al.stack([vx_b * th.cos() - vy_b * th.sin(), vx_b * th.sin() + vy_b * th.cos()])


def _c3bf(x_i, x_j):
  p_rel = al.stack([x_j[0] - x_i[0], x_j[1] - x_i[1]])
  p_rel_sq = (p_rel * p_rel).sum()
  p_rel_norm = (p_rel_sq + 1e-8).sqrt()
  vi = _global_velocity(x_i)
  vj = _global_velocity(x_j)
  v_rel = vj - vi
  v_rel_norm = ((v_rel * v_rel).sum() + 1e-8).sqrt()
  eps = 1e-8
  cos_phi = (eps + _softplus(p_rel_sq - SAFETY_RADIUS**2 - eps)).sqrt() / p_rel_norm
  return ((p_rel * v_rel).sum() + v_rel_norm * p_rel_norm * cos_phi).scalar()


def _wall_residuals(xi, xi1):
  vi_k = _global_velocity(xi)
  vi_k1 = _global_velocity(xi1)
  h_k_xmin = (xi[0] - (ARENA_X_MIN + WALL_SAFETY_RADIUS)) + VEL_DAMPING * vi_k[0]
  h_k1_xmin = (xi1[0] - (ARENA_X_MIN + WALL_SAFETY_RADIUS)) + VEL_DAMPING * vi_k1[0]
  h_k_xmax = ((ARENA_X_MAX - WALL_SAFETY_RADIUS) - xi[0]) - VEL_DAMPING * vi_k[0]
  h_k1_xmax = ((ARENA_X_MAX - WALL_SAFETY_RADIUS) - xi1[0]) - VEL_DAMPING * vi_k1[0]
  h_k_ymin = (xi[1] - (ARENA_Y_MIN + WALL_SAFETY_RADIUS)) + VEL_DAMPING * vi_k[1]
  h_k1_ymin = (xi1[1] - (ARENA_Y_MIN + WALL_SAFETY_RADIUS)) + VEL_DAMPING * vi_k1[1]
  h_k_ymax = ((ARENA_Y_MAX - WALL_SAFETY_RADIUS) - xi[1]) - VEL_DAMPING * vi_k[1]
  h_k1_ymax = ((ARENA_Y_MAX - WALL_SAFETY_RADIUS) - xi1[1]) - VEL_DAMPING * vi_k1[1]
  return [
    (h_k1_xmin - (1 - GAMMA) * h_k_xmin).scalar(),
    (h_k1_xmax - (1 - GAMMA) * h_k_xmax).scalar(),
    (h_k1_ymin - (1 - GAMMA) * h_k_ymin).scalar(),
    (h_k1_ymax - (1 - GAMMA) * h_k_ymax).scalar(),
  ]


@al.function("unbumpercars_pair_c3bf", {"prev_si": NSTATE, "prev_sj": NSTATE, "si": NSTATE, "sj": NSTATE, "slack": 1})
def pair_c3bf_fn(prev_si, prev_sj, si, sj, slack):
  return {"h": al.stack([(_c3bf(si, sj) - (1 - GAMMA) * _c3bf(prev_si, prev_sj) + slack[0]).scalar()])}


@al.function("unbumpercars_wall_residuals", {"sk": NSTATE, "sk1": NSTATE, "slack": 1})
def wall_residuals_fn(sk, sk1, slack):
  return {"h": al.stack([(wall + slack[0]).scalar() for wall in _wall_residuals(sk, sk1)])}


def unbumpercars_ineq_function(ncars: int) -> al.Function:
  u = al.sym("u", n_dec(ncars))
  p = al.sym("p", n_param(ncars), diff=False)
  pw = p[:N_WEIGHTS]
  state_offset = N_WEIGHTS
  states_k_packed = p[state_offset : state_offset + NSTATE * ncars]
  u_packed = u[: NCTRL * ncars]
  slack_packed = u[NCTRL * ncars : NCTRL * ncars + 1]

  states_k1_packed = al.map_(
    dynamics_fn,
    ncars,
    [(states_k_packed, 0, NSTATE), (u_packed, 0, NCTRL), (pw, 0, 0)],
  )

  pieces: list[al.Expr] = []
  pairs = [(i, j) for i in range(ncars) for j in range(i + 1, ncars)]
  if pairs:
    idx_i = np.concatenate([np.arange(NSTATE, dtype=np.int64) + i * NSTATE for (i, _) in pairs])
    idx_j = np.concatenate([np.arange(NSTATE, dtype=np.int64) + j * NSTATE for (_, j) in pairs])
    gather_prev_si = al.gather(states_k_packed, idx_i)
    gather_prev_sj = al.gather(states_k_packed, idx_j)
    gather_si = al.gather(states_k1_packed, idx_i)
    gather_sj = al.gather(states_k1_packed, idx_j)
    pieces.append(
      al.map_(
        pair_c3bf_fn,
        len(pairs),
        [
          (gather_prev_si, 0, NSTATE),
          (gather_prev_sj, 0, NSTATE),
          (gather_si, 0, NSTATE),
          (gather_sj, 0, NSTATE),
          (slack_packed, 0, 0),
        ],
      )
    )

  pieces.append(
    al.map_(
      wall_residuals_fn,
      ncars,
      [(states_k_packed, 0, NSTATE), (states_k1_packed, 0, NSTATE), (slack_packed, 0, 0)],
    )
  )

  ineq = pieces[0] if len(pieces) == 1 else al.concat(pieces)
  return al.Function(f"unbumpercars_reduced_ineq_N{ncars}", [u, p], [ineq.scalar()], ["u", "p"], ["ineq"])


def _ca_softplus(x):
  return casadi.log(1 + casadi.exp(x))


def _ca_pose_continuous(pose, vf, beta_f, beta_r):
  theta = pose[2]
  omega = vf * casadi.sin(beta_f - beta_r) / ((LF + LR) * casadi.cos(beta_r))
  vx_b = vf * casadi.cos(beta_f)
  vy_b = vf * casadi.sin(beta_f) - LF * omega
  return casadi.vertcat(vx_b * casadi.cos(theta) - vy_b * casadi.sin(theta), vx_b * casadi.sin(theta) + vy_b * casadi.cos(theta), omega)


def _ca_rk4_pose(pose, vf, beta_f, beta_r):
  k1 = _ca_pose_continuous(pose, vf, beta_f, beta_r)
  k2 = _ca_pose_continuous(pose + DT / 2 * k1, vf, beta_f, beta_r)
  k3 = _ca_pose_continuous(pose + DT / 2 * k2, vf, beta_f, beta_r)
  k4 = _ca_pose_continuous(pose + DT * k3, vf, beta_f, beta_r)
  return pose + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _ca_unpack_weights(pw):
  return (
    casadi.reshape(pw[OFFSETS[0] : OFFSETS[1]], W0_SHAPE[1], W0_SHAPE[0]).T,
    pw[OFFSETS[1] : OFFSETS[2]],
    casadi.reshape(pw[OFFSETS[2] : OFFSETS[3]], W1_SHAPE[1], W1_SHAPE[0]).T,
    pw[OFFSETS[3] : OFFSETS[4]],
    casadi.reshape(pw[OFFSETS[4] : OFFSETS[5]], W2_SHAPE[1], W2_SHAPE[0]).T,
    pw[OFFSETS[5] : OFFSETS[6]],
  )


def _ca_dynamics(state, u, pw):
  vf, alpha_f, alpha_r, delta = state[3], state[4], state[5], state[6]
  throttle, steering = u[0], u[1]
  beta_f = alpha_f + delta
  beta_r = alpha_r
  pose_next = _ca_rk4_pose(casadi.vertcat(state[0], state[1], state[2]), vf, beta_f, beta_r)
  mlp_in = casadi.vertcat(vf / VF_NORM, alpha_f / AF_NORM, alpha_r / AR_NORM, delta / DELTA_NORM, steering, throttle)
  w0, b0, w1, b1, w2, b2 = _ca_unpack_weights(pw)
  hidden = _ca_softplus(w0 @ mlp_in + b0)
  hidden = _ca_softplus(w1 @ hidden + b1)
  vel_next = w2 @ hidden + b2
  delta_ref = steering * (STEERING_RANGE / 2)
  delta_next = delta + (delta_ref - delta) / TL_STEERING * DT
  return casadi.vertcat(pose_next[0], pose_next[1], pose_next[2], _ca_softplus(vel_next[0]), vel_next[1], vel_next[2], delta_next)


def _ca_global_velocity(state):
  th, v, bf, br = state[2], state[3], state[4], state[5]
  omega = v * casadi.sin(bf - br) / ((LF + LR) * casadi.cos(br))
  vx_b = v * casadi.cos(bf)
  vy_b = v * casadi.sin(bf) - LF * omega
  return casadi.vertcat(vx_b * casadi.cos(th) - vy_b * casadi.sin(th), vx_b * casadi.sin(th) + vy_b * casadi.cos(th))


def _ca_c3bf(x_i, x_j):
  p_rel = casadi.vertcat(x_j[0] - x_i[0], x_j[1] - x_i[1])
  p_rel_sq = casadi.dot(p_rel, p_rel)
  p_rel_norm = casadi.sqrt(p_rel_sq + 1e-8)
  vi = _ca_global_velocity(x_i)
  vj = _ca_global_velocity(x_j)
  v_rel = vj - vi
  v_rel_norm = casadi.sqrt(casadi.dot(v_rel, v_rel) + 1e-8)
  eps = 1e-8
  cos_phi = casadi.sqrt(eps + _ca_softplus(p_rel_sq - SAFETY_RADIUS**2 - eps)) / p_rel_norm
  return casadi.dot(p_rel, v_rel) + v_rel_norm * p_rel_norm * cos_phi


def _ca_unbumpercars_ineq_jac(ncars: int, sym_t=casadi.MX, name: str | None = None):
  u = sym_t.sym("u", n_dec(ncars))
  p = sym_t.sym("p", n_param(ncars))
  slack = u[NCTRL * ncars]
  pw = p[:N_WEIGHTS]
  state_offset = N_WEIGHTS
  states_k = [p[state_offset + NSTATE * i : state_offset + NSTATE * (i + 1)] for i in range(ncars)]
  states_k1 = [_ca_dynamics(states_k[i], u[NCTRL * i : NCTRL * i + NCTRL], pw) for i in range(ncars)]
  constraints = []
  for i in range(ncars):
    for j in range(i + 1, ncars):
      constraints.append(_ca_c3bf(states_k1[i], states_k1[j]) - (1 - GAMMA) * _ca_c3bf(states_k[i], states_k[j]) + slack)
  for i in range(ncars):
    vi_k = _ca_global_velocity(states_k[i])
    vi_k1 = _ca_global_velocity(states_k1[i])
    constraints += [
      (states_k1[i][0] - (ARENA_X_MIN + WALL_SAFETY_RADIUS))
      + VEL_DAMPING * vi_k1[0]
      - (1 - GAMMA) * ((states_k[i][0] - (ARENA_X_MIN + WALL_SAFETY_RADIUS)) + VEL_DAMPING * vi_k[0])
      + slack,
      ((ARENA_X_MAX - WALL_SAFETY_RADIUS) - states_k1[i][0])
      - VEL_DAMPING * vi_k1[0]
      - (1 - GAMMA) * (((ARENA_X_MAX - WALL_SAFETY_RADIUS) - states_k[i][0]) - VEL_DAMPING * vi_k[0])
      + slack,
      (states_k1[i][1] - (ARENA_Y_MIN + WALL_SAFETY_RADIUS))
      + VEL_DAMPING * vi_k1[1]
      - (1 - GAMMA) * ((states_k[i][1] - (ARENA_Y_MIN + WALL_SAFETY_RADIUS)) + VEL_DAMPING * vi_k[1])
      + slack,
      ((ARENA_Y_MAX - WALL_SAFETY_RADIUS) - states_k1[i][1])
      - VEL_DAMPING * vi_k1[1]
      - (1 - GAMMA) * (((ARENA_Y_MAX - WALL_SAFETY_RADIUS) - states_k[i][1]) - VEL_DAMPING * vi_k[1])
      + slack,
    ]
  ineq = casadi.vertcat(*constraints)
  return casadi.Function(name or f"ca_unbumpercars_reduced_ineq_jac_N{ncars}", [u, p], [casadi.jacobian(ineq, u)])


def _sample_inputs(ncars: int) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(4)
  uv = rng.normal(scale=0.2, size=n_dec(ncars))
  uv[-1] = 0.05
  pv = rng.normal(scale=0.3, size=n_param(ncars))
  pv[:N_WEIGHTS] = official_weights()
  state_offset = N_WEIGHTS
  for i in range(ncars):
    pv[state_offset + NSTATE * i + 3] = 1.0 + 0.1 * i
    pv[state_offset + NSTATE * i + 5] = 0.1
  return uv, pv


def test_unbumpercars_reduced_ineq_jacobian_matches_casadi_mx() -> None:
  ncars = 2
  fn = unbumpercars_ineq_function(ncars)
  jf = fn.factory("unbumpercars_reduced_ineq_jac_N2", ["u", "p"], ["jac:ineq:u"])
  ca_jf = _ca_unbumpercars_ineq_jac(ncars, casadi.MX)
  uv, pv = _sample_inputs(ncars)

  np.testing.assert_allclose(jf(uv, pv), np.asarray(ca_jf(uv, pv)), rtol=1e-9, atol=1e-9)


def test_unbumpercars_reduced_colored_sparse_jacobian_matches_dense_and_casadi_structure() -> None:
  ncars = 2
  fn = unbumpercars_ineq_function(ncars)
  spjf = fn.factory("unbumpercars_reduced_ineq_spjac_N2", ["u", "p"], ["spjac:ineq:u"])
  jf = fn.factory("unbumpercars_reduced_ineq_jac_N2", ["u", "p"], ["jac:ineq:u"])
  ca_jf = _ca_unbumpercars_ineq_jac(ncars, casadi.MX)
  uv, pv = _sample_inputs(ncars)

  sparsity = spjf.output_sparsities[0]
  assert sparsity is not None
  dense = jf(uv, pv)
  compact = spjf(uv, pv)
  assert isinstance(dense, np.ndarray)
  assert isinstance(compact, np.ndarray)
  flat = np.asarray(sparsity.rows) * dense.shape[1] + np.asarray(sparsity.cols)
  np.testing.assert_allclose(compact, np.ravel(dense)[flat], rtol=1e-9, atol=1e-9)

  ca_mask = np.zeros(dense.shape, dtype=bool)
  ca_rows, ca_cols = ca_jf.sparsity_out(0).get_triplet()
  ca_mask[np.asarray(ca_rows, dtype=np.int64), np.asarray(ca_cols, dtype=np.int64)] = True
  np.testing.assert_array_equal(sparsity.to_mask(), ca_mask)


def test_unbumpercars_reduced_fixture_marks_dense_mlp_and_sparse_constraints() -> None:
  assert any(node.op == al.Ops.MATMUL and node.lowering == "block" for node in topo(dynamics_fn.outputs))
  fn = unbumpercars_ineq_function(2)
  assert fn.outputs[0].lowering == "scalar"
