"""Thin cross-check for the CT-DTCBF bumpercars filter problem in benchmarks/problems/bumpercars_filter."""

import numpy as np
import pytest

pytest.importorskip("casadi")

from benchmarks.problems.bumpercars_filter.common import (
  CarPhysics,
  ClosedLoopConfig,
  FilterConfig,
  load_ct_full_weights,
  pair_h,
  rk4_step_np,
  sample_initial_states,
)
from benchmarks.problems.bumpercars_filter.filters import CasadiDTCBFSafetyFilter, build_alloy_oracle


def test_ctdt_alloy_oracle_matches_casadi():
  loop_cfg = ClosedLoopConfig(ncars=2)
  filt_cfg = FilterConfig()
  weights = load_ct_full_weights()
  oracle = build_alloy_oracle(loop_cfg, filt_cfg)
  ca_filt = CasadiDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)

  rng = np.random.default_rng(3)
  bar_x = sample_initial_states(loop_cfg).reshape(-1)
  u_des = rng.uniform(-0.5, 0.5, 2 * loop_cfg.ncars)
  z = np.concatenate([np.clip(u_des + rng.normal(scale=0.1, size=u_des.size), -1.0, 1.0), [0.02]])
  physics, dt = loop_cfg.physics.array(), np.array([loop_cfg.dt])
  p = np.concatenate([bar_x, u_des, weights.packed, physics, dt])

  cost_al, g_al = oracle(z, bar_x, u_des, weights.packed, physics, dt)
  np.testing.assert_allclose(np.asarray(cost_al).reshape(-1), np.asarray(ca_filt.cost_fn(z, p)).reshape(-1), rtol=1e-9, atol=1e-9)
  np.testing.assert_allclose(np.asarray(g_al).reshape(-1), np.asarray(ca_filt.g_fn(z, p)).reshape(-1), rtol=1e-9, atol=1e-9)
  assert np.asarray(g_al).size == loop_cfg.ncars * (loop_cfg.ncars - 1) // 2 + 4 * loop_cfg.ncars


def test_ctdt_alloy_oracle_g_matches_numpy_at_asymmetric_physics():
  # independent NumPy reference (plant dynamics + hand-built h rows) with distinct per-entry physics pins the p-tail ordering
  physics = CarPhysics(lf=0.9, lr=1.3, max_delta=0.7, steering_time_constant=0.45, x_min=-3.0, x_max=11.0, y_min=1.0, y_max=17.0)
  loop_cfg = ClosedLoopConfig(ncars=2, physics=physics, dt=0.17)
  weights = load_ct_full_weights()
  oracle = build_alloy_oracle(loop_cfg, FilterConfig())

  rng = np.random.default_rng(9)
  states = sample_initial_states(loop_cfg)
  u = rng.uniform(-0.8, 0.8, (2, 2))
  slack = 0.03
  z = np.concatenate([u.reshape(-1), [slack]])
  _, g_al = oracle(z, states.reshape(-1), np.zeros(4), weights.packed, physics.array(), np.array([loop_cfg.dt]))

  # positions of the one-step prediction; rk4_step_np's theta wrap only touches entry 2, which no h row reads
  nxt = np.stack([rk4_step_np(states[i], u[i], loop_cfg.dt, weights, physics) for i in range(2)])
  rows = [pair_h(nxt[0], nxt[1], loop_cfg.safety_radius) - (1.0 - loop_cfg.pair_gamma) * pair_h(states[0], states[1], loop_cfg.safety_radius) + slack]
  m = loop_cfg.wall_margin
  for i in range(2):
    for h_next, h_cur in zip(
      [nxt[i, 0] - (physics.x_min + m), (physics.x_max - m) - nxt[i, 0], nxt[i, 1] - (physics.y_min + m), (physics.y_max - m) - nxt[i, 1]],
      [
        states[i, 0] - (physics.x_min + m),
        (physics.x_max - m) - states[i, 0],
        states[i, 1] - (physics.y_min + m),
        (physics.y_max - m) - states[i, 1],
      ],
      strict=True,
    ):
      rows.append(h_next - (1.0 - loop_cfg.wall_gamma) * h_cur + slack)
  np.testing.assert_allclose(np.asarray(g_al).reshape(-1), np.array(rows), rtol=1e-9, atol=1e-9)
