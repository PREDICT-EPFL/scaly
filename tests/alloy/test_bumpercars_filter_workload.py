"""Thin cross-check for the CT-DTCBF bumpercars filter problem in benchmarks/problems/bumpercars_filter."""

import numpy as np
import pytest

pytest.importorskip("casadi")

from benchmarks.problems.bumpercars_filter.common import ClosedLoopConfig, FilterConfig, load_ct_full_weights, sample_initial_states
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
  p = np.concatenate([bar_x, u_des, weights.packed])

  cost_al, g_al = oracle(z, bar_x, u_des, weights.packed)
  np.testing.assert_allclose(np.asarray(cost_al).reshape(-1), np.asarray(ca_filt.cost_fn(z, p)).reshape(-1), rtol=1e-9, atol=1e-9)
  np.testing.assert_allclose(np.asarray(g_al).reshape(-1), np.asarray(ca_filt.g_fn(z, p)).reshape(-1), rtol=1e-9, atol=1e-9)
  assert np.asarray(g_al).size == loop_cfg.ncars * (loop_cfg.ncars - 1) // 2 + 4 * loop_cfg.ncars
