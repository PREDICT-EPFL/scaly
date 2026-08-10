from __future__ import annotations

import numpy as np
import pytest

from alloy.toolchain import solver_diagnostic, solver_loadable
from benchmarks.problems.chain_of_masses import ChainParams, initial_state, n_dec, n_param, rk4_step_np
from benchmarks.problems.chain_of_masses.closed_loop import ClosedLoopConfig, extract_positions, plant_step, run_episode


def test_extract_positions_includes_fixed_anchor() -> None:
  state = initial_state(5)
  points = extract_positions(state, 5)

  assert points.shape == (5, 3)
  np.testing.assert_array_equal(points[0], np.zeros(3))
  np.testing.assert_allclose(points[:, 0], np.linspace(0.0, 7.0, 5))
  np.testing.assert_array_equal(points[:, 1:], np.zeros((5, 2)))
  with pytest.raises(ValueError, match="expected state shape"):
    extract_positions(state[:-1], 5)


def test_plant_step_is_parameterized_fixed_step_rk4() -> None:
  state = initial_state(3)
  control = np.array([0.1, -0.2, 0.3])
  params = ChainParams(mass=0.04, spring_d=0.8, rest_len=0.05, gravity=-8.0, dt=0.01)

  actual = plant_step(state, control, params)
  np.testing.assert_allclose(actual, rk4_step_np(state, control, params), rtol=0.0, atol=0.0)
  assert actual.shape == state.shape
  assert np.all(np.isfinite(actual))


@pytest.mark.skipif(not solver_loadable("ipopt"), reason=solver_diagnostic("ipopt"))
def test_tiny_alloy_ipopt_episode() -> None:
  config = ClosedLoopConfig(n_masses=3, horizon=1, steps=1, params=ChainParams(dt=0.05))
  episode = run_episode(config)

  assert episode.states.shape == (2, 9)
  assert episode.controls.shape == (1, 3)
  assert episode.points.shape == (2, 3, 3)
  assert episode.oracle_z.shape == (n_dec(3, 1),)
  assert episode.oracle_p.shape == (n_param(3),)
  assert len(episode.telemetry) == 1
  stats = episode.telemetry[0]
  assert stats.t_total >= 0.0
  assert stats.t_solver >= 0.0
  assert stats.t_fe >= 0.0
  assert stats.t_glue >= 0.0
  assert np.all(np.isfinite(episode.states))
  assert np.all(np.isfinite(episode.controls))
  assert np.all(np.isfinite(episode.points))
