from __future__ import annotations

import numpy as np
import pytest

from alloy.toolchain import solver_diagnostic, solver_loadable
from benchmarks.problems.tracking_nmpc import NX, NZ, TrackingParams
from benchmarks.problems.tracking_nmpc.closed_loop import (
  EpisodeConfig,
  ReferencePath,
  continuous_dynamics_np,
  reference_control,
  reference_state,
  rk4_step_np,
  run_episode,
)


def test_reference_is_closed_and_smooth() -> None:
  path = ReferencePath(radius=2.7, center_x=1.0, center_y=-0.4, speed=1.2)
  phases = np.linspace(0.0, 2.0 * np.pi, 65)
  ref = reference_state(phases, path)
  np.testing.assert_allclose(ref[0, :2], ref[-1, :2], atol=1e-14)
  assert ref[-1, 2] - ref[0, 2] == pytest.approx(2.0 * np.pi)
  np.testing.assert_allclose(np.linalg.norm(ref[:, :2] - [path.center_x, path.center_y], axis=1), path.radius)
  assert np.all(ref[:, 3] == path.speed)


def test_numpy_step_tracks_steady_circle_and_uses_parameter_dt() -> None:
  path = ReferencePath(radius=3.0, speed=1.4)
  params = TrackingParams(dt=0.02)
  x = reference_state(0.3, path)
  u = reference_control(path, params)
  stepped = rk4_step_np(x, u, params)
  half_step = rk4_step_np(x, u, TrackingParams(dt=0.5 * params.dt))
  np.testing.assert_allclose((stepped - x) / params.dt, continuous_dynamics_np(x, u, params), rtol=2e-2, atol=2e-2)
  np.testing.assert_allclose(half_step - x, 0.5 * (stepped - x), rtol=5e-3, atol=2e-5)
  assert stepped[2] > x[2]


@pytest.mark.skipif(not solver_loadable("ipopt"), reason=solver_diagnostic("ipopt"))
def test_very_short_solver_backed_episode() -> None:
  config = EpisodeConfig.smoke()
  result = run_episode(config)
  assert result.states.shape == (config.steps + 1, NX)
  assert result.references.shape == result.states.shape
  assert result.controls.shape == (config.steps, 2)
  assert result.oracle_z.shape == (NZ * (config.horizon + 1),)
  assert result.oracle_p.shape == (NX * (config.horizon + 1) + 7,)
  assert len(result.telemetry) == config.steps
  assert np.all(np.isfinite(result.states))
  assert np.all(np.isfinite(result.references))
  assert np.all(np.isfinite(result.controls))
  assert np.all(np.diff(result.progress) > -1e-3)
  for sample in result.telemetry:
    assert sample.stats.iter >= 0
    assert sample.solver_time >= 0.0
    assert sample.function_evaluation_time >= 0.0
    assert sample.glue_time >= 0.0
