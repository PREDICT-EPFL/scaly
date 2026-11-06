"""Check steady straight-line motion at 3 m/s, with zero steering and throttle balancing drag.

The throttle is (297.03 + 16.665 * 3 + 0.6784 * 9) / 4.95. Every cost residual
is zero along this feasible reference, so the known global minimum is zero.
"""

from pathlib import Path
import runpy

import numpy as np
import pytest


def test_acados_race_car_straight_line(tmp_path: Path) -> None:
  pytest.importorskip("acados_template")
  example = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples" / "acados_race_car.py"))
  reference = np.tile([0.0, 0.0, 0.0, 3.0], (7, 1))
  reference[:, 0] = 0.15 * np.arange(7)
  solver = example["solve_race_car"](tmp_path, reference[0], reference)
  assert solver.status == 0
  np.testing.assert_allclose(solver.get_cost(), 0.0, atol=1e-10)
  for stage in range(7):
    np.testing.assert_allclose(solver.get(stage, "x"), reference[stage], atol=1e-7, rtol=0)
    if stage < 6:
      np.testing.assert_allclose(solver.get(stage, "u"), [71.33951515151515, 0.0], atol=1e-6, rtol=0)
