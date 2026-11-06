"""Compare the race-car example's Scaly and CasADi acados SQP iterates."""

from pathlib import Path
import runpy

import numpy as np
import pytest


def test_acados_race_car_iterates(tmp_path: Path) -> None:
  pytest.importorskip("acados_template")
  example = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples" / "acados_race_car.py"))
  actual, reference = [example["solve_race_car"](tmp_path, horizon=6, scaly=use_scaly) for use_scaly in (True, False)]
  for solver in (actual, reference):
    assert solver.status == 0
    assert 1 < solver.get_stats("sqp_iter") < 50
    assert np.max(solver.get_residuals()) < 1e-8
  assert actual.get_stats("sqp_iter") == reference.get_stats("sqp_iter")
  for iteration in range(int(actual.get_stats("sqp_iter")) + 1):
    for field in ("x", "u", "pi", "lam"):
      for stage, (left, right) in enumerate(
        zip(getattr(actual.get_iterate(iteration), field), getattr(reference.get_iterate(iteration), field), strict=True)
      ):
        np.testing.assert_allclose(left, right, atol=1e-7, rtol=1e-7, err_msg=f"{field} at SQP iterate {iteration}, stage {stage}")
  np.testing.assert_allclose(actual.get_cost(), reference.get_cost(), atol=1e-8, rtol=1e-8)
