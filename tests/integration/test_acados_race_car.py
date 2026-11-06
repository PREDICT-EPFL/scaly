"""Compare the race-car example's Scaly and CasADi acados SQP iterates."""

from pathlib import Path
import runpy

import numpy as np
import pytest


def test_acados_race_car_iterates(tmp_path: Path) -> None:
  pytest.importorskip("acados_template")
  example = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples" / "acados_race_car.py"))
  actual, reference = [example["solve_race_car"](tmp_path, horizon=6, scaly=use_scaly) for use_scaly in (True, False)]
  example["compare_iterates"](actual, reference)
  for solver in (actual, reference):
    assert 1 < solver.get_stats("sqp_iter") < 50
    assert np.max(solver.get_residuals()) < 1e-8
