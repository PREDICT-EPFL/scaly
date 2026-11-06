"""Compare the race-car example's Scaly and CasADi acados SQP iterates."""

from pathlib import Path
import runpy

import numpy as np
import pytest


def test_acados_race_car_iterates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  pytest.importorskip("acados_template")
  example = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples" / "acados_race_car.py"))

  def reject_casadi_model(*args):
    raise AssertionError("The Scaly solver must not construct the CasADi dynamics or cost")

  with monkeypatch.context() as blocked:
    for name in ("casadi_dynamics", "casadi_cost"):
      blocked.setitem(example["solve_race_car"].__globals__, name, reject_casadi_model)
    actual = example["solve_race_car"](tmp_path, horizon=6, scaly=True)
  reference = example["solve_race_car"](tmp_path, horizon=6, scaly=False)
  example["compare_iterates"](actual, reference)
  for solver in (actual, reference):
    assert 1 < solver.get_stats("sqp_iter") < 50
    assert np.max(solver.get_residuals()) < 1e-8
