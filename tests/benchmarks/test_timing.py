"""Deployment modes use measured build and solve costs without inventing samples."""

import pytest

from benchmarks.harness.timing import SolveTiming, mode_rows


def test_modes_derive_startup_and_exclude_first_step(monkeypatch):
  clock = iter([0.0, 0.1, 0.2, 0.23, 0.3, 0.305, 0.4, 0.407])
  monkeypatch.setattr("benchmarks.harness.timing.perf_counter", lambda: next(clock))
  timing = SolveTiming()
  timing.prepared()
  for _ in range(3):
    timing.start_step()
    timing.end_step()
  summary = {"problem": "synthetic", "solver": "ipopt", "oracle": "scaly", **timing.summary()}
  jit, prebuilt = mode_rows(summary)
  assert jit["time_to_first_solve_ms"] == pytest.approx(130.0)
  assert prebuilt["time_to_first_solve_ms"] == pytest.approx(30.0)
  assert jit["per_step_ms"] == prebuilt["per_step_ms"] == pytest.approx(6.0)
  interpreted = mode_rows({**summary, "oracle": "casadi"}, interpreted=True)
  assert len(interpreted) == 1 and interpreted[0]["mode"] == "interpreted"
  assert interpreted[0]["time_to_first_solve_ms"] == pytest.approx(130.0)


def test_one_step_has_no_steady_state_sample():
  timing = SolveTiming()
  timing.prepared()
  timing.start_step()
  timing.end_step()
  summary = {"problem": "synthetic", "solver": "ipopt", "oracle": "scaly", **timing.summary()}
  assert all(row["per_step_ms"] is None for row in mode_rows(summary))


@pytest.mark.solver("ipopt")
def test_interpreted_casadi_mode_solves_and_records_stats():
  import casadi as ca
  import numpy as np

  from benchmarks.harness.casadi_ipopt import INTERPRETED, make_casadi_ipopt

  z, p = ca.MX.sym("z"), ca.MX.sym("p", 0)
  nlp = ca.Function("timing_nlp", [z, p], [(z - 2) ** 2, ca.MX.zeros(0)])
  token = INTERPRETED.set(True)
  try:
    solver = make_casadi_ipopt("timing_solver", nlp, {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False})
  finally:
    INTERPRETED.reset(token)
  result = solver(np.zeros(1), np.zeros(0), np.array([-10.0]), np.array([10.0]), np.zeros(0), np.zeros(0), np.zeros(1), np.zeros(0))
  assert result[0] == pytest.approx([2.0])
  assert not solver.compiled
  assert solver.last_stats.status.value == 0
  assert solver.last_stats.t_total > 0
  assert solver.last_stats.n_eval_f > 0
  from scaly.utils.env import shared_lib_ext

  from pathlib import Path

  assert solver.ipopt_library.endswith(f"libipopt{shared_lib_ext()}")
  assert Path(solver.ipopt_library).is_file()


def test_repeated_modes_report_dispersion_and_missing_samples(tmp_path):
  import csv

  from benchmarks.harness.timing import summarize_modes

  paths = []
  for i, first in enumerate([10.0, 20.0, None]):
    summary = {
      "problem": "synthetic",
      "solver": "ipopt",
      "oracle": "scaly",
      "build_ms": 100.0,
      "first_solve_ms": first,
      "steady_step_ms": 5.0 if first is not None else None,
    }
    rows = mode_rows(summary)
    path = tmp_path / f"run{i}.csv"
    with path.open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)
    paths.append(path)
  rows = summarize_modes(paths, tmp_path / "summary.csv")
  assert len(rows) == 2
  jit = next(row for row in rows if row["mode"] == "jit")
  assert jit["runs"] == 3 and jit["time_to_first_solve_ms_count"] == 2
  assert jit["time_to_first_solve_ms_mean"] == 115.0
  assert jit["time_to_first_solve_ms_stdev"] == pytest.approx(50**0.5)
  assert jit["time_to_first_solve_ms_cv"] == pytest.approx(50**0.5 / 115)


@pytest.mark.parametrize("exception", [RuntimeError, ValueError])
def test_failed_episode_keeps_unavailable_mode_rows(tmp_path, monkeypatch, exception):
  import csv

  from benchmarks.harness import closed_loop

  def fail(*args, **kwargs):
    raise exception("synthetic solve failure")

  monkeypatch.setattr(closed_loop, "_run", fail)
  with pytest.raises(exception, match="synthetic solve failure"):
    closed_loop.run("synthetic", smoke=True, solver="ipopt", oracle="scaly", out_dir=tmp_path, cli_args=[])
  with (tmp_path / "synthetic" / "ipopt+scaly" / "modes.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
  assert len(rows) == 2
  assert all(row["time_to_first_solve_ms"] == row["per_step_ms"] == "" for row in rows)


@pytest.mark.solver("ipopt")
def test_prepared_solver_never_compiles_during_first_solve(tmp_path, monkeypatch):
  import scaly as sc
  import numpy as np
  from scaly.codegen import jit
  from benchmarks.harness import solve_problem

  monkeypatch.setenv("SCALY_CACHE_DIR", str(tmp_path))

  @sc.problem(vars=sc.L("x", 1), params=sc.L("target", 1))
  def problem(x, target):
    return sc.ProblemSpec(minimize=((x - target) ** 2).sum())

  solver = sc.solver(problem, "ipopt", name="mode_warmup", options={"print_level": 0, "sb": "yes"})
  x = sc.sym("diagnostic_x", 1)
  diagnostic = sc.Function._from_exprs("mode_diagnostic", [x], [x.sin()], ["x"], ["y"])
  setattr(solver, "_benchmark_base", diagnostic)
  timing = SolveTiming()
  timing.prepared(solver)

  def unexpected_compile(*args, **kwargs):
    raise AssertionError("compilation leaked into the first solve")

  monkeypatch.setattr(jit, "CompiledFunction", unexpected_compile)
  result = solve_problem(solver, np.zeros(1), np.zeros(0), np.zeros(0), np.zeros(1), np.array([0.5]))
  np.testing.assert_allclose(result["x"], [0.5], atol=1e-8)
  np.testing.assert_allclose(diagnostic.numerical_call(np.array([0.5])), np.sin([0.5]))


@pytest.mark.parametrize(
  "native,code,status",
  [
    ("Feasible_Point_Found", 6, 1),
    ("Maximum_CpuTime_Exceeded", -4, 2),
    ("Maximum_WallTime_Exceeded", -5, 2),
    ("Diverging_Iterates", 4, 5),
    ("User_Requested_Stop", 5, 6),
  ],
)
def test_interpreted_statuses_match_compiled_outcomes(native, code, status):
  import numpy as np
  from benchmarks.harness.casadi_ipopt import InterpretedCasadiIpopt

  class NativeSolver:
    def __call__(self, *values):
      return [np.zeros(1), np.array(0.0), *[np.zeros(1)] * 4]

    def stats(self):
      return {"return_status": native, "iter_count": 1}

  solver = InterpretedCasadiIpopt.__new__(InterpretedCasadiIpopt)
  solver.solver = NativeSolver()
  solver()
  assert solver.last_stats is not None
  assert solver.last_stats.status.value == status
  assert solver.last_stats.native_status == code
