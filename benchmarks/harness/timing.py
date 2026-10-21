"""Closed-loop construction and solve timings used by the deployment mode table."""

from scaly.function.concrete import ConcreteFunction

from scaly.function.model import as_concrete

from benchmarks.harness import registered_base


from time import perf_counter

import numpy as np
from typing import NotRequired, TypedDict


class SolveTimingSummary(TypedDict):
  """Construction and solve times measured for one episode."""

  build_ms: float
  first_solve_ms: float | None
  steady_step_ms: float | None
  solve_wall_ms: list[float]


class ModeSummary(TypedDict):
  """Episode identity and the timing measurements available for its mode table."""

  problem: str
  solver: str
  oracle: str | None
  build_ms: NotRequired[float]
  first_solve_ms: NotRequired[float | None]
  steady_step_ms: NotRequired[float | None]
  solve_wall_ms: NotRequired[list[float]]


class _ModeCommon(TypedDict):
  problem: str
  solver: str
  oracle: str | None
  per_step_ms: float | None
  build_ms: float | None
  timing_scope: str
  ipopt_provider: str
  cache_policy: str


class ModeRow(_ModeCommon):
  """Startup and steady solve costs for one deployment mode."""

  mode: str
  time_to_first_solve_ms: float | None


def prepare_solver(controller) -> None:
  """Compile the solver so that its first solve call does not."""
  import scaly as sc

  if isinstance(controller, sc.Solver):
    controller.function.compile()


def prepare_checks(controller) -> None:
  """Compile the standalone oracles used by benchmark result checks, which no controller needs."""
  import scaly as sc

  if isinstance(controller, sc.Solver):
    plain = controller.function
    for function in (as_concrete(plain).descriptor.base, registered_base(plain)):
      if isinstance(function, sc.Function | ConcreteFunction):
        function.compile()


class SolveTiming:
  """Measure construction separately from the first and subsequent solve calls."""

  def __init__(self) -> None:
    self.started = perf_counter()
    self.build_ms = 0.0
    self.solve_ms: list[float] = []

  def prepared(self, controller=None) -> None:
    prepare_solver(controller)
    self.build_ms = (perf_counter() - self.started) * 1000.0
    prepare_checks(controller)

  def start_step(self) -> None:
    self.started = perf_counter()

  def end_step(self) -> None:
    self.solve_ms.append((perf_counter() - self.started) * 1000.0)

  def summary(self) -> SolveTimingSummary:
    return {
      "build_ms": self.build_ms,
      "first_solve_ms": self.solve_ms[0] if self.solve_ms else None,
      "steady_step_ms": float(np.mean(self.solve_ms[1:])) if len(self.solve_ms) > 1 else None,
      "solve_wall_ms": self.solve_ms,
    }


def mode_rows(summary: ModeSummary, *, interpreted: bool = False) -> list[ModeRow]:
  """Derive startup and steady costs from a completed episode, without rerunning it."""
  common: _ModeCommon = {
    "problem": summary["problem"],
    "solver": summary["solver"],
    "oracle": summary["oracle"],
    "per_step_ms": summary.get("steady_step_ms"),
    "build_ms": summary.get("build_ms"),
    "timing_scope": "wall solve call; build excludes result-check oracles; steady excludes first step; prebuilt excludes construction and loading",
    "ipopt_provider": ("CasADi wheel" if interpreted else "configured Scaly solver plugin") if summary["solver"] == "ipopt" else "not applicable",
    "cache_policy": "process cache as configured; jit startup includes observed build or cache load",
  }
  first = summary.get("first_solve_ms")
  if first is None:
    return [{**common, "mode": mode, "time_to_first_solve_ms": None} for mode in (["interpreted"] if interpreted else ["jit", "prebuilt"])]
  if interpreted:
    return [{**common, "mode": "interpreted", "time_to_first_solve_ms": summary["build_ms"] + first}]
  return [
    {**common, "mode": "jit", "time_to_first_solve_ms": summary["build_ms"] + first},
    {**common, "mode": "prebuilt", "time_to_first_solve_ms": first},
  ]


def summarize_modes(paths, out):
  """Aggregate repeated episode mode tables and retain unavailable measurements."""
  import csv
  from pathlib import Path
  from statistics import mean, median, stdev

  groups = {}
  for path in paths:
    source = Path(path)
    files = [source] if source.is_file() else sorted(source.rglob("modes.csv"))
    for file in files:
      with file.open(newline="") as stream:
        for row in csv.DictReader(stream):
          key = tuple(row[name] for name in ("problem", "solver", "oracle", "mode", "ipopt_provider", "timing_scope", "cache_policy"))
          groups.setdefault(key, []).append(row)
  rows = []
  for key, samples in sorted(groups.items()):
    row = dict(zip(("problem", "solver", "oracle", "mode", "ipopt_provider", "timing_scope", "cache_policy"), key, strict=True))
    row["runs"] = len(samples)
    for metric in ("time_to_first_solve_ms", "per_step_ms"):
      values = [float(sample[metric]) for sample in samples if sample[metric]]
      average = mean(values) if values else None
      deviation = stdev(values) if len(values) > 1 else None
      for name, value in {
        "count": len(values),
        "mean": average,
        "median": median(values) if values else None,
        "stdev": deviation,
        "cv": deviation / average if deviation is not None and average else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
      }.items():
        row[f"{metric}_{name}"] = value
    rows.append(row)
  output = Path(out)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", newline="") as stream:
    if rows:
      writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)
  return rows
