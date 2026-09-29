"""A receding-horizon controller as the examples write one, for the tests: an OCP's solver, its warm
start shifted after every solve, and what a solve gives the tests to read."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import scaly as sc


@dataclass(frozen=True)
class Solved:
  """One solve: the trajectory, the point reached, and the solver's report."""

  xs: np.ndarray
  us: np.ndarray
  point: np.ndarray
  status: sc.Status
  cost: float
  iterations: int
  leaves: dict[str, np.ndarray]

  @property
  def slack(self) -> np.ndarray | None:
    return self.leaves.get("slack")

  @property
  def zs(self) -> np.ndarray | None:
    return self.leaves.get("zs")


class Controller:
  """``sc.ocp.solver`` and ``sc.ocp.shift`` of one OCP and method, and the warm start between solves."""

  def __init__(self, problem: sc.ocp.DiscreteOCP, method: Any, *, name: str | None = None) -> None:
    self.problem, self.method = problem, method
    self.solve_fn = sc.ocp.solver(problem, method, name=name)
    self.shift_fn = sc.ocp.shift(problem, method)
    self.layout = method.layout(problem) if hasattr(method, "layout") else None  # a Direct method's point has leaves
    self.warm: np.ndarray | None = None

  def initial_guess(self, x0: Any, u: Any = None) -> np.ndarray:
    return sc.ocp.initial_guess(self.problem, self.method, x0, u)

  def solve(self, x0: Any, *, warm: np.ndarray | None = None, **params: Any) -> Solved:
    x0 = np.ravel(np.asarray(x0, dtype=np.float64))
    start = warm if warm is not None else self.warm if self.warm is not None else self.initial_guess(x0)
    values = [np.ravel(np.asarray(params[p.name], dtype=np.float64)) for p in self.problem.params]
    xs, us, point, info = self.solve_fn(x0, *values, start)
    point = np.asarray(point)
    self.warm = np.asarray(self.shift_fn(point))
    cut, leaves = 0, {}
    for name, size in zip(self.layout.var_names, self.layout.var_sizes, strict=True) if self.layout is not None else ():
      leaves[name] = point[cut : cut + size]
      cut += size
    k = self.problem.interval.n_internal
    if k and "zs" in leaves:
      leaves["zs"] = leaves["zs"].reshape(self.problem.N, k)
    status = sc.Status(int(info.status))
    return Solved(np.asarray(xs), np.asarray(us), point, status, float(info.objective), int(info.iter), leaves)

  def __call__(self, x0: Any, **params: Any) -> np.ndarray:
    """The control to apply at ``x0``: the first of the solution, from the shifted last one."""
    return self.solve(x0, **params).us[0]


def closed_loop(controller: Controller, plant: Any, x0: Any, steps: int, **params: Any) -> tuple[np.ndarray, np.ndarray, list[sc.Status], np.ndarray]:
  """``steps`` intervals of ``controller`` on ``plant(x, u) -> x_next``; a parameter may be a
  callable of the step number. The states, the controls, and per step the status and iterations."""
  x = np.ravel(np.asarray(x0, dtype=np.float64))
  xs, us, statuses, iterations = [x], [], [], []
  for k in range(steps):
    solved = controller.solve(x, **{name: value(k) if callable(value) else value for name, value in params.items()})
    statuses.append(solved.status)
    iterations.append(solved.iterations)
    u = solved.us[0]
    x = np.ravel(np.asarray(plant(x, u), dtype=np.float64))
    xs.append(x)
    us.append(u)
  return np.array(xs), np.array(us), statuses, np.array(iterations)
