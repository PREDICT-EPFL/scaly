"""The methods of ``scaly.ocp``: their registry, the ``Info`` they report, and ``solver``."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from ..function.method import Info as MethodInfo
from ..function.method import registry
from ..function.model import ConcreteFunction
from ..function.tree import L, Record
from ..ir.types import TensorType
from .problem import ContinuousOCP, DiscreteOCP


@dataclass(frozen=True)
class Info(MethodInfo):
  """What every OCP solver returns beside the trajectory: ``status`` (a ``Status`` code), ``iter``,
  the ``objective`` at the solution and its ``primal_residual``, the largest violation of the
  dynamics and constraints. All four are ``float64`` scalars that carry no derivative."""

  objective: Any
  primal_residual: Any

  @classmethod
  def tree(cls, prefix: str = "info:") -> Record:
    return Record(cls, **{f.name: L(prefix + f.name, TensorType((), diff=False)) for f in fields(cls)})


REGISTRY = registry("ocp", preference=("direct",))
"""Every OCP method by short name (``"direct"``); ``auto`` formulates the problem and hands it to the
first ``sc.opt`` method that takes it."""


def solver(problem: DiscreteOCP, method: Any = "auto", /, *, name: str | None = None) -> ConcreteFunction[Any, Any, Any, Any]:
  """The Function that solves ``problem`` by ``method``: an OCP method with its options
  (``sc.ocp.Direct(sc.opt.PIQP(sparse=True))``), its name, or ``"auto"``. Every OCP method's
  Function has one signature::

      xs, us, point, info = solve(x0, *params, warm)

  the initial state, the OCP's parameters in ``problem.params`` order (a varying one with ``N + 1``
  values, flat), and ``warm``, the method's primal-dual point to start from; the states ``(N + 1,
  nx)``, the controls ``(N, nu)``, the point the solve reached, and an ``Info``. ``shift(problem,
  method)`` moves a point up one stage for the next solve of a receding horizon, and
  ``initial_guess(problem, method, x0)`` gives a first one. The Function nests in a graph like any
  other: a control law is ``us[0]`` and the shifted point, one C function."""
  chosen = REGISTRY.resolve(method, discrete(problem))
  return chosen.build(problem, name=name or f"{problem.name}_{chosen.label}")


def discrete(problem: Any) -> DiscreteOCP:
  """``problem`` if it is a ``DiscreteOCP``; a ``ContinuousOCP`` is refused with the way to one."""
  if isinstance(problem, ContinuousOCP):
    raise TypeError(
      f"{problem.name} is a ContinuousOCP, which no method solves: transcribe it first, sc.ocp.transcribe(problem, transcription, N=...)"
    )
  return problem


__all__ = ["REGISTRY", "Info", "discrete", "solver"]
