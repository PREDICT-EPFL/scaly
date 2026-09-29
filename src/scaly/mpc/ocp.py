"""The OCP: dynamics, costs, constraints, horizon and transcription, and the ``sc.opt.problem`` it builds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from ..function.model import ConcreteFunction, Function
from ..function.tree import L, param_list
from ..ir.expr import Expr
from ..ocp.formulate import to_problem
from ..ocp.problem import ContinuousOCP, DiscreteOCP, Path, Quadratic, TerminalEquality
from ..ocp.transcription import Transcription

__all__ = ["OCP", "Path", "Quadratic", "TerminalEquality", "linear"]


def linear(a: Any, b: Any, *, name: str = "linear") -> ConcreteFunction[Any, Any, Any, Any]:
  """The discrete-time map ``x_next = A x + B u``, for an OCP's ``step=``: with ``Quadratic`` costs and
  polytopic constraints the OCP is a QP. A continuous-time pair goes through ``si.zoh`` first."""
  a, b = np.atleast_2d(np.asarray(a, dtype=np.float64)), np.asarray(b, dtype=np.float64)
  b = b.reshape(a.shape[0], -1)
  if a.shape[0] != a.shape[1]:
    raise ValueError(f"A must be square, got {a.shape}")

  def body(x: Expr, u: Expr) -> Expr:
    return Expr.const(a) @ x + Expr.const(b) @ u

  return ConcreteFunction(name, body, param_list(L("x", a.shape[0]), L("u", b.shape[1])), L("xnext", a.shape[0]))


class OCP:
  """An optimal control problem over a horizon of ``horizon`` intervals, formulated as an
  ``sc.opt.problem``: a ``scaly.ocp.DiscreteOCP`` (transcribed from a ``ContinuousOCP`` for ``ode``)
  and its ``to_problem`` formulation, sparse or ``condensed``. The ``scaly.ocp`` classes are the
  vocabulary; this keeps the controller's constructor until it moves there.

  Args:
    ode: a continuous-time model ``f(x, u, *params) -> xdot``; or
    step: a discrete-time map ``F(x, u, *params) -> x_next``. Exactly one.
    horizon: the number of intervals ``N``.
    dt: the interval's length. Required with ``ode``; with ``step`` it only sets the times.
    transcription: how ``ode`` becomes constraints; ``ocp.MultipleShooting(si.RK4())`` by default.
    condensed: the condensed form, as ``to_problem(..., form="condensed")``.
    stage_cost, terminal_cost, x_bounds, u_bounds, constraints, terminal, varying, cost, name: as
      for ``scaly.ocp.ContinuousOCP``.
  """

  def __init__(
    self,
    *,
    ode: Function[Any, Any, Any, Any] | None = None,
    step: Function[Any, Any, Any, Any] | None = None,
    horizon: int,
    dt: float | None = None,
    transcription: Transcription | None = None,
    stage_cost: Function[Any, Any, Any, Any] | Quadratic | None = None,
    terminal_cost: Function[Any, Any, Any, Any] | Quadratic | None = None,
    x_bounds: tuple[Any, Any] | None = None,
    u_bounds: tuple[Any, Any] | None = None,
    constraints: Sequence[Path] = (),
    terminal: Any = None,
    varying: Sequence[str] = (),
    cost: Literal["points", "integral"] | None = None,
    condensed: bool = False,
    name: str | None = None,
  ) -> None:
    if (ode is None) == (step is None):
      raise ValueError("an OCP takes exactly one of ode= (a continuous-time model) and step= (a discrete-time map)")
    if int(horizon) != horizon or horizon < 1:
      raise ValueError(f"horizon must be a positive integer, got {horizon}")
    if ode is not None and not (dt is not None and float(dt) > 0):
      raise ValueError("a continuous-time OCP needs a positive dt, the interval's length")
    if cost not in (None, "points", "integral") or (cost == "integral" and ode is None):
      raise ValueError("cost is 'points', or 'integral' for a continuous-time model")
    if transcription is not None and ode is None:
      raise ValueError("a transcription goes with ode=; a discrete-time map is its own transcription")
    common: dict[str, Any] = dict(
      stage_cost=stage_cost,
      terminal_cost=terminal_cost,
      x_bounds=x_bounds,
      u_bounds=u_bounds,
      constraints=constraints,
      terminal=terminal,
      varying=varying,
    )
    if ode is not None:
      assert dt is not None
      continuous = ContinuousOCP(ode, T=float(dt) * int(horizon), cost=cost, name=name, **common)
      self.docp = DiscreteOCP._transcribed(continuous, transcription, int(horizon), float(dt))
    else:
      assert step is not None
      self.docp = DiscreteOCP(step=step, N=int(horizon), dt=dt, name=name, **common)
    self.condensed = bool(condensed)
    self.problem, self.layout = to_problem(self.docp, "condensed" if condensed else "sparse")

  @property
  def horizon(self) -> int:
    return self.docp.N

  @property
  def states(self) -> Any:
    return self.layout.states

  def __getattr__(self, name: str) -> Any:
    return getattr(self.__dict__["docp"], name)
