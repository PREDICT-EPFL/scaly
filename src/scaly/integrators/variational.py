"""Explicit Runge-Kutta steps with their sensitivities by the variational equation, the control held zero- or first-order over the step."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..function.api import jacobian
from ..function.model import ConcreteFunction, Function
from ..function.tree import G, L, param_list
from ..ir.expr import Expr
from .model import check_model
from .tableau import Tableau, tableau


def variational(
  f: Function[Any, Any, Any, Any], method: str | Tableau = "tsit5", *, dt: float, steps: int = 1, hold: Any = None, name: str | None = None
) -> ConcreteFunction[Any, Any, Any, Any]:
  """A map over ``[0, dt]`` of ``xdot = f(x, u, *params)`` by ``steps`` steps of an explicit
  Runge-Kutta ``method``, and its Jacobian carried stage by stage by the variational equation
  ``dPhi/dt = f_x Phi + f_u du/dz``, so exactly the forward-mode derivative of the steps.

  ``hold`` says how the control varies over the interval, per component: 0 for a zero-order hold
  (``u(t) = u0``), 1 for a first-order hold (``u(t) = u0 + (t / dt) (u1 - u0)``); a scalar applies to
  every component, and ``None`` is a zero-order hold throughout. With a zero-order hold throughout
  the map is ``(x0, u, *params) -> (x, Phi)`` with ``Phi = d x / d(x0, u)``; otherwise
  ``(x0, u0, u1, *params) -> (x, Phi)`` with ``Phi = d x / d(x0, u0, u1)``, ``(nx, nx + 2 nu)``.
  The parameters are held and carry no sensitivity. Stages that no weight reads are skipped (the
  first-same-as-last stage of ``tsit5``, say)."""
  model = f.concrete
  check_model(model, model.name)
  if len(model.input_tree.parts) < 2 or len(model.inputs[1].shape) != 1:
    raise ValueError(f"{model.name}: variational takes a model f(x, u, *params), the control one vector")
  tab = tableau(method)
  if not tab.explicit:
    raise ValueError(f"variational takes an explicit method, not {tab.name}")
  if steps < 1:
    raise ValueError("steps must be at least 1")
  nx, nu = model.inputs[0].size, model.inputs[1].size
  mask = np.zeros(nu) if hold is None else np.broadcast_to(np.asarray(hold, dtype=np.float64), (nu,)).copy()
  if not np.isin(mask, (0.0, 1.0)).all():
    raise ValueError("hold is 0 (zero-order) or 1 (first-order) per control component")
  foh = bool(mask.any())
  label = name or f"{model.name}_{tab.name}_variational"
  a, b, c = tab.a, tab.b, tab.c
  needed = _needed(a, b)
  h = float(dt) / steps
  params = [L(n, e.type) for n, e in zip(model.input_names[2:], model.inputs[2:], strict=True)]
  width = nx + (2 * nu if foh else nu)

  def rates_body(x: Expr, u: Expr, *p: Expr) -> tuple[Expr, Expr, Expr]:
    xdot = model(x, u, *p)
    return xdot, jacobian(xdot, x), jacobian(xdot, u)

  # One call node per stage: the rates and their Jacobians by AD of the model alone.
  rates = ConcreteFunction(
    f"{label}_rates", rates_body, param_list(L("x", nx), L("u", nu), *params), G(L("xdot", nx), L("J_x", (nx, nx)), L("J_u", (nx, nu)))
  )
  m = np.diag(mask)

  def body(x0: Expr, *rest: Expr) -> tuple[Expr, Expr]:
    u0, u1, p = (rest[0], rest[1], rest[2:]) if foh else (rest[0], rest[0], rest[1:])
    x = x0
    phi = Expr.const(np.hstack([np.eye(nx), np.zeros((nx, width - nx))]))
    for step in range(steps):
      t0 = step * h
      ks: list[Any] = [None] * len(b)
      kphis: list[Any] = [None] * len(b)
      for i in range(len(b)):
        if not needed[i]:
          continue
        xi, phii = x, phi
        for j in range(i):
          if a[i, j] != 0.0:
            xi = xi + (h * a[i, j]) * ks[j]
            phii = phii + (h * a[i, j]) * kphis[j]
        frac = (t0 + c[i] * h) / float(dt)  # the hold's fraction of the interval
        if foh:
          u = u0 + Expr.const(mask * frac) * (u1 - u0)
          du = Expr.const(np.hstack([np.zeros((nu, nx)), np.eye(nu) - frac * m, frac * m]))  # d u / d(x0, u0, u1)
        else:
          u = u0
          du = Expr.const(np.hstack([np.zeros((nu, nx)), np.eye(nu)]))  # d u / d(x0, u)
        rate, jx, ju = rates(xi, u, *p)
        ks[i] = rate
        kphis[i] = jx @ phii + ju @ du
      for i in range(len(b)):
        if b[i] != 0.0:
          x = x + (h * b[i]) * ks[i]
          phi = phi + (h * b[i]) * kphis[i]
    return x, phi

  controls = [L("u0", nu), L("u1", nu)] if foh else [L("u", nu)]
  return ConcreteFunction(label, body, param_list(L("x0", nx), *controls, *params), G(L("x", nx), L("Phi", (nx, width))))


def _needed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """The stages a step evaluates: those with a weight, and those a needed later stage reads."""
  needed = np.asarray(b != 0.0)
  for i in reversed(range(len(b))):
    if needed[i]:
      needed[:i] |= a[i, :i] != 0.0
  return needed


__all__ = ["variational"]
