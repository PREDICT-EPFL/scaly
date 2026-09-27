"""Explicit Runge-Kutta steps of a continuous-time model, as Functions with the model's signature."""

from __future__ import annotations

from typing import Any

from ..function.model import Function
from ..ir.expr import Expr
from .model import Rhs, discrete_map
from .tableau import Tableau, tableau

__all__ = ["explicit", "rk4"]


def explicit(f: Function[Any, Any, Any, Any], method: str | Tableau = "rk4", *, dt: float | None, steps: int = 1, name: str | None = None) -> Any:
  """The discrete map ``F(x, ...) -> xnext`` of an explicit Runge-Kutta method over an interval ``dt``.

  Args:
    f: the model, a Function whose first parameter is the state ``x`` and whose one output is its
      derivative. Its other parameters (inputs, parameters) become ``F``'s and are held fixed over
      the interval: a zero-order hold.
    method: a name from ``TABLEAUS`` (``"euler"``, ``"heun"``, ``"midpoint"``, ``"ralston"``,
      ``"rk3"``, ``"ssprk3"``, ``"rk4"``, ``"rk38"``, and the pairs ``"bs32"`` and ``"dopri5"``, whose
      higher-order solution is the step) or an explicit ``Tableau``.
    dt: the interval, folded into the generated code; ``None`` appends a ``dt`` parameter instead.
    steps: equal substeps per interval. Up to ``UNROLL_STEPS`` are unrolled, more run as a loop.
    name: the Function's name, by default ``{f.name}_{method}``.

  Each stage calls ``f`` once, as a call node, so the model is generated once and differentiated
  once however many stages and substeps there are. A template model gives a template.
  """
  tab = tableau(method)
  if not tab.explicit:
    raise ValueError(f"{tab.name} is an implicit method; use sc.integrators.implicit")
  return discrete_map(f, tab.name, dt, steps, name, lambda rhs, x, h: rk_step(tab, rhs, x, h))


def rk4(f: Function[Any, Any, Any, Any], *, dt: float | None, steps: int = 1, name: str | None = None) -> Any:
  """The classical fourth-order Runge-Kutta map of ``f``: ``explicit(f, "rk4", ...)``."""
  return explicit(f, "rk4", dt=dt, steps=steps, name=name)


def rk_step(tab: Tableau, rhs: Rhs, x: Expr, h: Expr | float) -> Expr:
  """One explicit step: stage ``i`` evaluates ``rhs`` at ``x + h sum_j a_ij k_j``, and the step is
  ``x + h sum_i b_i k_i``."""
  ks: list[Expr] = []
  for i in range(tab.stages):
    ks.append(rhs(increment(x, tab.a[i, :i], ks, h)))
  return increment(x, tab.b, ks, h)


def increment(x: Expr, weights: Any, ks: list[Expr], h: Expr | float) -> Expr:
  """``x + h sum_j w_j k_j`` with as few multiplications as the weights allow: a numerical ``h`` folds
  into the weights, a zero weight drops its term, and the most frequent weight (the smallest on a
  tie) is factored out, so its terms need no multiplication. RK4's step is then
  ``x + h/6 (k1 + 2 k2 + 2 k3 + k4)``, as written by hand."""
  folded = 1.0 if isinstance(h, Expr) else float(h)
  terms = [(float(w) * folded, k) for w, k in zip(weights, ks) if w != 0]
  if not terms:
    return x
  magnitudes = [abs(c) for c, _ in terms]
  scale = min(set(magnitudes), key=lambda m: (-magnitudes.count(m), m))
  total: Expr | None = None
  for c, k in terms:
    ratio = c / scale
    if total is not None and ratio == -1:
      total = total - k
      continue
    term = k if ratio == 1 else ratio * k
    total = term if total is None else total + term
  assert total is not None
  return x + (h * scale if isinstance(h, Expr) else scale) * total
