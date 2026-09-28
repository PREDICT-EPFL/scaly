"""Explicit Runge-Kutta steps of a continuous-time model, as Functions with the model's signature."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from ..function.model import ConcreteFunction, Function
from ..function.sugar import while_loop
from ..ir.expr import Expr, cast, concat, maximum, minimum, where
from ..ir.types import dtypes
from .model import Rhs, Step, discrete_map, model_rhs
from .tableau import Tableau, order_conditions, tableau

__all__ = ["adaptive", "explicit", "rk4", "symplectic"]


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
  return discrete_map(f, tab.name, dt, steps, name, lambda model, _name, _h: lambda x, others, h: rk_step(tab, model_rhs(model, others), x, h))


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


def adaptive(
  f: Function[Any, Any, Any, Any],
  pair: str | Tableau = "dopri5",
  *,
  dt: float | None = None,
  rtol: float = 1e-6,
  atol: float = 1e-9,
  max_steps: int = 10_000,
  h0: float | None = None,
  name: str | None = None,
) -> Any:
  """The map ``S(x, ..., dt) -> x(dt)`` of an embedded pair with error control: the steps are chosen
  as the integration goes, in a ``while_loop`` of at most ``max_steps`` steps, as ``solve_ivp`` does.

  Args:
    f: the model, as for ``explicit``.
    pair: an explicit tableau with embedded weights, ``"dopri5"`` (Dormand-Prince 5(4)) or ``"bs32"``
      (Bogacki-Shampine 3(2)); the step keeps the higher-order solution.
    dt: the interval, folded into the code; ``None`` (the default) makes it the last input.
    rtol, atol: a step is accepted when the difference between the two solutions, divided by
      ``atol + rtol * max(|x|, |x_next|)`` entry by entry, has a root mean square of at most 1.
    max_steps: the bound on accepted and rejected steps together. When it is reached before the end
      of the interval, the result is NaN.
    h0: the first step; by default ``dt / 100``.
    name: the Function's name, by default ``{f.name}_{pair}_adaptive``.

  The next step is ``0.9 err^(-1/(q+1))`` times the last, within ``[0.2, 5]``, ``q`` the order of the
  embedded solution. For plant models in closed-loop simulation, where accuracy matters more than a
  fixed cost. Derivatives take the step sizes as they were chosen: the controller's factor is rounded
  to a power of ``2^(1/1024)`` through an integer, which carries no derivative, so a derivative is the
  derivative of the steps taken, as for a fixed-step method, plus the last step's dependence on
  ``dt``.
  """
  tab = tableau(pair)
  if tab.b_err is None or not tab.explicit:
    raise ValueError(f"adaptive needs an explicit embedded pair, 'dopri5' or 'bs32', got {tab.name}")
  if not rtol > 0 or not atol > 0 or int(max_steps) != max_steps or max_steps < 1 or (h0 is not None and not h0 > 0):
    raise ValueError(f"rtol and atol must be positive, max_steps a positive integer and h0 positive; got {rtol}, {atol}, {max_steps}, {h0}")
  exponent = -1.0 / (order_conditions(tab.a, tab.b_err, tab.order) + 1)

  def method(model: ConcreteFunction[Any, Any, Any, Any], fname: str, _h: float | None) -> Step:
    n = model.inputs[0].size

    def step(x: Expr, others: tuple[Expr, ...], span: Expr | float) -> Expr:
      span = span if isinstance(span, Expr) else Expr.const(float(span))
      start = Expr.const(float(h0)) if h0 is not None else 0.01 * span.abs()
      carry = concat([x, Expr.const(np.zeros(1)), start.reshape((1,))])
      params = [*others, span]
      syms = [Expr.sym(f"q{i}", p.shape, dtype=p.type.dtype) for i, p in enumerate(params)]
      state = Expr.sym("xth", (n + 2,))
      y, time, h = state[:n], state[n], state[n + 1]
      rhs, total = model_rhs(model, tuple(syms[:-1])), syms[-1]
      hs = minimum(h, total - time)
      ks: list[Expr] = []
      for i in range(tab.stages):
        ks.append(rhs(increment(y, tab.a[i, :i], ks, hs)))
      high, low = increment(y, tab.b, ks, hs), increment(y, tab.b_err, ks, hs)
      scale = atol + rtol * maximum(y.abs(), high.abs())
      err = ((((high - low) / scale) ** 2).sum() / n).sqrt()
      accept = err <= 1.0
      nxt = concat([where(accept, high, y), where(accept, time + hs, time).reshape((1,)), (h * _controller(err, exponent)).reshape((1,))])
      labels = ["xth", *(f"q{i}" for i in range(len(syms)))]
      body = ConcreteFunction.from_exprs(f"{fname}_step", [state, *syms], [nxt], labels, ["xth_next"])
      left = total - time
      cond = ConcreteFunction.from_exprs(f"{fname}_running", [state, *syms], [left > 1e-12 * maximum(1.0, total.abs())], labels, ["go"])
      out, _ = while_loop(cond, body, carry, max_iter=int(max_steps), params=params)
      done = (span - out[n]) <= 1e-12 * maximum(1.0, span.abs())
      return where(done, out[:n], Expr.const(np.full(n, np.nan)))

    return step

  return discrete_map(f, f"{tab.name}_adaptive", dt, 1, name, method)


def _controller(err: Expr, exponent: float) -> Expr:
  """The factor from one step to the next: ``0.9 err^exponent`` within ``[0.2, 5]``, rounded to a
  power of ``2^(1/1024)`` so that it carries no derivative (``_power_of_two``)."""
  return _power_of_two(minimum(5.0, maximum(0.2, 0.9 * maximum(err, 1e-10) ** exponent)), 1024)


def _power_of_two(value: Expr, resolution: int = 1) -> Expr:
  """``value`` rounded towards 1 to a power of ``2^(1/resolution)``, through a cast of its exponent
  to ``int64``, which carries no derivative: the result is a constant for derivatives."""
  exponent = cast(cast(value.log() * (resolution / math.log(2.0)), dtypes.int64), dtypes.float64)
  return (exponent * (math.log(2.0) / resolution)).exp()


def symplectic(
  f: Function[Any, Any, Any, Any],
  method: Literal["stormer_verlet", "symplectic_euler"] = "stormer_verlet",
  *,
  split: int,
  dt: float | None,
  steps: int = 1,
  name: str | None = None,
) -> Any:
  """The map of a symplectic method for a model whose state is positions then velocities (or
  momenta), ``x = [q, v]`` with ``q = x[:split]``.

  ``"stormer_verlet"`` (order 2) is a half kick, a drift, a half kick::

      v_half = v + h/2 a(q, v);  q_next = q + h q'(q, v_half);  v_next = v_half + h/2 a(q_next, v_half)

  and ``"symplectic_euler"`` (order 1) a kick then a drift. For a separable model, where ``q'``
  depends only on the velocities and ``a`` only on the positions (and the held inputs), both are
  symplectic: over long horizons the energy error stays bounded instead of drifting, which a
  Runge-Kutta method does not give. For any other model they are explicit partitioned methods of
  the same form. Each half is read from one call of the model, at the state it needs.
  """
  if method not in ("stormer_verlet", "symplectic_euler"):
    raise ValueError(f"method must be 'stormer_verlet' or 'symplectic_euler', got {method!r}")

  def prepare(model: ConcreteFunction[Any, Any, Any, Any], fname: str, _h: float | None) -> Step:
    n = model.inputs[0].size
    if int(split) != split or not 0 < split < n:
      raise ValueError(f"{fname}: split must count the positions, between 1 and {n - 1}, got {split}")

    def step(x: Expr, others: tuple[Expr, ...], h: Expr | float) -> Expr:
      rhs = model_rhs(model, others)
      q, v = x[:split], x[split:]
      if method == "symplectic_euler":
        v1 = v + h * rhs(x)[split:]
        return concat([q + h * rhs(concat([q, v1]))[:split], v1])
      half = v + (0.5 * h) * rhs(x)[split:]
      q1 = q + h * rhs(concat([q, half]))[:split]
      return concat([q1, half + (0.5 * h) * rhs(concat([q1, half]))[split:]])

    return step

  return discrete_map(f, method, dt, steps, name, prepare)
