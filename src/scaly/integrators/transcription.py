"""Transcriptions: how one interval of a horizon becomes variables, equality constraints and a cost."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..function.model import ConcreteFunction, Function
from ..function.tree import G, L, param_list
from ..ir.expr import Expr, concat
from .method import ODE, solver
from .methods import RK4
from .model import check_model, model_rhs
from .polynomial import differentiation_matrix, gauss_nodes, interpolation_matrix, lagrange_integrals, radau_nodes

__all__ = ["Collocation", "Interval", "MultipleShooting", "Pseudospectral", "Transcription"]


@dataclass(frozen=True)
class Interval:
  """One interval of a transcribed horizon: a Function that a horizon maps over its intervals.

  Attributes:
    fn: ``(x, u, [z], xnext, *params, [dt]) -> r`` or ``-> (r, cost)``: the interval's
      ``n_residual`` equality residuals, zero when the interval is consistent, and its cost (one
      value) when the transcription was given one. Two outputs rather than one, so that a map of
      the residuals is a map a sparse Jacobian can take stage by stage.
      ``x`` and ``xnext`` are the states at the interval's ends, ``u`` the control at its start,
      ``z`` its own variables (present when ``n_internal > 0``), ``params`` the model's remaining
      inputs as flat leaves, and ``dt`` its length when that is an input.
    n_internal: the size of ``z``: the states at ``state_times``, then the controls at
      ``control_times``, each flat.
    n_residual: how many residuals come before the cost.
    state_times: the times in ``[0, 1]``, as fractions of the interval, of the states in ``z``.
    control_times: the times of the controls in ``z``; the control ``u`` is at 0.
    has_cost: whether ``fn`` has the cost as its second output.
  """

  fn: ConcreteFunction[Any, Any, Any, Any]
  n_internal: int
  n_residual: int
  state_times: np.ndarray
  control_times: np.ndarray
  has_cost: bool

  def guess(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """A starting point for ``z``: the interval's states all ``x`` and its controls all ``u``."""
    return np.concatenate([np.tile(np.ravel(x), self.state_times.size), np.tile(np.ravel(u), self.control_times.size)])


class Transcription:
  """How a continuous-time model and a running cost become an ``Interval``; ``MultipleShooting``,
  ``Collocation`` and ``Pseudospectral`` are the three kinds."""

  label: str

  def interval(
    self, model: Function[Any, Any, Any, Any], cost: Function[Any, Any, Any, Any] | None = None, *, dt: float | None, name: str | None = None
  ) -> Interval:
    """The interval of ``model`` (``f(x, u, *params) -> xdot``, the control its second input) of
    length ``dt`` (``None``: an input), with ``cost`` (``l(x, u, *params)``, one value) integrated
    over it when given. The Function is named ``name``, by default ``{model}_{label}``."""
    model_c, cost_c = _checked(model, cost)
    return self._interval(model_c, cost_c, dt, name or f"{model_c.name}_{self.label}")

  def _interval(
    self, model: ConcreteFunction[Any, Any, Any, Any], cost: ConcreteFunction[Any, Any, Any, Any] | None, dt: float | None, name: str
  ) -> Interval:
    raise NotImplementedError


def _checked(model: Any, cost: Any) -> tuple[ConcreteFunction[Any, Any, Any, Any], ConcreteFunction[Any, Any, Any, Any] | None]:
  if not isinstance(model, Function) or (cost is not None and not isinstance(cost, Function)):
    raise TypeError("a transcription takes a model and a cost that are sc.Functions")
  model_c = model.concrete
  check_model(model_c, model_c.name)
  if len(model_c.input_tree.parts) < 2 or len(model_c.input_tree.parts[1].names) != 1 or len(model_c.inputs[1].shape) != 1:
    raise ValueError(f"{model_c.name}: a transcribed model takes the state, then the control, one vector: f(x, u, *params)")
  if cost is None:
    return model_c, None
  cost_c = cost.concrete
  if [e.type.shape for e in cost_c.inputs] != [e.type.shape for e in model_c.inputs] or len(cost_c.outputs) != 1 or cost_c.outputs[0].size != 1:
    raise ValueError(f"{cost_c.name}: a running cost takes the model's inputs, l(x, u, *params), and gives one value")
  return model_c, cost_c


def _leaves(model: ConcreteFunction[Any, Any, Any, Any], internal: int, dt: float | None) -> list[L]:
  """The interval Function's parameters: ``x, u, [z], xnext, *params, [dt]``, each one flat leaf."""
  x, u = model.inputs[0], model.inputs[1]
  params = [L(f"p{i}", e.type) for i, e in enumerate(model.inputs[2:])]
  return [
    L("x", x.type),
    L("u", u.type),
    *([L("z", (internal,))] if internal else []),
    L("xnext", x.type),
    *params,
    *([L("dt", ())] if dt is None else []),
  ]


def _interval_fn(
  name: str, model: ConcreteFunction[Any, Any, Any, Any], internal: int, dt: float | None, has_cost: bool, body: Callable[..., Any]
) -> ConcreteFunction[Any, Any, Any, Any]:
  """The interval Function over ``_leaves``; ``body(x, u, z, xnext, params, h)`` gets ``z = None``
  without internal variables and ``h`` the length, a number or the ``dt`` input, and returns the
  residuals, or ``(residuals, cost)`` when ``has_cost``."""

  def fn(*args: Expr) -> Any:
    x, u, rest = args[0], args[1], list(args[2:])
    z = rest.pop(0) if internal else None
    xnext = rest.pop(0)
    h: Expr | float = rest.pop() if dt is None else float(dt)
    return body(x, u, z, xnext, tuple(rest), h)

  output = G(L("r", ...), L("cost", (1,))) if has_cost else L("r", ...)
  return ConcreteFunction(name, fn, param_list(*_leaves(model, internal, dt)), output)


class MultipleShooting(Transcription):
  """Each interval's end is the model integrated from its start by ``method``, an integrator method
  (``si.RK4(steps=2)``, ``si.RadauIIA(2)``, ``si.StormerVerlet(split=2)``, ...) or its name; the
  residual is that end minus ``xnext``, and no variables are added. A running cost is integrated by
  the same method, as one more state, so the step and its cost are one call of the map."""

  def __init__(self, method: Any = None) -> None:
    self.method = method if method is not None else RK4()
    self.label = f"{self.method if isinstance(self.method, str) else self.method.label}_shooting"

  def _interval(
    self, model: ConcreteFunction[Any, Any, Any, Any], cost: ConcreteFunction[Any, Any, Any, Any] | None, dt: float | None, name: str
  ) -> Interval:
    n = model.inputs[0].size
    target = model if cost is None else _augmented(model, cost, f"{name}_augmented")
    step = solver(ODE(target, dt=dt), self.method, name=f"{name}_step")

    def body(x: Expr, u: Expr, _z: Any, xnext: Expr, params: tuple[Expr, ...], h: Expr | float) -> Any:
      start = x if cost is None else concat([x, Expr.const(np.zeros(1))])
      end = step(*target.input_tree.unflatten((start, u, *params)), *([h] if dt is None else []))
      return end - xnext if cost is None else (end[:n] - xnext, end[n:])

    return Interval(_interval_fn(name, model, 0, dt, cost is not None, body), 0, n, np.zeros(0), np.zeros(0), cost is not None)


def _augmented(
  model: ConcreteFunction[Any, Any, Any, Any], cost: ConcreteFunction[Any, Any, Any, Any], name: str
) -> ConcreteFunction[Any, Any, Any, Any]:
  """``[f; l]`` over ``[x; q]``: the model with its running cost as one more state."""
  n = model.inputs[0].size

  def fn(xq: Expr, *rest: Any) -> Expr:
    return concat([model(xq[:n], *rest), cost(xq[:n], *rest).reshape((1,))])

  return ConcreteFunction(name, fn, param_list(L("x", (n + 1,)), *model.input_tree.parts[1:]), L("xdot", (n + 1,)))


class _Polynomial(Transcription):
  """A transcription by a polynomial per interval through the states at ``nodes`` (fractions of the
  interval), its derivative matched to the model at the ``collocated`` nodes.

  The state at node 0 is ``x``; the state at 1 is ``xnext`` when 1 is a node (``end_is_node``), and
  otherwise the polynomial's value there, one more residual. The controls are at the collocated
  nodes (``node_controls``) or held at ``u`` over the interval. The cost is the quadrature of the
  running cost at the collocated nodes, with weights exact for the polynomials the nodes allow."""

  nodes: np.ndarray
  collocated: np.ndarray
  node_controls: bool

  def _setup(self, nodes: np.ndarray, collocated: np.ndarray, node_controls: bool) -> None:
    self.nodes, self.collocated, self.node_controls = nodes, collocated, node_controls
    self.end_is_node = bool(nodes[-1] == 1.0)
    self.derivatives = differentiation_matrix(nodes)[collocated]
    self.end = interpolation_matrix(nodes, np.ones(1))[0]
    self.weights = lagrange_integrals(nodes[collocated], np.ones(1))[0]

  def _interval(
    self, model: ConcreteFunction[Any, Any, Any, Any], cost: ConcreteFunction[Any, Any, Any, Any] | None, dt: float | None, name: str
  ) -> Interval:
    n, nu = model.inputs[0].size, model.inputs[1].size
    inner = list(range(1, self.nodes.size - int(self.end_is_node)))  # nodes whose states are in z
    moving = [int(i) for i in self.collocated if i != 0] if self.node_controls else []  # nodes whose controls are in z
    internal = len(inner) * n + len(moving) * nu

    def body(x: Expr, u: Expr, z: Expr | None, xnext: Expr, params: tuple[Expr, ...], h: Expr | float) -> Any:
      states = {0: x, **({self.nodes.size - 1: xnext} if self.end_is_node else {})}
      states.update({node: z[k * n : (k + 1) * n] for k, node in enumerate(inner)} if z is not None else {})
      offset = len(inner) * n
      controls = {node: z[offset + k * nu : offset + (k + 1) * nu] for k, node in enumerate(moving)} if z is not None else {}
      values = [states[j] for j in range(self.nodes.size)]
      residuals, costs = [], []
      for row, node in enumerate(self.collocated):
        u_node = controls.get(int(node), u)
        slope = sum((float(c) * values[j] for j, c in enumerate(self.derivatives[row]) if c != 0), Expr.const(np.zeros(n)))
        rhs = model_rhs(model, (u_node, *params))
        residuals.append(slope - h * rhs(values[node]))
        if cost is not None:
          costs.append(float(self.weights[row]) * cost(*model.input_tree.unflatten((values[node], u_node, *params))).reshape((1,)))
      if not self.end_is_node:
        residuals.append(xnext - sum((float(c) * values[j] for j, c in enumerate(self.end) if c != 0), Expr.const(np.zeros(n))))
      if cost is None:
        return concat(residuals)
      return concat(residuals), h * sum(costs[1:], costs[0])

    n_residual = self.collocated.size * n + (0 if self.end_is_node else n)
    fn = _interval_fn(name, model, internal, dt, cost is not None, body)
    return Interval(fn, internal, n_residual, self.nodes[inner], self.nodes[moving], cost is not None)


class Collocation(_Polynomial):
  """Local orthogonal collocation: per interval, a polynomial through the start and ``degree``
  collocation points, its derivative the model's at each point; the control is held over the
  interval.

  ``points`` are ``"radau"`` (right Radau, order ``2 degree - 1``: 1 is a point, whose state is
  ``xnext``, so ``degree - 1`` states are added) or ``"legendre"`` (Gauss, order ``2 degree``:
  ``degree`` states are added, and the polynomial's value at 1 must equal ``xnext``, ``n`` residuals
  more). At a fixed control each is the implicit Runge-Kutta method of the same nodes, Radau IIA or
  Gauss-Legendre. Lobatto points are not offered: they include the start, so the polynomial through
  the states at the points is one degree short of the collocation conditions, and Lobatto IIIA is
  ``si.implicit`` instead."""

  def __init__(self, degree: int = 3, points: Literal["radau", "legendre"] = "radau") -> None:
    nodes_of = {"radau": radau_nodes, "legendre": gauss_nodes}
    if points not in nodes_of:
      raise ValueError(f"points must be 'radau' or 'legendre', got {points!r}")
    if int(degree) != degree or degree < 1:
      raise ValueError(f"collocation needs a degree of at least 1, got {degree}")
    c = nodes_of[points](int(degree))
    self._setup(np.concatenate([[0.0], c]), np.arange(1, c.size + 1), node_controls=False)
    self.degree, self.points, self.label = int(degree), points, f"collocation_{points}{int(degree)}"


class Pseudospectral(_Polynomial):
  """Radau pseudospectral collocation (as GPOPS-II): per interval (a segment; one interval over the
  horizon is the global method), the states and the controls at ``nodes`` Legendre-Gauss-Radau
  points, 0 among them, the polynomial through them and the end collocated at every point. The end
  is ``xnext``; the states at the other points but 0 and the controls at the points but 0 are the
  interval's own variables. Accuracy grows faster than any fixed order as ``nodes`` does, for a
  smooth solution.

  The node at 0 carries the control a receding horizon applies. The Gauss scheme has no node there,
  and the Lobatto one collocates at both ends, one condition more than its unknowns at a fixed
  control, so neither is offered."""

  def __init__(self, nodes: int = 10) -> None:
    if int(nodes) != nodes or nodes < 2:
      raise ValueError(f"pseudospectral collocation needs at least 2 nodes, got {nodes}")
    points = 1.0 - radau_nodes(int(nodes))[::-1]
    self._setup(np.concatenate([points, [1.0]]), np.arange(points.size), node_controls=True)
    self.n_nodes, self.label = int(nodes), f"pseudospectral{int(nodes)}"
