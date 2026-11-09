"""Discrete-time multistage optimal-control problems and their general and stage-form problems."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, cast

import numpy as np

from ..function.model import Function, as_concrete
from ..function.concrete import ConcreteFunction
from ..function.sugar import broadcast, vmap, window
from ..function.tree import _Leaf, flat_parameters, flat_tree
from ..ir.expr import Expr, concat
from ..ir.types import TensorType
from .problem import Problem, ProblemSpec, bounded, problem
from .stages import StageLayout

_RESERVED = ("x", "u", "initial_state", "z", "x_N")


@dataclass(frozen=True, slots=True)
class StageConstraint:
  """The constraint ``lo <= fn(...) <= hi`` on one stage.

  ``fn`` takes the state and the control for a path constraint, which holds at every stage before
  the last, and the state alone for a terminal constraint. Its further inputs are matched by name, as
  for the costs. A side is ``None``, a number, or an array of the output's size, and ``lo == hi``
  makes an equality.
  """

  fn: Function[Any, Any, Any, Any]
  lo: Any = None
  hi: Any = None

  def __post_init__(self) -> None:
    if self.lo is None and self.hi is None:
      raise ValueError("StageConstraint needs at least one of lo / hi")


@dataclass(frozen=True)
class StageFunctions:
  """The Functions of one stage of a stage-form problem, which map over its stages.

  Every stage ``k < N`` takes ``(x, u, *params)`` and the terminal stage ``(x, *params)``, with ``x``
  the stage's state, the global variables appended. The parameters are the problem's after
  ``initial_state``: a name in ``stage_params`` takes stage ``k``'s row of its parameter, every other
  the whole value. ``initial`` gives the rows ``layout.eq[0]``, ``dynamics`` the next state,
  ``path`` the rows ``layout.ineq[k]`` and ``terminal`` the rows ``layout.ineq[N]``. A cost or a
  constraint the problem lacks is ``None``. The bounds of those rows and of the variables are the
  problem's own, from its bounds oracle.
  """

  initial: Function[Any, Any, Any, Any]
  dynamics: Function[Any, Any, Any, Any]
  cost: Function[Any, Any, Any, Any] | None
  path: Function[Any, Any, Any, Any] | None
  terminal_cost: Function[Any, Any, Any, Any] | None
  terminal: Function[Any, Any, Any, Any] | None
  stage_params: tuple[str, ...]


def _side(value: Any, size: int, fill: float, what: str) -> np.ndarray:
  if isinstance(value, Expr):
    raise TypeError(f"{what} is an expression, but a bound is a number or an array")
  array = np.full(size, fill) if value is None else np.asarray(value, dtype=np.float64)
  if array.shape not in ((), (size,)):
    raise ValueError(f"{what} has shape {array.shape}, expected a number or ({size},)")
  return np.broadcast_to(array, (size,))


@dataclass(frozen=True, eq=False)
class DiscreteOCP:
  """A discrete-time optimal-control problem over ``N`` stages, stated by its stage Functions.

  It minimizes ``sum_k l(x_k, u_k) + Vf(x_N)`` subject to ``x_{k+1} = F(x_k, u_k)``, ``x_0`` equal to
  the parameter ``initial_state``, the path constraints and control bounds at every stage ``k < N``,
  the state bounds at every stage after the first, and the terminal constraints on ``x_N``.

  ``dynamics`` takes the state and the control, two vectors, and returns the next state. The stage
  cost and the path constraints take the same two, and the terminal cost and constraints take the
  state. A Function's further inputs are matched by name across Functions: a name in
  ``global_vars`` is a decision variable shared by every stage, a name in ``stage_params`` a
  parameter with one row per stage, ``N + 1`` of them, and any other name a parameter. ``x_bounds``
  and ``u_bounds`` are ``(lo, hi)`` pairs, each side ``None``, a number or an array.

  ``problem()`` and ``stage_problem()`` lower it to a ``Problem``, and ``pack`` and ``unpack``
  convert between trajectories and their variables.
  """

  N: int
  dynamics: Function[Any, Any, Any, Any]
  stage_cost: Function[Any, Any, Any, Any] | None = None
  terminal_cost: Function[Any, Any, Any, Any] | None = None
  path: Sequence[StageConstraint] = ()
  terminal: Sequence[StageConstraint] = ()
  x_bounds: tuple[Any, Any] | None = None
  u_bounds: tuple[Any, Any] | None = None
  stage_params: Sequence[str] = ()
  global_vars: Sequence[str] = ()
  name: str | None = None
  _inputs: dict[str, TensorType] = field(init=False, repr=False)

  def __post_init__(self) -> None:
    if isinstance(self.N, bool) or not isinstance(self.N, int | np.integer) or self.N < 1:
      raise ValueError(f"N must be a positive integer, got {self.N!r}")
    object.__setattr__(self, "N", int(self.N))
    for name, value in (("path", self.path), ("terminal", self.terminal), ("stage_params", self.stage_params), ("global_vars", self.global_vars)):
      object.__setattr__(self, name, tuple(value))
    for constraint in (*self.path, *self.terminal):
      if not isinstance(constraint, StageConstraint):
        raise TypeError(f"path and terminal take StageConstraint values, got {type(constraint).__name__}")
    if self.name is None:
      object.__setattr__(self, "name", f"{self.dynamics.name}_ocp")
    dynamics = as_concrete(self.dynamics)
    types = dynamics.input_tree.types
    if len(types) < 2 or any(len(type_.shape) != 1 for type_ in types[:2]):
      raise ValueError(f"{dynamics.name} takes the state, then the control, two vectors: F(x, u, ...)")
    if len(dynamics.outputs) != 1 or dynamics.outputs[0].shape != types[0].shape:
      raise ValueError(f"{dynamics.name} returns the next state, one vector of the state's shape {types[0].shape}")
    inputs: dict[str, TensorType] = {}
    for fn, leading, scalar in self._functions():
      concrete = as_concrete(fn)
      if any(not isinstance(part, _Leaf) for part in getattr(concrete.input_tree, "parts", ())):
        raise ValueError(f"{concrete.name} takes one tensor per input")
      names, declared = concrete.input_names, concrete.input_tree.types
      if names[:1] == ("u",) or names[1:leading] == ("x",):
        raise ValueError(f"{concrete.name} takes the state first, then the control, but its inputs are named {names[:leading]}")
      if tuple(type_.shape for type_ in declared[:leading]) != tuple(type_.shape for type_ in types[:leading]):
        raise ValueError(
          f"{concrete.name} takes the state{', then the control' if leading == 2 else ''} first, of shapes {[t.shape for t in types[:leading]]}"
        )
      if len(concrete.outputs) != 1 or (scalar and concrete.outputs[0].size != 1) or len(concrete.outputs[0].shape) > 1:
        raise ValueError(f"{concrete.name} returns {'one value' if scalar else 'one vector'}")
      for name, type_ in zip(names[leading:], declared[leading:], strict=True):
        if name in _RESERVED:
          raise ValueError(f"{concrete.name}: an input may not be named {name!r}")
        if inputs.setdefault(name, type_).shape != type_.shape:
          raise ValueError(f"{concrete.name}: input {name!r} is {type_.shape}, but another Function takes it as {inputs[name].shape}")
    for field_name in ("stage_params", "global_vars"):
      for name in getattr(self, field_name):
        if name not in inputs:
          raise ValueError(f"{field_name} names {name!r}, which no Function takes")
    if both := sorted(set(self.stage_params) & set(self.global_vars)):
      raise ValueError(f"{both[0]!r} is both a stage parameter and a global variable")
    for pair, size, what in ((self.x_bounds, types[0].size, "x_bounds"), (self.u_bounds, types[1].size, "u_bounds")):
      for side in pair or ():
        _side(side, size, 0.0, what)
    for constraint in (*self.path, *self.terminal):
      concrete = as_concrete(constraint.fn)
      for side in ("lo", "hi"):
        _side(getattr(constraint, side), concrete.outputs[0].size, 0.0, f"{concrete.name} {side}")
    object.__setattr__(self, "_inputs", inputs)

  def _functions(self) -> tuple[tuple[Function[Any, Any, Any, Any], int, bool], ...]:
    """Every user Function with its number of leading inputs (state, control) and whether it is a cost."""
    return (
      (self.dynamics, 2, False),
      *(() if self.stage_cost is None else ((self.stage_cost, 2, True),)),
      *(() if self.terminal_cost is None else ((self.terminal_cost, 1, True),)),
      *((c.fn, 2, False) for c in self.path),
      *((c.fn, 1, False) for c in self.terminal),
    )

  @property
  def _nx(self) -> int:
    return as_concrete(self.dynamics).input_tree.types[0].shape[0]

  @property
  def _nu(self) -> int:
    return as_concrete(self.dynamics).input_tree.types[1].shape[0]

  @property
  def _globals(self) -> dict[str, TensorType]:
    return {name: self._inputs[name] for name in self.global_vars}

  @property
  def _params(self) -> dict[str, TensorType]:
    return {name: type_ for name, type_ in self._inputs.items() if name not in self.global_vars}

  @property
  def _n_globals(self) -> int:
    return sum(type_.size for type_ in self._globals.values())

  def problem(self) -> Problem[Any, Any, Any, Any]:
    """The general form: variables ``z`` of shape ``(N, nx + nu)``, one row ``[x_k, u_k]`` per stage,
    ``x_N``, then each global variable, shared by every stage."""
    return self._general

  def stage_problem(self) -> Problem[Any, Any, Any, Any]:
    """The stage form: each global variable becomes part of the state, held constant from stage to
    stage, so every stage depends on its own variables only. The problem records the sizes of its
    stages and its stage Functions for the solvers that read them."""
    return self._staged

  @cached_property
  def _general(self) -> Problem[Any, Any, Any, Any]:
    return self._lower(stage_form=False)

  @cached_property
  def _staged(self) -> Problem[Any, Any, Any, Any]:
    return self._lower(stage_form=True)

  def _stage_functions(self, stage_form: bool, name: str) -> StageFunctions:
    nx, nu = self._nx, self._nu
    globals_ = self._globals
    nxa = nx + (self._n_globals if stage_form else 0)
    params = {param: TensorType(type_.shape, type_.dtype, diff=False) for param, type_ in self._params.items()}
    extra = params if stage_form else {**globals_, **params}

    def build(suffix: str, inputs: dict[str, TensorType], body: Any) -> Function[Any, Any, Any, Any]:
      tree = flat_parameters(tuple(inputs), tuple(inputs.values()))
      symbols = tuple(tree.symbols())
      env = dict(zip(tree.names, symbols, strict=True))
      if stage_form:
        offset = nx
        for variable, type_ in globals_.items():
          env[variable] = env["x"][offset : offset + type_.size].reshape(type_.shape)
          offset += type_.size
      output = body(env)
      fn = ConcreteFunction.build(f"{name}_{suffix}", tree, symbols, flat_tree((suffix,), (output.type,)), (output,))
      return Function._from_instance(fn)

    def call(fn: Function[Any, Any, Any, Any], env: dict[str, Expr], leading: int) -> Expr:
      names = as_concrete(fn).input_names
      x = env["x"][:nx] if stage_form else env["x"]
      return fn(*(x, env.get("u"))[:leading], *(env[input_name] for input_name in names[leading:]))

    def stacked(constraints: Sequence[StageConstraint], leading: int) -> Any:
      return lambda env: concat([value.reshape((value.size,)) for value in (call(c.fn, env, leading) for c in constraints)])

    state = {"x": TensorType((nxa,))}
    stage = {**state, "u": TensorType((nu,)), **extra}
    terminal = {**state, **extra}

    def dynamics(env: dict[str, Expr]) -> Expr:
      next_state = call(self.dynamics, env, 2)
      return concat([next_state, env["x"][nx:]]) if stage_form and nxa > nx else next_state

    stage_cost, terminal_cost = self.stage_cost, self.terminal_cost
    return StageFunctions(
      initial=build("initial", {**state, "initial_state": TensorType((nx,), diff=False)}, lambda env: env["x"][:nx] - env["initial_state"]),
      dynamics=build("dynamics", stage, dynamics),
      cost=None if stage_cost is None else build("cost", stage, lambda env: call(stage_cost, env, 2).reshape(())),
      path=build("path", stage, stacked(self.path, 2)) if self.path else None,
      terminal_cost=None if terminal_cost is None else build("terminal_cost", terminal, lambda env: call(terminal_cost, env, 1).reshape(())),
      terminal=build("terminal", terminal, stacked(self.terminal, 1)) if self.terminal else None,
      stage_params=tuple(self.stage_params),
    )

  def _lower(self, *, stage_form: bool) -> Problem[Any, Any, Any, Any]:
    N, nx, nu = self.N, self._nx, self._nu
    name = f"{self.name}_stages" if stage_form else cast(str, self.name)
    functions = self._stage_functions(stage_form, name)
    globals_ = {} if stage_form else self._globals
    nxa = nx + (self._n_globals if stage_form else 0)
    nz = nxa + nu
    params = self._params
    variables = flat_tree(("z", "x_N", *globals_), (TensorType((N, nz)), TensorType((nxa,)), *globals_.values()))
    parameters = flat_tree(
      ("initial_state", *params),
      (
        TensorType((nx,), diff=False),
        *(
          TensorType((N + 1, *type_.shape) if param in self.stage_params else type_.shape, type_.dtype, diff=False) for param, type_ in params.items()
        ),
      ),
    )

    def bounds(side: int, fill: float) -> tuple[Expr, ...] | None:
      declared = [pair for pair in (self.x_bounds, self.u_bounds) if pair is not None]
      if not any(pair[side] is not None for pair in declared):
        return None
      x = _side(None if self.x_bounds is None else self.x_bounds[side], nx, fill, "x_bounds")
      u = _side(None if self.u_bounds is None else self.u_bounds[side], nu, fill, "u_bounds")
      state = np.concatenate([x, np.full(nxa - nx, fill)])
      rows = np.tile(np.concatenate([state, u]), (N, 1))
      rows[0, :nxa] = fill
      return (Expr.const(rows), Expr.const(state), *(Expr.const(np.full(type_.shape, fill)) for type_ in globals_.values()))

    def constraint(fn: Function[Any, Any, Any, Any], constraints: Sequence[StageConstraint], values: Expr, copies: int) -> Any:
      sizes = [as_concrete(c.fn).outputs[0].size for c in constraints]
      lo, hi = (
        np.tile(
          np.concatenate(
            [_side(getattr(c, side), size, fill, f"{as_concrete(c.fn).name} {side}") for c, size in zip(constraints, sizes, strict=True)]
          ),
          copies,
        )
        for side, fill in (("lo", -np.inf), ("hi", np.inf))
      )
      return bounded(values, lo=None if np.isneginf(lo).all() else lo, hi=None if np.isposinf(hi).all() else hi, name=as_concrete(fn).name)

    def body(variables: tuple[Expr, ...], parameters: Any) -> ProblemSpec[Any]:
      z, x_last, *shared = variables
      initial_state, *values = parameters if isinstance(parameters, tuple) else (parameters,)
      flat = concat([z.vec(), x_last])
      mapped = [window(flat, 0, nz), window(flat, nxa, nz), *(broadcast(value) for value in shared)]
      last = [x_last, *shared]
      for param, value in zip(params, values, strict=True):
        mapped.append(value[:N] if param in self.stage_params else broadcast(value))
        last.append(value[N] if param in self.stage_params else value)
      next_states = concat([z[1:, :nxa].vec(), x_last]) if N > 1 else x_last
      cost = Expr.const(0.0)
      if functions.cost is not None:
        cost = cost + vmap(functions.cost, N)(*mapped).sum()
      if functions.terminal_cost is not None:
        cost = cost + functions.terminal_cost(*last)
      ineq = []
      if functions.path is not None:
        ineq.append(constraint(functions.path, self.path, vmap(functions.path, N)(*mapped).vec(), N))
      if functions.terminal is not None:
        ineq.append(constraint(functions.terminal, self.terminal, functions.terminal(*last), 1))
      return ProblemSpec(
        minimize=cost,
        eq=(functions.initial(flat[:nxa], initial_state), vmap(functions.dynamics, N)(*mapped).vec() - next_states),
        ineq=tuple(ineq),
        lb=bounds(0, -np.inf),
        ub=bounds(1, np.inf),
      )

    lowered = problem(vars=variables, params=parameters, name=name)(body)
    if not stage_form:
      return lowered
    n_eq = nx + N * nxa
    n_path = 0 if functions.path is None else as_concrete(functions.path).outputs[0].size
    n_terminal = 0 if functions.terminal is None else as_concrete(functions.terminal).outputs[0].size
    layout = StageLayout(
      nx=(nxa,) * (N + 1),
      nu=(nu,) * N + (0,),
      dynamics=tuple(tuple(range(nx + k * nxa, nx + (k + 1) * nxa)) for k in range(N)),
      eq=(tuple(range(nx)),) + ((),) * N,
      ineq=(
        *(tuple(range(n_eq + k * n_path, n_eq + (k + 1) * n_path)) for k in range(N)),
        tuple(range(n_eq + N * n_path, n_eq + N * n_path + n_terminal)),
      ),
    )
    return replace(lowered, stage_layout=layout, stage_functions=functions)

  def pack(self, xs: Any, us: Any, *global_values: Any, stage_form: bool = False) -> tuple[np.ndarray, ...]:
    """The variables of ``problem()``, or of ``stage_problem()``, for states ``xs`` of shape
    ``(N + 1, nx)``, controls ``us`` of shape ``(N, nu)`` and one value per global variable. Each
    broadcasts, so ``pack(x0, 0.0)`` holds the initial state with zero controls."""
    globals_ = self._globals
    if len(global_values) != len(globals_):
      raise ValueError(f"{self.name} has {len(globals_)} global variable(s) {tuple(globals_)}, got {len(global_values)} value(s)")
    xs = np.broadcast_to(np.asarray(xs, dtype=np.float64), (self.N + 1, self._nx))
    us = np.broadcast_to(np.asarray(us, dtype=np.float64), (self.N, self._nu))
    values = [
      np.broadcast_to(np.asarray(value, dtype=np.float64), type_.shape).copy() for value, type_ in zip(global_values, globals_.values(), strict=True)
    ]
    if not stage_form:
      return (np.hstack([xs[:-1], us]), xs[-1].copy(), *values)
    shared = np.concatenate([value.ravel() for value in values]) if values else np.zeros(0)
    return np.hstack([xs[:-1], np.tile(shared, (self.N, 1)), us]), np.concatenate([xs[-1], shared])

  def unpack(self, variables: tuple[Any, ...], *, stage_form: bool = False) -> tuple[np.ndarray, ...]:
    """``(xs, us, *global values)`` from the variables of ``problem()``, or of ``stage_problem()``,
    taking the global variables from its last stage."""
    nx, nxa = self._nx, self._nx + (self._n_globals if stage_form else 0)
    z, x_last, *values = (np.asarray(value) for value in variables)
    if stage_form:
      offsets = np.cumsum([nx, *(type_.size for type_ in self._globals.values())])
      values = [x_last[start:end].reshape(type_.shape) for start, end, type_ in zip(offsets[:-1], offsets[1:], self._globals.values(), strict=True)]
    return (np.vstack([z[:, :nx], x_last[:nx]]), z[:, nxa:], *values)
