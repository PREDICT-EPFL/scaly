"""Assemble dense stage derivatives and map them over a builder's recorded stage form."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..ad.derivatives import gradient, hessian, jacobian
from ..function.concrete import ConcreteFunction
from ..function.model import Function, as_concrete
from ..function.sugar import broadcast, vmap, window
from ..ir.expr import Expr, concat
from .problem import Problem
from .model import StageOracles


def stage_oracles(problem: Problem[Any, Any, Any, Any]) -> StageOracles:
  """Differentiate one stage before mapping, without constructing global sparse derivatives."""
  functions, layout = problem.stage_functions, problem.stage_layout
  assert functions is not None and layout is not None
  dynamics = as_concrete(functions.dynamics)
  nx, nu = (type_.size for type_ in dynamics.input_tree.types[:2])
  N = layout.N
  nz = nx + nu
  ni = len(layout.eq[0])
  np_ = len(layout.ineq[0])
  params = problem._param_symbols
  x = problem._nlp.x

  def multiplier(name: str, shape: tuple[int, ...]) -> Expr:
    while name in problem.params.names:
      name = "_" + name
    return Expr.sym(name, shape, diff=False)

  scale = multiplier("lam:f", ())
  lam = multiplier("lam:g", (problem.n_eq + problem.n_ineq,))

  def build(name: str, inputs: tuple[Expr, ...], outputs: tuple[Expr, ...], names: tuple[str, ...]) -> ConcreteFunction:
    return ConcreteFunction._from_exprs(f"{problem.name}_{name}", inputs, outputs, tuple(str(v.name) for v in inputs), names)

  def local(terminal: bool) -> tuple[ConcreteFunction, ConcreteFunction, ConcreteFunction]:
    width = nx if terminal else nz
    z = Expr.sym("z", (width,))
    local_params = tuple(dynamics.inputs[2:])
    env = dict(zip(problem.params.names[1:], local_params, strict=True))

    def call(fn: Function | None) -> Expr:
      if fn is None:
        return Expr.const(np.zeros(0))
      args = (z,) if terminal else (z[:nx], z[nx:])
      return fn(*args, *(env[name] for name in as_concrete(fn).input_names[len(args) :]))

    cost_fn = functions.terminal_cost if terminal else functions.cost
    cost = Expr.const(0.0) if cost_fn is None else call(cost_fn)
    values = call(functions.terminal) if terminal else concat([call(functions.dynamics), call(functions.path)])
    multipliers = multiplier("lam:g", (values.size,))
    lag = scale * cost + (multipliers * values).sum()
    grad = gradient(lag, z).reshape((1, width))
    block = concat([hessian(lag, z), grad], axis=0)
    prefix = "terminal" if terminal else "stage"
    return (
      build(f"{prefix}_grad", (z, *local_params), (gradient(cost, z),), ("grad",)),
      build(f"{prefix}_jac", (z, *local_params), (jacobian(values, z),), ("jac",)),
      build(f"{prefix}_hess", (z, *local_params, scale, multipliers), (block,), ("hess",)),
    )

  stage_grad, stage_jac, stage_hess = local(False)
  terminal_grad, terminal_jac, terminal_hess = local(True)
  mapped = [window(x, 0, nz)]
  last = [x[N * nz :]]
  for name, value in zip(problem.params.names[1:], params[1:], strict=True):
    mapped.append(value[:N] if name in functions.stage_params else broadcast(value))
    last.append(value[N] if name in functions.stage_params else value)

  initial_x = Expr.sym("x", (nx,))
  initial_value = functions.initial(initial_x, params[0])
  initial_lag = (lam[:ni] * initial_value).sum()
  initial_jac = build("initial_jac", (initial_x, params[0]), (jacobian(initial_value, initial_x),), ("jac",))
  initial_block = concat([hessian(initial_lag, initial_x), gradient(initial_lag, initial_x).reshape((1, nx))], axis=0)
  initial_hess = build("initial_hess", (initial_x, params[0], lam), (initial_block,), ("hess",))

  grad = concat([vmap(stage_grad, N)(*mapped).vec(), terminal_grad(*last)])
  jac = vmap(stage_jac, N)(*mapped)
  stage_lam = concat([lam[ni : problem.n_eq].reshape((N, nx)), lam[problem.n_eq : problem.n_eq + N * np_].reshape((N, np_))], axis=1)
  blocks = vmap(stage_hess, N)(*mapped, broadcast(scale), stage_lam)
  first = initial_hess(x[:nx], params[0], lam)
  first = concat([first[:nx], Expr.const(np.zeros((nu, nx))), first[nx:]], axis=0)
  first = concat([first, Expr.const(np.zeros((nz + 1, nu)))], axis=1)
  blocks = concat([(blocks[0] + first).reshape((1, nz + 1, nz)), blocks[1:]], axis=0)
  return StageOracles(
    grad=build("mapped_grad", (x, *params), (grad,), ("grad",)),
    jac=build(
      "mapped_jac",
      (x, *params),
      (jac[:, :nx], initial_jac(x[:nx], params[0]), jac[:, nx:], terminal_jac(*last)),
      ("dynamics", "initial", "path", "terminal"),
    ),
    hess=build("mapped_hess", (x, *params, scale, lam), (blocks, terminal_hess(*last, scale, lam[problem.n_eq + N * np_ :])), ("stages", "terminal")),
  )
