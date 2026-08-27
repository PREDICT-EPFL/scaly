"""Test helpers for exercising plugins through the typed problem API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

import alloy as al
from alloy.function.tree import Tree, flat_tree
from alloy.ir.expr import Expr, as_expr, substitute
from alloy.ir.types import TensorType
from alloy.solvers.model import SolverDescriptor


def build_nlp(
  *,
  x: Expr,
  f: Any,
  p: Expr | Sequence[Expr] | None = None,
  h_eq: Any = None,
  g_ineq: Any = None,
  l_ineq: Any = None,
  u_ineq: Any = None,
  x_lb: Any = None,
  x_ub: Any = None,
  solver: str = "ipopt",
  name: str | None = None,
  options: dict[str, Any] | None = None,
) -> al.Function:
  """Express an old flat NLP test fixture through ProblemSpec."""
  if x.name is None:
    raise ValueError("test decision variable needs a name")
  variables = al.L(x.name, x.type)
  declared_params = () if p is None else ((p,) if isinstance(p, Expr) else tuple(p))
  for param in declared_params:
    if param.name is None:
      raise ValueError("test parameter needs a name")

  def spec(replacements: dict[Expr, Expr]) -> al.ProblemSpec[Expr]:
    def sub(value: Any) -> Expr:
      return substitute(as_expr(value), replacements)

    equalities = () if h_eq is None else (sub(h_eq),)
    inequalities = ()
    if g_ineq is not None:
      inequalities = (
        al.bounded(
          sub(g_ineq),
          None if l_ineq is None else sub(l_ineq),
          None if u_ineq is None else sub(u_ineq),
        ),
      )
    return al.ProblemSpec(
      minimize=sub(f),
      eq=equalities,
      ineq=inequalities,
      lb=None if x_lb is None else sub(x_lb),
      ub=None if x_ub is None else sub(x_ub),
    )

  if not declared_params:

    @al.problem(vars=variables, name=name)
    def problem_body(new_x: Expr) -> al.ProblemSpec[Expr]:
      return spec({x: new_x})

  else:
    param_tree: Tree[Any, Any] = flat_tree(
      cast(tuple[str, ...], tuple(param.name for param in declared_params)),
      tuple(TensorType(param.shape, param.type.dtype, param.type.sparsity, diff=False) for param in declared_params),
    )

    @al.problem(vars=variables, params=param_tree, name=name)
    def problem_body(new_x: Expr, new_params: Any) -> al.ProblemSpec[Expr]:
      replacements = {x: new_x}
      replacements.update(zip(declared_params, param_tree.flatten_symbolic(new_params, "test parameters"), strict=True))
      return spec(replacements)

  return al.solver(problem_body, solver, name=name, options=options)


def solve_nlp(
  solver: al.Function,
  x0: np.ndarray,
  lam_eq: np.ndarray,
  lam_ineq: np.ndarray,
  lam_box: np.ndarray,
  *params: np.ndarray,
) -> dict[str, np.ndarray]:
  """Run a typed one-block solver and expose oracle values for old assertions."""
  descriptor = cast(SolverDescriptor, solver.descriptor)
  if len(params) != len(descriptor.param_names):
    raise ValueError(f"expected {len(descriptor.param_names)} parameters, got {len(params)}")
  param_values: Any = () if not params else params[0] if len(params) == 1 else params
  x, lam_box, lam_eq, lam_ineq = solver.numerical_call((x0, lam_box, lam_eq, lam_ineq, param_values))
  base = descriptor.base
  if isinstance(base, al.Function):
    values = base.numerical_call((np.asarray(x).reshape(-1), param_values))
    if isinstance(values, tuple):
      f, constraints = values
    else:
      f, constraints = values, np.zeros(0)
  else:
    f, constraints = np.array(np.nan), np.zeros(descriptor.n_eq + descriptor.n_ineq)
  constraints = np.asarray(constraints).reshape(-1)
  return {
    "x": x,
    "f": f,
    "h_eq": constraints[: descriptor.n_eq],
    "g_ineq": constraints[descriptor.n_eq :],
    "lam_eq": lam_eq,
    "lam_ineq": lam_ineq,
    "lam_box": lam_box,
  }
