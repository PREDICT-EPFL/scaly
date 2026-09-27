"""Test helpers for exercising plugins through the typed problem API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

import scaly as sc
from scaly.function.tree import Tree, flat_tree
from scaly.ir.expr import Expr, as_expr, substitute
from scaly.ir.types import TensorType
from scaly.solvers.model import SolverDescriptor


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
) -> sc.ConcreteFunction:
  """Express an old flat NLP test fixture through ProblemSpec."""
  if x.name is None:
    raise ValueError("test decision variable needs a name")
  variables = sc.L(x.name, x.type)
  declared_params = () if p is None else ((p,) if isinstance(p, Expr) else tuple(p))
  for param in declared_params:
    if param.name is None:
      raise ValueError("test parameter needs a name")

  def spec(replacements: dict[Expr, Expr]) -> sc.ProblemSpec[Expr]:
    def sub(value: Any) -> Expr:
      return substitute(as_expr(value), replacements)

    equalities = () if h_eq is None else (sub(h_eq),)
    inequalities = ()
    if g_ineq is not None:
      inequalities = (
        sc.bounded(
          sub(g_ineq),
          None if l_ineq is None else sub(l_ineq),
          None if u_ineq is None else sub(u_ineq),
        ),
      )
    return sc.ProblemSpec(
      minimize=sub(f),
      eq=equalities,
      ineq=inequalities,
      lb=None if x_lb is None else sub(x_lb),
      ub=None if x_ub is None else sub(x_ub),
    )

  if not declared_params:

    @sc.problem(vars=variables, name=name)
    def problem_body(new_x: Expr) -> sc.ProblemSpec[Expr]:
      return spec({x: new_x})

  else:
    param_tree: Tree[Any, Any] = flat_tree(
      cast(tuple[str, ...], tuple(param.name for param in declared_params)),
      tuple(TensorType(param.shape, param.type.dtype, param.type.sparsity, diff=False) for param in declared_params),
    )

    @sc.problem(vars=variables, params=param_tree, name=name)
    def problem_body(new_x: Expr, new_params: Any) -> sc.ProblemSpec[Expr]:
      replacements = {x: new_x}
      replacements.update(zip(declared_params, param_tree.flatten_symbolic(new_params, "test parameters"), strict=True))
      return spec(replacements)

  return sc.solver(problem_body, solver, name=name, options=options)


def build_qp(
  *,
  P: Any,
  c: Any,
  A_eq: Any = None,
  b_eq: Any = None,
  G_ineq: Any = None,
  l_ineq: Any = None,
  u_ineq: Any = None,
  x_lb: Any = None,
  x_ub: Any = None,
  solver: str = "piqp",
  name: str | None = None,
  options: dict[str, Any] | None = None,
  sparse: bool = False,
) -> sc.ConcreteFunction:
  """Express an old matrix-form QP test fixture through ProblemSpec."""
  P_expr, c_expr = as_expr(P), as_expr(c)
  if len(P_expr.shape) != 2 or P_expr.shape[0] != P_expr.shape[1]:
    raise ValueError(f"P must be square 2D, got shape {P_expr.shape}")
  n = P_expr.shape[0]
  if c_expr.shape != (n,):
    raise ValueError(f"c must have shape ({n},), got {c_expr.shape}")
  variable = sc.L("decision", n)

  @sc.problem(vars=variable, name=name)
  def problem_body(x: Expr) -> sc.ProblemSpec[Expr]:
    equalities = () if A_eq is None else (as_expr(A_eq) @ x - as_expr(b_eq),)
    inequalities = ()
    if G_ineq is not None:
      inequalities = (sc.bounded(as_expr(G_ineq) @ x, l_ineq, u_ineq),)
    return sc.ProblemSpec(
      minimize=0.5 * (x @ P_expr @ x) + c_expr @ x,
      eq=equalities,
      ineq=inequalities,
      lb=None if x_lb is None else as_expr(x_lb),
      ub=None if x_ub is None else as_expr(x_ub),
    )

  return sc.solver(problem_body, solver, name=name, options={"sparse": sparse, **(options or {})})


def solve_qp(
  solver: sc.Function,
  x0: np.ndarray,
  lam_eq0: np.ndarray,
  lam_ineq0: np.ndarray,
  *params: np.ndarray,
  **named_params: np.ndarray,
) -> dict[str, np.ndarray]:
  """Run a typed one-block QP and expose the retired matrix-builder result names."""
  descriptor = cast(SolverDescriptor, solver.descriptor)
  if params and named_params:
    raise TypeError("pass positional or named QP parameters, not both")
  values = params or tuple(named_params[name] for name in descriptor.param_names)
  outputs = solver(*solver.input_tree.unflatten((x0, np.zeros_like(x0), lam_eq0, lam_ineq0, *values)))
  result = dict(zip(solver.output_names, outputs, strict=True))
  result["x"] = outputs[0]
  result["lam_box"] = outputs[1]
  result["cost"] = np.asarray(solver.solver_stats().obj)
  return result


def solve_nlp(
  solver: sc.Function,
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
  x, lam_box, lam_eq, lam_ineq = solver.numerical_call(x0, lam_box, lam_eq, lam_ineq, param_values)
  base = descriptor.base
  if isinstance(base, sc.Function):
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
