"""``Direct``: an OCP formulated as an optimization problem (``to_problem``) and solved by any ``sc.opt`` method, with the warm start shifted stage by stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..function.method import Support
from ..function.model import ConcreteFunction
from ..function.tree import G, L, param_list
from ..ir.expr import Expr, concat
from ..opt.solver import solver as opt_solver
from .formulate import Form, Layout, to_problem
from .method import Info
from .problem import METHOD_API, DiscreteOCP
from .warmstart import shifted, split


@dataclass(frozen=True)
class Direct:
  """The direct method: ``to_problem(ocp, form)`` solved by ``method``, an ``sc.opt`` method with its
  options (``sc.opt.PIQP(sparse=True)`` for a linear OCP's banded QP, ``sc.opt.IPOPT()``, ...), its
  name, or ``"auto"``. ``form="condensed"`` eliminates the states (``to_problem``). The warm start
  is the whole primal-dual point of the problem, flat: the variables, their bounds' multipliers,
  the equality and the inequality multipliers."""

  name: ClassVar[str] = "ocp.direct"
  problem: ClassVar[type] = DiscreteOCP
  api: ClassVar[int] = METHOD_API

  method: Any = "auto"
  form: Form = "sparse"

  def __post_init__(self) -> None:
    if self.form not in ("sparse", "condensed"):
      raise ValueError(f"form is 'sparse' or 'condensed', got {self.form!r}")

  @property
  def label(self) -> str:
    """How the method is spelled in the names of its Functions."""
    method = self.method if isinstance(self.method, str) else self.method.name.removeprefix("opt.")
    return method if self.form == "sparse" else f"{method}_condensed"

  def supports(self, problem: Any) -> Support:
    """Whether this method can solve ``problem``: the reasons it cannot, if any."""
    if not isinstance(problem, DiscreteOCP):
      return Support((f"{type(problem).__name__} is not a DiscreteOCP",))
    if self.form == "condensed" and (not problem.shooting or problem.cost_rule == "integral"):
      return Support(("the condensed form takes a discrete map, or multiple shooting with costs at the points",))
    return Support()

  def layout(self, problem: DiscreteOCP) -> Layout:
    """The layout of the formulated problem, whose point the solver takes and returns."""
    return to_problem(problem, self.form)[1]

  def warm_size(self, problem: DiscreteOCP) -> int:
    """The size of the flat primal-dual point."""
    nlp, layout = to_problem(problem, self.form)
    return 2 * layout.n_vars + nlp.n_eq + nlp.n_ineq

  def build(self, problem: DiscreteOCP, *, name: str) -> ConcreteFunction[Any, Any, Any, Any]:
    """The solver Function: ``(x0, *params, warm) -> (xs, us, point, info)``."""
    nlp, layout = to_problem(problem, self.form)
    solve = opt_solver(nlp, self.method, name=f"{name}_solve")
    size, n_eq = self.warm_size(problem), nlp.n_eq
    nx, nu, n = problem.nx, problem.nu, problem.N

    def body(x0: Expr, *rest: Expr) -> Any:
      *values, warm = rest
      primal, box, lam_eq, lam_ineq = split(layout, n_eq, warm)
      prm = (x0, concat([v.reshape((v.size,)) for v in values])) if problem.params else x0
      out = solve(_tree(primal), _tree(box), lam_eq, lam_ineq, prm)
      p, b = _leaves(out[0]), _leaves(out[1])
      point = concat([*p, *b, *([out[2]] if n_eq else []), *([out[3]] if nlp.n_ineq else [])])
      leaves = dict(zip(layout.var_names, p, strict=True))
      us = leaves["us"]
      xs = leaves["xs"] if layout.states is None else layout.states(x0, us, *values)
      opt_info = out[4]
      info = Info(status=opt_info.status, iter=opt_info.iter, objective=opt_info.objective, primal_residual=opt_info.primal_residual)
      return xs.reshape((n + 1, nx)), us.reshape((n, nu)), point, info

    inputs = param_list(L("x0", nx), *(L(p.name, (problem.param_size(p),)) for p in problem.params), L("warm", size))
    outputs = G(L("xs", (n + 1, nx)), L("us", (n, nu)), L("point", size), Info.tree())
    return ConcreteFunction(name, body, inputs, outputs)

  def shift(self, problem: DiscreteOCP) -> ConcreteFunction[Any, Any, Any, Any]:
    """``point -> warm``, the point moved up one stage (``sc.ocp.shift``)."""
    nlp, layout = to_problem(problem, self.form)
    size = self.warm_size(problem)

    def body(point: Expr) -> Expr:
      return shifted(problem, layout, nlp.n_eq, nlp.n_ineq, point)

    suffix = "" if self.form == "sparse" else "_condensed"
    return ConcreteFunction(f"{problem.name}_shift{suffix}", body, param_list(L("point", size)), L("warm", size))

  def initial_guess(self, problem: DiscreteOCP, x0: Any, u: Any = None) -> np.ndarray:
    """A first warm start (``sc.ocp.initial_guess``)."""
    layout = self.layout(problem)
    x0 = np.ravel(np.asarray(x0, dtype=np.float64))
    u = np.zeros(problem.nu) if u is None else np.ravel(np.asarray(u, dtype=np.float64))
    values = {"xs": np.tile(x0, problem.N + 1), "us": np.tile(u, problem.N), "zs": np.tile(problem.interval.guess(x0, u), problem.N)}
    primal = [values.get(name, np.zeros(size)) for name, size in zip(layout.var_names, layout.var_sizes, strict=True)]
    return np.concatenate([*primal, np.zeros(self.warm_size(problem) - layout.n_vars)])


def _tree(leaves: list[Any]) -> Any:
  return leaves[0] if len(leaves) == 1 else tuple(leaves)


def _leaves(tree: Any) -> list[Any]:
  return list(tree) if isinstance(tree, tuple) else [tree]


__all__ = ["Direct"]
