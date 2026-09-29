"""``SCvx`` (experimental): sequential convex programming by a penalized trust region, each iteration a QP solved by an ``sc.opt`` method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..function.api import function, jacobian
from ..function.method import Status, Support
from ..function.model import ConcreteFunction
from ..function.sugar import vmap, while_loop
from ..function.tree import G, L, param_list
from ..ir.expr import Expr, cast, concat, less, logical_and, maximum, where
from ..opt.problem import NO_UB, ProblemSpec, bounded
from ..opt.problem import problem as opt_problem
from ..opt.solver import solver as opt_solver
from ..sets import Polytope
from .formulate import step_map
from .method import Info
from .problem import METHOD_API, DiscreteOCP, TerminalEquality
from ..utils.experimental import warn_experimental

warn_experimental(__name__)


@dataclass(frozen=True)
class SCvx:
  """Sequential convex programming with the penalized trust region (PTR) of SCvx and OpenSCvx
  (experimental), for a ``DiscreteOCP`` whose stage is a map, with convex ``Quadratic`` costs at the
  points and hard constraints.

  Each iteration linearizes the map and the path constraints about the reference trajectory
  ``(X, U)`` and solves the convex subproblem with the QP method ``qp``: the problem's costs, plus
  ``prox`` times the squared step from the reference (the trust region, as a penalty), plus
  ``virtual`` times the l1 norm of a virtual control that keeps the linearized dynamics feasible;
  the bounds, the linearized paths and a terminal equality or polytope as constraints. The solution
  is the next reference. It stops when the squared step is below ``tol_trust`` and the virtual
  control's norm below ``tol_virtual``, with ``OK``, or after ``max_iter`` iterations, with
  ``MAX_ITER``; at convergence the step vanishes, so the point is a stationary point of the problem.

  The warm start, and the point a solve returns, is the reference ``(X, U)``: the states over the
  ``N + 1`` knots, then the controls, flat. ``Info.primal_residual`` is the largest defect of the
  nonlinear dynamics at the solution."""

  name: ClassVar[str] = "ocp.scvx"
  problem: ClassVar[type] = DiscreteOCP
  api: ClassVar[int] = METHOD_API
  label: ClassVar[str] = "scvx"

  qp: Any = "ipm"
  prox: float = 1e-3
  virtual: float = 1e4
  tol_trust: float = 1e-12
  tol_virtual: float = 1e-9
  max_iter: int = 100

  def supports(self, problem: Any) -> Support:
    """Whether this method can solve ``problem``: the reasons it cannot, if any."""
    if not isinstance(problem, DiscreteOCP):
      return Support((f"{type(problem).__name__} is not a DiscreteOCP",))
    reasons = []
    if not problem.shooting:
      reasons.append("SCvx takes a discrete map or multiple shooting")
    if problem.cost_rule == "integral" and problem.stage_cost is not None:
      reasons.append("SCvx takes the running cost at the points")
    if problem.stage_quadratic is None or (problem.terminal_cost is not None and problem.terminal_quadratic is None):
      reasons.append("SCvx takes convex Quadratic costs")
    if any(c.soft is not None for c in problem.constraints):
      reasons.append("SCvx takes hard path constraints")
    if problem.terminal is not None and not isinstance(problem.terminal, TerminalEquality | Polytope):
      reasons.append("SCvx takes a terminal equality or a polytope, whose constraints are linear")
    return Support(tuple(reasons))

  def warm_size(self, problem: DiscreteOCP) -> int:
    """The size of the reference trajectory: the states over the ``N + 1`` knots and the controls."""
    return (problem.N + 1) * problem.nx + problem.N * problem.nu

  def build(self, problem: DiscreteOCP, *, name: str) -> ConcreteFunction[Any, Any, Any, Any]:
    """The solver Function: ``(x0, *params, warm) -> (xs, us, point, info)``."""
    p = problem
    nx, nu, n = p.nx, p.nu, p.N
    nX, nU = (n + 1) * nx, n * nu
    step = step_map(p)
    scale = p.dt if p.continuous and p.dt is not None else 1.0
    sizes = [q.type.size for q in p.params]
    varying = [q.name in p.varying for q in p.params]
    psize = max(sum(p.param_size(q) for q in p.params), 1)
    path_rows = [fn.outputs[0].size for fn in p.paths]
    slots = [L(q.name, q.type) for q in p.params]

    def split(pv: Expr) -> list[Expr]:
      out, cut = [], 0
      for q in p.params:
        out.append(pv[cut : cut + p.param_size(q)])
        cut += p.param_size(q)
      return out

    def stage_specs(values: list[Expr]) -> list[tuple[Expr, int, int]]:
      return [(v, 0, s if vary else 0) for v, s, vary in zip(values, sizes, varying, strict=True)]

    def ends(values: list[Expr]) -> list[Expr]:
      return [p.at_end({q.name: v}, q) for q, v in zip(p.params, values, strict=True)]

    # The linearization of one stage about (x, u): the map's value and Jacobians, and each path's.
    def linear_body(x: Expr, u: Expr, *values: Expr) -> Expr:
      nxt = step(x, u, *values)
      parts = [nxt, jacobian(nxt, x).reshape((nx * nx,)), jacobian(nxt, u).reshape((nx * nu,))]
      for fn in p.paths:
        g = fn(x, u, *values)
        parts += [g, jacobian(g, x).reshape((g.size * nx,)), jacobian(g, u).reshape((g.size * nu,))]
      return concat(parts)

    width = nx + nx * nx + nx * nu + sum(r * (1 + nx + nu) for r in path_rows)
    linearize = ConcreteFunction(f"{name}_linearize", linear_body, param_list(L("x", nx), L("u", nu), *slots), L("lin", width))

    def unpack(lin: Expr, k: int) -> tuple[Expr, Expr, Expr, list[tuple[Expr, Expr, Expr]]]:
      row = lin[k * width : (k + 1) * width]
      fk, ak, bk = row[:nx], row[nx : nx + nx * nx].reshape((nx, nx)), row[nx + nx * nx : nx + nx * nx + nx * nu].reshape((nx, nu))
      cut, paths = nx + nx * nx + nx * nu, []
      for r in path_rows:
        g, gx, gu = (
          row[cut : cut + r],
          row[cut + r : cut + r + r * nx].reshape((r, nx)),
          row[cut + r + r * nx : cut + r * (1 + nx + nu)].reshape((r, nu)),
        )
        paths.append((g, gx, gu))
        cut += r * (1 + nx + nu)
      return fk, ak, bk, paths

    x_lo, x_hi = _pair(p.x_bounds, nx)
    u_lo, u_hi = _pair(p.u_bounds, nu)
    lb = (
      Expr.const(np.r_[np.full(nx, -np.inf), np.tile(x_lo, n)]),
      Expr.const(np.tile(u_lo, n)),
      Expr.const(np.zeros(n * nx)),
      Expr.const(np.zeros(n * nx)),
    )
    ub = (Expr.const(np.r_[np.full(nx, np.inf), np.tile(x_hi, n)]), Expr.const(np.tile(u_hi, n)), NO_UB, NO_UB)

    @opt_problem(
      vars=G(L("xs", nX), L("us", nU), L("vp", n * nx), L("vm", n * nx)),
      params=G(L("x0", nx), L("xr", nX), L("ur", nU), L("lin", n * width), L("p", psize)),
      name=f"{name}_subproblem",
    )
    def subproblem(variables: Any, params: Any) -> ProblemSpec[Any]:
      xs, us, vp, vm = variables
      x0, xr, ur, lin, pv = params
      values = split(pv)
      specs = [(xs, 0, nx), (us, 0, nu), *stage_specs(values)]
      assert p.stage_cost is not None
      cost = vmap(p.stage_cost, n, specs).sum() * scale
      if p.terminal_cost is not None:
        cost = cost + p.terminal_cost(xs[n * nx :], *ends(values))
      dx, du = xs - xr, us - ur
      cost = cost + self.prox * ((dx * dx).sum() + (du * du).sum()) + self.virtual * (vp + vm).sum()
      eq, ineq = [xs[:nx] - x0], []
      for k in range(n):
        fk, ak, bk, paths = unpack(lin, k)
        xk, uk = xs[k * nx : (k + 1) * nx], us[k * nu : (k + 1) * nu]
        dxk, duk = xk - xr[k * nx : (k + 1) * nx], uk - ur[k * nu : (k + 1) * nu]
        eq.append(xs[(k + 1) * nx : (k + 2) * nx] - (fk + ak @ dxk + bk @ duk) - (vp[k * nx : (k + 1) * nx] - vm[k * nx : (k + 1) * nx]))
        for (g, gx, gu), spec in zip(paths, p.constraints, strict=True):
          model_g = g + gx @ dxk + gu @ duk
          lo = None if spec.lo is None else Expr.const(np.broadcast_to(np.asarray(spec.lo, dtype=np.float64), (g.size,)).copy())
          hi = None if spec.hi is None else Expr.const(np.broadcast_to(np.asarray(spec.hi, dtype=np.float64), (g.size,)).copy())
          ineq.append(bounded(model_g, lo=lo, hi=hi, name=f"path_{k}"))
      x_end = xs[n * nx :]
      if isinstance(p.terminal, TerminalEquality):
        ref = p.terminal.x_ref
        if isinstance(ref, str):
          value = values[[q.name for q in p.params].index(ref)]
          target = value[n * nx :] if ref in p.varying else value
        else:
          target = Expr.const(np.zeros(nx) if ref is None else np.asarray(ref, dtype=np.float64))
        eq.append(x_end - target)
      elif isinstance(p.terminal, Polytope):
        for i, (g, lo, hi) in enumerate(p.terminal.constraints(x_end)):
          ineq.append(bounded(g, lo=lo, hi=hi, name=f"terminal_set{i}"))
      return ProblemSpec(minimize=cost, eq=tuple(eq), ineq=tuple(ineq), lb=lb, ub=ub)

    qp: Any = opt_solver(subproblem, self.qp, name=f"{name}_qp")  # called with trees of Exprs
    shapes = subproblem.vars.shapes
    zeros = tuple(Expr.const(np.zeros(s)) for s in shapes)
    eq_zero, ineq_zero = Expr.const(np.zeros(subproblem.n_eq)), Expr.const(np.zeros(subproblem.n_ineq))
    tol_trust, tol_virtual = self.tol_trust, self.tol_virtual

    # The carry is the reference (X, U) and the stop flag; the initial state and the parameters are loop parameters.
    @function(name=f"{name}_iteration")
    def iteration(carry: Expr, x0: Expr, pv: Expr) -> Expr:
      values = split(pv)
      xr, ur = carry[:nX], carry[nX : nX + nU]
      lin = vmap(linearize, n, [(xr, 0, nx), (ur, 0, nu), *stage_specs(values)]).reshape((n * width,))
      (xs, us, vp, vm), *_ = qp(zeros, zeros, eq_zero, ineq_zero, (x0, xr, ur, lin, pv))
      dx, du = xs - xr, us - ur
      trust = (dx * dx).sum() + (du * du).sum()
      done = logical_and(less(trust, tol_trust), less((vp + vm).sum(), tol_virtual))
      return concat([xs, us, cast(done, "float64").reshape((1,))])

    @function(name=f"{name}_running")
    def running(carry: Expr, x0: Expr, pv: Expr) -> Expr:
      return less(carry[nX + nU], 0.5)

    def body(x0: Expr, *rest: Expr) -> Any:
      *raw, warm = rest
      flat = [v.reshape((v.size,)) for v in raw]
      pv = concat(flat) if flat else Expr.const(np.zeros(1))
      start = concat([concat([x0, warm[nx:nX]]), warm[nX:], Expr.const(np.zeros(1))])
      carry, count = while_loop(running, iteration, start, max_iter=self.max_iter, params=(x0, pv))
      xs, us = carry[:nX], carry[nX : nX + nU]
      specs = [(xs, 0, nx), (us, 0, nu), *stage_specs(flat)]
      assert p.stage_cost is not None
      cost = vmap(p.stage_cost, n, specs).sum() * scale
      if p.terminal_cost is not None:
        cost = cost + p.terminal_cost(xs[n * nx :], *ends(flat))
      rollout = vmap(step, n, specs).reshape((n * nx,))
      defect = maximum((rollout - xs[nx:]).abs().max(), 0.0)
      status = where(less(carry[nX + nU], 0.5), float(Status.MAX_ITER), float(Status.OK))
      info = Info(status=status, iter=count.cast("float64"), objective=cost, primal_residual=defect)
      return xs.reshape((n + 1, nx)), us.reshape((n, nu)), carry[: nX + nU], info

    size = self.warm_size(p)
    inputs = param_list(L("x0", nx), *(L(q.name, (p.param_size(q),)) for q in p.params), L("warm", size))
    outputs = G(L("xs", (n + 1, nx)), L("us", (n, nu)), L("point", size), Info.tree())
    return ConcreteFunction(name, body, inputs, outputs)

  def shift(self, problem: DiscreteOCP) -> ConcreteFunction[Any, Any, Any, Any]:
    """``point -> warm``: the states and the controls moved up one stage, the last repeated."""
    nx, nu, n = problem.nx, problem.nu, problem.N
    nX, size = (n + 1) * nx, self.warm_size(problem)

    def body(point: Expr) -> Expr:
      xs, us = point[:nX], point[nX:]
      moved_u = concat([us[nu:], us[us.size - nu :]]) if n > 1 else us
      return concat([xs[nx:], xs[nX - nx :], moved_u])

    return ConcreteFunction(f"{problem.name}_shift_scvx", body, param_list(L("point", size)), L("warm", size))

  def initial_guess(self, problem: DiscreteOCP, x0: Any, u: Any = None) -> np.ndarray:
    """A first reference (``sc.ocp.initial_guess``): every state ``x0``, every control ``u`` (zeros by default)."""
    x0 = np.ravel(np.asarray(x0, dtype=np.float64))
    u = np.zeros(problem.nu) if u is None else np.ravel(np.asarray(u, dtype=np.float64))
    return np.concatenate([np.tile(x0, problem.N + 1), np.tile(u, problem.N)])


def _pair(bounds: tuple[Any, Any] | None, size: int) -> tuple[np.ndarray, np.ndarray]:
  lo, hi = (None, None) if bounds is None else bounds
  side = lambda v, fill: np.full(size, fill) if v is None else np.broadcast_to(np.asarray(v, dtype=np.float64), (size,)).copy()  # noqa: E731
  return side(lo, -np.inf), side(hi, np.inf)


__all__ = ["SCvx"]
