"""``ILQR``: iterative LQR on an unconstrained ``DiscreteOCP``, the whole solve generated as loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..function.method import Status, Support
from ..function.model import ConcreteFunction
from ..function.api import function, gradient, hessian, jacobian
from ..function.sugar import scan, while_loop
from ..function.tree import G, L, param_list
from ..ir.expr import Expr, concat, greater, less, where
from ..linalg import cho_solve, cholesky
from .formulate import step_map
from .method import Info
from .problem import METHOD_API, DiscreteOCP


@dataclass(frozen=True)
class ILQR:
  """Iterative LQR (Li and Todorov 2004) with the regularization and line search of Tassa et al.
  (2012), for a ``DiscreteOCP`` whose stage is a map (a discrete-time model, or multiple shooting),
  with costs at the points and no constraints, bounds or terminal set.

  Each iteration rolls the map out from ``x0`` under the controls, runs the backward Riccati pass on
  local models from AD (the map's Jacobians, the costs' gradients and Hessians), and searches along
  the feedback policy ``u = u_k + alpha k_k + K_k (x - x_k)``, halving ``alpha`` up to
  ``max_line_search`` times until the cost falls by ``armijo`` of the predicted change. The
  Levenberg-Marquardt term ``mu I`` on ``Q_uu`` falls tenfold (to ``mu_min``) after an accepted step
  and grows tenfold after a rejected one; a factorization that fails gives NaN gains, so no step, so
  a larger ``mu``. The solve stops with ``OK`` when an accepted step improves the cost by less than
  ``tol``, relative, or the backward pass predicts a smaller decrease than that; with ``NUMERICS``
  when ``mu`` passes ``mu_max``; and with ``MAX_ITER`` after ``max_iter`` iterations.

  The warm start, and the point a solve returns, is the controls ``u_0 .. u_{N-1}``, flat."""

  name: ClassVar[str] = "ocp.ilqr"
  problem: ClassVar[type] = DiscreteOCP
  api: ClassVar[int] = METHOD_API
  label: ClassVar[str] = "ilqr"

  max_iter: int = 200
  tol: float = 1e-10
  max_line_search: int = 12
  armijo: float = 1e-4
  mu: float = 1.0
  mu_min: float = 1e-8
  mu_max: float = 1e10

  def __post_init__(self) -> None:
    if self.max_iter < 1 or self.max_line_search < 1:
      raise ValueError("max_iter and max_line_search must be positive")
    if not 0 < self.mu_min <= self.mu <= self.mu_max:
      raise ValueError("the regularization needs 0 < mu_min <= mu <= mu_max")

  def supports(self, problem: Any) -> Support:
    """Whether this method can solve ``problem``: the reasons it cannot, if any."""
    if not isinstance(problem, DiscreteOCP):
      return Support((f"{type(problem).__name__} is not a DiscreteOCP",))
    reasons = []
    if not problem.shooting:
      reasons.append("iLQR takes a discrete map or multiple shooting, not a transcription with variables of its own")
    if problem.cost_rule == "integral" and problem.stage_cost is not None:
      reasons.append("iLQR takes the running cost at the points, not integrated")
    if problem.constraints:
      reasons.append("iLQR takes no path constraints")
    if _bounded(problem.x_bounds) or _bounded(problem.u_bounds):
      reasons.append("iLQR takes no bounds")
    if problem.terminal is not None:
      reasons.append("iLQR takes no terminal set or equality")
    return Support(tuple(reasons))

  def warm_size(self, problem: DiscreteOCP) -> int:
    """The size of the warm start: the controls, flat."""
    return problem.N * problem.nu

  def build(self, problem: DiscreteOCP, *, name: str) -> ConcreteFunction[Any, Any, Any, Any]:
    """The solver Function: ``(x0, *params, warm) -> (xs, us, point, info)``."""
    nx, nu, n = problem.nx, problem.nu, problem.N
    params = problem.params
    sizes = [p.type.size for p in params]
    flat_sizes = [problem.param_size(p) for p in params]
    varying = [p.name in problem.varying for p in params]
    step, stage_fn, terminal_fn = step_map(problem), problem.stage_cost, problem.terminal_cost
    scale = problem.dt if problem.continuous and problem.dt is not None else 1.0
    n_gain = nu + nu * nx
    slots = [L(p.name, p.type) for p in params]

    def stage(x: Expr, u: Expr, values: tuple[Expr, ...]) -> Expr:
      return stage_fn(x, u, *values) * scale if stage_fn is not None else Expr.const(0.0) * x[0]

    def terminal(x: Expr, values: list[Expr]) -> Expr:
      ends = [v[n * s :] if vary else v for v, s, vary in zip(values, sizes, varying, strict=True)]
      shaped = [e.reshape(p.type.shape) for e, p in zip(ends, params, strict=True)]
      return terminal_fn(x, *shaped) if terminal_fn is not None else Expr.const(0.0) * x[0]

    def forward_specs(values: list[Expr]) -> list[tuple[Expr, int, int]]:
      return [(v, 0, s if vary else 0) for v, s, vary in zip(values, sizes, varying, strict=True)]

    def backward_specs(values: list[Expr]) -> list[tuple[Expr, int, int]]:
      return [(v, (n - 1) * s, -s) if vary else (v, 0, 0) for v, s, vary in zip(values, sizes, varying, strict=True)]

    def split(pv: Expr) -> list[Expr]:
      out, cut = [], 0
      for size in flat_sizes:
        out.append(pv[cut : cut + size])
        cut += size
      return out

    # Rollout under the controls: the next state, the state stacked, the stage cost stacked.
    def rollout_body(x: Expr, u: Expr, *values: Expr) -> tuple[Expr, Expr, Expr]:
      return step(x, u, *values), x, stage(x, u, values).reshape((1,))

    rollout_step = ConcreteFunction(
      f"{name}_rollout", rollout_body, param_list(L("x", nx), L("u", nu), *slots), G(L("xnext", nx), L("state", nx), L("cost", 1))
    )

    def rollout(x0: Expr, us: Expr, values: list[Expr]) -> tuple[Expr, Expr, Expr]:
      x_final, xs, costs = scan(rollout_step, x0, [(us, 0, nu), *forward_specs(values)], length=n)
      return x_final, xs, costs.sum() + terminal(x_final, values)

    # Backward pass. Carry: V_x and V_xx; sliced backwards: x_k, u_k and a varying parameter's value.
    def backward_body(value: Expr, x: Expr, u: Expr, mu: Expr, *values: Expr) -> tuple[Expr, Expr, Expr]:
      vx, vxx = value[:nx], value[nx:].reshape((nx, nx))
      x_next, cost = step(x, u, *values), stage(x, u, values)
      fx, fu = jacobian(x_next, x), jacobian(x_next, u)
      lx, lu = gradient(cost, x), gradient(cost, u)
      qx = lx + fx.T @ vx
      qu = lu + fu.T @ vx
      qxx = hessian(cost, x) + fx.T @ vxx @ fx
      quu = hessian(cost, u) + fu.T @ vxx @ fu
      qux = jacobian(lu, x) + fu.T @ vxx @ fx
      chol = cholesky(quu + Expr.const(np.eye(nu)) * mu[0])
      k = -cho_solve(chol, qu)
      gain = -cho_solve(chol, qux)
      vx_prev = qx + gain.T @ (quu @ k) + gain.T @ qu + qux.T @ k
      vxx_prev = qxx + gain.T @ quu @ gain + gain.T @ qux + qux.T @ gain
      vxx_prev = 0.5 * (vxx_prev + vxx_prev.T)
      dv = concat([(k @ qu).reshape((1,)), (0.5 * (k @ (quu @ k))).reshape((1,))])
      return concat([vx_prev, vxx_prev.reshape((nx * nx,))]), concat([k, gain.reshape((nu * nx,))]), dv

    backward_step = ConcreteFunction(
      f"{name}_backward",
      backward_body,
      param_list(L("value", nx + nx * nx), L("x", nx), L("u", nu), L("mu", 1), *slots),
      G(L("value_prev", nx + nx * nx), L("gains", n_gain), L("dv", 2)),
    )

    # Forward pass under the policy; the gains come out of the backward pass last stage first.
    def policy_body(x: Expr, x_nom: Expr, u_nom: Expr, gains: Expr, alpha: Expr, *values: Expr) -> tuple[Expr, Expr, Expr]:
      u = u_nom + alpha[0] * gains[:nu] + gains[nu:].reshape((nu, nx)) @ (x - x_nom)
      return step(x, u, *values), u, stage(x, u, values).reshape((1,))

    policy_step = ConcreteFunction(
      f"{name}_policy",
      policy_body,
      param_list(L("x", nx), L("x_nom", nx), L("u_nom", nu), L("gains", n_gain), L("alpha", 1), *slots),
      G(L("xnext", nx), L("u", nu), L("cost", 1)),
    )

    armijo, tol, mu_min, mu_max = self.armijo, self.tol, self.mu_min, self.mu_max

    # The line search's carry is (accepted, cost, alpha, controls); the rest are loop parameters.
    @function(name=f"{name}_line_search")
    def line_search(carry: Expr, trial: Expr, x0: Expr, pv: Expr, xs: Expr, us: Expr, gains: Expr, j_dv: Expr) -> Expr:
      values = split(pv)
      alpha = Expr.const(0.5) ** trial.cast("float64")
      specs = [(xs, 0, nx), (us, 0, nu), (gains, (n - 1) * n_gain, -n_gain), (alpha.reshape((1,)), 0, 0), *forward_specs(values)]
      x_final, u_new, costs = scan(policy_step, x0, specs, length=n)
      cost = costs.sum() + terminal(x_final, values)
      expected = alpha * j_dv[1] + alpha * alpha * j_dv[2]  # negative: the predicted change
      accepted = less(cost - j_dv[0], armijo * expected)
      return concat([accepted.cast("float64").reshape((1,)), cost.reshape((1,)), alpha.reshape((1,)), u_new])

    @function(name=f"{name}_not_accepted")
    def not_accepted(carry: Expr, x0: Expr, pv: Expr, xs: Expr, us: Expr, gains: Expr, j_dv: Expr) -> Expr:
      return less(carry[0], 0.5)

    # The outer carry is (controls, cost, mu, stop): stop 0 to go on, 1 converged, 2 mu past its limit.
    @function(name=f"{name}_iteration")
    def iteration(carry: Expr, x0: Expr, pv: Expr) -> Expr:
      values = split(pv)
      us, mu = carry[: n * nu], carry[n * nu + 1]
      x_final, xs, cost = rollout(x0, us, values)
      end = terminal(x_final, values)
      value_n = concat([gradient(end, x_final), hessian(end, x_final).reshape((nx * nx,))])
      specs = [(xs, (n - 1) * nx, -nx), (us, (n - 1) * nu, -nu), (mu.reshape((1,)), 0, 0), *backward_specs(values)]
      _, gains, dv = scan(backward_step, value_n, specs, length=n)
      dv = dv.reshape((n, 2))
      j_dv = concat([cost.reshape((1,)), dv[:, 0].sum().reshape((1,)), dv[:, 1].sum().reshape((1,))])
      start = concat([Expr.const(np.zeros(3)), us])
      searched, _ = while_loop(not_accepted, line_search, start, max_iter=self.max_line_search, index=True, params=(x0, pv, xs, us, gains, j_dv))
      accepted = greater(searched[0], 0.5)
      us_next = where(accepted, searched[3:], us)
      cost_next = where(accepted, searched[1], cost)
      mu_next = where(accepted, (mu * 0.1).maximum(mu_min), mu * 10.0)
      scale = cost.abs().maximum(1.0)
      # Converged: an accepted step that improves little, or a model that predicts no decrease at
      # all, where the line search would only compare rounding errors.
      converged = (accepted & less((cost - cost_next) / scale, tol)) | less(-(j_dv[1] + j_dv[2]), tol * scale)
      stop = where(converged, 1.0, where(greater(mu_next, mu_max), 2.0, 0.0))
      return concat([us_next, cost_next.reshape((1,)), mu_next.reshape((1,)), stop.reshape((1,))])

    @function(name=f"{name}_not_done")
    def not_done(carry: Expr, x0: Expr, pv: Expr) -> Expr:
      return less(carry[n * nu + 2], 0.5)

    def body(x0: Expr, *rest: Expr) -> Any:
      *values, warm = rest
      flat = [v.reshape((v.size,)) for v in values]
      pv = concat(flat) if flat else Expr.const(np.zeros(1))
      start = concat([warm, Expr.const(np.array([np.inf, self.mu, 0.0]))])
      carry, n_iter = while_loop(not_done, iteration, start, max_iter=self.max_iter, params=(x0, pv))
      us = carry[: n * nu]
      x_final, xs, cost = rollout(x0, us, flat)
      stop = carry[n * nu + 2]
      status = where(greater(stop, 1.5), float(Status.NUMERICS), where(greater(stop, 0.5), float(Status.OK), float(Status.MAX_ITER)))
      info = Info(status=status, iter=n_iter.cast("float64"), objective=cost, primal_residual=Expr.const(0.0))
      return concat([xs, x_final]).reshape((n + 1, nx)), us.reshape((n, nu)), us, info

    size = self.warm_size(problem)
    inputs = param_list(L("x0", nx), *(L(p.name, (problem.param_size(p),)) for p in params), L("warm", size))
    outputs = G(L("xs", (n + 1, nx)), L("us", (n, nu)), L("point", size), Info.tree())
    return ConcreteFunction(name, body, inputs, outputs)

  def shift(self, problem: DiscreteOCP) -> ConcreteFunction[Any, Any, Any, Any]:
    """``point -> warm``: the controls moved up one stage, the last repeated (``sc.ocp.shift``)."""
    nu, size = problem.nu, self.warm_size(problem)

    def body(point: Expr) -> Expr:
      return concat([point[nu:], point[size - nu :]]) if problem.N > 1 else point

    return ConcreteFunction(f"{problem.name}_shift_ilqr", body, param_list(L("point", size)), L("warm", size))

  def initial_guess(self, problem: DiscreteOCP, x0: Any, u: Any = None) -> np.ndarray:
    """A first warm start (``sc.ocp.initial_guess``): every control ``u``, zeros by default."""
    u = np.zeros(problem.nu) if u is None else np.ravel(np.asarray(u, dtype=np.float64))
    return np.tile(u, problem.N)


def _bounded(bounds: tuple[Any, Any] | None) -> bool:
  """Whether ``(lo, hi)`` bounds anything: a finite side."""
  if bounds is None:
    return False
  return any(side is not None and np.isfinite(np.asarray(side, dtype=np.float64)).any() for side in bounds)


__all__ = ["ILQR"]
