"""``GaussNewton`` and ``LevenbergMarquardt``: nonlinear least squares as generated loops."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..function.method import Support
from ..function.model import ConcreteFunction
from ..function.sugar import while_loop
from ..ir.expr import Expr, concat, isfinite, maximum, minimum, norm_inf, stack, sumsqr, where
from ..linalg.dense import cho_solve, cholesky
from .implicit import Residual, jacobian_at
from .method import METHOD_API, Info, status
from .problem import LeastSquares


def _positive(value: Any, what: str, *, integer: bool = False) -> None:
  if not value > 0 or (integer and int(value) != value):
    raise ValueError(f"{what} must be a positive {'integer' if integer else 'number'}, got {value!r}")


def _stationarity(residual: Residual, z: Expr, params: Sequence[Expr]) -> Expr:
  """``|J^T r|_inf`` at ``z``, what a least-squares solver drives to zero."""
  return norm_inf(jacobian_at(residual, z, params).T @ residual(z, params))


def _loop(name: str, z0: Expr, params: Sequence[Expr], extra: Expr, step: Any, tol: float, max_iter: int) -> tuple[Expr, Expr]:
  """A ``while_loop`` over the carry ``(z, extra..., last step)`` from ``(z0, extra, inf)``: ``step(z,
  extra, params)`` gives the next ``(z, extra, step length)``; it stops once a step moves ``z`` by no
  more than ``tol (1 + |z|_inf)``. Returns the final carry and the iterations."""
  n, k = z0.size, extra.size
  params = list(params)
  carry = Expr.sym("c", (n + k + 1,))
  syms = [Expr.sym(f"q{i}", p.shape, dtype=p.type.dtype) for i, p in enumerate(params)]
  labels = ["c", *(f"q{i}" for i in range(len(syms)))]
  z_next, extra_next, length = step(carry[:n], carry[n : n + k], syms)
  body = ConcreteFunction.from_exprs(f"{name}_step", [carry, *syms], [concat([z_next, extra_next, length.reshape((1,))])], labels, ["c_next"])
  going = carry[n + k] > tol * (1.0 + norm_inf(carry[:n]))
  cond = ConcreteFunction.from_exprs(f"{name}_go", [carry, *syms], [going], labels, ["go"])
  return while_loop(cond, body, concat([z0, extra, Expr.const(np.array([np.inf]))]), max_iter=max_iter, params=params)


@dataclass(frozen=True)
class GaussNewton:
  """Gauss-Newton for ``minimize 1/2 |r(z; p)|^2``: ``z <- z - (J^T J)^{-1} J^T r``, the normal
  equations factored by Cholesky, so ``J`` needs full column rank along the way. It stops once a step
  moves ``z`` by no more than ``tol (1 + |z|_inf)``, at most ``max_iter`` times; ``max_step`` caps
  each step's largest entry. The status is ``OK`` when it stopped by that test. Converges quickly on
  a small-residual problem and may not converge on a large-residual one, where
  ``LevenbergMarquardt`` is the safer choice."""

  name: ClassVar[str] = "roots.gauss_newton"
  problem: ClassVar[type] = LeastSquares
  api: ClassVar[int] = METHOD_API

  tol: float = 1e-10
  max_iter: int = 50
  max_step: float | None = None

  def __post_init__(self) -> None:
    _positive(self.tol, "tol")
    _positive(self.max_iter, "max_iter", integer=True)
    if self.max_step is not None:
      _positive(self.max_step, "max_step")

  def supports(self, problem: Any) -> Support:
    return Support() if isinstance(problem, LeastSquares) else Support((f"{type(problem).__name__} is not a LeastSquares",))

  def build(self, problem: LeastSquares[Any, Any, Any, Any], *, name: str) -> ConcreteFunction[Any, Any, Any, Any]:
    return problem.function(lambda p, z0, params, fname: self.iterate(p.residual, z0, params, name=fname), name=name)

  def iterate(self, residual: Residual, z0: Expr, params: Sequence[Expr], *, name: str) -> tuple[Expr, Info]:
    """The least-squares point from ``z0`` and its ``Info``, as expressions."""

    def step(z: Expr, extra: Expr, held: Sequence[Expr]) -> tuple[Expr, Expr, Expr]:
      j = jacobian_at(residual, z, held)
      delta = -cho_solve(cholesky(j.T @ j), j.T @ residual(z, held))
      if self.max_step is not None:
        delta = delta * minimum(1.0, self.max_step / norm_inf(delta))
      return z + delta, extra, norm_inf(delta)

    out, n_iter = _loop(name, z0, params, Expr.const(np.zeros(0)), step, self.tol, self.max_iter)
    n = z0.size
    z, g = out[:n], _stationarity(residual, out[:n], params)
    return z, Info(status=status(out[n] <= self.tol * (1.0 + norm_inf(z)), z, g), iter=n_iter, residual=g)


@dataclass(frozen=True)
class LevenbergMarquardt:
  """Levenberg-Marquardt for ``minimize 1/2 |r(z; p)|^2``: the step ``-(J^T J + mu I)^{-1} J^T r``,
  taken when it lowers the cost, with Nielsen's update of the damping ``mu`` from the ratio of the
  actual to the predicted decrease (``mu`` starts at ``damping`` times the largest diagonal entry of
  ``J^T J``). It stops once a step, taken or not, would move ``z`` by no more than
  ``tol (1 + |z|_inf)``, at most ``max_iter`` times, and a rejected step raises ``mu`` until one is.
  Works where ``J`` loses rank and from a start far from the solution."""

  name: ClassVar[str] = "roots.levenberg_marquardt"
  problem: ClassVar[type] = LeastSquares
  api: ClassVar[int] = METHOD_API

  tol: float = 1e-10
  max_iter: int = 100
  damping: float = 1e-3

  def __post_init__(self) -> None:
    _positive(self.tol, "tol")
    _positive(self.max_iter, "max_iter", integer=True)
    _positive(self.damping, "damping")

  def supports(self, problem: Any) -> Support:
    return Support() if isinstance(problem, LeastSquares) else Support((f"{type(problem).__name__} is not a LeastSquares",))

  def build(self, problem: LeastSquares[Any, Any, Any, Any], *, name: str) -> ConcreteFunction[Any, Any, Any, Any]:
    return problem.function(lambda p, z0, params, fname: self.iterate(p.residual, z0, params, name=fname), name=name)

  def iterate(self, residual: Residual, z0: Expr, params: Sequence[Expr], *, name: str) -> tuple[Expr, Info]:
    """The least-squares point from ``z0`` and its ``Info``, as expressions."""
    n = z0.size
    eye = Expr.const(np.eye(n))

    def step(z: Expr, extra: Expr, held: Sequence[Expr]) -> tuple[Expr, Expr, Expr]:
      mu, nu = extra[0], extra[1]
      r, j = residual(z, held), jacobian_at(residual, z, held)
      g = j.T @ r
      delta = -cho_solve(cholesky(j.T @ j + mu * eye), g)
      trial = z + delta
      actual = 0.5 * (sumsqr(r) - sumsqr(residual(trial, held)))
      predicted = 0.5 * (delta * (mu * delta - g)).sum()
      rho = actual / predicted
      taken = (rho > 0.0) & isfinite(actual)
      cube = (2.0 * rho - 1.0) * (2.0 * rho - 1.0) * (2.0 * rho - 1.0)
      mu_next = where(taken, mu * maximum(1.0 / 3.0, 1.0 - cube), mu * nu)
      nu_next = where(taken, 2.0, 2.0 * nu)
      return where(taken, trial, z), stack([mu_next, nu_next]), norm_inf(delta)

    j0 = jacobian_at(residual, z0, list(params))
    mu0 = self.damping * maximum(norm_inf(Expr.const(np.ones(j0.shape[0])) @ (j0 * j0)), 1e-300)  # the largest diagonal entry of J^T J
    out, n_iter = _loop(name, z0, params, stack([mu0, Expr.const(2.0)]), step, self.tol, self.max_iter)
    z, g = out[:n], _stationarity(residual, out[:n], params)
    return z, Info(status=status(out[n + 2] <= self.tol * (1.0 + norm_inf(z)), z, g), iter=n_iter, residual=g)


__all__ = ["GaussNewton", "LevenbergMarquardt"]
