"""Calibrating a hanging chain from a photograph: Newton equilibrium with an implicit derivative.

A chain of ``SEGMENTS`` springs hangs between two fixed hooks. Its shape minimizes the potential
energy

    E(p; k, m, w) = sum_i k/2 (|p_{i+1} - p_i| - rest)^2 + m g sum_i y_i + w g y_middle

over the free node positions ``p``, for spring stiffness ``k``, node mass ``m`` and an extra weight
``w`` hung from the middle node. The equilibrium ``p*(k, m, w)`` is found by Newton's method in a ``sc.while_loop``: the gradient and Hessian of ``E``
come from Scaly's AD inside the loop body, the step from the generated dense ``cholesky`` and
``cho_solve``. The step is safeguarded: if the factorization breaks down (a Hessian that is not
positive definite gives a non-finite step, which ``sc.isfinite`` detects) the body takes a short
gradient step instead, and every step is capped in length with ``sc.minimum``.

Two photographs, one without and one with a known 50 g weight at the middle node, give the
positions of a few nodes, with noise. (One photograph would not do: without the weight the shape
depends on ``m / k`` only.) Fitting ``theta = (log k, log m)`` to them means differentiating ``p*`` in ``theta``, and differentiating through the Newton iterations
would be both costly and wrong in the early ones. ``sc.custom_derivative`` attaches the
implicit-function rule instead: at the equilibrium ``grad_p E(p*, theta) = 0``, so

    dp*/d(theta, w) = -H^{-1} d(grad_p E)/d(theta, w),

one more solve with the Hessian. The fit is L-BFGS on the gradient that rule gives.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

import scaly as sc
from scaly import linalg

SEGMENTS, SPAN, REST, GRAVITY = 10, 1.2, 0.15, 9.81
FREE = SEGMENTS - 1
NP = 2 * FREE
TOL, MAX_NEWTON, MAX_STEP = 1e-11, 100, 0.05
OBSERVED = np.array([2, 4, 6])  # free nodes seen in the photographs
MIDDLE, WEIGHT = FREE // 2, 0.05


def energy(p: sc.Expr, params: sc.Expr) -> sc.Expr:
  """``params = (log k, log m, w)``."""
  k, m, w = params[0].exp(), params[1].exp(), params[2]
  nodes = sc.concat([sc.const(np.zeros((1, 2))), p.reshape((FREE, 2)), sc.const(np.array([[SPAN, 0.0]]))])
  d = nodes[1:] - nodes[:-1]
  stretch = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]).sqrt() - REST
  heights = p.reshape((FREE, 2))[:, 1]
  return 0.5 * k * sc.sumsqr(stretch) + m * GRAVITY * heights.sum() + w * GRAVITY * heights[MIDDLE]


def initial_guess() -> np.ndarray:
  s = np.linspace(0.0, 1.0, SEGMENTS + 1)[1:-1]
  return np.stack([SPAN * s, -2.4 * s * (1 - s)], axis=1).reshape(-1)  # sagging 0.6, every spring stretched


# A loop body reads only its carry, so the parameters travel in it: carry = [p (NP) | params (3) | |grad E|].
_carry = sc.sym("carry", NP + 4)
_p, _theta = _carry[:NP], _carry[NP : NP + 3]
_grad = sc.gradient(energy(_p, _theta), _p)
_newton = -linalg.cho_solve(linalg.cholesky(sc.hessian(energy(_p, _theta), _p)), _grad)
_step = sc.where(sc.isfinite(sc.norm_inf(_newton)), _newton, -0.01 * _grad)
_step = _step * sc.minimum(1.0, MAX_STEP / sc.maximum(sc.norm_inf(_step), 1e-300))
newton_iteration = sc.Function._from_exprs(
  "chain_newton", [_carry], [sc.concat([_p + _step, _theta, sc.norm_inf(_grad).reshape((1,))])], ["carry"], ["next"]
)
not_converged = sc.Function._from_exprs("chain_not_converged", [_carry], [sc.greater(_carry[NP + 3], TOL)], ["carry"], ["go_on"])


def _equilibrium() -> sc.Function:
  params = sc.sym("params", 3)
  start = sc.concat([sc.const(initial_guess()), params, sc.const(np.ones(1))])
  carry, _ = sc.while_loop(not_converged, newton_iteration, start, max_iter=MAX_NEWTON)
  return sc.Function._from_exprs("chain_equilibrium", [params], [carry[:NP]], ["params"], ["p"])


def _implicit_vjp() -> sc.Function:
  """``(params, p*, p_bar) -> params_bar = -(d grad_p E / d params)^T H^{-1} p_bar``."""
  params, p, p_bar = sc.sym("params", 3), sc.sym("p", NP), sc.sym("p_bar", NP)
  grad = sc.gradient(energy(p, params), p)
  w = linalg.cho_solve(linalg.cholesky(sc.jacobian(grad, p)), p_bar)
  return sc.Function._from_exprs(
    "chain_equilibrium_vjp", [params, p, p_bar], [-(sc.jacobian(grad, params).T @ w)], ["params", "p", "p_bar"], ["params_bar"]
  )


equilibrium = sc.custom_derivative(_equilibrium(), vjp=_implicit_vjp())


@sc.function(sc.G(sc.L("theta", 2), sc.L("photos", (2, len(OBSERVED), 2))), sc.G(sc.L("loss", ()), sc.L("gradient", 2)))
def misfit(inputs: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr]:
  theta, photos = inputs
  loss = sc.const(0.0)
  for photo, weight in enumerate((0.0, WEIGHT)):
    p = equilibrium(sc.concat([theta, sc.const(np.array([weight]))])).reshape((FREE, 2))
    seen = sc.stack([p[int(i)] for i in OBSERVED])
    loss = loss + 0.5 * sc.sumsqr(seen - photos[photo])
  return loss, sc.gradient(loss, theta)


def shape(theta: np.ndarray, weight: float = 0.0) -> np.ndarray:
  """The equilibrium free-node positions, ``(FREE, 2)``."""
  return equilibrium(np.r_[np.asarray(theta, dtype=float), weight]).reshape(FREE, 2)


def main(seed: int = 0, noise: float = 2e-3) -> dict[str, np.ndarray]:
  truth = np.log([60.0, 0.03])
  rng = np.random.default_rng(seed)
  photo = np.stack([shape(truth, w)[OBSERVED] for w in (0.0, WEIGHT)]) + noise * rng.standard_normal((2, len(OBSERVED), 2))
  start = np.log([10.0, 0.2])
  result = optimize.minimize(lambda th: tuple(misfit((th, photo))), start, jac=True, method="L-BFGS-B")
  return {"truth": truth, "photo": photo, "start": start, "theta": result.x, "loss": np.array(result.fun), "evaluations": np.array(result.nfev)}


if __name__ == "__main__":
  out = main()
  k0, m0 = np.exp(out["start"])
  k, m = np.exp(out["theta"])
  kt, mt = np.exp(out["truth"])
  print(f"start k = {k0:.1f}, m = {m0:.3f}; fitted k = {k:.2f}, m = {m:.4f}; true k = {kt:.1f}, m = {mt:.3f}")
  print(f"{int(out['evaluations'])} equilibrium solves with gradients; residual {np.sqrt(2 * float(out['loss']) / out['photo'].size):.1e} per coordinate")
  print(f"lowest point of the fitted chain: {shape(out['theta'])[:, 1].min():.3f} m")
