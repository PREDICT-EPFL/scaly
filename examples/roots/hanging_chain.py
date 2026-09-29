# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly"]
# ///
"""Calibrating a hanging chain from a photograph: Newton equilibrium with an implicit derivative.

A chain of ``SEGMENTS`` springs hangs between two fixed hooks. Its shape minimizes the potential
energy

    E(p; k, m, w) = sum_i k/2 (|p_{i+1} - p_i| - rest)^2 + m g sum_i y_i + w g y_middle

over the free node positions ``p``, for spring stiffness ``k``, node mass ``m`` and an extra weight
``w`` hung from the middle node. The equilibrium ``p*(k, m, w)`` is where ``grad_p E = 0``, an
``sc.roots.root`` solved by ``sc.roots.Newton``: the gradient and Hessian of ``E`` come from Scaly's
AD, the step from the generated dense LU, every step capped in length (``max_step``). At the initial
guess the middle springs are compressed and the Hessian is indefinite, so a Cholesky step would fail
there; the capped Newton steps carry the chain to the minimum, where it is positive definite.

Two photographs, one without and one with a known 50 g weight at the middle node, give the
positions of a few nodes, with noise. (One photograph would not do: without the weight the shape
depends on ``m / k`` only.) Fitting ``theta = (log k, log m)`` to them means differentiating ``p*`` in ``theta``, and differentiating through the Newton iterations
would be both costly and wrong in the early ones. A roots solver's solution carries the
implicit-function derivative instead: at the equilibrium ``grad_p E(p*, theta) = 0``, so

    dp*/d(theta, w) = -H^{-1} d(grad_p E)/d(theta, w),

one more solve with the Hessian. The fit is L-BFGS on the gradient that rule gives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import optimize

import scaly as sc
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parents[1] / "generated" / "hanging_chain"

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
  return np.stack([SPAN * s, -2.4 * s * (1 - s)], axis=1).reshape(-1)  # sagging 0.6


@sc.roots.root(vars=sc.L("p", NP), params=sc.L("params", 3), name="chain")
def stationary(p: sc.Expr, params: sc.Expr) -> sc.Expr:
  return sc.gradient(energy(p, params), p)


_solve = sc.roots.solver(stationary, sc.roots.Newton(tol=TOL, max_iter=MAX_NEWTON, max_step=MAX_STEP), name="chain_newton")


@sc.function(3, output="p")
def equilibrium(params: sc.Expr) -> sc.Expr:
  p, _ = _solve(sc.const(initial_guess()), params)
  return p


@sc.function(2, (2, len(OBSERVED), 2), output=sc.G("loss", "gradient"))
def misfit(theta: sc.Expr, photos: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
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
  result = optimize.minimize(lambda th: tuple(misfit(th, photo)), start, jac=True, method="L-BFGS-B")
  return {"truth": truth, "photo": photo, "start": start, "theta": result.x, "loss": np.array(result.fun), "evaluations": np.array(result.nfev)}


if __name__ == "__main__":
  out = main()
  k0, m0 = np.exp(out["start"])
  k, m = np.exp(out["theta"])
  kt, mt = np.exp(out["truth"])
  print(f"start k = {k0:.1f}, m = {m0:.3f}; fitted k = {k:.2f}, m = {m:.4f}; true k = {kt:.1f}, m = {mt:.3f}")
  print(
    f"{int(out['evaluations'])} equilibrium solves with gradients; residual {np.sqrt(2 * float(out['loss']) / out['photo'].size):.1e} per coordinate"
  )
  print(f"lowest point of the fitted chain: {shape(out['theta'])[:, 1].min():.3f} m")
  write_module(misfit, GENERATED)
  print(f"generated C in {GENERATED}")
