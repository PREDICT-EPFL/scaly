"""The Van der Pol limit cycle by shooting: Newton on a flow map, differentiated through a scan.

The oscillator ``x'' - mu (1 - x^2) x' + x = 0``, as ``x = (x1, x2)``, has one stable periodic orbit.
Single shooting finds it as a root of

    F(a, T) = phi_T((a, 0); mu) - (a, 0) = 0,

the flow over one period ``T`` from a point on the axis ``x2 = 0`` (the phase condition), with two
unknowns ``(a, T)``, solved by Newton steps capped in length. The flow map is ``K`` classical RK4 steps written as one ``sc.scan`` whose step
size ``T / K`` enters every step as a broadcast (stride-0) input, so ``sc.jacobian`` of the final state
in ``(a, T)`` is forward mode through the loop. The Newton iteration around it is a
``sc.while_loop`` with ``mu`` as a loop parameter: the whole shooting solver is one C function.

At the solution, the Jacobian of the flow in the initial state is the monodromy matrix. Its
eigenvalues are the Floquet multipliers: one is exactly 1 (a shift along the orbit), and by
Liouville's formula their product is ``exp(int_0^T mu (1 - x1^2) dt)``, an integral the same scan
accumulates as a third state. Both are checked, and the period for ``mu = 1`` is the known
6.6632868593. Natural-parameter continuation then traces the period as ``mu`` grows; the stable
multiplier it reports comes from Liouville's formula, since once it falls below machine precision
relative to the trivial one the monodromy matrix can no longer resolve it.

The generated C lands in ``examples/generated/periodic_orbit/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "periodic_orbit"
K = 2000  # RK4 steps per period
TOL, MAX_NEWTON, MAX_STEP = 1e-12, 50, 0.5
PERIOD_MU_1 = 6.6632868593


def vector_field(z: sc.Expr, mu: sc.Expr) -> sc.Expr:
  x1, x2 = z[0], z[1]
  return sc.stack([x2, mu * (1 - x1 * x1) * x2 - x1, mu * (1 - x1 * x1)])  # the third state integrates div f


@sc.function(3, 2)
def rk4(z: sc.Expr, mu_dt: sc.Expr) -> sc.Expr:
  mu, dt = mu_dt[0], mu_dt[1]
  k1 = vector_field(z, mu)
  k2 = vector_field(z + 0.5 * dt * k1, mu)
  k3 = vector_field(z + 0.5 * dt * k2, mu)
  k4 = vector_field(z + dt * k3, mu)
  return z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def flow(x0: sc.Expr, period: sc.Expr, mu: sc.Expr) -> sc.Expr:
  """``(phi_T(x0), int_0^T div f dt)`` as one scan."""
  z0 = sc.concat([x0, sc.const(np.zeros(1))])
  mu_dt = sc.stack([mu, period / K])
  return sc.scan(rk4, z0, [(mu_dt, 0, 0)], length=K)[0]


@sc.function
def shooting(y: sc.Expr, mu: sc.Expr) -> tuple[sc.Expr, sc.Expr]:  # y = (a, T)
  x0 = sc.stack([y[0], sc.const(0.0)])
  f = flow(x0, y[1], mu)[:2] - x0
  return f, sc.jacobian(f, y)


@sc.function
def newton_step(carry: sc.Expr, mu: sc.Expr) -> sc.Expr:
  y = carry[:2]
  f, j = shooting(y, mu)
  det = j[0, 0] * j[1, 1] - j[0, 1] * j[1, 0]
  step = sc.stack([j[1, 1] * f[0] - j[0, 1] * f[1], j[0, 0] * f[1] - j[1, 0] * f[0]]) / det  # the 2 x 2 inverse
  y_next = y - step * sc.minimum(1.0, MAX_STEP / sc.norm_inf(step))  # a damped step far from the orbit
  f_next, _ = shooting(y_next, mu)
  return sc.concat([y_next, sc.norm_inf(f_next).reshape((1,))])


@sc.function
def not_converged(carry: sc.Expr, mu: sc.Expr) -> sc.Expr:
  return sc.greater(carry[2], TOL)


@sc.function(2, (), output=sc.G("amplitude", "period", "monodromy", "liouville", "iterations"))
def limit_cycle(guess: sc.Expr, mu: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
  carry, n_iter = sc.while_loop(not_converged, newton_step, sc.concat([guess, sc.const(np.ones(1))]), max_iter=MAX_NEWTON, params=(mu,))
  x0 = sc.stack([carry[0], sc.const(0.0)])
  z_final = flow(x0, carry[1], mu)
  monodromy = sc.jacobian(z_final[:2], x0)
  return carry[0], carry[1], monodromy, z_final[2].exp(), n_iter


def main() -> dict:
  amplitude, period, monodromy, liouville, iterations = limit_cycle(np.array([2.0, 6.0]), np.array(1.0))
  multipliers = np.sort(np.abs(np.linalg.eigvals(monodromy)))
  sweep = []
  guess = np.array([float(amplitude), float(period)])
  for mu in np.arange(1.0, 4.01, 0.25):
    a, t, _, liou, it = limit_cycle(guess, np.array(mu))
    guess = np.array([float(a), float(t)])
    sweep.append((float(mu), float(a), float(t), float(liou), int(it)))
  return {
    "amplitude": float(amplitude),
    "period": float(period),
    "iterations": int(iterations),
    "multipliers": multipliers,
    "det_monodromy": float(np.linalg.det(monodromy)),
    "liouville": float(liouville),
    "sweep": sweep,
  }


if __name__ == "__main__":
  out = main()
  print(f"mu = 1: amplitude {out['amplitude']:.10f}, period {out['period']:.10f} (known {PERIOD_MU_1}) after {out['iterations']} Newton steps")
  print(f"Floquet multipliers: 1 {out['multipliers'][1] - 1:+.1e} and {out['multipliers'][0]:.6e}")
  print(f"det(monodromy) = {out['det_monodromy']:.9e}, Liouville's exp(int div f) = {out['liouville']:.9e}")
  print("continuation in mu:")
  for mu, a, t, rho, it in out["sweep"]:
    print(f"  mu = {mu:4.2f}: amplitude {a:.6f}, period {t:.6f}, stable multiplier {rho:.3e} ({it} Newton steps)")
  write_module(limit_cycle, GENERATED)
  print(f"generated C in {GENERATED}")
