# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly"]
# ///
"""The Van der Pol limit cycle by shooting: Newton on a flow map, differentiated through a scan.

The oscillator ``x'' - mu (1 - x^2) x' + x = 0``, as ``x = (x1, x2)``, has one stable periodic orbit.
Single shooting finds it as a root of

    F(a, T) = phi_T((a, 0); mu) - (a, 0) = 0,

the flow over one period ``T`` from a point on the axis ``x2 = 0`` (the phase condition), with two
unknowns ``(a, T)``: an ``sc.roots.root`` with ``mu`` its parameter, solved by ``sc.roots.Newton``
with its steps capped in length (``max_step``). The flow map is ``K`` classical RK4 steps,
``si.rk4`` with the interval ``T`` an input, which runs them as one ``sc.scan`` whose step size
``T / K`` enters every step as a broadcast (stride-0) input, so the Newton step's Jacobian in
``(a, T)`` is forward mode through the loop. The Newton iteration around it is a ``while_loop`` in
the generated code: the whole shooting solver is one C function.

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
from scaly import integrators as si
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parents[1] / "generated" / "periodic_orbit"
K = 2000  # RK4 steps per period
TOL, MAX_NEWTON, MAX_STEP = 1e-12, 50, 0.5
PERIOD_MU_1 = 6.6632868593


@sc.function(3, (), output="zdot")
def vector_field(z: sc.Expr, mu: sc.Expr) -> sc.Expr:
  x1, x2 = z[0], z[1]
  return sc.stack([x2, mu * (1 - x1 * x1) * x2 - x1, mu * (1 - x1 * x1)])  # the third state integrates div f


# rk4(z, mu, dt) -> znext: K steps of dt / K, the interval an input (dt=None), run as one scan.
rk4 = si.rk4(vector_field, dt=None, steps=K)


def flow(x0: sc.Expr, period: sc.Expr, mu: sc.Expr) -> sc.Expr:
  """``(phi_T(x0), int_0^T div f dt)`` as one scan."""
  return rk4(sc.concat([x0, sc.const(np.zeros(1))]), mu, period)


@sc.roots.root(vars=sc.L("y", 2), params=sc.L("mu", ()), name="shooting")
def shooting(y: sc.Expr, mu: sc.Expr) -> sc.Expr:  # y = (a, T)
  x0 = sc.stack([y[0], sc.const(0.0)])
  return flow(x0, y[1], mu)[:2] - x0


# Steps capped at MAX_STEP in their largest entry: a damped Newton far from the orbit.
shoot = sc.roots.solver(shooting, sc.roots.Newton(tol=TOL, max_iter=MAX_NEWTON, max_step=MAX_STEP), name="shooting_newton")


@sc.function(2, (), output=sc.G("amplitude", "period", "monodromy", "liouville", "iterations"))
def limit_cycle(guess: sc.Expr, mu: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
  y, info = shoot(guess, mu)
  x0 = sc.stack([y[0], sc.const(0.0)])
  z_final = flow(x0, y[1], mu)
  monodromy = sc.jacobian(z_final[:2], x0)
  return y[0], y[1], monodromy, z_final[2].exp(), info.iter


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
