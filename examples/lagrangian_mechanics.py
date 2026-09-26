"""Equations of motion by differentiation: an n-link pendulum from its Lagrangian alone.

For a planar chain of ``n`` point masses on massless rods, with angles ``q`` from the vertical, the
only modelling is the positions of the masses as functions of ``q`` and the Lagrangian

    L(q, qd) = T - V,   T = 1/2 sum_k m_k |d p_k/dt|^2,   V = g sum_k m_k y_k,   d p_k/dt = (dp_k/dq) qd.

Everything else is derivatives taken with respect to *expressions* (``q`` and ``qd`` inside a
graph, not function inputs): the Euler-Lagrange equations

    M(q) qdd + (d^2 L / dqd dq) qd - dL/dq = tau,   M(q) = d^2 L / dqd^2,

give the mass matrix with ``sc.hessian``, the velocity-product terms with ``sc.jacobian`` of
``sc.gradient`` and the generalized forces with ``sc.gradient``, and the forward dynamics
``qdd = M^{-1}(tau + dL/dq - C qd)`` is a dense ``cholesky`` / ``cho_solve``. The resulting
``forward_dynamics`` function is what one would embed in a simulator or an MPC.

A classical RK4 integrator runs ``STEPS`` steps as one ``sc.scan`` that also records the energy,
whose drift checks the derivation. The double pendulum is compared with its textbook equations.
The triple pendulum is chaotic: ``sc.jacobian`` of the final state in the initial one, through the
whole scan, gives the finite-time Lyapunov exponent ``log(sigma_max) / t``, and the same number from
two nearby simulations agrees.

The generated C lands in ``examples/generated/lagrangian_mechanics/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scaly as sc
from scaly import linalg
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "lagrangian_mechanics"
G = 9.81
DT, STEPS = 1e-3, 5000


def model(masses: np.ndarray, lengths: np.ndarray) -> dict[str, sc.Function]:
  """Forward dynamics, energy, an RK4 step and a simulator for one chain."""
  n = len(masses)
  lower = np.tril(np.ones((n, n)))  # p_k sums the rods up to k

  def positions(q: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return sc.const(lower * lengths) @ q.sin(), -(sc.const(lower * lengths) @ q.cos())

  def lagrangian(q: sc.Expr, qd: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    x, y = positions(q)
    vx, vy = sc.jacobian(x, q) @ qd, sc.jacobian(y, q) @ qd
    kinetic = 0.5 * (sc.const(masses) * (vx * vx + vy * vy)).sum()
    potential = G * (sc.const(masses) * y).sum()
    return kinetic - potential, kinetic + potential

  @sc.function(sc.G(sc.L("q", n), sc.L("qd", n), sc.L("tau", n)), sc.G(sc.L("qdd", n), sc.L("M", (n, n))))
  def forward_dynamics(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr]:
    q, qd, tau = inputs
    lag, _ = lagrangian(q, qd)
    p = sc.gradient(lag, qd)  # the generalized momenta
    mass = sc.jacobian(p, qd)  # = hessian(lag, qd)
    coupling = sc.jacobian(p, q)
    rhs = tau + sc.gradient(lag, q) - coupling @ qd
    return linalg.cho_solve(linalg.cholesky(mass), rhs), mass

  @sc.function(sc.G(sc.L("q", n), sc.L("qd", n)), sc.L("E", ()))
  def energy(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    return lagrangian(*inputs)[1]

  zero = sc.const(np.zeros(n))

  def f(x: sc.Expr) -> sc.Expr:
    return sc.concat([x[n:], forward_dynamics((x[:n], x[n:], zero))[0]])

  @sc.function(sc.L("x", 2 * n), sc.G(sc.L("x_next", 2 * n), sc.L("E", 1)))
  def rk4_step(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    k1 = f(x)
    k2 = f(x + 0.5 * DT * k1)
    k3 = f(x + 0.5 * DT * k2)
    k4 = f(x + DT * k3)
    return x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), energy((x[:n], x[n:])).reshape((1,))

  @sc.function(sc.L("x0", 2 * n), sc.G(sc.L("x_final", 2 * n), sc.L("energies", STEPS), sc.L("sensitivity", (2 * n, 2 * n))))
  def simulate(x0: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    x_final, energies = sc.scan(rk4_step, x0, [], length=STEPS)
    return x_final, energies, sc.jacobian(x_final, x0)

  return {"forward_dynamics": forward_dynamics, "energy": energy, "rk4_step": rk4_step, "simulate": simulate}


def double_pendulum_textbook(q: np.ndarray, qd: np.ndarray, m: np.ndarray, lengths: np.ndarray) -> np.ndarray:
  """The classical closed-form double-pendulum accelerations."""
  (t1, t2), (w1, w2), (m1, m2), (l1, l2) = q, qd, m, lengths
  delta = t1 - t2
  den = 2 * m1 + m2 - m2 * np.cos(2 * delta)
  a1 = (-G * (2 * m1 + m2) * np.sin(t1) - m2 * G * np.sin(t1 - 2 * t2) - 2 * np.sin(delta) * m2 * (w2**2 * l2 + w1**2 * l1 * np.cos(delta))) / (
    l1 * den
  )
  a2 = (2 * np.sin(delta) * (w1**2 * l1 * (m1 + m2) + G * (m1 + m2) * np.cos(t1) + w2**2 * l2 * m2 * np.cos(delta))) / (l2 * den)
  return np.array([a1, a2])


def main() -> dict:
  rng = np.random.default_rng(0)
  m2, l2 = np.array([1.0, 0.5]), np.array([1.0, 0.7])
  double = model(m2, l2)
  q, qd = rng.uniform(-2, 2, 2), rng.uniform(-1, 1, 2)
  qdd, _ = double["forward_dynamics"]((q, qd, np.zeros(2)))
  textbook_error = np.abs(qdd - double_pendulum_textbook(q, qd, m2, l2)).max()

  triple = model(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
  x0 = np.array([2.0, 2.2, 2.4, 0.0, 0.0, 0.0])
  x_final, energies, sens = triple["simulate"](x0)
  t_final = STEPS * DT
  ftle = np.log(np.linalg.svd(sens, compute_uv=False)[0]) / t_final
  # The same growth rate from two simulations a tiny distance apart along the most unstable direction.
  u0 = np.linalg.svd(sens)[2][0]
  eps = 1e-8
  x_pert, _, _ = triple["simulate"](x0 + eps * u0)
  ftle_twin = np.log(np.linalg.norm(x_pert - x_final) / eps) / t_final
  return {
    "textbook_error": textbook_error,
    "energy_drift": np.abs(energies - energies[0]).max() / abs(energies[0]),
    "ftle": ftle,
    "ftle_twin": ftle_twin,
    "x_final": x_final,
    "functions": triple,
  }


if __name__ == "__main__":
  out = main()
  print(f"double pendulum: accelerations from the Lagrangian match the textbook equations to {out['textbook_error']:.1e}")
  print(f"triple pendulum, {STEPS} RK4 steps in one scan: relative energy drift {out['energy_drift']:.1e}")
  print(
    f"finite-time Lyapunov exponent over {STEPS * DT:.0f} s: {out['ftle']:.3f} /s from the Jacobian through the scan, {out['ftle_twin']:.3f} /s from two runs"
  )
  for name in ("forward_dynamics", "rk4_step", "simulate"):
    write_module(out["functions"][name], GENERATED)
  print(f"generated C in {GENERATED}")
