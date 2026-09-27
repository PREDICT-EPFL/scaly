"""System identification by maximum likelihood: differentiating an extended Kalman filter end to end.

A forced Duffing oscillator (a stiffening spring, a damper, a known forcing ``u``),

    x1' = x2,   x2' = -k x1 - k3 x1^3 - c x2 + u(t) + w,

is sampled every ``dt`` with its position measured in noise. The unknowns are
``theta = (log c, log k3, log q, log r)``: damping, spring nonlinearity and the process and
measurement noise intensities. The EKF gives the one-step-ahead innovations ``e_k`` and their
variances ``S_k``, hence the negative log-likelihood

    NLL(theta) = 1/2 sum_k (log S_k + e_k^2 / S_k),

and maximum likelihood fits ``theta`` to the record.

The whole filter is one ``sc.scan``: its carry is the estimate and the covariance, the
measurements and the forcing are sliced one per step, and ``theta`` is broadcast (stride 0). Inside
the step the transition Jacobian of an RK4 discretization comes from ``sc.jacobian``, and the
covariance update is the Joseph form. ``sc.gradient`` of the NLL runs reverse mode backwards
through all ``T`` steps, including the derivative of the Jacobians the filter used, which is the
awkward part to write by hand (it needs second derivatives of the dynamics). L-BFGS uses it.

The generated C lands in ``examples/generated/ekf_identification/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import optimize

import scaly as sc
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "ekf_identification"
T, DT, K_LIN = 600, 0.05, 1.0
THETA_TRUE = np.log([0.3, 0.8, 0.02, 0.01])  # c, k3, q, r


def dynamics(x: sc.Expr, u: sc.Expr, c: sc.Expr, k3: sc.Expr) -> sc.Expr:
  return sc.stack([x[1], -K_LIN * x[0] - k3 * x[0] * x[0] * x[0] - c * x[1] + u])


def rk4(x: sc.Expr, u: sc.Expr, c: sc.Expr, k3: sc.Expr) -> sc.Expr:
  k1 = dynamics(x, u, c, k3)
  k2 = dynamics(x + 0.5 * DT * k1, u, c, k3)
  k3_ = dynamics(x + 0.5 * DT * k2, u, c, k3)
  k4 = dynamics(x + DT * k3_, u, c, k3)
  return x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3_ + k4)


@sc.function(6, 2, 4)
def ekf_step(belief: sc.Expr, yu: sc.Expr, theta: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  x, p = belief[:2], belief[2:].reshape((2, 2))
  y, u = yu[0], yu[1]
  c, k3, q, r = theta[0].exp(), theta[1].exp(), theta[2].exp(), theta[3].exp()
  # Predict.
  x_pred = rk4(x, u, c, k3)
  f = sc.jacobian(x_pred, x)
  p_pred = f @ p @ f.T + sc.const(np.array([[DT**3 / 3, DT**2 / 2], [DT**2 / 2, DT]])) * q
  # Update with the position measurement, H = [1, 0].
  e = y - x_pred[0]
  s = p_pred[0, 0] + r
  gain = p_pred[:, 0] / s
  i_kh = sc.const(np.eye(2)) - sc.stack([gain, sc.const(np.zeros(2))], axis=1)
  p_next = i_kh @ p_pred @ i_kh.T + r * gain.reshape((2, 1)) @ gain.reshape((1, 2))
  x_next = x_pred + gain * e
  nll = 0.5 * (s.log() + e * e / s + np.log(2 * np.pi))
  return sc.concat([x_next, p_next.reshape((4,))]), nll.reshape((1,)), e.reshape((1,))


@sc.function(4, T, T, output=sc.G("nll", "grad", "innovations"))
def likelihood(theta: sc.Expr, y: sc.Expr, u: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  yu = sc.stack([y, u], axis=1).reshape((2 * T,))
  belief0 = sc.const(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]))
  _, nlls, innovations = sc.scan(ekf_step, belief0, [(yu, 0, 2), (theta, 0, 0)], length=T)
  value = nlls.sum()
  return value, sc.gradient(value, theta), innovations


def simulate(theta: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
  """A record from the true system, with process noise integrated over each step."""
  c, k3, q, r = np.exp(theta)
  rng = np.random.default_rng(seed)
  u = 1.2 * np.cos(0.9 * DT * np.arange(T))
  cov = q * np.array([[DT**3 / 3, DT**2 / 2], [DT**2 / 2, DT]])
  x, y = np.array([1.0, 0.0]), np.zeros(T)

  def f(x: np.ndarray, uk: float) -> np.ndarray:
    return np.array([x[1], -K_LIN * x[0] - k3 * x[0] ** 3 - c * x[1] + uk])

  for k in range(T):
    k1 = f(x, u[k])
    k2 = f(x + 0.5 * DT * k1, u[k])
    k3_ = f(x + 0.5 * DT * k2, u[k])
    k4 = f(x + DT * k3_, u[k])
    x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3_ + k4) + rng.multivariate_normal(np.zeros(2), cov)
    y[k] = x[0] + np.sqrt(r) * rng.standard_normal()
  return y, u


def main() -> dict:
  y, u = simulate(THETA_TRUE)
  theta0 = np.log([1.0, 0.1, 0.1, 0.1])
  value, grad, _ = likelihood(theta0, y, u)
  eps = 1e-6
  fd = np.array([(likelihood(theta0 + eps * e, y, u)[0] - likelihood(theta0 - eps * e, y, u)[0]) / (2 * eps) for e in np.eye(4)])
  fit = optimize.minimize(lambda t: likelihood(t, y, u)[:2], theta0, jac=True, method="L-BFGS-B")
  _, _, innovations = likelihood(fit.x, y, u)
  _, _, innovations0 = likelihood(theta0, y, u)
  return {
    "grad_error": np.abs(grad - fd).max() / np.abs(fd).max(),
    "theta": fit.x,
    "nll_start": float(value),
    "nll_fit": float(fit.fun),
    "nll_true": float(likelihood(THETA_TRUE, y, u)[0]),
    "evaluations": int(fit.nfev),
    "innovation_rms": (float(np.sqrt(np.mean(innovations0**2))), float(np.sqrt(np.mean(innovations**2)))),
  }


if __name__ == "__main__":
  out = main()
  print(f"gradient of the NLL through {T} EKF steps vs central differences: relative error {out['grad_error']:.1e}")
  print(
    f"L-BFGS: NLL {out['nll_start']:.1f} -> {out['nll_fit']:.1f} in {out['evaluations']} evaluations (at the true parameters {out['nll_true']:.1f})"
  )
  for name, fitted, true in zip(("c", "k3", "q", "r"), np.exp(out["theta"]), np.exp(THETA_TRUE), strict=True):
    print(f"  {name:>2} = {fitted:.4f}  (true {true:.4f})")
  print(f"innovation RMS {out['innovation_rms'][0]:.3f} with the initial guess, {out['innovation_rms'][1]:.3f} fitted")
  write_module(likelihood, GENERATED)
  print(f"generated C in {GENERATED}")
