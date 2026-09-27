"""Gaussian-process regression: the marginal likelihood through a Cholesky factor, to second order.

A zero-mean GP with the squared-exponential kernel and noise,

    k(x, x') = sf^2 exp(-|x - x'|^2 / (2 l^2)),   K = k(X, X) + sn^2 I,

has the negative log marginal likelihood

    NLL(theta) = 1/2 y^T K^{-1} y + sum_i log L_ii + n/2 log(2 pi),   K = L L^T,

in the log-hyperparameters ``theta = (log l, log sf, log sn)``. Scaly writes it with the generated
dense ``cholesky`` and ``solve_triangular`` (loops with triangular bounds at this order, ``n = 120``)
and differentiates *through the factorization*: ``sc.gradient`` gives the gradient L-BFGS needs,
checked against the textbook formula ``1/2 tr((K^{-1} - alpha alpha^T) dK/dtheta)``, and
``sc.hessian`` gives the curvature at the optimum, whose inverse is the Laplace approximation of
the hyperparameters' posterior covariance.

The predictive mean ``k_*^T K^{-1} y`` and variance ``k(x_*, x_*) - |L^{-1} k_*|^2`` on a test grid are
a second generated function, the part one would deploy.

The generated C lands in ``examples/generated/gaussian_process/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import optimize

import scaly as sc
from scaly import linalg
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "gaussian_process"
N_TRAIN, N_TEST = 120, 200


def kernel(xa: sc.Expr, xb: sc.Expr, length: sc.Expr, scale: sc.Expr) -> sc.Expr:
  d = xa.reshape((xa.size, 1)) - xb.reshape((1, xb.size))
  return scale * scale * (-(d * d) / (2.0 * length * length)).exp()


def factor(theta: sc.Expr, x: sc.Expr) -> sc.Expr:
  length, scale, noise = theta[0].exp(), theta[1].exp(), theta[2].exp()
  return linalg.cholesky(kernel(x, x, length, scale) + sc.const(np.eye(N_TRAIN)) * (noise * noise))


def nll(theta: sc.Expr, x: sc.Expr, y: sc.Expr) -> sc.Expr:
  chol = factor(theta, x)
  white = linalg.solve_triangular(chol, y)  # L^{-1} y
  log_det = sc.gather(chol, np.arange(N_TRAIN) * (N_TRAIN + 1)).log().sum()
  return 0.5 * sc.sumsqr(white) + log_det + 0.5 * N_TRAIN * np.log(2 * np.pi)


@sc.function(
  sc.G(sc.L("theta", 3), sc.L("x", N_TRAIN), sc.L("y", N_TRAIN)),
  output=sc.G(sc.L("nll", ()), sc.L("grad", 3), sc.L("hess", (3, 3))),
)
def marginal_likelihood(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  theta, x, y = inputs
  value = nll(theta, x, y)
  return value, sc.gradient(value, theta), sc.hessian(value, theta)


@sc.function(
  sc.G(sc.L("theta", 3), sc.L("x", N_TRAIN), sc.L("y", N_TRAIN), sc.L("x_test", N_TEST)),
  output=sc.G(sc.L("mean", N_TEST), sc.L("var", N_TEST)),
)
def predict(inputs: tuple[sc.Expr, ...]) -> tuple[sc.Expr, sc.Expr]:
  theta, x, y, x_test = inputs
  length, scale = theta[0].exp(), theta[1].exp()
  chol = factor(theta, x)
  alpha = linalg.cho_solve(chol, y)
  k_star = kernel(x, x_test, length, scale)  # (N_TRAIN, N_TEST)
  v = linalg.solve_triangular(chol, k_star)
  return k_star.T @ alpha, scale * scale - (v * v).T @ sc.const(np.ones(N_TRAIN))


def data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  x = np.sort(rng.uniform(-4, 4, N_TRAIN))
  y = np.sin(1.5 * x) + 0.3 * np.cos(4 * x) + 0.1 * rng.standard_normal(N_TRAIN)
  return x, y


def reference(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
  """NLL and its gradient by the trace formula, in NumPy."""
  ell, sf, sn = np.exp(theta)
  d2 = (x[:, None] - x[None, :]) ** 2
  k_f = sf**2 * np.exp(-d2 / (2 * ell**2))
  k = k_f + sn**2 * np.eye(len(x))
  chol = np.linalg.cholesky(k)
  alpha = np.linalg.solve(k, y)
  value = 0.5 * y @ alpha + np.log(np.diag(chol)).sum() + 0.5 * len(x) * np.log(2 * np.pi)
  inner = np.linalg.inv(k) - np.outer(alpha, alpha)
  dk = [k_f * d2 / ell**2, 2 * k_f, 2 * sn**2 * np.eye(len(x))]
  return float(value), np.array([0.5 * np.sum(inner * d) for d in dk])


def main() -> dict:
  x, y = data()
  theta0 = np.log([1.0, 1.0, 0.5])
  value, grad, hess = marginal_likelihood((theta0, x, y))
  ref_value, ref_grad = reference(theta0, x, y)
  eps = 1e-6
  hess_fd = np.stack(
    [(marginal_likelihood((theta0 + eps * e, x, y))[1] - marginal_likelihood((theta0 - eps * e, x, y))[1]) / (2 * eps) for e in np.eye(3)]
  )

  fit = optimize.minimize(lambda t: marginal_likelihood((t, x, y))[:2], theta0, jac=True, method="L-BFGS-B")
  _, grad_opt, hess_opt = marginal_likelihood((fit.x, x, y))
  covariance = np.linalg.inv(hess_opt)
  x_test = np.linspace(-5, 5, N_TEST)
  mean, var = predict((fit.x, x, y, x_test))
  truth = np.sin(1.5 * x_test) + 0.3 * np.cos(4 * x_test)
  inside = np.abs(x_test) < 4
  return {
    "value_error": abs(value - ref_value),
    "grad_error": np.abs(grad - ref_grad).max(),
    "hess_error": np.abs(hess - hess_fd).max(),
    "theta": fit.x,
    "theta_std": np.sqrt(np.diag(covariance)),
    "grad_at_optimum": np.abs(grad_opt).max(),
    "hess_eigenvalues": np.linalg.eigvalsh(hess_opt),
    "rmse": float(np.sqrt(np.mean((mean[inside] - truth[inside]) ** 2))),
    "coverage": float(np.mean(np.abs(mean[inside] - truth[inside]) < 2 * np.sqrt(var[inside]))),
    "evaluations": int(fit.nfev),
  }


if __name__ == "__main__":
  out = main()
  print(
    f"NLL vs NumPy {out['value_error']:.1e}; gradient through the Cholesky vs the trace formula {out['grad_error']:.1e}; Hessian vs differences {out['hess_error']:.1e}"
  )
  ell, sf, sn = np.exp(out["theta"])
  print(
    f"L-BFGS in {out['evaluations']} evaluations: l = {ell:.3f}, sf = {sf:.3f}, sn = {sn:.3f} (true noise 0.1); |grad| = {out['grad_at_optimum']:.1e}"
  )
  print(
    f"Laplace posterior std of (log l, log sf, log sn): {np.array2string(out['theta_std'], precision=3)}; Hessian positive definite: {bool(out['hess_eigenvalues'].min() > 0)}"
  )
  print(f"prediction on [-4, 4]: RMSE {out['rmse']:.3f} against the noise-free function, {100 * out['coverage']:.0f}% inside two standard deviations")
  for fn in (marginal_likelihood, predict):
    write_module(fn, GENERATED)
  print(f"generated C in {GENERATED}")
