# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""System identification of a nonlinear mass-spring-damper from 2000 samples, by Gauss-Newton (CasADi).

    M y'' + c y' + k y + k_NL y^3 = u

The four parameters are fitted to simulated output data (10 RK4 steps per sample), first by single
shooting (the parameters are the only variables), then by multiple shooting (the states at every
sample are variables too, tied by continuity constraints). IPOPT is given the Gauss-Newton Hessian
J'J of the residuals, and the oracles are just-in-time compiled.

After casadi/docs/examples/python/sysid.py (Joris Gillis, 2018).
"""

import casadi as ca
import numpy as np

from _common import as_arrays, casadi_ipopt_options, casadi_jit, show

N = 2000  # samples
FS = 610.1  # sampling frequency [Hz]
STEPS_PER_SAMPLE = 10
PARAM_TRUTH = np.array([5.625e-6, 2.3e-4, 1, 4.69])
PARAM_GUESS = np.array([5, 2, 1, 5])
SCALE = np.array([1e-6, 1e-4, 1, 1])


def build(verbose: bool = False):
  y, dy, u = ca.MX.sym("y"), ca.MX.sym("dy"), ca.MX.sym("u")
  states = ca.vertcat(y, dy)
  M, c, k, k_NL = ca.MX.sym("M"), ca.MX.sym("c"), ca.MX.sym("k"), ca.MX.sym("k_NL")
  params = ca.vertcat(M, c, k, k_NL)
  ode = ca.Function("ode", [states, u, params], [ca.vertcat(dy, (u - k_NL * y**3 - k * y - c * dy) / M)])
  dt = 1 / FS / STEPS_PER_SAMPLE
  k1 = ode(states, u, params)
  k2 = ode(states + dt / 2.0 * k1, u, params)
  k3 = ode(states + dt / 2.0 * k2, u, params)
  k4 = ode(states + dt * k3, u, params)
  one_step = ca.Function("one_step", [states, u, params], [states + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)])
  X = states
  for _ in range(STEPS_PER_SAMPLE):
    X = one_step(X, u, params)
  one_sample = ca.Function("one_sample", [states, u, params], [X])
  all_samples = one_sample.mapaccum("all_samples", N)

  # The measured data: a simulation with the true parameters.
  np.random.seed(0)
  u_data = ca.DM(0.1 * np.random.random(N))
  x0 = ca.DM([0, 0])
  y_data = all_samples(x0, u_data, ca.repmat(PARAM_TRUTH, 1, N))[0, :].T

  jit = casadi_jit(force=True, expand=False)  # the original always compiles its oracles

  def gauss_newton(e, nlp, V):
    J = ca.jacobian(e, V)
    sigma = ca.MX.sym("sigma")
    hess_lag = ca.Function(
      "nlp_hess_l", {"x": V, "lam_f": sigma, "hess_gamma_x_x": sigma * ca.triu(J.T @ J)}, ["x", "p", "lam_f", "lam_g"], ["hess_gamma_x_x"], jit
    )
    return ca.nlpsol("solver", "ipopt", nlp, {**casadi_ipopt_options(verbose), "hess_lag": hess_lag, **jit})

  # Single shooting
  X_symbolic = all_samples(x0, u_data, ca.repmat(params * SCALE, 1, N))
  e = y_data - X_symbolic[0, :].T
  single = gauss_newton(e, {"x": params, "f": 0.5 * ca.dot(e, e)}, params)

  # Multiple shooting
  Xs = ca.MX.sym("X", 2, N)
  Xn = one_sample.map(N)(Xs, u_data.T, ca.repmat(params * SCALE, 1, N))
  e = y_data - Xn[0, :].T
  V = ca.veccat(params, Xs)
  multiple = gauss_newton(e, {"x": V, "f": 0.5 * ca.dot(e, e), "g": ca.vec(Xn[:, :-1] - Xs[:, 1:])}, V)
  yd = np.diff(y_data.full(), axis=0) * FS
  V0 = ca.veccat(PARAM_GUESS, ca.horzcat(y_data, ca.vertcat(yd, yd[-1])).T)

  def run():
    sol_single = single(x0=PARAM_GUESS)
    it_single = single.stats()["iter_count"]
    sol_multiple = multiple(x0=V0, lbg=0, ubg=0)
    it_multiple = multiple.stats()["iter_count"]
    return as_arrays(
      {
        "params_single": sol_single["x"] * SCALE,
        "params_multiple": sol_multiple["x"][:4] * SCALE,
        "iter_single": it_single,
        "iter_multiple": it_multiple,
      }
    )

  return run


if __name__ == "__main__":
  out = build(verbose=False)()
  show(out)
  assert np.max(np.abs(out["params_single"] - PARAM_TRUTH)) < 1e-8
  assert np.max(np.abs(out["params_multiple"] - PARAM_TRUTH)) < 1e-8
