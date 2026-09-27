"""System identification of a nonlinear mass-spring-damper from 2000 samples (Scaly).

    M y'' + c y' + k y + k_NL y^3 = u

The four parameters are fitted to simulated output data (10 RK4 steps per sample), first by single
shooting (the parameters are the only variables; the 2000 samples are a ``scan``), then by multiple
shooting (the states at every sample are variables too, the samples one ``vmap``). The 10 RK4 steps
of a sample are a ``scan`` too, and the data are parameters of the problems rather than constants
baked into the generated code.

Scaly's IPOPT backend takes the exact Lagrangian Hessian and has no way to pass the Gauss-Newton
one the CasADi original uses, so the iterations differ; at a zero-residual fit both reach the true
parameters.

After casadi/docs/examples/python/sysid.py (Joris Gillis, 2018).
"""

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show

N = 2000  # samples
FS = 610.1  # sampling frequency [Hz]
STEPS_PER_SAMPLE = 10
PARAM_TRUTH = np.array([5.625e-6, 2.3e-4, 1, 4.69])
PARAM_GUESS = np.array([5.0, 2, 1, 5])
SCALE = np.array([1e-6, 1e-4, 1, 1])
DT = 1 / FS / STEPS_PER_SAMPLE


def ode(x: sc.Expr, u: sc.Expr, p: sc.Expr) -> sc.Expr:
  y, dy = x[0], x[1]
  M, c, k, k_NL = p[0], p[1], p[2], p[3]
  return sc.stack([dy, (u[0] - k_NL * y**3 - k * y - c * dy) / M])


@sc.function(sc.G(sc.L("x", 2), sc.L("u", 1), sc.L("p", 4)), output=sc.L("xnext", ...))
def one_step(inputs):
  x, u, p = inputs
  k1 = ode(x, u, p)
  k2 = ode(x + DT / 2.0 * k1, u, p)
  k3 = ode(x + DT / 2.0 * k2, u, p)
  k4 = ode(x + DT * k3, u, p)
  return x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


@sc.function(sc.G(sc.L("x", 2), sc.L("u", 1), sc.L("p", 4)), output=sc.L("xnext", ...))
def one_sample(inputs):
  x, u, p = inputs
  (x,) = sc.scan(one_step, x, [(u, 0, 0), (p, 0, 0)], length=STEPS_PER_SAMPLE)
  return x


@sc.function(sc.G(sc.L("x", 2), sc.L("u", 1), sc.L("p", 4)), output=sc.G(sc.L("xnext", ...), sc.L("y", ...)))
def sample_and_output(inputs):
  xnext = one_sample(inputs)
  return xnext, xnext[0:1]


@sc.function(sc.G(sc.L("u", N), sc.L("p", 4)), output=sc.L("y", ...))
def all_samples(inputs):
  u, p = inputs
  _, y = sc.scan(sample_and_output, sc.const(np.zeros(2)), [(u, 0, 1), (p, 0, 0)], length=N)
  return y


# The measured data: a simulation with the true parameters.
np.random.seed(0)
U_DATA = 0.1 * np.random.random(N)
Y_DATA = all_samples((U_DATA, PARAM_TRUTH))


@sc.problem(vars=sc.L("params", 4), params=sc.G(sc.L("u", N), sc.L("y", N)))
def single_shooting(params, data):
  u, y = data
  e = y - all_samples((u, params * SCALE))
  return sc.ProblemSpec(minimize=0.5 * sc.dot(e, e))


@sc.problem(vars=sc.G(sc.L("params", 4), sc.L("X", 2 * N)), params=sc.G(sc.L("u", N), sc.L("y", N)))
def multiple_shooting(variables, data):
  params, X = variables
  u, y = data
  Xn = sc.vmap(one_sample, N, [(X, 0, 2), (u, 0, 1), (params * SCALE, 0, 0)]).reshape((N, 2))
  e = y - Xn[:, 0]
  gaps = Xn[:-1] - X.reshape((N, 2))[1:]
  return sc.ProblemSpec(minimize=0.5 * sc.dot(e, e), eq=(gaps.reshape((2 * (N - 1),)),))


def build(verbose: bool = False):
  single = sc.solver(single_shooting, "ipopt", options=scaly_ipopt_options(verbose))
  multiple = sc.solver(multiple_shooting, "ipopt", options=scaly_ipopt_options(verbose))
  yd = np.diff(Y_DATA) * FS
  X0 = np.stack([Y_DATA, np.r_[yd, yd[-1]]], axis=1).reshape(-1)

  def run():
    p_single, *_ = single((PARAM_GUESS, np.zeros(4), np.zeros(0), np.zeros(0), (U_DATA, Y_DATA)))
    it_single = single.solver_stats().iter
    (p_multiple, _), *_ = multiple(((PARAM_GUESS, X0), (np.zeros(4), np.zeros(2 * N)), np.zeros(2 * (N - 1)), np.zeros(0), (U_DATA, Y_DATA)))
    it_multiple = multiple.solver_stats().iter
    return {
      "params_single": p_single * SCALE,
      "params_multiple": p_multiple * SCALE,
      "iter_single": np.array([it_single]),
      "iter_multiple": np.array([it_multiple]),
    }

  return run


if __name__ == "__main__":
  out = build(verbose=False)()
  show(out)
  assert np.max(np.abs(out["params_single"] - PARAM_TRUTH)) < 1e-8
  assert np.max(np.abs(out["params_multiple"] - PARAM_TRUTH)) < 1e-8
