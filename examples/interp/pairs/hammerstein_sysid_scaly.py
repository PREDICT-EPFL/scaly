# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly", "scaly-ipopt"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Identification of a Hammerstein model from 2 000 samples by single shooting (Scaly).

    v[k] = N(u[k]),   y[k+1] = a1 y[k] + a2 y[k-1] + v[k],   N a cubic B-spline with 20 coefficients c

The static nonlinearity is ``interp.BSpline`` with the coefficients an ``Expr``: inside the step it
is the local basis at ``u[k]`` times four of them, and its derivative in them is exact. The 2 000
steps are one ``scan``, the parameters broadcast into it. IPOPT fits ``(a1, a2, c)``.
"""

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show
from _data import HAMMERSTEIN_KNOTS, hammerstein_data
from scaly import interp

DATA = hammerstein_data()
N = DATA["u"].size


@sc.function(2, 1, 2, 20, output=sc.G("xnext", "y"))
def step(x, u, a, c):
  v = interp.BSpline(HAMMERSTEIN_KNOTS, c, 3)(u[0])
  y = a[0] * x[0] + a[1] * x[1] + v
  return sc.stack([y, x[0]]), sc.stack([y])


@sc.opt.problem(vars=sc.G(sc.L("a", 2), sc.L("c", 20)))
def sysid(theta):
  a, c = theta
  _, y = sc.scan(step, sc.const(np.zeros(2)), [(sc.const(DATA["u"]), 0, 1), (a, 0, 0), (c, 0, 0)], length=N)
  residual = y - DATA["y"]
  return sc.opt.ProblemSpec(minimize=(residual**2).sum())


def build(verbose: bool = False):
  solve = sc.opt.solver(sysid, sc.opt.IPOPT(options=scaly_ipopt_options(verbose)))

  def run():
    (a, c), *_ = solve((DATA["a_guess"], DATA["c_guess"]), (np.zeros(2), np.zeros(20)), np.zeros(0), np.zeros(0), ())
    stats = sc.opt.solver_stats(solve)
    return {"a": a, "c": c, "f": np.array([stats.obj]), "iter": np.array([stats.iter])}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
