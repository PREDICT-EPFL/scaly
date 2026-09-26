"""Hock-Schittkowski problem 15: Rosenbrock's valley cut by two nonconvex constraints (CasADi).

    minimize  100 (x1 - x0^2)^2 + (1 - x0)^2
    subject to  x0 x1 >= 1,  x0 + x1^2 >= 0,  x0 <= 1/2

The original solves it with Uno in its IPOPT preset, as here; Scaly has no Uno backend and uses IPOPT.

After casadi/docs/examples/python/hs015.py (David Kiessling, 2026).
"""

import casadi as ca
import numpy as np

from _common import as_arrays, casadi_jit, show


def build(verbose: bool = False):
  x = ca.MX.sym("x", 2)
  f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
  g = ca.vertcat(x[0] * x[1], x[0] + x[1] ** 2)
  opts = {"uno": {"preset": "ipopt"}, "print_time": verbose, **casadi_jit()}
  if not verbose:
    opts["uno"]["logger"] = "SILENT"
  solver = ca.nlpsol("solver", "uno", {"x": x, "f": f, "g": g}, opts)

  def run():
    res = solver(x0=[-2.0, 1.0], lbg=[1, 0], ubg=[np.inf, np.inf], lbx=[-np.inf, -np.inf], ubx=[0.5, np.inf])
    return as_arrays({"f": res["f"], "x": res["x"], "lam_x": res["lam_x"], "lam_g": res["lam_g"]})

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
