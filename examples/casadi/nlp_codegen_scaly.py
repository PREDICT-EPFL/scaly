# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly", "scaly-ipopt"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Generate C for an NLP's oracles, compile it and solve with IPOPT from the compiled code (Scaly).

    minimize  x^2 + y^2   subject to  x + y - 10 = 0

This is what ``sc.opt.solver`` does anyway: the first call generates C for the oracles and the IPOPT
wrapper, compiles it and caches the shared library. ``write_module`` writes the same C to a
directory, with a header, for building into another program.

After casadi/docs/examples/python/nlp_codegen.py (Joel Andersson, 2016).
"""

import tempfile
from pathlib import Path

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show
from scaly.codegen import write_module


@sc.opt.problem(vars=sc.L("w", 2))
def nlp(w):
  x, y = w[0], w[1]
  return sc.opt.ProblemSpec(minimize=x * x + y * y, eq=(x + y - 10,))


def build(verbose: bool = False):
  solve = sc.opt.solver(nlp, sc.opt.IPOPT(options=scaly_ipopt_options(verbose)))
  write_module(solve, Path(tempfile.mkdtemp(prefix="nlp_codegen_")))  # nlp_ipopt.c and .h, for use elsewhere

  def run():
    w, lam_w, lam_g, _, _ = solve(np.zeros(2), np.zeros(2), np.zeros(1), np.zeros(0), ())
    return {"f": np.array([sc.opt.solver_stats(solve).obj]), "x": w, "lam_x": lam_w, "lam_g": lam_g}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
