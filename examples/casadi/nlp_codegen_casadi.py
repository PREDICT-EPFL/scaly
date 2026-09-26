"""Generate C for an NLP's oracles, compile it and solve with IPOPT from the compiled code (CasADi).

    minimize  x^2 + y^2   subject to  x + y - 10 = 0

The original shows two routes, ``jit`` and ``external``; this is ``external``: the solver's
dependencies go to ``nlp.c``, which is compiled with ``gcc -O3`` and loaded back as the oracles of
a new ``nlpsol``. Scaly's route is the same one, taken by ``sc.solver`` on its own.

After casadi/docs/examples/python/nlp_codegen.py (Joel Andersson, 2016).
"""

import contextlib
import subprocess
import tempfile
from pathlib import Path

import casadi as ca
import numpy as np

from _common import as_arrays, casadi_ipopt_options, show


def build(verbose: bool = False):
  x, y = ca.MX.sym("x"), ca.MX.sym("y")
  nlp = {"x": ca.vertcat(x, y), "f": x * x + y * y, "g": x + y - 10}
  workdir = Path(tempfile.mkdtemp(prefix="nlp_codegen_"))
  with contextlib.chdir(workdir):  # generate_dependencies writes to the working directory
    ca.nlpsol("solver", "ipopt", nlp).generate_dependencies("nlp.c")
    subprocess.run(["gcc", "-fPIC", "-shared", "-O3", "nlp.c", "-o", "nlp.so"], check=True)
  solver = ca.nlpsol("solver", "ipopt", str(workdir / "nlp.so"), casadi_ipopt_options(verbose))

  def run():
    res = solver(lbx=-np.inf, ubx=np.inf, lbg=0, ubg=0, x0=0)
    return as_arrays({"f": res["f"], "x": res["x"], "lam_x": res["lam_x"], "lam_g": res["lam_g"]})

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
