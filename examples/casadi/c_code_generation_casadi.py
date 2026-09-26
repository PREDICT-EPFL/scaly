"""C code generation: the gradient of the determinant of a 7x7 matrix, compiled at three levels (CasADi).

``Function.generate`` writes the C, gcc compiles it, and ``ca.external`` loads the shared library
back as a ``Function``. CasADi's symbolic ``det`` expands by minors, which gives an expression of
about 45 000 elementary operations for the gradient.

After casadi/docs/examples/python/c_code_generation.py (Joel Andersson, 2016).
"""

import contextlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import casadi as ca
import numpy as np

from _common import show

N = 7
X0 = np.random.default_rng(0).random((N, N))
FLAGS = ["-march=native", "-fno-math-errno"]  # the flags Scaly's JIT adds, so both sides compile alike


def build(verbose: bool = False, opt: str = "-O3"):
  x = ca.SX.sym("x", N, N)
  grad_det = ca.Function("grad_det", [x], [ca.gradient(ca.det(x), x)], ["x"], ["gd"])
  workdir = Path(tempfile.mkdtemp(prefix="grad_det_"))
  with contextlib.chdir(workdir):
    cname = grad_det.generate()
    subprocess.run(["gcc", "-fPIC", "-shared", opt, *FLAGS, cname, "-o", "grad_det.so"], check=True)
  compiled = ca.external("grad_det", str(workdir / "grad_det.so"))
  if verbose:
    print(f"{grad_det.n_nodes()} elementary operations, {(workdir / cname).stat().st_size / 1e3:.0f} kB of C")

  def run():
    return {"gd": compiled(X0).full().reshape(-1)}

  return run


if __name__ == "__main__":
  for opt in ["-O0", "-O3", "-Os"] if len(sys.argv) < 2 else sys.argv[1:]:
    t0 = time.perf_counter()
    run = build(verbose=True, opt=opt)
    print(f"{opt}: generated and compiled in {time.perf_counter() - t0:.2f} s")
    t0 = time.perf_counter()
    for _ in range(1000):
      out = run()
    print(f"{opt}: {(time.perf_counter() - t0) * 1e3:.3f} us per evaluation")
  show(out)
