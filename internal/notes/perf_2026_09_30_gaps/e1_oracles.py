"""E1's IPOPT oracles and stage oracles on the 3-D chain of masses, timed from C, without the CasADi side.

    uv run internal/notes/perf_2026_09_30_gaps/e1_oracles.py [--masses 3 5] [--horizon 25]
    PYTHONPATH=<worktree>/src uv run --no-sync internal/notes/perf_2026_09_30_gaps/e1_oracles.py   # another tree

The case study's own sweep (`examples/case_studies/fatrop_chain/sweep.py`) checks every kernel
against rockit's NLP, which needs the paper's code. This builds the same two sets of Scaly kernels
with the same point and times them with the study's `time_kernel.c`: the IPOPT drop-in's sparse
Lagrangian Hessian and constraint Jacobian over the whole horizon (`sc.opt.solver(problem, "ipopt")`'s
descriptor), and the Fatrop drop-in's dense stage blocks, a map over stages. CS-2 is the ratio of
the first Jacobian to the second: the same derivatives, assembled over the horizon.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "examples" / "case_studies" / "fatrop_chain"
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY.parent))


def main() -> None:
  import scaly as sc
  from fatrop_dropin import oracles, to_fatrop
  from scaly.codegen import write_module
  from scaly.codegen.jit import compile_flags
  from scaly.opt.external.graph import solver_descriptor
  from scaly_impl import build, sizes
  from sweep import point, time_call

  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--masses", type=int, nargs="+", default=[3, 5])
  parser.add_argument("--horizon", type=int, default=25)
  parser.add_argument("--repeats", type=int, default=2000)
  args = parser.parse_args()
  print(f"scaly from {Path(sc.__file__).parent}")
  for masses in args.masses:
    z, x0, lam_rockit, lam_scaly, _ = point(3, masses, args.horizon)
    nx, nu = sizes(3, masses)
    cells = []
    desc = solver_descriptor(sc.opt.solver(build(3, masses, args.horizon)["problem"], "ipopt", name=f"e1_M{masses}"))
    cells.append(("ipopt hess", desc.hess, [z, x0, np.array([1.0]), lam_scaly], int(desc.hess_sparsity.nnz)))
    cells.append(("ipopt jac", desc.jac, [z, x0], int(desc.jac_sparsity.nnz)))
    stage = oracles(3, masses, args.horizon)
    w = to_fatrop(z, nx, nu, args.horizon)
    lam_dyn = -lam_rockit[np.concatenate([np.arange(k * nx, (k + 1) * nx) + (0 if k == 0 else nx + k * nu) for k in range(args.horizon)])]
    cells.append(("stage hess", stage["hess"], [w, lam_dyn, np.array([1.0])], int(stage["hess"].outputs[0].size)))
    cells.append(("stage jac", stage["jac"], [w], int(stage["jac"].outputs[0].size)))
    for label, fn, inputs, n_out in cells:
      with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        module = write_module(fn, work)
        source = work / module.source_name
        lib = work / "kernel.so"
        subprocess.run(["cc", *compile_flags(), "-shared", "-fPIC", "-o", str(lib), str(source), "-lm"], check=True)
        time_call(lib, fn.name, inputs, [n_out], (module.workspace_size, 0, 0, 0), 400, work)  # warm
        best = min(time_call(lib, fn.name, inputs, [n_out], (module.workspace_size, 0, 0, 0), args.repeats, work)[0] for _ in range(5))
        print(f"masses {masses:2d}  {label:11s} {best * 1e6:9.2f} us   C {source.stat().st_size / 1e3:8.1f} KB   outputs {n_out}", flush=True)


if __name__ == "__main__":
  main()
