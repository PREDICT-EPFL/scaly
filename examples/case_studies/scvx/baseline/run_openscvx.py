"""Run OpenSCvx's `examples/rocket/6DoF_pdg.py` once, in this process, and write what it took.

    <OpenSCvx python> run_openscvx.py <OpenSCvx checkout> <cache dir> <out.json>

The example's module is executed as it is (its `__main__` block, which starts viser servers, is not),
then the phases OpenSCvx separates are timed: `initialize()` (tracing, XLA compilation, CVXPY
canonicalization and one warm-up iteration, "Total Initialization Time" in OpenSCvx's printout),
`solve()` (the PTR iterations) and `post_process()` (the propagated trajectory). The imports of JAX,
CVXPY and OpenSCvx and the construction of `Problem` are timed as well. OPENSCVX_CACHE_DIR is set to
`<cache dir>`: empty for a cold start, reused for a warm one.
"""

import json
import os
import sys
import time

t_start = time.perf_counter()
checkout, cache, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["OPENSCVX_CACHE_DIR"] = cache
import importlib.util  # noqa: E402

import numpy as np  # noqa: E402

t0 = time.perf_counter()
import cvxpy  # noqa: E402, F401
import jax  # noqa: E402, F401
import openscvx  # noqa: E402, F401

t_import = time.perf_counter() - t0
path = os.path.join(checkout, "examples", "rocket", "6DoF_pdg.py")
sys.path.insert(0, checkout)
t0 = time.perf_counter()
spec = importlib.util.spec_from_file_location("pdg6", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
t_problem = time.perf_counter() - t0
problem = mod.problem
problem.settings.dev.printing = False
t0 = time.perf_counter()
problem.initialize()
t_init = time.perf_counter() - t0
t0 = time.perf_counter()
problem.solve()
t_solve = time.perf_counter() - t0
t0 = time.perf_counter()
problem.post_process()
t_post = time.perf_counter() - t0
st = problem._state
x, u = np.asarray(st.x), np.asarray(st.u)
row = {
  "t_import": t_import,
  "t_problem": t_problem,
  "t_initialize": t_init,
  "t_solve": t_solve,
  "t_post_process": t_post,
  "t_process_python": time.perf_counter() - t_start,
  "iterations": int(st.k),
  "J_tr": float(st.J_tr),
  "J_vc": float(st.J_vc),
  "final_mass": float(x[-1, 0]),
  "t_f": float(x[-1, 14]),
  "X": x.tolist(),
  "U": u.tolist(),
}
with open(out, "w") as fp:
  json.dump(row, fp)
print(json.dumps({k: v for k, v in row.items() if k not in ("X", "U")}))
