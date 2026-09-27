"""Generate and compile the IPM solvers of one checkout, for C-side timing by ``timing.py``.

    uv run internal/notes/perf_2026_09_27_ipm_speed/gen.py --variant new [--problems a,b] [--backends sparse,dense] [--jobs 8]

To build the variant of another checkout (a ``git worktree`` of the base commit, say), put its
``src`` first on the path: ``PYTHONPATH=<worktree>/src uv run --no-sync .../gen.py --variant base``.

Per problem and backend, in a fresh process with its own JIT cache: the generated solver as a
``Function`` of the problem data (``x``, ``status``, ``iter`` out), its C, the shared library built
with the JIT's flags, the input blob ``time_entry.c`` reads, and ``meta.json`` (status, iterations,
``x``, graph-build, generation and compile times, C lines, object size, workspace). Everything goes
to ``build/<variant>/<problem>_<backend>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "qp_solvers"))
BUILD = HERE / "build"
ORDER = ("P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u")

# A quick set: sizes from 10 to 1000 unknowns, both kinds of structure, the qp_solvers families.
QUICK = (
  "HS118",
  "QAFIRO",
  "DUALC1",
  "CVXQP1_S",
  "QPCBLEND",
  "QSC205",
  "QBORE3D",
  "QCAPRI",
  "QSHARE1B",
  "PRIMALC1",
  "CVXQP2_S",
  "mpc_4_2_10",
  "mpc_12_4_20",
  "mpc_27_6_30",
  "ex_mpc_N20",
  "ex_portfolio",
  "ex_svm",
  "ex_dense",
)


def all_names() -> list[str]:
  from tests.solvers.ipm.problems import maros_meszaros_names

  return [*maros_meszaros_names(), "mpc_4_2_10", "mpc_12_4_20", "mpc_27_6_30", "ex_mpc_N20", "ex_portfolio", "ex_svm", "ex_dense"]


def problem(name: str):
  """A ``tests.solvers.ipm.problems.QP`` by name: a stored Maros-Meszaros problem, a generated MPC
  (``mpc_<nx>_<nu>_<N>``) or a family of ``examples/qp_solvers`` at its parameters (``ex_<case>``)."""
  from scipy import sparse

  from tests.solvers.ipm.problems import _qp, maros_meszaros, mpc_qp

  if name.startswith("mpc_"):
    nx, nu, horizon = (int(v) for v in name.split("_")[1:])
    return mpc_qp(nx, nu, horizon, name=name)
  if name.startswith("ex_"):
    import generated_piqp
    import problems as families

    case = families.cases()[name.removeprefix("ex_")]
    (P, c), (A, b), (G, g_lb, g_ub), (x_lb, x_ub), _ = generated_piqp.qp_data(case.problem)(case.params())

    def sp(m):
      return sparse.csc_array(np.where(np.abs(m) > 0.0, m, 0.0))

    return _qp(name, sp(P), c, A=sp(A), b=b, G=sp(G), h_l=g_lb, h_u=g_ub, x_l=x_lb, x_u=x_ub)
  return maros_meszaros(name)


def blob(flat: list[np.ndarray], out_sizes: list[int], workspace: int) -> bytes:
  head = np.array([len(flat), len(out_sizes), workspace, *(a.size for a in flat), *out_sizes], dtype="<i8").tobytes()
  return head + b"".join(np.ascontiguousarray(a, dtype="<f8").tobytes() for a in flat)


def one(name: str, backend: str, out: Path) -> dict:
  import scaly as sc
  from scaly.codegen import render_c_module
  from scaly.codegen.abi import c_ident
  from scaly.codegen.jit import compile_flags
  from scaly.codegen.toolchain import find_c_compiler
  from scaly.solvers.ipm import QPValues, Solver
  from tests.solvers.ipm.problems import ipm_inputs

  qp = problem(name)
  s, values = ipm_inputs(qp)
  t0 = time.perf_counter()
  syms = {k: sc.sym(k, np.shape(values[k])) for k in ORDER}
  res = Solver(s, backend, name=f"g_{c_ident(name)}").solve(QPValues.preprocess(s, **syms))
  keys = ["x", "status", "iter"]
  fn = sc.Function._from_exprs(f"g_{c_ident(name)}_{backend}", [syms[k] for k in ORDER], [res[k] for k in keys], list(ORDER), keys)
  build = time.perf_counter() - t0
  t0 = time.perf_counter()
  module = render_c_module(fn)
  generate = time.perf_counter() - t0
  out.mkdir(parents=True, exist_ok=True)
  src = out / "solver.c"
  src.write_text(module.body)
  lib = out / "lib.so"
  cc = find_c_compiler().cc
  t0 = time.perf_counter()
  subprocess.run([cc, *compile_flags(), "-fPIC", "-shared", str(src), *module.link_flags, "-lm", "-o", str(lib)], check=True)
  compile_s = time.perf_counter() - t0
  args = tuple(np.asarray(values[k], dtype=float) for k in ORDER)
  x, status, iters = fn(args)
  flat = [np.ravel(a) for a in args]
  (out / "inputs.bin").write_bytes(blob(flat, [int(e.size) for e in fn.outputs], int(module.workspace_size)))
  meta = {
    "problem": name,
    "backend": backend,
    "symbol": c_ident(fn.name),
    "n": s.n,
    "p": s.p,
    "m": s.m,
    "status": int(status),
    "iter": int(iters),
    "x": np.asarray(x).tolist(),
    "build_s": build,
    "generate_s": generate,
    "compile_s": compile_s,
    "c_lines": sum(1 for line in module.body.splitlines() if line.strip()),
    "object_bytes": lib.stat().st_size,
    "workspace": int(module.workspace_size),
  }
  (out / "meta.json").write_text(json.dumps(meta))
  return meta


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--variant", required=True)
  parser.add_argument("--problems", help="comma-separated, 'quick' (default) or 'all'")
  parser.add_argument("--backends", default="sparse,dense")
  parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
  parser.add_argument("--one", nargs=3, metavar=("PROBLEM", "BACKEND", "OUT"), help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.one:
    print(json.dumps({k: v for k, v in one(args.one[0], args.one[1], Path(args.one[2])).items() if k != "x"}))
    return
  names = list(QUICK) if args.problems in (None, "quick") else all_names() if args.problems == "all" else args.problems.split(",")
  cells = [(n, b) for n in names for b in args.backends.split(",")]
  base = BUILD / args.variant

  def run(cell: tuple[str, str]) -> str:
    name, backend = cell
    with tempfile.TemporaryDirectory(prefix="scaly-cache-") as cache:
      env = {**os.environ, "SCALY_CACHE_DIR": cache}
      t = time.perf_counter()
      done = subprocess.run([sys.executable, __file__, "--variant", args.variant, "--one", name, backend, str(base / f"{name}_{backend}")], capture_output=True, text=True, env=env)
    if done.returncode:
      return f"{name:>14} {backend:>6} FAILED {done.stderr.strip().splitlines()[-1][-300:] if done.stderr.strip() else done.returncode}"
    m = json.loads(done.stdout.strip().splitlines()[-1])
    return f"{name:>14} {backend:>6} {time.perf_counter() - t:6.1f} s  status {m['status']} iter {m['iter']:3d}  gen {m['generate_s']:.2f} s cc {m['compile_s']:.2f} s  {m['c_lines']} lines"

  with ThreadPoolExecutor(args.jobs) as pool:
    for line in pool.map(run, cells):
      print(line, flush=True)


if __name__ == "__main__":
  main()
