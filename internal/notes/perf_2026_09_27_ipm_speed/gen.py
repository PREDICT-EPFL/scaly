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
sys.path.insert(0, str(ROOT / "examples" / "opt" / "qp_solvers"))
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
  from scaly.testing.qp import maros_meszaros_names

  return [*maros_meszaros_names(), "mpc_4_2_10", "mpc_12_4_20", "mpc_27_6_30", "ex_mpc_N20", "ex_portfolio", "ex_svm", "ex_dense"]


def problem(name: str):
  """A ``tests.opt.ipm.problems.QP`` by name: a stored Maros-Meszaros problem, a generated MPC
  (``mpc_<nx>_<nu>_<N>``, or ``mpc_<nx>_<nu>_<N>_<r>`` with ``r`` path rows a stage), a Hessian of
  ``K`` dense blocks of ``B`` coupled with their neighbours under boxes (``btri_<K>_<B>``) or a
  family of ``examples/opt/qp_solvers`` at its parameters (``ex_<case>``)."""
  from scipy import sparse

  from scaly.testing.qp import maros_meszaros, mpc_qp
  from scaly.testing.qp import make_qp as _qp

  if name.startswith("mpc_"):
    nx, nu, horizon, *rows = (int(v) for v in name.split("_")[1:])
    return mpc_qp(nx, nu, horizon, path_rows=rows[0] if rows else 0, name=name)
  if name.startswith("btri_"):
    blocks, size = (int(v) for v in name.split("_")[1:])
    rng = np.random.default_rng(blocks * 1000 + size)
    n = blocks * size
    hess = np.zeros((n, n))
    for k in range(blocks):
      x = rng.standard_normal((size, size))
      hess[k * size : (k + 1) * size, k * size : (k + 1) * size] = x @ x.T / size
      if k + 1 < blocks:
        e = 0.3 * rng.standard_normal((size, size)) / np.sqrt(size)
        hess[(k + 1) * size : (k + 2) * size, k * size : (k + 1) * size] = e
        hess[k * size : (k + 1) * size, (k + 1) * size : (k + 2) * size] = e.T
    hess += np.diag(0.2 * np.abs(hess).sum(axis=1) + 1.0)
    return _qp(name, sparse.csc_array(hess), rng.standard_normal(n), x_l=-np.ones(n), x_u=np.ones(n))
  if name.startswith("ex_"):
    import compare
    import problems as families

    case = families.cases()[name.removeprefix("ex_")]
    (P, c), (A, b), (G, g_lb, g_ub), (x_lb, x_ub), _ = compare.qp_data(case.problem)(case.params())

    def sp(m):
      return sparse.csc_array(np.where(np.abs(m) > 0.0, m, 0.0))

    return _qp(name, sp(P), c, A=sp(A), b=b, G=sp(G), h_l=g_lb, h_u=g_ub, x_l=x_lb, x_u=x_ub)
  return maros_meszaros(name)


def blob(flat: list[np.ndarray], out_sizes: list[int], workspace: int) -> bytes:
  head = np.array([len(flat), len(out_sizes), workspace, *(a.size for a in flat), *out_sizes], dtype="<i8").tobytes()
  return head + b"".join(np.ascontiguousarray(a, dtype="<f8").tobytes() for a in flat)


def one(name: str, backend: str, out: Path, split: bool = False) -> dict:
  import scaly as sc
  from scaly.codegen import render_c_module
  from scaly.codegen.abi import c_ident
  from scaly.codegen.jit import compile_flags
  from scaly.codegen.toolchain import find_c_compiler
  from scaly.opt.ipm import QPValues, Scaling, Solver
  from tests.opt.ipm.problems import ipm_inputs

  qp = problem(name)
  s, values = ipm_inputs(qp)
  t0 = time.perf_counter()
  syms = {k: sc.sym(k, np.shape(values[k])) for k in ORDER}
  solver = Solver(s, backend, name=f"g_{c_ident(name)}")
  qv = QPValues.preprocess(s, **syms)
  args = tuple(np.asarray(values[k], dtype=float) for k in ORDER)
  names, inputs, scaling = list(ORDER), [syms[k] for k in ORDER], None
  if split:  # PIQP's setup once, outside the timed solve, as PIQP's own timer counts
    setup = sc.Function.from_exprs(f"g_{c_ident(name)}_{backend}_setup", inputs, [solver.setup(qv).flat()], names, ["scaling"])
    scaling_sym = sc.sym("scaling", (Scaling.size(s),))
    scaling = Scaling.unflat(s, scaling_sym)
    args = (*args, np.asarray(setup(args), dtype=float))
    names, inputs = [*names, "scaling"], [*inputs, scaling_sym]
  res = solver.solve(qv, scaling=scaling)
  keys = ["x", "status", "iter"]
  fn = sc.Function.from_exprs(f"g_{c_ident(name)}_{backend}", inputs, [res[k] for k in keys], names, keys)
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
  parser.add_argument("--split", action="store_true", help="time the solve alone, with PIQP's setup (Ruiz) done once beforehand")
  parser.add_argument("--one", nargs=3, metavar=("PROBLEM", "BACKEND", "OUT"), help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.one:
    print(json.dumps({k: v for k, v in one(args.one[0], args.one[1], Path(args.one[2]), args.split).items() if k != "x"}))
    return
  names = list(QUICK) if args.problems in (None, "quick") else all_names() if args.problems == "all" else args.problems.split(",")
  cells = [(n, b) for n in names for b in args.backends.split(",")]
  base = BUILD / args.variant

  def run(cell: tuple[str, str]) -> str:
    name, backend = cell
    with tempfile.TemporaryDirectory(prefix="scaly-cache-") as cache:
      env = {**os.environ, "SCALY_CACHE_DIR": cache}
      t = time.perf_counter()
      done = subprocess.run(
        [sys.executable, __file__, "--variant", args.variant, *(["--split"] if args.split else []), "--one", name, backend, str(base / f"{name}_{backend}")],
        capture_output=True,
        text=True,
        env=env,
      )
    if done.returncode:
      return f"{name:>14} {backend:>6} FAILED {done.stderr.strip().splitlines()[-1][-300:] if done.stderr.strip() else done.returncode}"
    m = json.loads(done.stdout.strip().splitlines()[-1])
    return f"{name:>14} {backend:>6} {time.perf_counter() - t:6.1f} s  status {m['status']} iter {m['iter']:3d}  gen {m['generate_s']:.2f} s cc {m['compile_s']:.2f} s  {m['c_lines']} lines"

  with ThreadPoolExecutor(args.jobs) as pool:
    for line in pool.map(run, cells):
      print(line, flush=True)


if __name__ == "__main__":
  main()
