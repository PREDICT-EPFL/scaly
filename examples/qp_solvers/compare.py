"""The QP families of ``problems.py`` through five solvers: build, generate, compile, size, solve, check.

    uv run examples/qp_solvers/compare.py [--cases mpc_N20,portfolio,...] [--solvers piqp_sparse,...]
                                          [--budget 2.0] [--out examples/qp_solvers/build/results.json]

The solvers, each a plain Scaly ``Function`` from the problem's parameters to its solution:

| Key | Solver | Behind the generated C |
| --- | --- | --- |
| ``piqp_sparse``, ``piqp_dense`` | ``sc.opt.solver(problem, "piqp")``, sparse and dense | oracles for the QP data, and a call into the vendored PIQP 0.6.2 library |
| ``scaly_sparse``, ``scaly_dense`` | ``generated_piqp.solver(problem, backend)`` | nothing: PIQP's algorithm is the generated C, specialised to the problem's sparsity |
| ``ipopt`` | ``sc.opt.solver(problem, "ipopt")`` | oracles, and a call into the vendored IPOPT 3.14 with MUMPS |

Every (case, solver) cell runs in a fresh process with an empty JIT cache, so nothing a previous
cell lowered or compiled is reused. Measured per cell:

- **build**: constructing the ``Function`` in Python (proving the problem quadratic, extracting the
  data, for the generated solver also tracing the algorithm);
- **generate**: lowering and rendering to C (``render_c_module``);
- **C code**: lines and bytes of the generated translation unit, written with ``write_module`` to
  ``examples/generated/qp_solvers/<case>/<solver>/``;
- **compile**: ``cc`` with the JIT's flags on that translation unit, best of two, and the object
  size (``size``: text + data) of the result, with the shared libraries it links from the solver
  plugins counted separately;
- **solve**: the entry point called from C by ``time_entry.c`` (no Python in the loop), best and
  median over as many cold solves as fit in ``--budget`` seconds (5 to 2000); and the same through
  the Python ``Function`` call, for reference;
- **result**: status, iterations, the objective and the primal residual of the returned ``x``
  against the extracted QP, and ``x`` itself for cross-checks in the notebook.

``compare.ipynb`` runs this and reads the JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GENERATED = HERE.parent / "generated" / "qp_solvers"
BUILD = HERE / "build"

SOLVERS = {
  "piqp_sparse": "PIQP library, sparse",
  "piqp_dense": "PIQP library, dense",
  "scaly_sparse": "generated PIQP, sparse",
  "scaly_dense": "generated PIQP, dense",
  "ipopt": "IPOPT library (MUMPS)",
}
REFERENCE = "reference"  # vendored sparse PIQP at 1e-12, for the answers only
TIGHT = {"eps_abs": 1e-12, "eps_rel": 1e-12, "eps_duality_gap_abs": 1e-12, "eps_duality_gap_rel": 1e-12}
IPOPT_OPTIONS = {"tol": 1e-8, "print_level": 0, "hessian_constant": "yes", "jac_c_constant": "yes", "jac_d_constant": "yes"}


def all_cases() -> dict[str, Any]:
  import problems

  out = problems.cases()
  for horizon in (5, 10, 20, 40):
    case = problems.mpc(horizon)
    out.setdefault(case.name, case)
  return out


def build(case: Any, key: str) -> Any:
  import scaly as sc

  import generated_piqp

  name = f"{case.name}_{key}"
  if key in ("piqp_sparse", "piqp_dense"):
    return sc.opt.solver(case.problem, sc.opt.PIQP(sparse=key == "piqp_sparse"), name=name)
  if key == REFERENCE:
    return sc.opt.solver(case.problem, sc.opt.PIQP(sparse=True, options={**TIGHT}), name=name)
  if key in ("scaly_sparse", "scaly_dense"):
    return generated_piqp.solver(case.problem, "sparse" if key == "scaly_sparse" else "dense", name=name)
  if key == "ipopt":
    return sc.opt.solver(case.problem, sc.opt.IPOPT(options=IPOPT_OPTIONS), name=name)
  raise KeyError(key)


def call_args(fun: Any, case: Any, key: str) -> tuple[tuple[Any, ...], list[np.ndarray]]:
  """The call's arguments, one per parameter, and their flat arrays, in the order of the entry's inputs."""
  params = case.params()
  if key.startswith("scaly_"):
    flat = [np.asarray(a, dtype=float) for a in case.problem.params.flatten_numerical(params, "parameters")]
    return fun.input_tree.unflatten(tuple(flat)), flat
  p = case.problem
  zeros = p.vars.unflatten(tuple(np.zeros(s) for s in p.vars.shapes))
  args = (zeros, zeros, np.zeros(p.n_eq), np.zeros(p.n_ineq), params)
  return args, [np.asarray(a, dtype=float) for a in fun.input_tree.flatten_numerical(args, "solver inputs")]


def flat_x(out: Any, key: str, problem: Any) -> np.ndarray:
  if key.startswith("scaly_"):
    return np.asarray(out[0], dtype=float)
  return np.concatenate([np.ravel(v) for v in problem.vars.flatten_numerical(out[0], "solution")])


def sh(cmd: list[str]) -> str:
  done = subprocess.run(cmd, capture_output=True, text=True)
  if done.returncode:
    raise RuntimeError(f"{' '.join(cmd)} failed:\n{done.stderr[-3000:]}")
  return done.stdout


def object_size(path: Path) -> int:
  """text + data of a shared object, as ``size`` reports them."""
  text, data = sh(["size", str(path)]).splitlines()[1].split()[:2]
  return int(text) + int(data)


def plugin_libraries(lib: Path) -> dict[str, int]:
  """The shared libraries ``lib`` loads from the solver plugins' packages, with their object sizes
  (system runtimes such as libc, libm, libstdc++ are left out)."""
  out: dict[str, int] = {}
  if platform.system() != "Linux":
    return out
  for line in sh(["ldd", str(lib)]).splitlines():
    m = re.match(r"\s*(\S+) => (\S+)", line)
    if m and ("scaly_piqp" in m.group(2) or "scaly_ipopt" in m.group(2)):
      out[m.group(1)] = object_size(Path(m.group(2)))
  return out


def time_driver() -> Path:
  exe = BUILD / "time_entry"
  if not exe.exists() or exe.stat().st_mtime < (HERE / "time_entry.c").stat().st_mtime:
    BUILD.mkdir(parents=True, exist_ok=True)
    sh([os.environ.get("SCALY_CC", "cc"), "-O2", "-o", str(exe), str(HERE / "time_entry.c"), "-ldl"])
  return exe


def residuals(x: np.ndarray, data: Any) -> dict[str, float]:
  """The problem's objective at ``x`` and its largest constraint violation, from ``qp_data``."""
  (P, c), (A, b), (G, g_lb, g_ub), (x_lb, x_ub), f0 = data
  viol = [np.abs(A @ x - b)] if A.size else []
  if G.size:
    gx = G @ x
    viol += [np.maximum(g_lb - gx, 0.0), np.maximum(gx - g_ub, 0.0)]
  viol += [np.maximum(x_lb - x, 0.0), np.maximum(x - x_ub, 0.0)]
  return {"objective": float(0.5 * x @ P @ x + c @ x + f0), "primal_res": float(max((v.max() for v in viol if v.size), default=0.0))}


def measure(case_name: str, key: str, budget: float) -> dict[str, Any]:
  """One cell, in this (fresh) process."""
  from scaly.codegen import render_c_module, write_module
  from scaly.codegen.abi import c_ident
  from scaly.codegen.jit import compile_flags
  from scaly.opt import solver_stats

  import generated_piqp

  case = all_cases()[case_name]
  t = time.perf_counter()
  fun = build(case, key)
  t_build = time.perf_counter() - t
  t = time.perf_counter()
  module = render_c_module(fun)
  t_generate = time.perf_counter() - t
  row: dict[str, Any] = {"case": case_name, "solver": key, "build_s": t_build, "generate_s": t_generate}

  if key != REFERENCE:
    write_module(fun, GENERATED / case_name / key)
  code = module.body
  row["c_lines"] = sum(1 for line in code.splitlines() if line.strip())
  row["c_bytes"] = len(code.encode())
  row["workspace_doubles"] = int(module.workspace_size)

  work = Path(tempfile.mkdtemp(prefix=f"{case_name}_{key}_"))
  src, lib = work / f"{c_ident(fun.name)}.c", work / f"lib{c_ident(fun.name)}.so"
  src.write_text(code)
  cc = os.environ.get("SCALY_CC", "cc")
  cmd = [cc, *compile_flags(), "-fPIC", "-shared", str(src), *module.link_flags, "-lm", "-o", str(lib)]
  times = []
  for _ in range(2):
    t = time.perf_counter()
    sh(cmd)
    times.append(time.perf_counter() - t)
  row["compile_s"] = min(times)
  row["object_bytes"] = object_size(lib)
  row["library_bytes"] = plugin_libraries(lib)

  args, flat = call_args(fun, case, key)
  out = fun(*args)  # JIT: compiles once more, into this process's empty cache
  x = flat_x(out, key, case.problem)
  if key.startswith("scaly_"):
    status, iters = int(out[4]), int(out[5])
    row["status"] = "solved" if status == 1 else f"PIQP status {status}"
  else:
    stats = solver_stats(fun)
    iters = int(stats.iter)
    row["status"] = "solved" if stats.status.name in ("OK", "ACCEPTABLE") else stats.status.name.lower()
  row["iterations"] = iters
  row["x"] = x.tolist()
  row.update(residuals(x, generated_piqp.qp_data(case.problem)(case.params())))

  # Solve time, through Python and from C.
  t = time.perf_counter()
  fun(*args)
  once = max(time.perf_counter() - t, 1e-7)
  reps = int(min(2000, max(5, budget / once)))
  py = []
  for _ in range(reps):
    t = time.perf_counter()
    fun(*args)
    py.append(time.perf_counter() - t)
  row["python_best_s"], row["python_median_s"] = min(py), statistics.median(py)
  if not key.startswith("scaly_"):
    row["native_total_s"] = float(solver_stats(fun).t_total)  # the solver's own timer, last solve

  sizes_in = [a.size for a in flat]
  sizes_out = [int(e.size) for e in fun.outputs]
  blob = np.array([len(flat), len(sizes_out), module.workspace_size, *sizes_in, *sizes_out], dtype="<i8").tobytes()
  blob += b"".join(np.ascontiguousarray(a, dtype="<f8").tobytes() for a in flat)
  (work / "inputs.bin").write_bytes(blob)
  best, median = sh([str(time_driver()), str(lib), c_ident(fun.name), str(work / "inputs.bin"), str(reps), str(work / "outputs.bin")]).split()
  row["c_best_s"], row["c_median_s"], row["repeats"] = float(best) * 1e-9, float(median) * 1e-9, reps
  c_out = np.fromfile(work / "outputs.bin", dtype="<f8")
  row["c_matches_python"] = bool(np.allclose(c_out[: x.size], x, rtol=0, atol=1e-12))
  shutil.rmtree(work, ignore_errors=True)
  return row


def run_cell(case_name: str, key: str, budget: float) -> dict[str, Any]:
  """``measure`` in a fresh process with an empty JIT cache."""
  with tempfile.TemporaryDirectory(prefix="scaly-cache-") as cache:
    env = {**os.environ, "SCALY_CACHE_DIR": cache}
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", case_name, key, "--budget", str(budget)]
    done = subprocess.run(cmd, capture_output=True, text=True, env=env)
  if done.returncode:
    return {"case": case_name, "solver": key, "failed": (done.stderr.strip().splitlines() or [f"exit {done.returncode}"])[-1][-400:]}
  return json.loads(done.stdout.strip().splitlines()[-1])


def machine() -> dict[str, str]:
  cc = os.environ.get("SCALY_CC", "cc")
  from scaly.codegen.jit import compile_flags

  cpu = platform.processor() or platform.machine()
  if Path("/proc/cpuinfo").exists():
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), cpu)
  elif platform.system() == "Darwin":
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or cpu
  return {
    "platform": platform.platform(),
    "cpu": cpu,
    "compiler": subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0],
    "flags": " ".join(compile_flags()),
    "python": platform.python_version(),
  }


def run(
  cases: list[str] | None = None, solvers: list[str] | None = None, budget: float = 2.0, out: Path | None = None, verbose: bool = True
) -> dict[str, Any]:
  """Every requested cell, one fresh process each, plus the reference answers; written to ``out``."""
  cases = cases or list(all_cases())
  solvers = solvers or list(SOLVERS)
  rows = []
  for case_name in cases:
    for key in [*solvers, REFERENCE]:
      t = time.perf_counter()
      row = run_cell(case_name, key, budget if key != REFERENCE else 0.05)
      rows.append(row)
      if verbose:
        note = row.get("failed") or f"{row['status']}, {row['iterations']} it, {1e6 * row['c_best_s']:.1f} us"
        print(f"{case_name:>12} {key:>13}  {time.perf_counter() - t:6.1f} s   {note}", flush=True)
  result = {"machine": machine(), "solvers": SOLVERS, "rows": rows}
  if out is not None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--cases", help="comma-separated case names (default: all)")
  parser.add_argument("--solvers", help=f"comma-separated, of {', '.join(SOLVERS)} (default: all)")
  parser.add_argument("--budget", type=float, default=2.0, help="seconds of solves per timing (default 2)")
  parser.add_argument("--out", type=Path, default=BUILD / "results.json")
  parser.add_argument("--one", nargs=2, metavar=("CASE", "SOLVER"), help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.one:
    case_name, key = args.one
    print(json.dumps(measure(case_name, key, args.budget)))
    return
  run(args.cases.split(",") if args.cases else None, args.solvers.split(",") if args.solvers else None, args.budget, args.out)
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
