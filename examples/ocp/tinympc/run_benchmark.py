# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Generated TinyMPC (Scaly) against the TinyMPC library on the microcontroller-benchmark problems.

    uv run python examples/ocp/tinympc/run_benchmark.py [--families random_mpc,safety_filter,rocket_landing]
                                                       [--reps 5] [--opt -O3] [--quick]

For every problem instance of the sweeps in ``problems.py``:

1. The library is set up on the instance (``tiny_setup`` computes its Riccati cache, which the
   driver writes out), and Scaly generates ``tiny_solve`` for the instance on that same cache.
2. The closed loop runs once in Python with the generated solver (JIT), recording what each step
   feeds the solver. Both C drivers then replay that recorded sequence of warm-started solves, so
   the library and the generated code solve exactly the same problems, step by step.
3. Measured, per solver: object-code size (``size``: text, data, bss) of the solver alone,
   compile time (and for Scaly the Python generation time), solve time per MPC step (best of the
   repetitions per step, and the median over repetitions of the whole episode), ADMM iterations,
   and the largest difference of the applied input between the two.

The library is built once from a pinned commit, with one patch: the ``std::cout`` that ``solve``
prints on convergence is removed, since it would be timed with the solve. Both sides are compiled
by the same compiler with the same flags (``--opt``, the host CPU, ``-fno-math-errno``; Eigen also
gets ``-DNDEBUG``). Outputs go to ``examples/ocp/tinympc/benchmark/build/`` (not committed): a CSV and
a JSON per run, which ``report.py`` turns into an HTML report.
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmark"  # the drivers, the pinned library and the builds

from problem import Cache, Problem
from problems import (
  RANDOM_MPC_SWEEPS,
  ROCKET_LANDING_SWEEP,
  SAFETY_FILTER_SWEEPS,
  Scenario,
  closed_loop,
  random_mpc,
  rocket_landing,
  safety_filter,
)
from solver import Solver, build_solver

from scaly.codegen import render_c_module
from scaly.codegen.jit import HOST_CFLAGS
from scaly.ir.types import Lowering

TINYMPC_URL = "https://github.com/TinyMPC/TinyMPC.git"
TINYMPC_COMMIT = "023f36bd5b27267e1a96ed47a774d120b5707f13"  # main, 2026-09-02
BENCH_URL = "https://github.com/RoboticExplorationLab/mcu-solver-benchmarks.git"
BENCH_COMMIT = "dfa77a5c86e4c86226b928f02732f625b16e263b"
THIRD_PARTY = BENCH / "third_party"
BUILD = BENCH / "build"
TINYMPC_SOURCES = ("admm.cpp", "tiny_api.cpp", "rho_benchmark.cpp")
CONVERGED_PRINT = 'std::cout << "Solver converged in " << solver->work->iter << " iterations" << std::endl;'


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
  done = subprocess.run(cmd, capture_output=True, text=True, **kw)
  if done.returncode:
    raise RuntimeError(f"{' '.join(cmd)} failed ({done.returncode}):\n{done.stderr[-4000:]}")
  return done


def checkout(url: str, commit: str, dest: Path) -> Path:
  if not (dest / ".git").exists():
    dest.parent.mkdir(parents=True, exist_ok=True)
    sh(["git", "clone", "-q", url, str(dest)])
  if sh(["git", "-C", str(dest), "rev-parse", "HEAD"]).stdout.strip() != commit:
    sh(["git", "-C", str(dest), "fetch", "-q", "origin", commit])
    sh(["git", "-C", str(dest), "checkout", "-q", commit])
  return dest


def timed_compile(cmd: list[str], repeats: int = 3) -> float:
  """Best wall time of ``repeats`` compiles, in seconds."""
  best = float("inf")
  for _ in range(repeats):
    t = time.perf_counter()
    sh(cmd)
    best = min(best, time.perf_counter() - t)
  return best


def object_size(paths: list[Path]) -> dict[str, int]:
  out = {"text": 0, "data": 0, "bss": 0}
  for path in paths:
    text, data, bss = sh(["size", str(path)]).stdout.splitlines()[1].split()[:3]
    out["text"] += int(text)
    out["data"] += int(data)
    out["bss"] += int(bss)
  return out


def flags(opt: str) -> list[str]:
  return [opt, *HOST_CFLAGS]


# --- the library -----------------------------------------------------------------------------------


def build_tinympc(opt: str, cxx: str) -> dict:
  src = checkout(TINYMPC_URL, TINYMPC_COMMIT, THIRD_PARTY / "TinyMPC")
  out = BUILD / f"tinympc{opt}"
  lib = out / "src" / "tinympc"
  lib.mkdir(parents=True, exist_ok=True)
  for f in (src / "src" / "tinympc").iterdir():
    if f.suffix in (".cpp", ".hpp"):
      shutil.copy(f, lib / f.name)
  admm = (lib / "admm.cpp").read_text()
  if CONVERGED_PRINT not in admm:
    raise RuntimeError("the TinyMPC source changed: the convergence print to remove was not found")
  (lib / "admm.cpp").write_text(admm.replace(CONVERGED_PRINT, "/* print removed for timing */"))
  common = ["-std=c++17", *flags(opt), "-DNDEBUG", "-I", str(out / "src"), "-I", str(src / "include" / "Eigen")]
  objects, seconds = [], {}
  for name in TINYMPC_SOURCES:
    obj = out / (Path(name).stem + ".o")
    seconds[name] = timed_compile([cxx, *common, "-c", str(lib / name), "-o", str(obj)])
    objects.append(obj)
  driver = out / "tinympc_driver"
  sh([cxx, *common, "-I", str(BENCH), str(BENCH / "tinympc_driver.cpp"), *map(str, objects), "-o", str(driver)])
  return {"driver": driver, "compile_s": sum(seconds.values()), "compile_s_by_file": seconds, "size": object_size(objects)}


# --- episodes --------------------------------------------------------------------------------------


def write_episode(path: Path, s: Scenario, x0: np.ndarray, xref: np.ndarray, uref: np.ndarray, fixed_bounds: bool = False) -> None:
  p, st = s.problem, s.problem.settings
  header = [
    0x594E4954,
    1,
    p.nx,
    p.nu,
    p.N,
    len(x0),
    st.max_iter,
    int(st.en_state_bound),
    int(st.en_input_bound),
    len(p.state_cones),
    len(p.input_cones),
  ]
  header.append(int(fixed_bounds))
  parts = [np.array([p.rho, st.abs_pri_tol, st.abs_dua_tol]), p.A, p.B, p.fdyn, p.Q, p.R]
  parts += [np.array([c.start, c.dim, c.mu], float) for c in (*p.state_cones, *p.input_cones)]
  parts += [s.bounds.x_min, s.bounds.x_max, s.bounds.u_min, s.bounds.u_max]
  steps = [np.concatenate([x0[k], xref[k].ravel(), uref[k].ravel()]) for k in range(len(x0))]
  with path.open("wb") as fp:
    fp.write(np.array(header, dtype="<i4").tobytes())
    for part in parts + steps:
      fp.write(np.ascontiguousarray(part, dtype="<f8").tobytes())


def read_cache(path: Path, p: Problem) -> Cache:
  v = np.fromfile(path, dtype="<f8")
  nx, nu, at = p.nx, p.nu, 0
  out = []
  for shape in ((nu, nx), (nx, nx), (nu, nu), (nx, nx), (nx,), (nu,)):
    size = int(np.prod(shape))
    out.append(v[at : at + size].reshape(shape))
    at += size
  return Cache(*out)


def read_results(path: Path) -> dict:
  raw = path.read_bytes()
  steps, nu, reps = np.frombuffer(raw[:12], dtype="<i4")
  v = np.frombuffer(raw[12:], dtype="<f8")
  rows = v[: steps * (3 + nu)].reshape(steps, 3 + nu)
  return {"iterations": rows[:, 0], "solved": rows[:, 1] > 0.5, "best_ns": rows[:, 2], "u0": rows[:, 3:], "total_ns": v[steps * (3 + nu) :]}


# --- one instance ----------------------------------------------------------------------------------


def run_instance(family: str, s: Scenario, lib: dict, opt: str, cc: str, reps: int, lowering: Lowering = "auto", fixed_bounds: bool = False) -> dict:
  variant = lowering + ("-fixed-bounds" if fixed_bounds else "")
  out = BUILD / f"{family}{opt}-{variant}" / s.name
  out.mkdir(parents=True, exist_ok=True)
  p = s.problem
  empty = np.zeros((0, p.nx)), np.zeros((0, p.N, p.nx)), np.zeros((0, p.N - 1, p.nu))
  write_episode(out / "setup.bin", s, *empty)
  sh([str(lib["driver"]), str(out / "setup.bin"), str(out / "setup.out"), "0", str(out / "cache.bin")])
  c = read_cache(out / "cache.bin", p)

  # Generation runs in a fresh process: Scaly caches lowered callees, and the stage steps of one
  # instance can be those of the previous one, which would flatter every instance but the first.
  with (out / "generate.pkl").open("wb") as fp:
    pickle.dump({"problem": p, "cache": c, "lowering": lowering, "bounds": s.bounds if fixed_bounds else None, "out": str(out)}, fp)
  gen = json.loads(sh([sys.executable, str(Path(__file__).resolve()), "--generate", str(out / "generate.pkl")]).stdout.splitlines()[-1])
  generate_s, c_lines, workspace = gen["seconds"], gen["c_lines"], gen["workspace_doubles"]
  obj = out / "tinympc_solve.o"
  compile_s = timed_compile([cc, "-std=c11", *flags(opt), "-c", str(out / "tinympc_solve.c"), "-o", str(obj)])
  driver = out / "scaly_driver"
  sh([cc, "-std=c11", *flags(opt), "-I", str(out), "-I", str(BENCH), str(BENCH / "scaly_driver.c"), str(obj), "-lm", "-o", str(driver)])

  t = time.perf_counter()
  episode = closed_loop(s, Solver(p, s.bounds, c, lowering=lowering, fixed_bounds=fixed_bounds))
  loop_s = time.perf_counter() - t
  write_episode(out / "episode.bin", s, episode.x0, episode.xref, episode.uref, fixed_bounds)

  runs = {}
  for name, cmd in (("tinympc", [str(lib["driver"])]), ("scaly", [str(driver)])):
    sh([*cmd, str(out / "episode.bin"), str(out / f"{name}.out"), str(reps)])
    runs[name] = read_results(out / f"{name}.out")
  a, b = runs["tinympc"], runs["scaly"]
  steps = len(episode.x0)

  def timing(r: dict) -> dict:
    total = statistics.median(r["total_ns"]) if len(r["total_ns"]) else float("nan")
    iters = float(r["iterations"].sum())
    return {
      "iterations": iters,
      "solved": int(r["solved"].sum()),
      "mean_solve_us": total / steps / 1e3,
      "best_mean_solve_us": float(r["best_ns"].mean()) / 1e3,
      "max_best_solve_us": float(r["best_ns"].max()) / 1e3,
      "per_iteration_ns": total / max(iters, 1.0),
    }

  row = {
    "family": family,
    "name": s.name,
    "nx": p.nx,
    "nu": p.nu,
    "N": p.N,
    "steps": steps,
    "opt": opt,
    "lowering": lowering,
    "fixed_bounds": fixed_bounds,
    "tinympc": {**timing(a), "compile_s": lib["compile_s"], "size": lib["size"]},
    "scaly": {**timing(b), "generate_s": generate_s, "compile_s": compile_s, "size": object_size([obj]), "c_lines": c_lines, "workspace_doubles": workspace},
    "agreement": {
      "max_u0_diff": float(np.abs(a["u0"] - b["u0"]).max()),
      "max_u0_diff_vs_python": float(np.abs(b["u0"] - episode.u0).max()),
      "same_iterations": int((a["iterations"] == b["iterations"]).sum()),
    },
    "python_closed_loop_s": loop_s,
  }
  return row


def instances(family: str, quick: bool) -> list[Scenario]:
  if family == "random_mpc":
    data = checkout(BENCH_URL, BENCH_COMMIT, THIRD_PARTY / "mcu-solver-benchmarks") / "ICRA_benchmarks" / "qp_mpc_problem" / "random_problems"
    out, seen = [], set()
    for axis, triples in RANDOM_MPC_SWEEPS.items():
      for nx, nu, n in triples[:2] if quick else triples:
        if (nx, nu, n) in seen:
          continue
        seen.add((nx, nu, n))
        folder = {"nx": f"prob_nx_{nx}", "nu": f"prob_nu_{nu}", "N": f"prob_Nh_{n}"}[axis]
        out.append(random_mpc(nx, nu, n, data=data / folder / "rand_prob_osqp_params.npz"))
    return out
  if family == "safety_filter":
    seen = []
    for pairs in SAFETY_FILTER_SWEEPS.values():
      seen += [pn for pn in (pairs[:2] if quick else pairs) if pn not in seen]
    return [safety_filter(nx, n) for nx, n in seen]
  if family == "rocket_landing":
    return [rocket_landing(n) for n in (ROCKET_LANDING_SWEEP[:3] if quick else ROCKET_LANDING_SWEEP)]
  raise ValueError(family)


def generate(spec: Path) -> None:
  """``--generate SPEC``: build and render the solver described by a pickle, timing that alone."""
  with spec.open("rb") as fp:
    job = pickle.load(fp)
  t = time.perf_counter()
  module = render_c_module(build_solver(job["problem"], job["cache"], lowering=job["lowering"], fixed_bounds=job["bounds"]))
  seconds = time.perf_counter() - t
  out = Path(job["out"])
  (out / module.header_name).write_text(module.header)
  (out / module.source_name).write_text(module.source)
  print(json.dumps({"seconds": seconds, "c_lines": module.source.count("\n"), "workspace_doubles": module.workspace_size}))


def main() -> None:
  if len(sys.argv) == 3 and sys.argv[1] == "--generate":
    generate(Path(sys.argv[2]))
    return
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--families", default="random_mpc,safety_filter,rocket_landing")
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--opt", default="-O3", help="optimization flag for both sides; write it as --opt=-O2")
  ap.add_argument("--quick", action="store_true", help="two instances per sweep, for a toolchain check")
  ap.add_argument("--only", default="", help="regular expression on instance names")
  ap.add_argument("--cc", default="gcc")
  ap.add_argument("--cxx", default="g++")
  ap.add_argument("--workdir", type=Path, default=BENCH, help="where build/ and third_party/ go (local disk keeps compile times honest)")
  ap.add_argument("--lowering", default="auto", choices=("auto", "block", "scalar"), help="lowering of the generated stage steps")
  ap.add_argument("--fixed-bounds", action="store_true", help="bounds as constants of the generated code (the library still gets them at run time)")
  ap.add_argument("--budget", type=float, default=float("inf"), help="start no new instance after this many seconds; rerun to resume")
  args = ap.parse_args()
  global BUILD, THIRD_PARTY
  BUILD, THIRD_PARTY = args.workdir / "build", args.workdir / "third_party"
  BUILD.mkdir(parents=True, exist_ok=True)
  start = time.perf_counter()
  results = BUILD / f"results{args.opt}-{args.lowering}{'-fixed-bounds' if args.fixed_bounds else ''}.json"
  rows = json.loads(results.read_text())["rows"] if results.exists() else []
  done = {r["name"] for r in rows}
  lib_json = BUILD / f"tinympc{args.opt}" / "lib.json"
  if lib_json.exists() and json.loads(lib_json.read_text()).get("commit") == TINYMPC_COMMIT:
    lib = json.loads(lib_json.read_text())
    lib["driver"] = Path(lib["driver"])
  else:
    lib = build_tinympc(args.opt, args.cxx)
    lib_json.write_text(json.dumps(lib | {"driver": str(lib["driver"]), "commit": TINYMPC_COMMIT}))
  print(f"TinyMPC library: {lib['compile_s']:.2f} s to compile, {lib['size']}", flush=True)
  todo = [(f, s) for f in args.families.split(",") for s in instances(f, args.quick) if s.name not in done and re.search(args.only, s.name)]
  for i, (family, s) in enumerate(todo):
    if time.perf_counter() - start > args.budget:
      print(f"budget spent: {len(todo) - i} instances left", flush=True)
      return
    row = run_instance(family, s, lib, args.opt, args.cc, args.reps, args.lowering, args.fixed_bounds)
    rows.append(row)
    t, g = row["tinympc"], row["scaly"]
    print(
      f"{s.name:32s} iters {t['iterations']:7.0f} / {g['iterations']:7.0f}  solve {t['mean_solve_us']:9.2f} / {g['mean_solve_us']:9.2f} us"
      f"  ({t['mean_solve_us'] / g['mean_solve_us']:5.2f}x)  text {t['size']['text']} / {g['size']['text']}"
      f"  compile {t['compile_s']:.2f} / {g['generate_s']:.2f}+{g['compile_s']:.2f} s  du {row['agreement']['max_u0_diff']:.1e}",
      flush=True,
    )
    results.write_text(json.dumps(meta(args) | {"rows": rows}, indent=1))
  print("all instances done", flush=True)


def meta(args: argparse.Namespace) -> dict:
  cc = sh([args.cc, "--version"]).stdout.splitlines()[0]
  return {
    "tinympc_commit": TINYMPC_COMMIT,
    "benchmarks_commit": BENCH_COMMIT,
    "compiler": cc,
    "flags": flags(args.opt),
    "machine": platform.machine(),
    "system": platform.platform(),
    "python": platform.python_version(),
    "reps": args.reps,
  }


if __name__ == "__main__":
  main()
