from __future__ import annotations

# ruff: noqa: E402 -- direct execution must add the repository root before package imports

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import multiprocessing
import os
import random
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import numpy as np

from scaly.function.model import as_concrete
import scaly as sc
from scaly.codegen.aot import render_c_module
from scaly.solvers.graph import solver_compile_flags
from scaly.solvers.paths import solver_loadable
from benchmarks.harness import (
  CLOSED_LOOP_PAIRS,
  SMOKE_RESULTS,
  SWEEP_RESULTS,
  closed_loop_results_root,
  configure_math_policy,
  gbench,
  solver_oracle_name,
)
from benchmarks.harness.closed_loop import run as run_closed_loop
from benchmarks.harness.recording import layout_path
from benchmarks.harness.provenance import collect, write, require_headline_settings
from benchmarks.harness.report import report
from benchmarks.harness.study import PARTS, PROBLEMS, run_study
from benchmarks.harness.sweep import BACKENDS, DEFAULT_SIZES, build_kernel, run_cell, run_sweep
from benchmarks.problems import chain, npmpc


def _csv(value: str) -> list[str]:
  return [item.strip() for item in value.split(",") if item.strip()]


def _ints(value: str) -> list[int]:
  try:
    return [int(item) for item in _csv(value)]
  except ValueError as e:
    raise argparse.ArgumentTypeError("expected comma-separated integers") from e


def _choices(values: list[str], allowed: tuple[str, ...], parser: argparse.ArgumentParser, flag: str) -> list[str]:
  invalid = [value for value in values if value not in allowed]
  if invalid:
    parser.error(f"{flag}: invalid choice(s): {', '.join(invalid)} (choose from {', '.join(allowed)})")
  return values


NX, NU, N_OBSTACLES = 4, 4, 3
SAFETY_MARGIN, ALPHA = 0.5, 1.0


def _qp_filter() -> sc.Function:
  obstacles = np.array([[1.0, 1.0], [-1.0, 1.5], [0.0, -2.0]], dtype=np.float64)

  @sc.function(sc.group(sc.arg("x", (NX,)), sc.arg("u_ref", (NU,))), outputs=sc.arg("u", NU), name="smoke_safety_filter_qp")
  def safety_filter_qp(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, u_ref = inputs
    cars = sc.stack([sc.stack([x[2 * i], x[2 * i + 1]], axis=0) for i in range(2)], axis=0)
    rows, bias = [], []
    for car in range(2):
      for obstacle in obstacles:
        diff = cars[car] - sc.const(obstacle)
        grad = 2.0 * diff
        rows.append(sc.stack([grad[d] if k == car else sc.const(0.0) for k in range(2) for d in range(2)], axis=0))
        bias.append(ALPHA * (sc.dot(diff, diff) - sc.const(SAFETY_MARGIN**2)))

    @sc.problem(vars=sc.arg("u", NU), name="smoke_safety_filter_qp_problem")
    def problem(u: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
      return sc.ProblemSpec(
        minimize=0.5 * sc.dot(u, u) - sc.dot(u_ref, u),
        ineq=(sc.bounded(sc.stack(rows, axis=0) @ u, lo=-sc.stack(bias, axis=0), name="obstacles"),),
      )

    solve = sc.solver(problem, "piqp", name="smoke_safety_filter_qp_solver")
    params = problem.params.unflatten(tuple({"x": x, "u_ref": u_ref}[name] for name in problem.params.names))
    return solve(params)[0]

  return safety_filter_qp


def _nlp_filter() -> sc.Function:
  obstacles = np.array([[1.0, 1.0], [-1.0, 1.5], [0.0, -2.0]], dtype=np.float64)

  @sc.function(sc.group(sc.arg("x", (NX,)), sc.arg("u_ref", (NU,))), outputs=sc.arg("u", NU), name="smoke_safety_filter_nlp")
  def safety_filter_nlp(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, u_ref = inputs
    cars = sc.stack([sc.stack([x[2 * i], x[2 * i + 1]], axis=0) for i in range(2)], axis=0)

    @sc.problem(vars=sc.arg("u", (NU,)), name="smoke_safety_filter_problem")
    def problem(u: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
      rows = []
      for car in range(2):
        for obstacle in obstacles:
          diff = cars[car] - sc.const(obstacle)
          grad = 2.0 * diff
          rows.append(grad[0] * u[2 * car] + grad[1] * u[2 * car + 1] + ALPHA * (sc.dot(diff, diff) - sc.const(SAFETY_MARGIN**2)))
      return sc.ProblemSpec(
        minimize=0.5 * sc.dot(u - u_ref, u - u_ref),
        ineq=(
          sc.bounded(
            sc.stack(rows, axis=0),
            lo=sc.const(np.zeros(len(rows))),
            hi=sc.const(np.full(len(rows), 1e30)),
            name="obstacles",
          ),
        ),
      )

    nlp = sc.solver(problem, "ipopt", name="smoke_safety_filter_nlp_solver")
    return nlp(problem.params.unflatten(tuple({"x": x, "u_ref": u_ref}[name] for name in problem.params.names)))[0]

  return safety_filter_nlp


def _solver_call_smoke(required: bool) -> str | None:
  if not solver_loadable("piqp") or not solver_loadable("ipopt"):
    if required:
      raise RuntimeError("PIQP/IPOPT solver plugins are not loadable but solver_call was explicitly selected")
    return "solver_call skipped: PIQP/IPOPT solver plugins are not loadable"
  out_dir = SMOKE_RESULTS / "solver_call"
  out_dir.mkdir(parents=True, exist_ok=True)
  functions = [_qp_filter(), _nlp_filter()]
  sources, flags = [], []
  for fun in functions:
    module = render_c_module(fun)
    (out_dir / module.header_name).write_text(module.header)
    (out_dir / module.source_name).write_text(module.source)
    sources.append(module.source_name)
    for flag in solver_compile_flags(as_concrete(fun)):
      if flag not in flags:
        flags.append(flag)
  cpp = out_dir / "solver_smoke.cpp"
  cpp.write_text(
    "\n".join(
      [
        "#include <array>",
        "#include <cmath>",
        f'#include "{functions[0].name}.h"',
        f'#include "{functions[1].name}.h"',
        "int main() {",
        "  std::array<double, 4> x{0.0, 0.0, 2.0, 0.0}, ref{0.1, -0.1, 0.2, 0.0}, out{};",
        "  const double* arg[] = {x.data(), ref.data()};",
        "  double* res[] = {out.data()};",
        f"  std::array<double, ({functions[0].name}_SZ_W > 0 ? {functions[0].name}_SZ_W : 1)> w_qp{{}};",
        f"  if ({functions[0].name}(arg, res, nullptr, w_qp.data(), 0) != 0) return 1;",
        "  for (double value : out) if (!std::isfinite(value)) return 2;",
        f"  std::array<double, ({functions[1].name}_SZ_W > 0 ? {functions[1].name}_SZ_W : 1)> w_nlp{{}};",
        f"  if ({functions[1].name}(arg, res, nullptr, w_nlp.data(), 0) != 0) return 3;",
        "  for (double value : out) if (!std::isfinite(value)) return 4;",
        "  return 0;",
        "}",
      ]
    )
    + "\n"
  )
  exe = out_dir / "solver_smoke"
  cmd = [gbench.compiler(), "-O2", "-std=c++17", "-I", str(out_dir), cpp.name, *sources, "-o", str(exe), *flags, "-lm"]
  proc = subprocess.run(cmd, cwd=out_dir, text=True, capture_output=True)
  (out_dir / "compile.log").write_text(f"$ {shlex.join(cmd)}\n{proc.stdout}{proc.stderr}")
  if proc.returncode:
    raise RuntimeError(f"solver_call compile failed: {proc.stderr[-500:].strip()}")
  run = subprocess.run([str(exe)], cwd=out_dir, text=True, capture_output=True, timeout=120)
  if run.returncode:
    raise RuntimeError(f"solver_call binary exited {run.returncode}: {(run.stderr + run.stdout)[-500:].strip()}")
  return None


def _benchmark_smoke() -> None:
  for backend in ("scaly", "casadi_sx", "casadi_call_mx", "casadi_map_sx"):
    result, info = run_cell(
      "race_cars",
      5,
      backend,
      SMOKE_RESULTS / "race_cars" / f"{backend}_N5",
      codegen_timeout=300,
      compile_timeout=180,
      max_source_mb=50,
      benchmark_min_time="0.01s",
      load_harvested=False,
    )
    runtime = f", runtime_ns={result['runtime_ns']}" if result["runtime_ns"] else ""
    print(f"smoke race_cars size=5 backend={backend}: {result['runtime_status']}{runtime}" + (f" ({result['note']})" if result["note"] else ""))
    if result["runtime_status"] != "ok" or info is None:
      raise RuntimeError(f"race_cars {backend} smoke failed: {result['note']}")
    assert info["nnz"] > 0, "race_cars nnz must be positive"
    assert info["w_size"] is not None, "race_cars workspace must be recorded"
    assert info["nnz"] < info["n_rows"] * info["n_cols"], "race_cars Hessian must be sparse"
  race_jac = []
  for size in (5, 50):
    out_dir = SMOKE_RESULTS / "race_cars" / f"scaly_jac_N{size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    race_jac.append(build_kernel("race_cars_jac", size, "scaly", out_dir))
  assert race_jac[1]["source_lines"] < 1.2 * race_jac[0]["source_lines"], (
    f"race_cars Jacobian loop preservation regressed: N=50 has {race_jac[1]['source_lines']} lines, N=5 has {race_jac[0]['source_lines']}"
  )
  assert int(race_jac[1]["w_size"]) <= 3 * 2100, f"race_cars Jacobian workspace regressed: N=50 needs {race_jac[1]['w_size']} doubles"
  print(f"smoke race_cars Jacobian loop preservation: ok ({race_jac[0]['source_lines']} lines at N=5, {race_jac[1]['source_lines']} at N=50)")
  for backend, size in (("scaly", 5), ("casadi_map_sx", 5), ("casadi_sx", 3)):
    result, info = run_cell(
      "chain",
      size,
      backend,
      SMOKE_RESULTS / "chain" / f"{backend}_M{size}",
      codegen_timeout=300,
      compile_timeout=180,
      max_source_mb=50,
      benchmark_min_time="0.01s",
      load_harvested=False,
    )
    runtime = f", runtime_ns={result['runtime_ns']}" if result["runtime_ns"] else ""
    print(f"smoke chain size={size} backend={backend}: {result['runtime_status']}{runtime}" + (f" ({result['note']})" if result["note"] else ""))
    if result["runtime_status"] != "ok" or info is None:
      raise RuntimeError(f"chain {backend} M={size} smoke failed: {result['note']}")
    assert info["nnz"] > 0 and info["w_size"] is not None and info["nnz"] < info["n_rows"] * info["n_cols"]
  assert chain.n_state(5) == 21 and chain.NU == 3
  chain_jac = []
  for size in (5, 33):
    out_dir = SMOKE_RESULTS / "chain" / f"scaly_jac_M{size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    chain_jac.append(build_kernel("chain_jac", size, "scaly", out_dir))
  assert chain_jac[1]["source_lines"] < 1.2 * chain_jac[0]["source_lines"], (
    f"chain Jacobian loop preservation regressed: M=33 has {chain_jac[1]['source_lines']} lines, M=5 has {chain_jac[0]['source_lines']}"
  )
  print(f"smoke chain Jacobian loop preservation: ok ({chain_jac[0]['source_lines']} lines at M=5, {chain_jac[1]['source_lines']} at M=33)")
  print(f"smoke chain Jacobian workspace (record only): {chain_jac[0]['w_size']} doubles at M=5, {chain_jac[1]['w_size']} at M=33")
  for backend in ("scaly", "casadi_mx"):
    result, info = run_cell(
      "unbumpercars",
      2,
      backend,
      SMOKE_RESULTS / "unbumpercars" / f"{backend}_C2",
      codegen_timeout=300,
      compile_timeout=180,
      max_source_mb=50,
      benchmark_min_time="0.01s",
      load_harvested=False,
    )
    runtime = f", runtime_ns={result['runtime_ns']}" if result["runtime_ns"] else ""
    print(f"smoke unbumpercars size=2 backend={backend}: {result['runtime_status']}{runtime}" + (f" ({result['note']})" if result["note"] else ""))
    if result["runtime_status"] != "ok" or info is None:
      raise RuntimeError(f"unbumpercars {backend} smoke failed: {result['note']}")
    assert info["nnz"] > 0 and info["w_size"] is not None and info["nnz"] < info["n_rows"] * info["n_cols"]
  _npmpc_smoke()


def _npmpc_smoke() -> None:
  """The neural-process MPC kernels: a dense decoder at every horizon node.

  Five gates. The long-paper constraint Jacobian must compile and beat a dense reference. Its generated
  source must not grow with the horizon or decoder width. The published exact Lagrangian Hessian must
  also stay invariant on both axes, which pins both the decoder and the objective to VMAP-based
  forms.
  """
  infos = []
  for backend in ("scaly", "casadi_sx"):
    result, info = run_cell(
      "npmpc_jac",
      6,
      backend,
      SMOKE_RESULTS / "npmpc" / f"{backend}_N6",
      codegen_timeout=300,
      compile_timeout=180,
      max_source_mb=50,
      benchmark_min_time="0.01s",
      load_harvested=False,
    )
    runtime = f", runtime_ns={result['runtime_ns']}" if result["runtime_ns"] else ""
    print(f"smoke npmpc size=6 backend={backend}: {result['runtime_status']}{runtime}" + (f" ({result['note']})" if result["note"] else ""))
    if result["runtime_status"] != "ok" or info is None:
      raise RuntimeError(f"npmpc {backend} smoke failed: {result['note']}")
    assert info["nnz"] > 0 and info["w_size"] is not None and info["nnz"] < info["n_rows"] * info["n_cols"]
    infos.append(info)
  assert npmpc.n_dec(npmpc.HORIZON) == 65 and npmpc.Decoder().n_pw == 1396
  out_dir = SMOKE_RESULTS / "npmpc" / "scaly_N100"
  out_dir.mkdir(parents=True, exist_ok=True)
  large = build_kernel("npmpc_jac", 100, "scaly", out_dir)
  assert large["source_lines"] < 1.2 * infos[0]["source_lines"], (
    f"npmpc loop preservation regressed: N=100 has {large['source_lines']} lines, N=6 has {infos[0]['source_lines']}"
  )
  print(f"smoke npmpc loop preservation: ok ({infos[0]['source_lines']} lines at N=6, {large['source_lines']} at N=100)")
  # baseline 1600 doubles at N=100 (VMAP buffers scale linearly with the horizon); 3x headroom catches superlinear regressions
  assert int(large["w_size"]) <= 3 * 1600, f"npmpc workspace regressed: N=100 needs {large['w_size']} doubles (baseline 1600)"
  print(f"smoke npmpc workspace: ok ({infos[0]['w_size']} doubles at N=6, {large['w_size']} at N=100)")
  wide_dir = out_dir.parent / "scaly_W128"
  wide_dir.mkdir(parents=True, exist_ok=True)
  wide = build_kernel("npmpc_decoder_jac", 128, "scaly", wide_dir)
  assert wide["source_lines"] < 1.2 * infos[0]["source_lines"], (
    f"npmpc decoder width leaked into source size: W=128 has {wide['source_lines']} lines, W=32 has {infos[0]['source_lines']}"
  )
  print(f"smoke npmpc decoder width: ok ({wide['source_lines']} lines at W=128 against {infos[0]['source_lines']} at the shipped width)")
  # The exact Lagrangian Hessian is the kernel the exact-Hessian IPOPT and SQP columns consume, and
  # it is the one that caught an unrolled objective: a cost built as a Python loop over stages grows
  # the Hessian source linearly and, past roughly 75 stages, exceeds the Program IR passes' recursion
  # depth. Scanning the cost keeps this constant, so the gate is on the horizon as well as the width.
  hess_dirs = [SMOKE_RESULTS / "npmpc" / f"scaly_hess_N{n}" for n in (6, 100)]
  hess = []
  for out, size in zip(hess_dirs, (6, 100), strict=True):
    out.mkdir(parents=True, exist_ok=True)
    hess.append(build_kernel("npmpc", size, "scaly", out))
  assert hess[1]["source_lines"] < 1.2 * hess[0]["source_lines"], (
    f"npmpc Hessian loop preservation regressed: N=100 has {hess[1]['source_lines']} lines, N=6 has {hess[0]['source_lines']}"
  )
  assert hess[0]["nnz"] < hess[1]["nnz"], "the Hessian pattern must grow with the horizon even though its source does not"
  print(f"smoke npmpc lagrangian hessian: ok ({hess[0]['source_lines']} lines at N=6, {hess[1]['source_lines']} at N=100)")

  hess_wide_dir = SMOKE_RESULTS / "npmpc" / "scaly_hess_W128"
  hess_wide_dir.mkdir(parents=True, exist_ok=True)
  hess_wide = build_kernel("npmpc_decoder", 128, "scaly", hess_wide_dir)
  assert hess_wide["source_lines"] < 1.2 * hess[0]["source_lines"], (
    f"npmpc Hessian decoder width leaked into source size: W=128 has {hess_wide['source_lines']} lines, W=32 has {hess[0]['source_lines']}"
  )
  print(f"smoke npmpc Hessian decoder width: ok ({hess_wide['source_lines']} lines at W=128)")


def _problem_checks(problem: str) -> None:
  checks = importlib.import_module(f"benchmarks.problems.{problem}.checks")
  for name, outcome in checks.run_checks():
    print(f"smoke {problem}/{name}: {outcome}", flush=True)
    if os.environ.get("SCALY_REQUIRE_SOLVERS") == "1" and outcome.startswith("skipped:"):
      raise RuntimeError(f"{problem}/{name} unexpectedly {outcome}")


def _problem_smoke() -> None:
  """Run each problem's formulation gates in a separate process."""
  for problem in ("race_cars", "chain", "unbumpercars", "npmpc"):
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
      pool.submit(_problem_checks, problem).result()


def smoke(args) -> bool:
  selected = set(args.select or ("benchmarks", "problems", "solver_call")) - set(args.skip or ())
  failures = []
  if "problems" in selected:
    try:
      _problem_smoke()
    except Exception as e:
      failures.append(f"problems: {e}")
      print(f"smoke problems: FAILED ({type(e).__name__}: {e})")
  if "benchmarks" in selected:
    try:
      _benchmark_smoke()
    except Exception as e:
      failures.append(f"benchmarks: {e}")
      print(f"smoke benchmarks: FAILED ({e})")
  if "solver_call" in selected:
    try:
      note = _solver_call_smoke(required="solver_call" in (args.select or ()))
      print(note or "smoke solver_call: ok")
    except Exception as e:
      failures.append(f"solver_call: {e}")
      print(f"smoke solver_call: FAILED ({e})")
  if failures:
    print("smoke failed: " + "; ".join(failures))
    return False
  print("smoke passed")
  return True


def main() -> None:
  parser = argparse.ArgumentParser(description="Scaly correctness-gated benchmark harness")
  subparsers = parser.add_subparsers(dest="command", required=True)
  sweep_parser = subparsers.add_parser("sweep", help="run the scalability cell grid")
  sweep_parser.add_argument("--workloads", type=_csv, default=["race_cars", "unbumpercars", "chain", "npmpc", "npmpc_decoder"])
  sweep_parser.add_argument("--sizes", type=_ints, help="comma-separated sizes (applied to each selected workload)")
  sweep_parser.add_argument("--backends", type=_csv, help="comma-separated backends; defaults to every encoding defined for each workload")
  sweep_parser.add_argument("--out", "--csv", type=Path, default=SWEEP_RESULTS / "scalability.csv")
  sweep_parser.add_argument("--compile-timeout", type=float, default=180.0)
  sweep_parser.add_argument("--codegen-timeout", type=float, default=300.0)
  sweep_parser.add_argument("--max-source-mb", type=float, default=50.0)
  sweep_parser.add_argument("--benchmark-min-time", default="0.1s")
  sweep_parser.add_argument("--repetitions", type=int, default=3, help="fresh processes per cell (default: 3)")
  sweep_parser.add_argument("--order-seed", type=int, default=0, help="seed for reproducible backend order variation")
  sweep_parser.add_argument("--headline", action="store_true", help="require performance governor and explicit boost setting")
  sweep_parser.add_argument("--boost", choices=("on", "off"), help="required CPU boost state for a headline run")
  sweep_parser.add_argument(
    "--casadi-transform",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="run CasADi 3.8's default Function.transform() simplification flow on every CasADi kernel (default: enabled)",
  )
  study_parser = subparsers.add_parser("study", help="run the frozen headline sweeps and closed loops into one directory, then report")
  study_parser.add_argument("--out-dir", type=Path, required=True, help="unused directory receiving sweep/<problem>/ and closed-loop/<problem>/")
  study_parser.add_argument("--problem", type=_csv, default=list(PROBLEMS), help="comma-separated subset of the study problems (default: all)")
  study_parser.add_argument("--only", type=_csv, default=list(PARTS), help="comma-separated subset of: sweep, closed-loop")
  study_parser.add_argument("--repetitions", type=int, default=5, help="fresh processes per cell and per provider (default: 5)")
  study_parser.add_argument("--order-seed", type=int, default=0)
  study_parser.add_argument(
    "--headline", action=argparse.BooleanOptionalAction, default=True, help="require performance governor and the --boost state (default: enabled)"
  )
  study_parser.add_argument("--boost", choices=("on", "off"), default="off")
  study_parser.add_argument("--overwrite", action="store_true", help="reuse an existing --out-dir, replacing episodes and cells it already holds")
  report_parser = subparsers.add_parser("report", help="re-render the tables of an existing study directory")
  report_parser.add_argument("study_dir", type=Path)
  modes_parser = subparsers.add_parser("modes", help="derive the deployment table from recorded episode mode files")
  modes_parser.add_argument("inputs", type=Path, nargs="+", help="modes.csv files or episode result directories")
  modes_parser.add_argument("--out", type=Path, default=SWEEP_RESULTS.parent / "modes.csv")
  smoke_parser = subparsers.add_parser("smoke", help="run fast correctness and invariant gates")
  smoke_parser.add_argument("--select", action="append", choices=("benchmarks", "problems", "solver_call"))
  smoke_parser.add_argument("--skip", action="append", choices=("benchmarks", "problems", "solver_call"))
  closed_loop_parser = subparsers.add_parser("closed-loop", help="run a model-in-the-loop episode and write Foxglove artifacts")
  closed_loop_parser.add_argument(
    "--problem", type=_csv, default=["unbumpercars"], help="comma-separated subset of chain, race_cars, unbumpercars, npmpc"
  )
  closed_loop_parser.add_argument("--solver", type=_csv, default=["ipopt"], help="comma-separated subset of ipopt, sqp, none (default: ipopt)")
  closed_loop_parser.add_argument("--oracle", type=_csv, help="comma-separated subset of scaly, casadi; defaults to scaly")
  closed_loop_parser.add_argument("--smoke", action="store_true", help="use a short toolchain-check episode instead of the canonical point")
  closed_loop_parser.add_argument("--out-dir", type=Path)
  closed_loop_parser.add_argument("--casadi-interpreted", action="store_true", help="measure CasADi virtual-machine mode with its wheel IPOPT")
  closed_loop_parser.add_argument("--repetitions", type=int, default=1, help="independent cold-cache episode processes")
  closed_loop_parser.add_argument("--order-seed", type=int, default=0, help="initial solver+oracle order, rotated between repetitions")
  closed_loop_parser.add_argument("--overwrite", action="store_true", help="replace episodes and caches already present under --out-dir")
  closed_loop_parser.add_argument("--headline", action="store_true", help="require performance governor and explicit boost setting")
  closed_loop_parser.add_argument("--boost", choices=("on", "off"), help="required CPU boost state")
  args = parser.parse_args()
  if args.command in {"sweep", "study", "closed-loop", "smoke"}:
    configure_math_policy(measured_jit=args.command != "sweep" and (args.command != "study" or "closed-loop" in args.only))
  if args.command == "sweep":
    if args.repetitions < 1 or (args.headline and args.repetitions < 3):
      parser.error("--repetitions must be positive, and at least 3 for --headline")
    if args.headline and args.boost is None:
      parser.error("--headline requires --boost on or off")
    args.workloads = _choices(args.workloads, tuple(DEFAULT_SIZES), parser, "--workloads")
    if args.backends is not None:
      args.backends = _choices(args.backends, BACKENDS, parser, "--backends")
    success = run_sweep(args, sys.argv[1:])
  elif args.command == "study":
    args.problem = _choices(args.problem, PROBLEMS, parser, "--problem")
    args.only = _choices(args.only, PARTS, parser, "--only")
    success = run_study(args, sys.argv[1:])
  elif args.command == "report":
    print(f"report written to {report(args.study_dir.resolve())}")
    success = True
  elif args.command == "modes":
    from benchmarks.harness.timing import summarize_modes

    rows = summarize_modes(as_concrete(args).inputs, args.out)
    print(f"{len(rows)} deployment mode rows written to {args.out}")
    success = bool(rows)
  elif args.command == "smoke":
    success = smoke(args)
  else:
    if args.repetitions < 1 or (args.headline and args.repetitions < 3):
      parser.error("--repetitions must be positive, and at least 3 for --headline")
    if args.headline and args.boost is None:
      parser.error("--headline requires --boost on or off")
    problems = _choices(args.problem, tuple(CLOSED_LOOP_PAIRS), parser, "--problem")
    solvers = _choices(args.solver, ("ipopt", "sqp", "none"), parser, "--solver")
    oracles = _choices(args.oracle or ["scaly"], ("scaly", "casadi"), parser, "--oracle")
    if solvers == ["none"] and args.oracle is not None:
      parser.error("--solver none does not accept --oracle")
    pairs = [(solver, oracle) for solver in solvers for oracle in ([None] if solver == "none" else oracles)]
    if args.casadi_interpreted and pairs != [("ipopt", "casadi")]:
      parser.error("--casadi-interpreted requires --solver ipopt --oracle casadi")
    for problem in problems:
      for pair in pairs:
        if pair not in CLOSED_LOOP_PAIRS[problem]:
          choices = ", ".join(solver_oracle_name(*allowed) for allowed in CLOSED_LOOP_PAIRS[problem])
          parser.error(f"{solver_oracle_name(*pair)} is not available for --problem {problem} (choose from {choices})")
    episodes = [(problem, *pair) for problem in problems for pair in pairs]
    output_root = closed_loop_results_root(smoke=args.smoke, out_dir=args.out_dir).resolve()
    if args.casadi_interpreted and args.repetitions == 1:
      output_root /= "interpreted"
    if args.repetitions > 1 or len(episodes) > 1:
      from benchmarks.harness.timing import summarize_modes

      random.Random(args.order_seed).shuffle(episodes)
      run_order = []
      for repetition in range(1, args.repetitions + 1):
        if args.headline:
          require_headline_settings(args.boost == "on")
        repeat_root = output_root / f"repeat_{repetition}"
        repeat_root.mkdir(parents=True, exist_ok=True)
        offset = (repetition - 1) % len(episodes)
        for problem, solver, provider in episodes[offset:] + episodes[:offset]:
          if args.headline:
            require_headline_settings(args.boost == "on")
          run_order.append({"repetition": repetition, "problem": problem, "solver": solver, "oracle": provider})
          command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "closed-loop",
            "--problem",
            problem,
            "--solver",
            solver,
            "--out-dir",
            str(repeat_root),
          ]
          if provider is not None:
            command += ["--oracle", provider]
          if args.smoke:
            command += ["--smoke"]
          if args.casadi_interpreted:
            command += ["--casadi-interpreted"]
          name = "casadi_interpreted" if args.casadi_interpreted else solver_oracle_name(solver, provider)
          cache = repeat_root / "cache" / problem / name
          if cache.exists():
            if not args.overwrite:
              parser.error(f"fresh-process run requires an unused output directory: {repeat_root} (or pass --overwrite)")
            shutil.rmtree(cache)
            shutil.rmtree(repeat_root / problem / name, ignore_errors=True)
          cache.mkdir(parents=True)
          subprocess.run(
            command, check=True, env={**os.environ, "SCALY_CACHE_DIR": str(cache), "SCALY_CASADI_IPOPT_CACHE": str(cache / "casadi-ipopt")}
          )
          if args.headline:
            require_headline_settings(args.boost == "on")
      summarize_modes(sorted(output_root.glob("repeat_*/**/modes.csv")), output_root / "modes.summary.csv")
      write(output_root / "modes.summary.csv", {**collect(ROOT, gbench.compiler(), sys.argv[1:]), "run_order": run_order})
      print(f"Repeated episode artifacts written to {output_root}")
      return
    ((problem, solver, oracle),) = episodes
    output = run_closed_loop(
      problem,
      smoke=args.smoke,
      solver=solver,
      oracle=oracle,
      out_dir=output_root,
      casadi_interpreted=args.casadi_interpreted,
      cli_args=sys.argv[1:],
    )
    print(f"closed-loop artifacts written to {output}")
    layout = layout_path(problem)
    hint = f"import {layout}" if layout.is_file() else f"no layout checked in yet; export one from Desktop to {layout}"
    print(f"Foxglove: {hint}, then open {output / 'episode.mcap'}")
    success = True
  raise SystemExit(0 if success else 1)


if __name__ == "__main__":
  main()
