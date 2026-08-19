from __future__ import annotations

# ruff: noqa: E402 -- direct execution must add the repository root before package imports

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import numpy as np

import alloy as al
from alloy.codegen.aot import render_c_module
from alloy.solvers.graph import solver_compile_flags
from alloy.solvers.paths import solver_loadable
from benchmarks.harness import SMOKE_RESULTS, SWEEP_RESULTS, closed_loop_results_root, gbench, solver_oracle_name
from benchmarks.harness.closed_loop import run as run_closed_loop
from benchmarks.harness.recording import layout_path
from benchmarks.harness.sweep import BACKENDS, DEFAULT_SIZES, build_kernel, run_cell, run_sweep
from benchmarks.problems import chain


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
CLOSED_LOOP_PAIRS: dict[str, tuple[tuple[str, str | None], ...]] = {
  "chain": (("ipopt", "alloy"), ("sqp", "alloy"), ("sqp", "casadi")),
  "race_cars": (("ipopt", "alloy"), ("ipopt", "casadi"), ("sqp", "alloy"), ("sqp", "casadi")),
  "unbumpercars": (("ipopt", "alloy"), ("ipopt", "casadi"), ("sqp", "alloy"), ("sqp", "casadi"), ("none", None)),
}


def _qp_filter() -> al.Function:
  obstacles = np.array([[1.0, 1.0], [-1.0, 1.5], [0.0, -2.0]], dtype=np.float64)

  @al.function("smoke_safety_filter_qp", {"x": (NX,), "u_ref": (NU,)})
  def safety_filter_qp(x, u_ref):
    cars = al.stack([al.stack([x[2 * i], x[2 * i + 1]], axis=0) for i in range(2)], axis=0)
    rows, bias = [], []
    for car in range(2):
      for obstacle in obstacles:
        diff = cars[car] - al.const(obstacle)
        grad = 2.0 * diff
        rows.append(al.stack([grad[d] if k == car else al.const(0.0) for k in range(2) for d in range(2)], axis=0))
        bias.append(ALPHA * (al.dot(diff, diff) - al.const(SAFETY_MARGIN**2)))
    qp = al.qp(
      P=al.const(np.eye(NU)),
      c=-u_ref,
      G_ineq=al.stack(rows, axis=0),
      l_ineq=-al.stack(bias, axis=0),
      u_ineq=al.const(np.full(len(bias), 1e30)),
    )
    return {
      "u": qp.call(
        x0=al.const(np.zeros(NU)),
        lam_eq0=al.const(np.zeros(0)),
        lam_ineq0=al.const(np.zeros(len(bias))),
        x=x,
        u_ref=u_ref,
      )[0]
    }

  return safety_filter_qp


def _nlp_filter() -> al.Function:
  obstacles = np.array([[1.0, 1.0], [-1.0, 1.5], [0.0, -2.0]], dtype=np.float64)

  @al.function("smoke_safety_filter_nlp", {"x": (NX,), "u_ref": (NU,)})
  def safety_filter_nlp(x, u_ref):
    u = al.sym("u", NU)
    cars = al.stack([al.stack([x[2 * i], x[2 * i + 1]], axis=0) for i in range(2)], axis=0)
    rows = []
    for car in range(2):
      for obstacle in obstacles:
        diff = cars[car] - al.const(obstacle)
        grad = 2.0 * diff
        rows.append(grad[0] * u[2 * car] + grad[1] * u[2 * car + 1] + ALPHA * (al.dot(diff, diff) - al.const(SAFETY_MARGIN**2)))
    nlp = al.nlp(
      x=u,
      f=0.5 * al.dot(u - u_ref, u - u_ref),
      p=(x, u_ref),
      g_ineq=al.stack(rows, axis=0),
      l_ineq=al.const(np.zeros(len(rows))),
      u_ineq=al.const(np.full(len(rows), 1e30)),
    )
    return {
      "u": nlp.call(
        x0=al.const(np.zeros(NU)),
        lam_eq0=al.const(np.zeros(0)),
        lam_ineq0=al.const(np.zeros(len(rows))),
        lam_box0=al.const(np.zeros(NU)),
        x=x,
        u_ref=u_ref,
      )[0]
    }

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
    module = render_c_module(fun, typed_buffers=False)
    (out_dir / module.header_name).write_text(module.header)
    (out_dir / module.source_name).write_text(module.source)
    sources.append(module.source_name)
    for flag in solver_compile_flags(fun):
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
        f"  if ({functions[0].name}(arg, res, nullptr, w_qp.data(), nullptr) != 0) return 1;",
        "  for (double value : out) if (!std::isfinite(value)) return 2;",
        f"  std::array<double, ({functions[1].name}_SZ_W > 0 ? {functions[1].name}_SZ_W : 1)> w_nlp{{}};",
        f"  if ({functions[1].name}(arg, res, nullptr, w_nlp.data(), nullptr) != 0) return 3;",
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
  infos = []
  for backend in ("alloy", "casadi_sx"):
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
    assert info["nnz"] < info["n_rows"] * info["n_cols"], "race_cars Jacobian must be sparse"
    infos.append(info)
  out_dir = SMOKE_RESULTS / "race_cars" / "alloy_N50"
  out_dir.mkdir(parents=True, exist_ok=True)
  large = build_kernel("race_cars", 50, "alloy", out_dir)
  assert large["source_lines"] < 1.2 * infos[0]["source_lines"], (
    f"loop preservation regressed: h=50 has {large['source_lines']} lines, h=5 has {infos[0]['source_lines']}"
  )
  print(f"smoke loop preservation: ok ({infos[0]['source_lines']} lines at h=5, {large['source_lines']} at h=50)")
  # baseline 2100 doubles at h=50 (2026-07-13, scan buffers scale linearly with horizon); 3x headroom catches superlinear regressions
  assert int(large["w_size"]) <= 3 * 2100, f"workspace regressed: h=50 needs {large['w_size']} doubles (baseline 2100)"
  print(f"smoke workspace: ok ({infos[0]['w_size']} doubles at h=5, {large['w_size']} at h=50)")
  chain_infos = []
  for backend in ("alloy", "casadi_sx"):
    result, info = run_cell(
      "chain",
      5,
      backend,
      SMOKE_RESULTS / "chain" / f"{backend}_M5",
      codegen_timeout=300,
      compile_timeout=180,
      max_source_mb=50,
      benchmark_min_time="0.01s",
      load_harvested=False,
    )
    runtime = f", runtime_ns={result['runtime_ns']}" if result["runtime_ns"] else ""
    print(f"smoke chain size=5 backend={backend}: {result['runtime_status']}{runtime}" + (f" ({result['note']})" if result["note"] else ""))
    if result["runtime_status"] != "ok" or info is None:
      raise RuntimeError(f"chain {backend} smoke failed: {result['note']}")
    assert info["nnz"] > 0 and info["w_size"] is not None and info["nnz"] < info["n_rows"] * info["n_cols"]
    chain_infos.append(info)
  assert chain.n_state(5) == 21 and chain.NU == 3
  out_dir = SMOKE_RESULTS / "chain" / "alloy_M33"
  out_dir.mkdir(parents=True, exist_ok=True)
  chain_large = build_kernel("chain", 33, "alloy", out_dir)
  assert chain_large["source_lines"] < 1.2 * chain_infos[0]["source_lines"], (
    f"chain loop preservation regressed: M=33 has {chain_large['source_lines']} lines, M=5 has {chain_infos[0]['source_lines']}"
  )
  print(f"smoke chain loop preservation: ok ({chain_infos[0]['source_lines']} lines at M=5, {chain_large['source_lines']} at M=33)")
  print(f"smoke chain workspace (record only): {chain_infos[0]['w_size']} doubles at M=5, {chain_large['w_size']} at M=33")
  for backend in ("alloy", "casadi_mx"):
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


def _problem_smoke() -> None:
  """Per-problem formulation gates, owned by the problems themselves.

  These check properties of a benchmark problem — its data, its reference generator, its
  parameter layout, and the agreement of its oracle providers — and must pass before any timing is
  recorded. They live here rather than in `tests/` because the pytest suite covers Alloy's
  core and does not depend on benchmark problems (see `AGENTS.md`)."""
  from benchmarks.problems.unbumpercars.checks import run_checks as unbumpercars_checks
  from benchmarks.problems.chain.checks import run_checks as chain_checks
  from benchmarks.problems.race_cars.checks import run_checks as race_cars_checks

  for problem, run_checks in (("race_cars", race_cars_checks), ("chain", chain_checks), ("unbumpercars", unbumpercars_checks)):
    for name, outcome in run_checks():
      print(f"smoke {problem}/{name}: {outcome}")
      if os.environ.get("ALLOY_REQUIRE_SOLVERS") == "1" and outcome.startswith("skipped:"):
        raise RuntimeError(f"{problem}/{name} unexpectedly {outcome}")


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
  parser = argparse.ArgumentParser(description="Alloy correctness-gated benchmark harness")
  subparsers = parser.add_subparsers(dest="command", required=True)
  sweep_parser = subparsers.add_parser("sweep", help="run the scalability cell grid")
  sweep_parser.add_argument("--workloads", type=_csv, default=["race_cars", "unbumpercars", "chain"])
  sweep_parser.add_argument("--sizes", type=_ints, help="comma-separated sizes (applied to each selected workload)")
  sweep_parser.add_argument("--backends", type=_csv, default=list(BACKENDS))
  sweep_parser.add_argument("--out", "--csv", type=Path, default=SWEEP_RESULTS / "scalability.csv")
  sweep_parser.add_argument("--compile-timeout", type=float, default=180.0)
  sweep_parser.add_argument("--codegen-timeout", type=float, default=300.0)
  sweep_parser.add_argument("--max-source-mb", type=float, default=50.0)
  sweep_parser.add_argument("--benchmark-min-time", default="0.1s")
  smoke_parser = subparsers.add_parser("smoke", help="run fast correctness and invariant gates")
  smoke_parser.add_argument("--select", action="append", choices=("benchmarks", "problems", "solver_call"))
  smoke_parser.add_argument("--skip", action="append", choices=("benchmarks", "problems", "solver_call"))
  closed_loop_parser = subparsers.add_parser("closed-loop", help="run a model-in-the-loop episode and write Foxglove artifacts")
  closed_loop_parser.add_argument("--problem", choices=("chain", "race_cars", "unbumpercars"), default="unbumpercars")
  closed_loop_parser.add_argument("--solver", choices=("ipopt", "sqp", "none"), default="ipopt")
  closed_loop_parser.add_argument("--oracle", choices=("alloy", "casadi"), help="oracle provider; defaults to alloy")
  closed_loop_parser.add_argument("--smoke", action="store_true", help="use a short toolchain-check episode instead of the canonical point")
  closed_loop_parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.command == "sweep":
    args.workloads = _choices(args.workloads, tuple(DEFAULT_SIZES), parser, "--workloads")
    args.backends = _choices(args.backends, BACKENDS, parser, "--backends")
    success = run_sweep(args, sys.argv[1:])
  elif args.command == "smoke":
    success = smoke(args)
  else:
    if args.solver == "none" and args.oracle is not None:
      parser.error("--solver none does not accept --oracle")
    oracle = None if args.solver == "none" else (args.oracle or "alloy")
    pair = (args.solver, oracle)
    allowed_pairs = CLOSED_LOOP_PAIRS[args.problem]
    if pair not in allowed_pairs:
      choices = ", ".join(solver_oracle_name(*allowed) for allowed in allowed_pairs)
      parser.error(f"{solver_oracle_name(*pair)} is not available for --problem {args.problem} (choose from {choices})")
    output = run_closed_loop(
      args.problem,
      smoke=args.smoke,
      solver=args.solver,
      oracle=oracle,
      out_dir=closed_loop_results_root(smoke=args.smoke, out_dir=args.out_dir),
      cli_args=sys.argv[1:],
    )
    print(f"closed-loop artifacts written to {output}")
    layout = layout_path(args.problem)
    hint = f"import {layout}" if layout.is_file() else f"no layout checked in yet; export one from Desktop to {layout}"
    print(f"Foxglove: {hint}, then open {output / 'episode.mcap'}")
    success = True
  raise SystemExit(0 if success else 1)


if __name__ == "__main__":
  main()
