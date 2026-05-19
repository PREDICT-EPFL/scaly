"""Google Benchmark harness for AOT-compiled Alloy safety filters.

Generates two flavors of a small CBF-style safety filter — a QP variant
(driven by PIQP) and an NLP variant (driven by IPOPT) — each as a single
``Function`` whose body contains a nested ``SOLVER_CALL``. The Function is
rendered through the same `render_c_module(...)` path that powers the
existing tracking / unbumpercars benchmarks; the only new wrinkle is the
vendored PIQP / IPOPT link, which comes from
``alloy.codegen.solver_c.solver_compile_flags(...)``.

Run via::

    uv run python benchmarks/alloy_safety_filter_benchmark.py \\
        -- --benchmark_min_time=0.05s

Pass ``--variant qp`` / ``--variant nlp`` / ``--variant both`` (default) to
pick which filter(s) to build. Per-call runtime, generated source size, and
compile time are printed alongside the Google Benchmark output.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

import alloy as al
from alloy.codegen.c import render_c_module
from alloy.codegen.solver_c import solver_compile_flags

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = ROOT / "benchmarks" / "gen" / "alloy_safety_filter"


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
  return subprocess.run(cmd, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _pkg_config(package: str, flag: str) -> list[str]:
  try:
    out = _run(["pkg-config", flag, package]).stdout.strip()
  except (FileNotFoundError, subprocess.CalledProcessError):
    return []
  return out.split() if out else []


def _compiler() -> str:
  return os.environ.get("CXX") or shutil.which("clang++") or shutil.which("c++") or "c++"


# ---------------------------------------------------------------------------
# Workload: input-affine CBF safety filter (toy version).
# ---------------------------------------------------------------------------

# State: x in R^NX (positions stacked: cars and obstacles).
# Control: u in R^NU (per-car velocity command).
# Objective: track a reference u_ref under a CBF safety constraint that keeps
# every car at least margin r away from every obstacle.
NX = 4  # state size (2 cars, 2D positions)
NU = 4  # control size (2 cars, 2D velocity)
N_OBSTACLES = 3
SAFETY_MARGIN = 0.5
ALPHA = 1.0  # class-K gain in the CBF condition


def _build_qp_safety_filter() -> al.Function:
  """``min 0.5 |u - u_ref|^2  s.t.  CBF row * u + alpha * h >= 0`` for every (car, obstacle).

  ``h(x) = |p_car - p_obs|^2 - r^2`` is the barrier; its gradient gives one
  inequality row per (car, obstacle). For NU = 4 (two cars) and N_OBSTACLES = 3
  we get 6 ineq rows.
  """
  obs_positions = np.array([[1.0, 1.0], [-1.0, 1.5], [0.0, -2.0]], dtype=np.float64)

  @al.function("safety_filter_qp", {"x": (NX,), "u_ref": (NU,)})
  def safety_filter_qp(x, u_ref):
    # Reshape state to (2 cars, 2 dims).
    p_cars = al.stack([al.stack([x[2 * i], x[2 * i + 1]], axis=0) for i in range(2)], axis=0)  # (2, 2)
    rows: list[al.Expr] = []
    bias: list[al.Expr] = []
    for car in range(2):
      for obs in range(N_OBSTACLES):
        diff = p_cars[car] - al.const(obs_positions[obs])
        # h = |diff|^2 - margin^2; grad_x h * G_x maps to u via the per-car block.
        # For toy single-integrator dynamics (\dot p = u), grad_p h = 2 * diff.
        # The CBF row over the full u-vector zeroes out other cars' slots.
        grad_p = 2.0 * diff
        row_entries: list[al.Expr] = []
        for k in range(2):
          for d in range(2):
            row_entries.append(grad_p[d] if k == car else al.const(0.0))
        rows.append(al.stack(row_entries, axis=0))  # (NU,)
        # h itself (scalar).
        h = al.dot(diff, diff) - al.const(SAFETY_MARGIN * SAFETY_MARGIN)
        bias.append(ALPHA * h)
    G = al.stack(rows, axis=0)  # (6, NU)
    l_ineq = -al.stack(bias, axis=0)  # CBF row * u >= -alpha h  ->  G u in [-alpha h, +inf)
    u_ineq = al.const(np.full(len(bias), 1e30))
    qp = al.qp(P=al.const(np.eye(NU)), c=-u_ref, G_ineq=G, l_ineq=l_ineq, u_ineq=u_ineq)
    out = qp.call(
      x0=al.const(np.zeros(NU)),
      lam_eq0=al.const(np.zeros(0)),
      lam_ineq0=al.const(np.zeros(len(bias))),
      x=x,
      u_ref=u_ref,
    )
    return {"u": out[0]}

  return safety_filter_qp


def _build_nlp_safety_filter() -> al.Function:
  """Same shape but solved as an NLP (the CBF constraint is still linear in u
  here, but we route it through IPOPT to exercise the AOT IPOPT path)."""
  obs_positions = np.array([[1.0, 1.0], [-1.0, 1.5], [0.0, -2.0]], dtype=np.float64)
  n_ineq = 2 * N_OBSTACLES

  @al.function("safety_filter_nlp", {"x": (NX,), "u_ref": (NU,)})
  def safety_filter_nlp(x, u_ref):
    u_var = al.sym("u_var", NU)
    f = 0.5 * al.dot(u_var - u_ref, u_var - u_ref)
    p_cars = al.stack([al.stack([x[2 * i], x[2 * i + 1]], axis=0) for i in range(2)], axis=0)
    rows: list[al.Expr] = []
    bias: list[al.Expr] = []
    for car in range(2):
      for obs in range(N_OBSTACLES):
        diff = p_cars[car] - al.const(obs_positions[obs])
        grad_p = 2.0 * diff
        contrib = grad_p[0] * u_var[2 * car] + grad_p[1] * u_var[2 * car + 1]
        h = al.dot(diff, diff) - al.const(SAFETY_MARGIN * SAFETY_MARGIN)
        rows.append(contrib + ALPHA * h)  # constraint: row >= 0
        bias.append(h)  # unused, kept for symmetry
    g = al.stack(rows, axis=0)  # (n_ineq,)
    nlp = al.nlp(
      x=u_var,
      f=f,
      p=(x, u_ref),
      g_ineq=g,
      l_ineq=al.const(np.zeros(n_ineq)),
      u_ineq=al.const(np.full(n_ineq, 1e30)),
    )
    out = nlp.call(
      x0=al.const(np.zeros(NU)),
      lam_eq0=al.const(np.zeros(0)),
      lam_ineq0=al.const(np.zeros(n_ineq)),
      x=x,
      u_ref=u_ref,
    )
    return {"u": out[0]}

  return safety_filter_nlp


# ---------------------------------------------------------------------------
# Render + compile + benchmark.
# ---------------------------------------------------------------------------


def _render(fun: al.Function, out_dir: Path) -> tuple[Path, Path, list[str]]:
  module = render_c_module(fun, typed_buffers=False)
  header_path = out_dir / module.header_name
  source_path = out_dir / module.source_name
  header_path.write_text(module.header)
  source_path.write_text(module.source)
  return header_path, source_path, solver_compile_flags(fun)


def _write_sample_inputs(out_dir: Path, name: str) -> tuple[Path, Path]:
  rng = np.random.default_rng(7)
  x = rng.normal(scale=0.7, size=NX)
  u_ref = rng.normal(scale=0.3, size=NU)
  x_path = out_dir / f"{name}_x.bin"
  u_path = out_dir / f"{name}_u_ref.bin"
  x_path.write_bytes(x.astype(np.float64).tobytes())
  u_path.write_bytes(u_ref.astype(np.float64).tobytes())
  return x_path, u_path


def _write_benchmark_cpp(
  variants: list[tuple[str, str, Path, Path]],  # (label, fn_name, x_path, u_path)
  out_dir: Path,
) -> Path:
  lines: list[str] = [
    '#include "benchmark/benchmark.h"',
    "#include <array>",
    "#include <cstddef>",
    "#include <cstdio>",
  ]
  for label, fn_name, _, _ in variants:
    lines.append(f'#include "{fn_name}.h"')
  lines += [
    "",
    "template <std::size_t N>",
    "static bool load_doubles(const char* path, std::array<double, N>& out) {",
    '  std::FILE* f = std::fopen(path, "rb");',
    "  if (!f) return false;",
    "  std::size_t got = std::fread(out.data(), sizeof(double), N, f);",
    "  std::fclose(f);",
    "  return got == N;",
    "}",
    "",
    f"static std::array<double, {NX}> g_x{{}};",
    f"static std::array<double, {NU}> g_u_ref{{}};",
    "",
  ]
  for label, _fn_name, x_path, u_path in variants:
    lines += [
      f"static bool load_{label}_inputs() {{",
      f'  return load_doubles("{x_path.name}", g_x) && load_doubles("{u_path.name}", g_u_ref);',
      "}",
      "",
    ]
  for label, fn_name, _, _ in variants:
    lines += [
      f"static void BM_{label}(benchmark::State& state) {{",
      f'  if (!load_{label}_inputs()) {{ state.SkipWithError("sample inputs missing"); return; }}',
      f"  static std::array<double, {NU}> u_out{{}};",
      f"  static std::array<double, ({fn_name}_SZ_W > 0 ? {fn_name}_SZ_W : 1)> w{{}};",
      "  const double* arg[] = {g_x.data(), g_u_ref.data()};",
      "  double* res[] = {u_out.data()};",
      "  for (auto _ : state) {",
      f"    int rc = {fn_name}(arg, res, nullptr, w.data(), nullptr);",
      "    benchmark::DoNotOptimize(rc);",
      "    benchmark::DoNotOptimize(u_out.data());",
      "  }",
      f"  state.SetItemsProcessed(state.iterations() * {NU});",
      "}",
      f"BENCHMARK(BM_{label});",
      "",
    ]
  lines += [
    "int main(int argc, char** argv) {",
    "  benchmark::Initialize(&argc, argv);",
    "  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;",
    "  benchmark::RunSpecifiedBenchmarks();",
    "  benchmark::Shutdown();",
    "  return 0;",
    "}",
  ]
  cpp_path = out_dir / "benchmark.cpp"
  cpp_path.write_text("\n".join(lines) + "\n")
  return cpp_path


def build_and_run(variant: str, build_dir: Path, benchmark_args: list[str]) -> None:
  build_dir.mkdir(parents=True, exist_ok=True)
  pieces: list[tuple[str, str, Path, Path, Path, list[str]]] = []  # (label, fn_name, header, source, x_path, flags)

  if variant in {"qp", "both"}:
    t0 = time.perf_counter()
    fun = _build_qp_safety_filter()
    _header, source, flags = _render(fun, build_dir)
    qp_codegen_ms = (time.perf_counter() - t0) * 1000
    x_path, u_path = _write_sample_inputs(build_dir, "qp")
    print(f"qp:  codegen_ms={qp_codegen_ms:.1f}  source_bytes={source.stat().st_size}  flags={flags}")
    pieces.append(("qp", fun.name, _header, source, x_path, flags))

  if variant in {"nlp", "both"}:
    t0 = time.perf_counter()
    fun = _build_nlp_safety_filter()
    _header, source, flags = _render(fun, build_dir)
    nlp_codegen_ms = (time.perf_counter() - t0) * 1000
    x_path, u_path = _write_sample_inputs(build_dir, "nlp")
    print(f"nlp: codegen_ms={nlp_codegen_ms:.1f}  source_bytes={source.stat().st_size}  flags={flags}")
    pieces.append(("nlp", fun.name, _header, source, x_path, flags))

  if not pieces:
    print(f"no variants selected for {variant!r}")
    return

  bench_variants = [(label, fn_name, x_path, build_dir / f"{label}_u_ref.bin") for label, fn_name, _, _, x_path, _ in pieces]
  cpp_path = _write_benchmark_cpp(bench_variants, build_dir)

  exe = build_dir / "safety_filter_benchmark"
  sources = [str(cpp_path), *[str(source) for _, _, _, source, _, _ in pieces]]
  # Merge all solver flags (deduped while preserving order).
  combined_flags: list[str] = []
  for _, _, _, _, _, flags in pieces:
    for f in flags:
      if f not in combined_flags:
        combined_flags.append(f)
  cmd = [
    _compiler(),
    "-O3",
    "-std=c++17",
    "-I",
    str(build_dir),
    *sources,
    "-o",
    str(exe),
    *_pkg_config("benchmark", "--cflags"),
    *_pkg_config("benchmark", "--libs"),
    *combined_flags,
    "-lm",
  ]
  print(f"\nCompiling {exe.name}...")
  t0 = time.perf_counter()
  _run(cmd, cwd=build_dir)
  print(f"compile_ms={(time.perf_counter() - t0) * 1000:.1f}")

  print(f"\nRunning {exe.name}...")
  result = subprocess.run([str(exe), *benchmark_args], cwd=build_dir, text=True, capture_output=True)
  if result.stdout:
    print(result.stdout)
  if result.stderr:
    print(result.stderr, end="")
  if result.returncode != 0:
    raise SystemExit(result.returncode)


def main() -> None:
  parser = argparse.ArgumentParser(description="AOT-compile and benchmark Alloy safety filters via Google Benchmark.")
  parser.add_argument("--variant", choices=("qp", "nlp", "both"), default="both")
  parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
  parser.add_argument("--clean", action="store_true")
  parser.add_argument("benchmark_args", nargs=argparse.REMAINDER, help="passed through to the benchmark binary after --")
  args = parser.parse_args()

  benchmark_args = args.benchmark_args
  if benchmark_args[:1] == ["--"]:
    benchmark_args = benchmark_args[1:]
  if args.clean and args.build_dir.exists():
    shutil.rmtree(args.build_dir)
  build_and_run(args.variant, args.build_dir.resolve(), benchmark_args)


if __name__ == "__main__":
  main()
