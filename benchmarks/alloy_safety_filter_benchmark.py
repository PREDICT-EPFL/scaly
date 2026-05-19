"""Google-benchmark sweep for the Alloy continuous-time CBF safety filter.

For each (ncars, variant) pair, this driver:

1. Builds the symbolic safety-filter fixture from ``tests/alloy/test_safety_filter_workload.py``.
2. Derives a small zoo of single-output Alloy ``Function``s — forward ineq vector, forward cost,
   constraint Jacobian wrt u (dense + sparse), cost gradient wrt u, and (NLP only) the sparse
   Lagrangian Hessian wrt u.
3. Renders each to standalone C via ``alloy.codegen.c.render_c_module``.
4. Writes sample inputs + Python-computed expected outputs as binary files.
5. Emits a Google-benchmark ``benchmark.cpp`` that exercises each entry point and validates the
   result against the expected output before the timing loop.
6. Compiles + runs the binary, prints google-benchmark's table.

The intent is purely to measure how fast each piece runs — no optimization is performed. Use this
to spot which functions deserve hand-tuning later.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import alloy as al
from alloy.codegen.c import _workspace_size, render_c_module

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = ROOT / "benchmarks" / "gen" / "alloy_safety_filter"
FIXTURE = ROOT / "tests" / "alloy" / "test_safety_filter_workload.py"


def _load_fixture():
  spec = importlib.util.spec_from_file_location("safety_filter_workload", FIXTURE)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load {FIXTURE}")
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


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


def _cpp_array_size(size: int) -> int:
  return max(int(size), 1)


# ---------------------------------------------------------------------------
# Function builders
# ---------------------------------------------------------------------------


@dataclass
class BenchEntry:
  label: str  # short token used in benchmark name, e.g. "ineq", "jac_u"
  fn: al.Function  # alloy Function whose .c we compile and time
  out_size: int  # flat number of doubles read off res[0]
  expected: np.ndarray  # reference output (flattened)
  inputs: list[str]  # input names in the order the C ABI expects them
  codegen_ms: float = 0.0
  source_bytes: int = 0
  source_lines: int = 0
  w_size: int = 0
  sparsity_nnz: int | None = None
  source: str = ""
  header: str = ""
  symbol: str = ""


@dataclass
class VariantBuild:
  variant: str
  ncars: int
  entries: list[BenchEntry] = field(default_factory=list)
  sample: dict[str, np.ndarray] = field(default_factory=dict)


def _single_output(fn: al.Function, output_name: str, new_name: str) -> al.Function:
  idx = fn.output_names.index(output_name)
  return al.Function(
    new_name,
    fn.inputs,
    [fn.outputs[idx]],
    fn.input_names,
    [fn.output_names[idx]],
    [fn.output_sparsities[idx]],
  )


def _build_entries(fixture, ncars: int, variant: str) -> VariantBuild:
  if variant == "affine":
    base = fixture.safety_filter_affine_fn(ncars)
  elif variant == "nonlin":
    base = fixture.safety_filter_nonlin_fn(ncars)
  else:
    raise ValueError(f"unknown variant {variant!r}")
  input_names = list(base.input_names)
  build = VariantBuild(variant=variant, ncars=ncars)

  # Forward ineq and forward cost — split into single-output functions so each timing column is
  # comparable across variants. (The joint base function would compute both at once.)
  ineq_only = _single_output(base, "ineq", f"safety_{variant}_N{ncars}_ineq")
  cost_only = _single_output(base, "cost", f"safety_{variant}_N{ncars}_cost")

  # First-order derivatives. We pass *all* base inputs into the factory so the derivative function
  # carries the same input signature — that makes the benchmark + correctness plumbing uniform.
  jac_u = base.factory(
    f"safety_{variant}_N{ncars}_jac_ineq_u",
    input_names,
    ["jac:ineq:u"],
  )
  spjac_u = base.factory(
    f"safety_{variant}_N{ncars}_spjac_ineq_u",
    input_names,
    ["spjac:ineq:u"],
  )
  grad_cost_u = base.factory(
    f"safety_{variant}_N{ncars}_grad_cost_u",
    input_names,
    ["grad:cost:u"],
  )

  entries_spec = [
    ("ineq", ineq_only),
    ("cost", cost_only),
    ("jac_u", jac_u),
    ("spjac_u", spjac_u),
    ("grad_cost_u", grad_cost_u),
  ]

  if variant == "nonlin":
    # Sparse Lagrangian Hessian wrt u — the per-iteration object IPOPT needs. Affine variant has
    # zero second-order, so we skip it there. Alloy's reverse-mode AD does not yet support
    # Ops.MAP (see src/alloy/ad.py::_local_vjp), so building this factory currently fails.
    # Catch and skip with a clear note rather than blow up the entire sweep.
    try:
      sphess_lag_u = base.factory(
        f"safety_{variant}_N{ncars}_sphess_lag_u",
        [*input_names, "lam:ineq"],
        ["sphess:gamma:u:u"],
        aux={"gamma": ["ineq"]},
      )
      entries_spec.append(("sphess_lag_u", sphess_lag_u))
    except NotImplementedError as exc:
      print(f"[skip] sphess_lag_u for {variant} N={ncars}: {exc}")

  # Generate the sample inputs once for the variant + extra lam:ineq draw for the Hessian.
  sample = fixture.sample_inputs(ncars, variant)
  rng = np.random.default_rng(seed=ncars * 100 + (0 if variant == "affine" else 1))
  sample["lam:ineq"] = rng.normal(scale=0.5, size=fixture.n_ineq(ncars))
  build.sample = sample

  for label, fn in entries_spec:
    args = [sample[name] for name in fn.input_names]
    raw = fn.eval_interpreter(*args)
    expected = np.asarray(raw[0], dtype=np.float64).reshape(-1)
    out_size = int(np.prod(fn.outputs[0].shape))
    if expected.size != out_size:
      raise RuntimeError(f"expected size {out_size} for {fn.name}, got {expected.size}")
    sparsity = fn.output_sparsities[0]
    entry = BenchEntry(
      label=label,
      fn=fn,
      out_size=out_size,
      expected=expected,
      inputs=list(fn.input_names),
      sparsity_nnz=(sparsity.nnz if sparsity is not None else None),
    )
    build.entries.append(entry)
  return build


# ---------------------------------------------------------------------------
# C-codegen / IO
# ---------------------------------------------------------------------------


def _emit_c_sources(build: VariantBuild, out_dir: Path) -> None:
  for entry in build.entries:
    t0 = time.perf_counter()
    module = render_c_module(entry.fn, header_name=f"{entry.fn.name}.h", source_name=f"{entry.fn.name}.c", typed_buffers=False)
    entry.codegen_ms = (time.perf_counter() - t0) * 1000.0
    entry.source = module.source_name
    entry.header = module.header_name
    entry.symbol = entry.fn.name
    entry.source_bytes = len(module.source)
    entry.source_lines = module.source.count("\n") + 1
    entry.w_size = _workspace_size(entry.fn)
    (out_dir / module.header_name).write_text(module.header)
    (out_dir / module.source_name).write_text(module.source)


def _write_sample_files(build: VariantBuild, out_dir: Path) -> dict[str, Path]:
  paths: dict[str, Path] = {}
  for name, arr in build.sample.items():
    safe = name.replace(":", "_")
    path = out_dir / f"sample_{build.variant}_N{build.ncars}_{safe}.bin"
    path.write_bytes(arr.astype(np.float64).tobytes())
    paths[name] = path
  return paths


def _write_expected_files(build: VariantBuild, out_dir: Path) -> dict[str, Path]:
  paths: dict[str, Path] = {}
  for entry in build.entries:
    path = out_dir / f"expected_{build.variant}_N{build.ncars}_{entry.label}.bin"
    path.write_bytes(entry.expected.astype(np.float64).tobytes())
    paths[entry.label] = path
  return paths


# ---------------------------------------------------------------------------
# benchmark.cpp emission
# ---------------------------------------------------------------------------


def _make_unique_input_names(build: VariantBuild) -> dict[str, str]:
  """Map (input name) -> sanitized C identifier suffix shared across all benchmarks."""
  return {name: name.replace(":", "_") for name in build.sample}


def _write_benchmark_cpp(build: VariantBuild, out_dir: Path, sample_paths: dict[str, Path], expected_paths: dict[str, Path]) -> None:
  variant = build.variant
  ncars = build.ncars
  cpp_lines: list[str] = [
    '#include "benchmark/benchmark.h"',
    "#include <array>",
    "#include <cmath>",
    "#include <cstddef>",
    "#include <cstdio>",
    "#include <cstdlib>",
    "#include <cstring>",
  ]
  for entry in build.entries:
    cpp_lines.append(f'#include "{entry.header}"')
  cpp_lines.append("")

  # IO helper
  cpp_lines += [
    "template <std::size_t N>",
    "static bool load_doubles(const char* path, std::array<double, N>& out) {",
    '  std::FILE* f = std::fopen(path, "rb");',
    '  if (!f) { std::fprintf(stderr, "could not open %s\\n", path); return false; }',
    "  std::size_t got = std::fread(out.data(), sizeof(double), N, f);",
    "  std::fclose(f);",
    "  if (got != N) {",
    '    std::fprintf(stderr, "expected %zu doubles in %s, got %zu\\n", N, path, got);',
    "    return false;",
    "  }",
    "  return true;",
    "}",
    "",
  ]

  # Globals for every input we have a sample for
  input_to_cident: dict[str, str] = {}
  for name, arr in build.sample.items():
    cident = "g_" + name.replace(":", "_")
    input_to_cident[name] = cident
    cpp_lines.append(f"static std::array<double, {_cpp_array_size(arr.size)}> {cident}{{}};")
  cpp_lines.append("")

  # Expected output globals
  for entry in build.entries:
    cpp_lines.append(f"static std::array<double, {_cpp_array_size(entry.expected.size)}> g_expected_{entry.label}{{}};")
  cpp_lines.append("")

  # load_all
  cpp_lines.append("static bool load_all() {")
  cpp_lines.append("  bool ok = true;")
  for name, arr in build.sample.items():
    path = sample_paths[name]
    cpp_lines.append(f'  ok = ok && load_doubles("{path.name}", {input_to_cident[name]});')
  for entry in build.entries:
    path = expected_paths[entry.label]
    cpp_lines.append(f'  ok = ok && load_doubles("{path.name}", g_expected_{entry.label});')
  cpp_lines.append("  return ok;")
  cpp_lines.append("}")
  cpp_lines.append("")

  # Correctness check
  cpp_lines += [
    "static int correctness_check() {",
    "  if (!load_all()) return 1;",
    "  const double atol = 1e-9;",
    "  const double rtol = 1e-7;",
  ]
  for entry in build.entries:
    arg_init = ", ".join(f"{input_to_cident[name]}.data()" for name in entry.inputs)
    cpp_lines += [
      "  {",
      f"    static std::array<double, {_cpp_array_size(entry.out_size)}> got{{}};",
      f"    static std::array<double, {_cpp_array_size(entry.w_size)}> w{{}};",
      f"    const double* arg[] = {{{arg_init}}};",
      "    double* res[] = {got.data()};",
      f"    int rc = {entry.symbol}(arg, res, nullptr, w.data(), nullptr);",
      f'    if (rc != 0) {{ std::fprintf(stderr, "{entry.symbol} returned %d\\n", rc); return 1; }}',
      f"    for (std::size_t k = 0; k < {entry.out_size}; ++k) {{",
      f"      double diff = std::fabs(got[k] - g_expected_{entry.label}[k]);",
      f"      double thresh = atol + rtol * std::fabs(g_expected_{entry.label}[k]);",
      "      if (!(std::isfinite(diff) && diff <= thresh)) {",
      f'        std::fprintf(stderr, "{entry.symbol}[%zu] got=%.17g exp=%.17g diff=%.3g thresh=%.3g\\n",',
      f"          k, got[k], g_expected_{entry.label}[k], diff, thresh);",
      "        return 1;",
      "      }",
      "    }",
      f'    std::printf("ok: {entry.symbol} (size {entry.out_size})\\n");',
      "  }",
    ]
  cpp_lines += ["  return 0;", "}", ""]

  # Benchmark each entry
  for entry in build.entries:
    arg_init = ", ".join(f"{input_to_cident[name]}.data()" for name in entry.inputs)
    bench_name = f"BM_Safety_{variant}_N{ncars}_{entry.label}"
    cpp_lines += [
      f"static void {bench_name}(benchmark::State& state) {{",
      f"  static std::array<double, {_cpp_array_size(entry.out_size)}> out{{}};",
      f"  static std::array<double, {_cpp_array_size(entry.w_size)}> w{{}};",
      f"  const double* arg[] = {{{arg_init}}};",
      "  double* res[] = {out.data()};",
      "  for (auto _ : state) {",
      f"    int rc = {entry.symbol}(arg, res, nullptr, w.data(), nullptr);",
      "    benchmark::DoNotOptimize(rc);",
      "    benchmark::DoNotOptimize(out.data());",
      "  }",
      f"  state.SetItemsProcessed(state.iterations() * {entry.out_size});",
      "}",
      f"BENCHMARK({bench_name});",
      "",
    ]

  cpp_lines += [
    "int main(int argc, char** argv) {",
    '  if (correctness_check() != 0) { std::fprintf(stderr, "correctness check failed\\n"); return 2; }',
    "  benchmark::Initialize(&argc, argv);",
    "  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;",
    "  benchmark::RunSpecifiedBenchmarks();",
    "  benchmark::Shutdown();",
    "  return 0;",
    "}",
  ]
  (out_dir / f"benchmark_{variant}_N{ncars}.cpp").write_text("\n".join(cpp_lines) + "\n")


# ---------------------------------------------------------------------------
# Build + run
# ---------------------------------------------------------------------------


def _compile_and_run(build: VariantBuild, build_dir: Path, benchmark_args: list[str]) -> str:
  variant, ncars = build.variant, build.ncars
  exe = build_dir / f"safety_filter_{variant}_N{ncars}"
  sources = [f"benchmark_{variant}_N{ncars}.cpp", *(entry.source for entry in build.entries)]
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
    "-lm",
  ]
  print(f"\nCompiling {variant} N={ncars}...")
  t0 = time.perf_counter()
  _run(cmd, cwd=build_dir)
  print(f"  compile_ms={(time.perf_counter() - t0) * 1000:.1f}")
  print(f"Running {variant} N={ncars}...")
  try:
    result = _run([str(exe), *benchmark_args], cwd=build_dir)
  except subprocess.CalledProcessError as e:
    if e.stderr:
      print(e.stderr, end="")
    if e.stdout:
      print(e.stdout)
    raise
  if result.stderr:
    print(result.stderr, end="")
  print(result.stdout)
  return result.stdout


def build_and_run(
  ncars_list: list[int],
  variants: list[str],
  build_dir: Path,
  benchmark_args: list[str],
  stats_only: bool = False,
) -> None:
  fixture = _load_fixture()
  build_dir.mkdir(parents=True, exist_ok=True)

  builds: list[VariantBuild] = []
  for variant in variants:
    for ncars in ncars_list:
      t0 = time.perf_counter()
      build = _build_entries(fixture, ncars, variant)
      build_ms = (time.perf_counter() - t0) * 1000.0
      _emit_c_sources(build, build_dir)
      sample_paths = _write_sample_files(build, build_dir)
      expected_paths = _write_expected_files(build, build_dir)
      _write_benchmark_cpp(build, build_dir, sample_paths, expected_paths)
      builds.append(build)
      print(f"\n[{variant} N={ncars}] build_ms={build_ms:.1f}")
      for entry in build.entries:
        nnz_str = "" if entry.sparsity_nnz is None else f" nnz={entry.sparsity_nnz}"
        print(
          f"  {entry.label:<16} codegen_ms={entry.codegen_ms:7.1f} "
          f"bytes={entry.source_bytes:>9d} lines={entry.source_lines:>6d} "
          f"out={entry.out_size:>6d} w={entry.w_size:>6d}{nnz_str}"
        )

  if stats_only:
    return

  for build in builds:
    _compile_and_run(build, build_dir, benchmark_args)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--ncars", nargs="+", type=int, default=[2, 4, 8])
  parser.add_argument("--variant", action="append", choices=("affine", "nonlin"), default=None, help="may be passed multiple times (default: both)")
  parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
  parser.add_argument("--clean", action="store_true")
  parser.add_argument("--stats-only", action="store_true", help="generate sources but skip compile + run")
  parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
  args = parser.parse_args()
  bench_args = args.benchmark_args
  if bench_args[:1] == ["--"]:
    bench_args = bench_args[1:]
  variants = args.variant or ["affine", "nonlin"]
  if args.clean and args.build_dir.exists():
    shutil.rmtree(args.build_dir)
  build_and_run(args.ncars, variants, args.build_dir.resolve(), bench_args, stats_only=args.stats_only)


if __name__ == "__main__":
  main()
