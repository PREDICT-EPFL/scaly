from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import casadi as ca
import numpy as np

from alloy.codegen.c import _workspace_size, render_c_module

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = ROOT / "benchmarks" / "gen" / "alloy_unbumpercars_ineq_jac"
FIXTURE = ROOT / "tests" / "alloy" / "test_unbumpercars_workload.py"


def _load_fixture():
  spec = importlib.util.spec_from_file_location("alloy_unbumpercars_workload", FIXTURE)
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


def _write_alloy(fixture, ncars: int, out_dir: Path) -> dict[str, int | str]:
  fn = fixture.unbumpercars_ineq_function(ncars)
  name = f"alloy_unbumpercars_ineq_jac_N{ncars}"
  spjf = fn.factory(name, ["u", "p"], ["spjac:ineq:u"])
  module = render_c_module(spjf, header_name=f"{name}.h", source_name=f"{name}.c", typed_buffers=False)
  (out_dir / module.header_name).write_text(module.header)
  (out_dir / module.source_name).write_text(module.source)
  sparsity = spjf.output_sparsities[0]
  assert sparsity is not None
  return {
    "backend": "alloy",
    "name": name,
    "source": module.source_name,
    "header": module.header_name,
    "u_size": fixture.n_dec(ncars),
    "p_size": fixture.n_param(ncars),
    "nnz": sparsity.nnz,
    "n_ineq": fn.outputs[0].shape[0],
    "n_dec": fixture.n_dec(ncars),
    "rows": tuple(int(r) for r in sparsity.rows),
    "cols": tuple(int(c) for c in sparsity.cols),
    "w_size": _workspace_size(spjf),
    "iw_size": 0,
    "source_bytes": len(module.source),
    "source_lines": module.source.count("\n") + 1,
    "spjf": spjf,
  }


def _casadi_triplet(fn) -> tuple[tuple[int, ...], tuple[int, ...]]:
  rows, cols = fn.sparsity_out(0).get_triplet()
  return tuple(int(r) for r in rows), tuple(int(c) for c in cols)


def _write_sample_inputs(fixture, ncars: int, out_dir: Path, alloy_info: dict) -> tuple[Path, Path, Path]:
  uv, pv = fixture._sample_inputs(ncars)
  spjf = alloy_info["spjf"]
  jf = fixture.unbumpercars_ineq_function(ncars).factory(f"dense_ref_N{ncars}", ["u", "p"], ["jac:ineq:u"])
  dense_expected = np.asarray(jf(uv, pv), dtype=np.float64).reshape(-1)
  expected_shape = (int(alloy_info["n_ineq"]), int(alloy_info["n_dec"]))
  if dense_expected.size != expected_shape[0] * expected_shape[1]:
    raise RuntimeError(f"expected dense shape {expected_shape}, got {dense_expected.shape}")
  # cross-check: scatter the compact alloy values into dense and confirm they match
  compact_alloy = np.asarray(spjf(uv, pv), dtype=np.float64).reshape(-1)
  scatter = np.zeros(dense_expected.size, dtype=np.float64)
  flat = np.asarray(alloy_info["rows"], dtype=np.int64) * expected_shape[1] + np.asarray(alloy_info["cols"], dtype=np.int64)
  scatter[flat] = compact_alloy
  if not np.allclose(scatter, dense_expected, atol=1e-9, rtol=1e-9):
    raise RuntimeError("Python alloy compact does not scatter to the dense reference - fix the fixture before benchmarking")
  u_path = out_dir / f"sample_u_N{ncars}.bin"
  p_path = out_dir / f"sample_p_N{ncars}.bin"
  expected_path = out_dir / f"expected_dense_N{ncars}.bin"
  u_path.write_bytes(uv.astype(np.float64).tobytes())
  p_path.write_bytes(pv.astype(np.float64).tobytes())
  expected_path.write_bytes(dense_expected.tobytes())
  return u_path, p_path, expected_path


def _write_casadi(fixture, ncars: int, out_dir: Path, kind: str) -> dict[str, int | str]:
  name = f"casadi_{kind}_unbumpercars_ineq_jac_N{ncars}"
  fn = fixture._ca_unbumpercars_ineq_jac(ncars, ca.SX if kind == "sx" else ca.MX, name=name)
  cwd = Path.cwd()
  os.chdir(out_dir)
  try:
    gen = ca.CodeGenerator(f"{name}.c", {"with_header": True, "casadi_int": "int"})
    gen.add(fn)
    gen.generate()
  finally:
    os.chdir(cwd)
  source = (out_dir / f"{name}.c").read_text()
  rows, cols = _casadi_triplet(fn)
  return {
    "backend": f"casadi_{kind}",
    "name": name,
    "source": f"{name}.c",
    "header": f"{name}.h",
    "u_size": fixture.n_dec(ncars),
    "p_size": fixture.n_param(ncars),
    "nnz": fn.sparsity_out(0).nnz(),
    "n_ineq": fn.size_out(0)[0],
    "n_dec": fn.size_out(0)[1],
    "rows": rows,
    "cols": cols,
    "w_size": fn.sz_w(),
    "iw_size": fn.sz_iw(),
    "source_bytes": len(source),
    "source_lines": source.count("\n") + 1,
  }


def _write_benchmark_cpp(
  alloy_info: dict[str, int | str], casadi_infos: list[dict[str, int | str]], out_dir: Path, sample_paths: tuple[Path, Path, Path]
) -> None:
  infos = [alloy_info, *casadi_infos]
  u_size = int(alloy_info["u_size"])
  p_size = int(alloy_info["p_size"])
  n_ineq = int(alloy_info["n_ineq"])
  n_dec = int(alloy_info["n_dec"])
  dense_size = n_ineq * n_dec
  for info in infos:
    if int(info["u_size"]) != u_size or int(info["p_size"]) != p_size:
      raise RuntimeError(f"correctness check requires matching shapes; got {info}")
    if int(info["n_ineq"]) != n_ineq or int(info["n_dec"]) != n_dec:
      raise RuntimeError(f"correctness check requires matching Jacobian shape; got {info}")

  u_path, p_path, expected_path = (p.name for p in sample_paths)

  lines = [
    '#include "benchmark/benchmark.h"',
    "#include <array>",
    "#include <cmath>",
    "#include <cstddef>",
    "#include <cstdio>",
    "#include <cstdlib>",
    "#include <cstring>",
    "",
  ]
  for info in infos:
    lines.append(f'#include "{info["header"]}"')
  lines += [
    "",
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
    f"static std::array<double, {_cpp_array_size(u_size)}> g_u{{}};",
    f"static std::array<double, {_cpp_array_size(p_size)}> g_p{{}};",
    f"static std::array<double, {_cpp_array_size(dense_size)}> g_expected_dense{{}};",
    "",
    "static bool load_sample_inputs() {",
    f'  return load_doubles("{u_path}", g_u)',
    f'      && load_doubles("{p_path}", g_p)',
    f'      && load_doubles("{expected_path}", g_expected_dense);',
    "}",
    "",
  ]

  for info in infos:
    sym = info["name"]
    rows = info["rows"]
    cols = info["cols"]
    nnz = int(info["nnz"])
    rows_arr = ", ".join(str(r) for r in rows) or "0"
    cols_arr = ", ".join(str(c) for c in cols) or "0"
    lines += [
      f"static const int {sym}_check_rows[{_cpp_array_size(nnz)}] = {{{rows_arr}}};",
      f"static const int {sym}_check_cols[{_cpp_array_size(nnz)}] = {{{cols_arr}}};",
    ]
  lines += [""]

  # Correctness check: each compiled backend's compact output is scattered into a dense
  # Jacobian using its own (rows, cols) and compared element-wise against the dense
  # reference produced by Python alloy (also cross-checked against the dense factory).
  lines += [
    "static int correctness_check() {",
    "  if (!load_sample_inputs()) return 1;",
    "  const double* arg[] = {g_u.data(), g_p.data()};",
    "  const double atol = 1e-9;",
    "  const double rtol = 1e-9;",
    f"  static std::array<double, {_cpp_array_size(dense_size)}> got_dense{{}};",
  ]
  for info in infos:
    name = str(info["name"])
    label = str(info["backend"])
    nnz_i = int(info["nnz"])
    w_size = int(info["w_size"])
    iw_size = int(info["iw_size"])
    needs_w = w_size > 0
    needs_iw = iw_size > 0
    if needs_w and needs_iw:
      res_args = "iw.data(), w.data(), 0"
    elif needs_w:
      res_args = "nullptr, w.data(), 0"
    else:
      res_args = "nullptr, nullptr, nullptr"
    w_lines = []
    if needs_w:
      w_lines.append(f"    static std::array<double, {_cpp_array_size(w_size)}> w{{}};")
    if needs_iw:
      w_lines.append(f"    static std::array<int, {_cpp_array_size(iw_size)}> iw{{}};")
    lines += [
      "  {",
      *w_lines,
      f"    static std::array<double, {_cpp_array_size(nnz_i)}> compact{{}};",
      "    double* res[] = {compact.data()};",
      f"    int rc = {name}(arg, res, {res_args});",
      f'    if (rc != 0) {{ std::fprintf(stderr, "{label} returned %d\\n", rc); return 1; }}',
      "    std::memset(got_dense.data(), 0, sizeof(got_dense));",
      f"    for (std::size_t i = 0; i < {nnz_i}; ++i) {{",
      f"      int row = {name}_check_rows[i];",
      f"      int col = {name}_check_cols[i];",
      f"      got_dense[row * {n_dec} + col] = compact[i];",
      "    }",
      f"    for (std::size_t k = 0; k < {dense_size}; ++k) {{",
      "      double diff = std::fabs(got_dense[k] - g_expected_dense[k]);",
      "      double thresh = atol + rtol * std::fabs(g_expected_dense[k]);",
      "      bool same_nan = std::isnan(got_dense[k]) && std::isnan(g_expected_dense[k]);",
      "      bool ok = same_nan || (std::isfinite(diff) && diff <= thresh);",
      "      if (!ok) {",
      f"        int row = static_cast<int>(k / {n_dec});",
      f"        int col = static_cast<int>(k % {n_dec});",
      f'        std::fprintf(stderr, "{label} dense[%d,%d]=%.17g vs expected=%.17g (|diff|=%.3g, thresh=%.3g)\\n", row, col, got_dense[k], g_expected_dense[k], diff, thresh);',
      "        return 1;",
      "      }",
      "    }",
      f'    std::printf("correctness ok: {label} dense Jacobian matches reference ({n_ineq}x{n_dec})\\n");',
      "  }",
    ]
  lines += ["  return 0;", "}", ""]

  name = str(alloy_info["name"])
  suffix = name.split("_N")[-1]
  alloy_w_size = int(alloy_info["w_size"])
  alloy_w_lines = [f"  static std::array<double, {_cpp_array_size(alloy_w_size)}> w{{}};"] if alloy_w_size > 0 else []
  alloy_call_args = "nullptr, w.data(), 0" if alloy_w_size > 0 else "nullptr, nullptr, nullptr"
  lines += [
    f"static void BM_AlloyUnbumpercarsIneqJacN{suffix}(benchmark::State& state) {{",
    f"  std::array<double, {_cpp_array_size(nnz)}> out{{}};",
    *alloy_w_lines,
    "  const double* arg[] = {g_u.data(), g_p.data()};",
    "  double* res[] = {out.data()};",
    "  for (auto _ : state) {",
    f"    int rc = {name}(arg, res, {alloy_call_args});",
    "    benchmark::DoNotOptimize(rc);",
    "    benchmark::DoNotOptimize(out.data());",
    "  }",
    f"  state.SetItemsProcessed(state.iterations() * {nnz});",
    "}",
    f"BENCHMARK(BM_AlloyUnbumpercarsIneqJacN{suffix});",
    "",
  ]

  for info in casadi_infos:
    name = str(info["name"])
    suffix = name.split("_N")[-1]
    bench_name = f"BM_{str(info['backend']).title().replace('_', '')}UnbumpercarsIneqJacN{suffix}"
    lines += [
      f"static void {bench_name}(benchmark::State& state) {{",
      f"  static std::array<double, {_cpp_array_size(nnz)}> out{{}};",
      f"  static std::array<double, {_cpp_array_size(int(info['w_size']))}> w{{}};",
      f"  static std::array<int, {_cpp_array_size(int(info['iw_size']))}> iw{{}};",
      "  const double* arg[] = {g_u.data(), g_p.data()};",
      "  double* res[] = {out.data()};",
      "  for (auto _ : state) {",
      f"    int rc = {name}(arg, res, iw.data(), w.data(), 0);",
      "    benchmark::DoNotOptimize(rc);",
      "    benchmark::DoNotOptimize(out.data());",
      "  }",
      f"  state.SetItemsProcessed(state.iterations() * {nnz});",
      "}",
      f"BENCHMARK({bench_name});",
      "",
    ]
  lines += [
    "int main(int argc, char** argv) {",
    '  if (correctness_check() != 0) { std::fprintf(stderr, "correctness check failed\\n"); return 2; }',
    "  benchmark::Initialize(&argc, argv);",
    "  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;",
    "  benchmark::RunSpecifiedBenchmarks();",
    "  benchmark::Shutdown();",
    "  return 0;",
    "}",
  ]
  (out_dir / "benchmark.cpp").write_text("\n".join(lines) + "\n")


def build_and_run(car_counts: list[int], build_dir: Path, benchmark_args: list[str], backends: tuple[str, ...], stats_only: bool = False) -> None:
  fixture = _load_fixture()
  build_dir.mkdir(parents=True, exist_ok=True)
  groups = []
  for ncars in car_counts:
    t0 = time.perf_counter()
    alloy_info = _write_alloy(fixture, ncars, build_dir)
    alloy_info["codegen_ms"] = int((time.perf_counter() - t0) * 1000)
    casadi_infos = []
    for kind in (k for k in ("sx", "mx") if f"casadi-{k}" in backends):
      t0 = time.perf_counter()
      info = _write_casadi(fixture, ncars, build_dir, kind)
      info["codegen_ms"] = int((time.perf_counter() - t0) * 1000)
      casadi_infos.append(info)
    groups.append((alloy_info, casadi_infos))

  print("Generated sources:")
  for alloy_info, casadi_infos in groups:
    for info in (alloy_info, *casadi_infos):
      print(
        f"  {info['backend']} N={str(info['name']).split('_N')[-1]} "
        f"codegen_ms={info['codegen_ms']} bytes={info['source_bytes']} lines={info['source_lines']} w={info['w_size']} nnz={info['nnz']}"
      )

  if stats_only:
    return

  for alloy_info, casadi_infos in groups:
    ncars_str = str(alloy_info["name"]).split("_N")[-1]
    ncars = int(ncars_str)
    sample_paths = _write_sample_inputs(fixture, ncars, build_dir, alloy_info)
    _write_benchmark_cpp(alloy_info, casadi_infos, build_dir, sample_paths)
    exe = build_dir / f"unbumpercars_ineq_jac_benchmark_N{ncars_str}"
    sources = ["benchmark.cpp", str(alloy_info["source"]), *(str(info["source"]) for info in casadi_infos)]
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
    print(f"\nCompiling benchmark N={ncars_str}...")
    t0 = time.perf_counter()
    _run(cmd, cwd=build_dir)
    print(f"compile_ms={(time.perf_counter() - t0) * 1000:.1f}")
    print(f"\nRunning benchmark N={ncars_str}...")
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


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate and run reduced unbumpercars inequality-Jacobian Google Benchmarks.")
  parser.add_argument("--car-counts", nargs="+", type=int, default=[2])
  parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
  parser.add_argument("--clean", action="store_true")
  parser.add_argument("--stats-only", action="store_true", help="generate sources and print size/workspace stats without compiling benchmarks")
  parser.add_argument(
    "--backend",
    action="append",
    choices=("alloy", "casadi-sx", "casadi-mx"),
    default=None,
    help="backend to include; may be passed multiple times (default: all)",
  )
  parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
  args = parser.parse_args()
  benchmark_args = args.benchmark_args[1:] if args.benchmark_args[:1] == ["--"] else args.benchmark_args
  if args.clean and args.build_dir.exists():
    shutil.rmtree(args.build_dir)
  backends = tuple(args.backend or ("alloy", "casadi-sx", "casadi-mx"))
  build_and_run(args.car_counts, args.build_dir, benchmark_args, backends, stats_only=args.stats_only)


if __name__ == "__main__":
  main()
