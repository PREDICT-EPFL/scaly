from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import casadi as ca
import numpy as np

import alloy as al
from alloy.codegen.c import render_c_module

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = ROOT / "benchmarks" / "gen" / "alloy_tracking_eq_jac"
TRACKING_FIXTURE = ROOT / "tests" / "alloy" / "test_tracking_workload.py"


def _load_tracking_fixture():
  spec = importlib.util.spec_from_file_location("alloy_tracking_workload", TRACKING_FIXTURE)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load {TRACKING_FIXTURE}")
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


def _write_alloy(tracking, horizon: int, out_dir: Path, *, variant: str = "unrolled") -> dict[str, int | str]:
  use_structured = False
  if variant == "unrolled":
    fn = tracking.tracking_eq_function(horizon)
    name = f"alloy_tracking_eq_jac_N{horizon}"
  elif variant == "map":
    fn = tracking.tracking_eq_function_map(horizon)
    name = f"alloy_map_tracking_eq_jac_N{horizon}"
  elif variant == "map_structured":
    fn = tracking.tracking_eq_function_map(horizon)
    name = f"alloy_map_structured_tracking_eq_jac_N{horizon}"
    use_structured = True
  else:
    raise ValueError(f"unknown alloy variant {variant!r}")
  sj = al.sparse_jacobian(fn.outputs[0], fn.inputs[0]) if use_structured else al.sparse_jacobian_colored(fn.outputs[0], fn.inputs[0])
  spjf = al.Function(name, [fn.inputs[0]], [sj.values], ["z"], ["spjac_eq_z"], [sj.sparsity])
  module = render_c_module(spjf, header_name=f"{name}.h", source_name=f"{name}.c", typed_buffers=False)
  (out_dir / module.header_name).write_text(module.header)
  (out_dir / module.source_name).write_text(module.source)
  workspace = int(re.search(rf"#define {name}_SZ_W (\d+)", module.header).group(1))
  return {
    "name": name,
    "variant": variant,
    "source": module.source_name,
    "header": module.header_name,
    "z_size": tracking.NZ * (horizon + 1),
    "p_size": tracking.NX * (horizon + 1),
    "input_args": ("z",),
    "nnz": sj.sparsity.nnz,
    "n_rows": fn.outputs[0].shape[0],
    "n_cols": fn.inputs[0].shape[0],
    "rows": tuple(int(r) for r in sj.sparsity.rows),
    "cols": tuple(int(c) for c in sj.sparsity.cols),
    "w_size": workspace,
    "source_bytes": len(module.source),
    "source_lines": module.source.count("\n") + 1,
    "spjf": spjf,
    "tracking_fn": fn,
  }


def _write_casadi(tracking, horizon: int, out_dir: Path, kind: str) -> dict[str, int | str]:
  name = f"casadi_{kind}_tracking_eq_jac_N{horizon}"
  fn = tracking._ca_tracking_eq_jac(horizon, name=name, sym_t=ca.SX if kind == "sx" else ca.MX)
  cwd = Path.cwd()
  os.chdir(out_dir)
  try:
    gen = ca.CodeGenerator(f"{name}.c", {"with_header": True, "casadi_int": "int"})
    gen.add(fn)
    gen.generate()
  finally:
    os.chdir(cwd)
  source_path = out_dir / f"{name}.c"
  source = source_path.read_text()
  rows, cols = fn.sparsity_out(0).get_triplet()
  return {
    "name": name,
    "backend": f"casadi_{kind}",
    "source": f"{name}.c",
    "header": f"{name}.h",
    "z_size": tracking.NZ * (horizon + 1),
    "p_size": tracking.NX * (horizon + 1),
    "input_args": ("z", "p"),
    "nnz": fn.sparsity_out(0).nnz(),
    "n_rows": fn.size_out(0)[0],
    "n_cols": fn.size_out(0)[1],
    "rows": tuple(int(r) for r in rows),
    "cols": tuple(int(c) for c in cols),
    "w_size": fn.sz_w(),
    "iw_size": fn.sz_iw(),
    "source_bytes": len(source),
    "source_lines": source.count("\n") + 1,
  }


def _write_tracking_sample_inputs(tracking, horizon: int, out_dir: Path, alloy_info: dict) -> tuple[Path, Path, Path]:
  rng = np.random.default_rng(7)
  z_size = int(alloy_info["z_size"])
  p_size = int(alloy_info["p_size"])
  zv = rng.normal(scale=0.4, size=z_size)
  pv = rng.normal(scale=0.4, size=p_size)
  # Always build the dense reference from the unrolled fixture: it uses single-seed jvp which has
  # no MAP rule yet. The variant under test is checked through its own compact sparse output below.
  ref_fn = tracking.tracking_eq_function(horizon)
  jf = ref_fn.factory(f"alloy_dense_ref_N{horizon}", ["z", "p"], ["jac:eq:z"])
  dense_expected = np.asarray(jf(zv, pv), dtype=np.float64).reshape(-1)
  n_rows = int(alloy_info["n_rows"])
  n_cols = int(alloy_info["n_cols"])
  if dense_expected.size != n_rows * n_cols:
    raise RuntimeError(f"expected dense {n_rows}x{n_cols}, got {dense_expected.shape}")
  compact_alloy = np.asarray(alloy_info["spjf"](zv), dtype=np.float64).reshape(-1)
  scatter = np.zeros(dense_expected.size, dtype=np.float64)
  rows = np.asarray(alloy_info["rows"], dtype=np.int64)
  cols = np.asarray(alloy_info["cols"], dtype=np.int64)
  scatter[rows * n_cols + cols] = compact_alloy
  if not np.allclose(scatter, dense_expected, atol=1e-9, rtol=1e-9):
    raise RuntimeError("Python alloy compact does not scatter to the dense reference - fix the fixture before benchmarking")
  z_path = out_dir / f"sample_z_N{horizon}.bin"
  p_path = out_dir / f"sample_p_N{horizon}.bin"
  expected_path = out_dir / f"expected_dense_N{horizon}.bin"
  z_path.write_bytes(zv.astype(np.float64).tobytes())
  p_path.write_bytes(pv.astype(np.float64).tobytes())
  expected_path.write_bytes(dense_expected.tobytes())
  return z_path, p_path, expected_path


def _cpp_array_size(size: int) -> int:
  return max(int(size), 1)


def _write_benchmark_cpp(
  alloy_info: dict[str, int | str],
  casadi_infos: list[dict[str, int | str]],
  out_dir: Path,
  sample_paths: tuple[Path, Path, Path],
  *,
  extra_alloy_infos: tuple[dict[str, int | str], ...] = (),
) -> None:
  alloy_infos = (alloy_info, *extra_alloy_infos)
  infos = [*alloy_infos, *casadi_infos]
  z_size = int(alloy_info["z_size"])
  p_size = int(alloy_info["p_size"])
  n_rows = int(alloy_info["n_rows"])
  n_cols = int(alloy_info["n_cols"])
  dense_size = n_rows * n_cols
  for info in infos:
    if int(info["z_size"]) != z_size or int(info["p_size"]) != p_size:
      raise RuntimeError(f"correctness check requires matching shapes; got {info}")
    if int(info["n_rows"]) != n_rows or int(info["n_cols"]) != n_cols:
      raise RuntimeError(f"correctness check requires matching Jacobian shape; got {info}")

  z_path, p_path, expected_path = (p.name for p in sample_paths)

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
    f"static std::array<double, {_cpp_array_size(z_size)}> g_z{{}};",
    f"static std::array<double, {_cpp_array_size(p_size)}> g_p{{}};",
    f"static std::array<double, {_cpp_array_size(dense_size)}> g_expected_dense{{}};",
    "",
    "static bool load_sample_inputs() {",
    f'  return load_doubles("{z_path}", g_z)',
    f'      && load_doubles("{p_path}", g_p)',
    f'      && load_doubles("{expected_path}", g_expected_dense);',
    "}",
    "",
  ]

  for info in infos:
    sym = info["name"]
    rows_arr = ", ".join(str(r) for r in info["rows"]) or "0"
    cols_arr = ", ".join(str(c) for c in info["cols"]) or "0"
    nnz_i = int(info["nnz"])
    lines += [
      f"static const int {sym}_check_rows[{_cpp_array_size(nnz_i)}] = {{{rows_arr}}};",
      f"static const int {sym}_check_cols[{_cpp_array_size(nnz_i)}] = {{{cols_arr}}};",
    ]
  lines += [""]

  lines += [
    "static int correctness_check() {",
    "  if (!load_sample_inputs()) return 1;",
    "  const double atol = 1e-9;",
    "  const double rtol = 1e-9;",
    f"  static std::array<double, {_cpp_array_size(dense_size)}> got_dense{{}};",
  ]
  for info in infos:
    name = str(info["name"])
    label = str(info["backend"])
    input_args = info["input_args"]
    nnz_i = int(info["nnz"])
    needs_iw_w = label.startswith("casadi")
    arg_init = ", ".join(f"g_{a}.data()" for a in input_args)
    res_args = "iw.data(), w.data(), 0" if needs_iw_w else "nullptr, w_alloy.data(), nullptr"
    w_size_i = int(info["w_size"])
    if needs_iw_w:
      w_lines = [
        f"    static std::array<double, {_cpp_array_size(w_size_i)}> w{{}};",
        f"    static std::array<int, {_cpp_array_size(int(info['iw_size']))}> iw{{}};",
      ]
    else:
      w_lines = [f"    static std::array<double, {_cpp_array_size(w_size_i)}> w_alloy{{}};"]
    lines += [
      "  {",
      f"    const double* arg[] = {{{arg_init}}};",
      *w_lines,
      f"    static std::array<double, {_cpp_array_size(nnz_i)}> compact{{}};",
      "    double* res[] = {compact.data()};",
      f"    int rc = {name}(arg, res, {res_args});",
      f'    if (rc != 0) {{ std::fprintf(stderr, "{label} returned %d\\n", rc); return 1; }}',
      "    std::memset(got_dense.data(), 0, sizeof(got_dense));",
      f"    for (std::size_t i = 0; i < {nnz_i}; ++i) {{",
      f"      int row = {name}_check_rows[i];",
      f"      int col = {name}_check_cols[i];",
      f"      got_dense[row * {n_cols} + col] = compact[i];",
      "    }",
      f"    for (std::size_t k = 0; k < {dense_size}; ++k) {{",
      "      double diff = std::fabs(got_dense[k] - g_expected_dense[k]);",
      "      double thresh = atol + rtol * std::fabs(g_expected_dense[k]);",
      "      bool same_nan = std::isnan(got_dense[k]) && std::isnan(g_expected_dense[k]);",
      "      bool ok = same_nan || (std::isfinite(diff) && diff <= thresh);",
      "      if (!ok) {",
      f"        int row = static_cast<int>(k / {n_cols});",
      f"        int col = static_cast<int>(k % {n_cols});",
      f'        std::fprintf(stderr, "{label} dense[%d,%d]=%.17g vs expected=%.17g (|diff|=%.3g, thresh=%.3g)\\n", row, col, got_dense[k], g_expected_dense[k], diff, thresh);',
      "        return 1;",
      "      }",
      "    }",
      f'    std::printf("correctness ok: {label} dense Jacobian matches reference ({n_rows}x{n_cols})\\n");',
      "  }",
    ]
  lines += ["  return 0;", "}", ""]

  for info in alloy_infos:
    name = str(info["name"])
    suffix = name.split("_N")[-1]
    nnz_i = int(info["nnz"])
    w_size_i = int(info["w_size"])
    variant_tag = "" if info.get("variant", "unrolled") == "unrolled" else str(info["variant"]).title()
    bench_name = f"BM_Alloy{variant_tag}TrackingEqJacN{suffix}"
    lines += [
      f"static void {bench_name}(benchmark::State& state) {{",
      f"  static std::array<double, {_cpp_array_size(nnz_i)}> out{{}};",
      f"  static std::array<double, {_cpp_array_size(w_size_i)}> w{{}};",
      "  const double* arg[] = {g_z.data()};",
      "  double* res[] = {out.data()};",
      "  for (auto _ : state) {",
      f"    int rc = {name}(arg, res, nullptr, w.data(), nullptr);",
      "    benchmark::DoNotOptimize(rc);",
      "    benchmark::DoNotOptimize(out.data());",
      "  }",
      f"  state.SetItemsProcessed(state.iterations() * {nnz_i});",
      "}",
      f"BENCHMARK({bench_name});",
      "",
    ]

  for info in casadi_infos:
    name = str(info["name"])
    suffix = name.split("_N")[-1]
    nnz_i = int(info["nnz"])
    w_size_i = int(info["w_size"])
    iw_size_i = int(info["iw_size"])
    bench_name = f"BM_{str(info['backend']).title().replace('_', '')}TrackingEqJacN{suffix}"
    lines += [
      f"static void {bench_name}(benchmark::State& state) {{",
      f"  static std::array<double, {_cpp_array_size(nnz_i)}> out{{}};",
      f"  static std::array<double, {_cpp_array_size(w_size_i)}> w{{}};",
      f"  static std::array<int, {_cpp_array_size(iw_size_i)}> iw{{}};",
      "  const double* arg[] = {g_z.data(), g_p.data()};",
      "  double* res[] = {out.data()};",
      "  for (auto _ : state) {",
      f"    int rc = {name}(arg, res, iw.data(), w.data(), 0);",
      "    benchmark::DoNotOptimize(rc);",
      "    benchmark::DoNotOptimize(out.data());",
      "  }",
      f"  state.SetItemsProcessed(state.iterations() * {nnz_i});",
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


def build_and_run(
  horizons: list[int],
  build_dir: Path,
  benchmark_args: list[str],
  variants: tuple[str, ...] = ("unrolled", "map"),
  casadi_kinds: tuple[str, ...] = ("sx", "mx"),
) -> None:
  tracking = _load_tracking_fixture()
  build_dir.mkdir(parents=True, exist_ok=True)

  groups: list[tuple[dict[str, int | str], tuple[dict[str, int | str], ...], list[dict[str, int | str]]]] = []
  for horizon in horizons:
    alloy_infos: list[dict[str, int | str]] = []
    for variant in variants:
      t0 = time.perf_counter()
      info = _write_alloy(tracking, horizon, build_dir, variant=variant)
      info["backend"] = "alloy" if variant == "unrolled" else f"alloy_{variant}"
      info["codegen_ms"] = int((time.perf_counter() - t0) * 1000)
      alloy_infos.append(info)
    casadi_infos: list[dict[str, int | str]] = []
    for kind in casadi_kinds:
      t0 = time.perf_counter()
      casadi_info = _write_casadi(tracking, horizon, build_dir, kind)
      casadi_info["codegen_ms"] = int((time.perf_counter() - t0) * 1000)
      casadi_infos.append(casadi_info)
    primary, *extra = alloy_infos
    groups.append((primary, tuple(extra), casadi_infos))

  print("Generated sources:")
  for primary, extra, casadi_infos in groups:
    for info in (primary, *extra, *casadi_infos):
      print(
        f"  {info['backend']} N={str(info['name']).split('_N')[-1]} "
        f"codegen_ms={info['codegen_ms']} bytes={info['source_bytes']} lines={info['source_lines']} w={info['w_size']} nnz={info['nnz']}"
      )
  for primary, extra, casadi_infos in groups:
    horizon_str = str(primary["name"]).split("_N")[-1]
    horizon = int(horizon_str)
    sample_paths = _write_tracking_sample_inputs(tracking, horizon, build_dir, primary)
    _write_benchmark_cpp(primary, casadi_infos, build_dir, sample_paths, extra_alloy_infos=extra)
    exe = build_dir / f"tracking_eq_jac_benchmark_N{horizon_str}"
    sources = ["benchmark.cpp", str(primary["source"]), *(str(info["source"]) for info in extra), *(str(info["source"]) for info in casadi_infos)]
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
    print(f"\nCompiling benchmark N={horizon_str}...")
    t0 = time.perf_counter()
    _run(cmd, cwd=build_dir)
    print(f"compile_ms={(time.perf_counter() - t0) * 1000:.1f}")
    print(f"\nRunning benchmark N={horizon_str}...")
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
  parser = argparse.ArgumentParser(description="Generate and run Alloy vs CasADi tracking equality-Jacobian Google Benchmarks.")
  parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 5, 10])
  parser.add_argument(
    "--alloy-variants",
    nargs="+",
    choices=("unrolled", "map", "map_structured"),
    default=["unrolled", "map", "map_structured"],
    help="which Alloy fixture builders to compile and benchmark; first variant is the correctness anchor",
  )
  parser.add_argument(
    "--casadi-kinds",
    nargs="+",
    choices=("sx", "mx"),
    default=["sx", "mx"],
    help="which CasADi backends to include; pass an empty list to skip CasADi entirely",
  )
  parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
  parser.add_argument("--clean", action="store_true")
  parser.add_argument("benchmark_args", nargs=argparse.REMAINDER, help="arguments passed through to the benchmark binary after --")
  args = parser.parse_args()

  benchmark_args = args.benchmark_args
  if benchmark_args[:1] == ["--"]:
    benchmark_args = benchmark_args[1:]
  if args.clean and args.build_dir.exists():
    shutil.rmtree(args.build_dir)
  build_and_run(args.horizons, args.build_dir.resolve(), benchmark_args, variants=tuple(args.alloy_variants), casadi_kinds=tuple(args.casadi_kinds))


if __name__ == "__main__":
  main()
