from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

from alloy.codegen.abi import c_ident
from benchmarks.harness import NATIVE_CFLAGS, ROOT


def compiler() -> str:
  return os.environ.get("CXX") or shutil.which("clang++") or shutil.which("c++") or "c++"


GBENCH_TAG = "v1.9.5"
GBENCH_REPO = "https://github.com/google/benchmark.git"


def _cmake_run(cmd: list[str], log: Path) -> None:
  try:
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
  except FileNotFoundError as e:
    raise RuntimeError(f"{cmd[0]} is required to build Google Benchmark {GBENCH_TAG}") from e
  with log.open("a") as fp:
    fp.write(f"$ {shlex.join(cmd)}\n{proc.stdout}{proc.stderr}")
  if proc.returncode:
    raise RuntimeError(f"{shlex.join(cmd[:2])} failed ({proc.returncode}), see {log}: {(proc.stderr or proc.stdout)[-400:].strip()}")


def _build_gbench(prefix: Path, header: Path, lib: Path) -> None:
  src, build = prefix / "src", prefix / "build"
  prefix.mkdir(parents=True, exist_ok=True)
  log = prefix / "build.log"
  log.write_text("")
  print(f"building Google Benchmark {GBENCH_TAG} in {prefix} (one-time)", flush=True)
  if not (src / "CMakeLists.txt").exists():
    shutil.rmtree(src, ignore_errors=True)  # a half-finished clone would otherwise poison every later run
    _cmake_run(["git", "clone", "--depth=1", "--branch", GBENCH_TAG, GBENCH_REPO, str(src)], log)
  _cmake_run(
    [
      "cmake",
      "-S",
      str(src),
      "-B",
      str(build),
      "-DCMAKE_BUILD_TYPE=Release",
      f"-DCMAKE_INSTALL_PREFIX={prefix}",
      "-DCMAKE_INSTALL_LIBDIR=lib",  # GNUInstallDirs would pick lib64 on some distributions
      "-DBUILD_SHARED_LIBS=OFF",
      "-DBENCHMARK_ENABLE_TESTING=OFF",
      "-DBENCHMARK_ENABLE_WERROR=OFF",
    ],
    log,
  )
  _cmake_run(["cmake", "--build", str(build), "--parallel"], log)
  _cmake_run(["cmake", "--install", str(build)], log)
  shutil.rmtree(build, ignore_errors=True)
  if not (header.exists() and lib.exists()):
    raise RuntimeError(f"Google Benchmark build left no {header.name}/{lib.name} under {prefix}, see {log}")


def gbench_flags() -> tuple[list[str], list[str]]:
  """Compile and link flags for the pinned Google Benchmark, building it inside this checkout on first use."""
  prefix = ROOT / "benchmarks" / "third_party" / "gbench" / GBENCH_TAG
  header, lib = prefix / "include" / "benchmark" / "benchmark.h", prefix / "lib" / "libbenchmark.a"
  if not (header.exists() and lib.exists()):
    _build_gbench(prefix, header, lib)
  return ["-I", str(prefix / "include"), "-pthread"], [str(lib)]


def parse_runtime_ns(stdout: str, prefix: str) -> float | None:
  start = stdout.find("{\n")
  if start < 0:
    start = stdout.find("{")
  if start < 0:
    return None
  try:
    data = json.loads(stdout[start:])
  except json.JSONDecodeError:
    return None
  for entry in data.get("benchmarks", []):
    if entry.get("name", "").startswith(prefix) and entry.get("run_type") == "iteration":
      return float(entry.get("cpu_time", entry.get("real_time", 0.0)))
  return None


def _array_size(value: int) -> int:
  return max(value, 1)


def _kernel_input_names(kernel) -> tuple[str, ...]:
  """Read a kernel's ABI input names without depending on its symbolic-library type."""
  name_in = getattr(kernel, "name_in", None)
  if callable(name_in):
    return tuple(str(name_in(index)) for index in range(kernel.n_in()))
  input_names = getattr(kernel, "input_names", ())
  return tuple(str(name) for name in input_names)


def _ordered_inputs(info: dict) -> tuple[tuple[str, str, int], ...]:
  """Return ``(sample key, C identifier, size)`` entries in the kernel's ABI order.

  Alloy keeps names such as ``lam:f`` in the Function object while the generated C ABI sanitizes
  them to ``lam_f``. The benchmark metadata historically used the sanitized spelling, so matching
  both forms lets the driver follow the kernel without changing sample-file names.
  """
  supplied = [(str(name), int(size)) for name, size in info["inputs"]]
  by_name = {name: (name, size) for name, size in supplied}
  by_c_name: dict[str, tuple[str, int]] = {}
  for name, size in supplied:
    c_name = c_ident(name)
    if c_name in by_c_name and by_c_name[c_name][0] != name:
      raise ValueError(f"input names {by_c_name[c_name][0]!r} and {name!r} share C identifier {c_name!r}")
    by_c_name[c_name] = (name, size)

  kernel_names = _kernel_input_names(info.get("callable")) if info.get("callable") is not None else ()
  if kernel_names:
    ordered: list[tuple[str, int]] = []
    used: set[str] = set()
    for kernel_name in kernel_names:
      entry = by_name.get(kernel_name) or by_c_name.get(c_ident(kernel_name))
      if entry is None:
        raise ValueError(f"no benchmark input metadata for kernel input {kernel_name!r}")
      if entry[0] in used:
        raise ValueError(f"benchmark input metadata maps more than one kernel input to {entry[0]!r}")
      used.add(entry[0])
      ordered.append(entry)
    if len(used) != len(supplied):
      extra = sorted(set(by_name) - used)
      raise ValueError(f"benchmark input metadata has inputs absent from kernel ABI: {extra}")
    supplied = ordered

  entries = tuple((name, c_ident(name), size) for name, size in supplied)
  c_names = [c_name for _, c_name, _ in entries]
  if len(set(c_names)) != len(c_names):
    raise ValueError(f"benchmark input names share C identifiers: {c_names}")
  return entries


def _kernel_symbol(info: dict) -> str:
  """Return the symbol exported by the generated source, not a construction label."""
  if info.get("symbol") is not None:
    return str(info["symbol"])
  kernel = info.get("callable")
  kernel_name = getattr(kernel, "name", None)
  if callable(kernel_name):
    kernel_name = kernel_name()
  if kernel_name is not None:
    return c_ident(str(kernel_name))
  return c_ident(str(info["name"]))


def write_cpp(info: dict, out_dir: Path, input_paths: dict[str, Path], expected_path: Path) -> Path:
  """Write the compiled C++ benchmark for one selected result layout.

  CasADi's ``nlp_jac_g`` has two result pointers, but IPOPT requests only ``jac_g_x`` (output 1).
  ``requested_output_indices`` keeps that callback contract visible in the generated driver and
  leaves every other result pointer null on every call.
  """
  label = info["backend"]
  symbol = _kernel_symbol(info)
  n_rows, n_cols, nnz = int(info["n_rows"]), int(info["n_cols"]), int(info["nnz"])
  dense_size, w_size, iw_size = n_rows * n_cols, int(info["w_size"]), int(info.get("iw_size", 0))
  input_specs = _ordered_inputs(info)
  inputs = [(sample_name, c_name, size) for sample_name, c_name, size in input_specs]
  arg_size, res_size = int(info.get("arg_size", len(inputs))), int(info.get("res_size", 1))
  output_index = int(info.get("output_index", 0))
  requested_outputs = tuple(int(index) for index in info.get("requested_output_indices", (output_index,)))
  output_nnz = tuple(int(value) for value in info.get("output_nnz", (nnz,)))
  if output_index not in requested_outputs:
    raise ValueError(f"selected output {output_index} is not among requested outputs {requested_outputs}")
  if len(set(requested_outputs)) != len(requested_outputs):
    raise ValueError(f"requested result pointers {requested_outputs} contain duplicates")
  if any(index < 0 or index >= res_size for index in requested_outputs):
    raise ValueError(f"requested result pointers {requested_outputs} do not fit result array of length {res_size}")
  if any(index >= len(output_nnz) for index in requested_outputs):
    raise ValueError(f"requested result pointers {requested_outputs} have no output sizes")
  if output_nnz[output_index] != nnz:
    raise ValueError(f"selected output {output_index} has {output_nnz[output_index]} values, expected {nnz}")
  row_values = tuple(int(x) for x in info["rows"])
  col_values = tuple(int(x) for x in info["cols"])
  if len(row_values) != nnz or len(col_values) != nnz:
    raise ValueError(f"selected output has {len(row_values)} row and {len(col_values)} column coordinates, expected {nnz}")
  rows = ", ".join(str(x) for x in row_values) or "0"
  cols = ", ".join(str(x) for x in col_values) or "0"
  check_order = tuple(sorted(range(nnz), key=lambda index: (row_values[index], col_values[index])))
  order = ", ".join(str(index) for index in check_order) or "0"
  output_buffers = {index: f"g_out_{index}" for index in requested_outputs}
  layout = str(info.get("layout", "full"))
  if layout not in {"full", "lower", "upper"}:
    raise ValueError(f"unsupported benchmark layout {layout!r}")
  selected_buffer = output_buffers[output_index]
  lines = [
    '#include "benchmark/benchmark.h"',
    "#include <array>",
    "#include <cmath>",
    "#include <cstddef>",
    "#include <cstdio>",
    "#include <cstring>",
    f'#include "{info["header"]}"',
    "",
    "template <std::size_t N>",
    "static bool load_doubles(const char* path, std::array<double, N>& out) {",
    '  std::FILE* f = std::fopen(path, "rb");',
    '  if (!f) { std::fprintf(stderr, "could not open %s\\n", path); return false; }',
    "  std::size_t got = std::fread(out.data(), sizeof(double), N, f);",
    "  std::fclose(f);",
    '  if (got != N) { std::fprintf(stderr, "expected %zu doubles in %s, got %zu\\n", N, path, got); return false; }',
    "  return true;",
    "}",
    "",
  ]
  for _, c_name, size in inputs:
    lines.append(f"static std::array<double, {_array_size(size)}> g_{c_name}{{}};")
  for index in requested_outputs:
    lines.append(f"static std::array<double, {_array_size(output_nnz[index])}> {output_buffers[index]}{{}};")
  lines += [
    f"static std::array<double, {_array_size(dense_size)}> g_expected_dense{{}};",
    f"static const int check_rows[{_array_size(nnz)}] = {{{rows}}};",
    f"static const int check_cols[{_array_size(nnz)}] = {{{cols}}};",
    *([f"static const int check_order[{_array_size(nnz)}] = {{{order}}};"] if layout != "full" else []),
    "",
    "static bool load_sample_inputs() {",
    "  return "
    + "\n      && ".join(
      [f'load_doubles("{input_paths[sample_name].name}", g_{c_name})' for sample_name, c_name, _ in inputs]
      + [f'load_doubles("{expected_path.name}", g_expected_dense)']
    )
    + ";",
    "}",
    "",
    "static int correctness_check() {",
    "  if (!load_sample_inputs()) return 1;",
    *([f"  static std::array<double, {_array_size(dense_size)}> dense{{}};"] if layout == "full" else []),
    f"  static std::array<double, {_array_size(w_size)}> w{{}};",
  ]
  if label.startswith("casadi"):
    lines.append(f"  static std::array<int, {_array_size(iw_size)}> iw{{}};")
  lines += [
    f"  std::array<const double*, {_array_size(arg_size)}> arg{{}};",
    *[f"  arg[{i}] = g_{c_name}.data();" for i, (_, c_name, _) in enumerate(inputs)],
    f"  std::array<double*, {_array_size(res_size)}> res{{}};",
    "  res.fill(nullptr);",
    *[f"  res[{index}] = {output_buffers[index]}.data();" for index in requested_outputs],
    f"  int rc = {symbol}(arg.data(), res.data(), " + ("iw.data(), w.data(), 0);" if label.startswith("casadi") else "nullptr, w.data(), nullptr);"),
    f'  if (rc != 0) {{ std::fprintf(stderr, "{label} returned %d\\n", rc); return 1; }}',
  ]
  if layout in {"lower", "upper"}:
    triangle_loop = (
      [
        f"  for (int row = 0; row < {n_rows}; ++row) {{",
        f"    int last_col = row + 1 < {n_cols} ? row + 1 : {n_cols};",
        "    for (int col = 0; col < last_col; ++col) {",
      ]
      if layout == "lower"
      else [
        f"  for (int row = 0; row < {n_rows}; ++row) {{",
        f"    for (int col = row; col < {n_cols}; ++col) {{",
      ]
    )
    lines += [
      "  std::size_t next = 0;",
      *triangle_loop,
      "      double got = 0.0;",
      f"      if (next < {nnz} && check_rows[check_order[next]] == row && check_cols[check_order[next]] == col) {{",
      f"        got = {selected_buffer}[check_order[next]];",
      "        ++next;",
      "      }",
      f"      double want = g_expected_dense[row * {n_cols} + col];",
      "      double diff = std::fabs(got - want);",
      "      bool exact = got == want;",  # inf == inf must pass: inf - inf is NaN and would fail the diff test
      "      bool same_nan = std::isnan(got) && std::isnan(want);",
      "      if (!(exact || same_nan || (std::isfinite(diff) && diff <= 1e-9 + 1e-9 * std::fabs(want)))) {",
      f'        std::fprintf(stderr, "{label} dense[%d,%d]=%.17g vs expected=%.17g\\n", row, col, got, want);',
      "        return 1;",
      "      }",
      "    }",
      "  }",
      f'  if (next != {nnz}) {{ std::fprintf(stderr, "{label} output has a coordinate outside the {layout} triangle\\n"); return 1; }}',
    ]
  else:
    lines += [
      "  std::memset(dense.data(), 0, sizeof(dense));",
      f"  for (std::size_t i = 0; i < {nnz}; ++i) dense[check_rows[i] * {n_cols} + check_cols[i]] = {selected_buffer}[i];",
      f"  for (std::size_t k = 0; k < {dense_size}; ++k) {{",
      "    double diff = std::fabs(dense[k] - g_expected_dense[k]);",
      "    double thresh = 1e-9 + 1e-9 * std::fabs(g_expected_dense[k]);",
      "    bool exact = dense[k] == g_expected_dense[k];",  # inf == inf must pass: inf - inf is NaN and would fail the diff test
      "    bool same_nan = std::isnan(dense[k]) && std::isnan(g_expected_dense[k]);",
      "    if (!(exact || same_nan || (std::isfinite(diff) && diff <= thresh))) {",
      f'      std::fprintf(stderr, "{label} dense[%zu,%zu]=%.17g vs expected=%.17g\\n", k / {n_cols}, k % {n_cols}, dense[k], g_expected_dense[k]);',
      "      return 1;",
      "    }",
      "  }",
    ]
  lines += [
    f'  std::printf("correctness ok: {label} {layout} layout matches reference ({n_rows}x{n_cols})\\n");',
    "  return 0;",
    "}",
    "",
    f"static void {info['benchmark']}(benchmark::State& state) {{",
    f"  static std::array<double, {_array_size(w_size)}> w{{}};",
  ]
  if label.startswith("casadi"):
    lines.append(f"  static std::array<int, {_array_size(iw_size)}> iw{{}};")
  lines += [
    f"  std::array<const double*, {_array_size(arg_size)}> arg{{}};",
    *[f"  arg[{i}] = g_{c_name}.data();" for i, (_, c_name, _) in enumerate(inputs)],
    f"  std::array<double*, {_array_size(res_size)}> res{{}};",
    "  for (auto _ : state) {",
    "    res.fill(nullptr);",
    *[f"    res[{index}] = {output_buffers[index]}.data();" for index in requested_outputs],
    f"    int rc = {symbol}(arg.data(), res.data(), "
    + ("iw.data(), w.data(), 0);" if label.startswith("casadi") else "nullptr, w.data(), nullptr);"),
    "    benchmark::DoNotOptimize(rc);",
    *[f"    benchmark::DoNotOptimize({output_buffers[index]}.data());" for index in requested_outputs],
    "  }",
    f"  state.SetItemsProcessed(state.iterations() * {nnz});",
    "}",
    f"BENCHMARK({info['benchmark']});",
    "",
    "int main(int argc, char** argv) {",
    '  if (correctness_check() != 0) { std::fprintf(stderr, "correctness check failed\\n"); return 2; }',
    "  benchmark::Initialize(&argc, argv);",
    "  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;",
    "  benchmark::RunSpecifiedBenchmarks();",
    "  benchmark::Shutdown();",
    "  return 0;",
    "}",
  ]
  path = out_dir / "benchmark.cpp"
  path.write_text("\n".join(lines) + "\n")
  return path


def compile_kernel(info: dict, out_dir: Path, timeout: float) -> tuple[str, float | None, str]:
  cflags, libs = gbench_flags()
  common = [compiler(), "-O3", *NATIVE_CFLAGS, "-std=c++17", "-I", str(out_dir), *cflags]
  commands = {
    "kernel_compile_ms": [*common, "-c", str(info["source"]), "-o", "kernel.o"],
    "wrapper_compile_ms": [*common, "-c", "benchmark.cpp", "-o", "wrapper.o"],
    "link_ms": [compiler(), *cflags, "kernel.o", "wrapper.o", "-o", str(out_dir / "benchmark"), *libs, *info.get("extra_libs", ()), "-lm"],
  }
  started = time.perf_counter()
  timings = info["compile_timings"] = {}
  with (out_dir / "compile.log").open("w") as log:
    for phase, cmd in commands.items():
      log.write(f"$ {shlex.join(cmd)}\n")
      phase_started = time.perf_counter()
      try:
        proc = subprocess.Popen(cmd, cwd=out_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
      except OSError as e:
        log.write(f"{e}\n")
        return "compile_error", None, str(e)
      try:
        stdout, stderr = proc.communicate(timeout=max(0.001, timeout - (phase_started - started)))
      except subprocess.TimeoutExpired:
        try:
          os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
          pass
        stdout, stderr = proc.communicate()
        log.write(stdout + stderr)
        return "timeout", None, f"compile > {timeout:.0f}s ({phase})"
      timings[phase] = (time.perf_counter() - phase_started) * 1000
      log.write(stdout + stderr)
      if proc.returncode:
        return "compile_error", (time.perf_counter() - started) * 1000, (stderr or stdout)[-400:].strip().replace("\n", " ")
  return "ok", (time.perf_counter() - started) * 1000, ""


def _text(value) -> str:
  return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")


def run_kernel(info: dict, out_dir: Path, benchmark_min_time: str, timeout: float = 120) -> tuple[str, float | None, str]:
  cmd = [
    str(out_dir / "benchmark"),
    f"--benchmark_min_time={benchmark_min_time}",
    "--benchmark_format=json",
    f"--benchmark_filter={info['benchmark']}",
  ]

  def archive(stdout, stderr) -> tuple[str, str]:
    stdout, stderr = _text(stdout), _text(stderr)
    (out_dir / "benchmark.json").write_text(stdout)
    (out_dir / "run.log").write_text(f"$ {shlex.join(cmd)}\n{stderr}")
    return stdout, stderr

  try:
    proc = subprocess.run(cmd, cwd=out_dir, check=True, text=True, capture_output=True, timeout=timeout)
  except subprocess.TimeoutExpired as e:
    archive(e.stdout, e.stderr)
    return "runtime_timeout", None, f"run > {timeout:.0f}s"
  except subprocess.CalledProcessError as e:
    stdout, stderr = archive(e.stdout, e.stderr)
    note = (stderr + stdout)[-400:].strip().replace("\n", " ")
    return ("correctness_fail" if e.returncode == 2 else "runtime_error"), None, note
  stdout, _ = archive(proc.stdout, proc.stderr)
  runtime = parse_runtime_ns(stdout, str(info["benchmark"]))
  return ("ok", runtime, "") if runtime is not None else ("parse_fail", None, "benchmark JSON contained no matching iteration")
