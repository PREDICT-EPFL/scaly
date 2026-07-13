from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time


def compiler() -> str:
  return os.environ.get("CXX") or shutil.which("clang++") or shutil.which("c++") or "c++"


def pkg_config(package: str, flag: str) -> list[str]:
  try:
    proc = subprocess.run(["pkg-config", flag, package], check=True, text=True, capture_output=True)
  except FileNotFoundError as e:
    raise RuntimeError("pkg-config is required to locate Google Benchmark") from e
  except subprocess.CalledProcessError as e:
    detail = (e.stderr or e.stdout).strip()
    raise RuntimeError(f"pkg-config could not find {package!r}: {detail}") from e
  return shlex.split(proc.stdout)


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


def write_cpp(info: dict, out_dir: Path, input_paths: dict[str, Path], expected_path: Path) -> Path:
  name, label = info["name"], info["backend"]
  n_rows, n_cols, nnz = int(info["n_rows"]), int(info["n_cols"]), int(info["nnz"])
  dense_size, w_size, iw_size = n_rows * n_cols, int(info["w_size"]), int(info.get("iw_size", 0))
  inputs = [(str(k), int(v)) for k, v in info["inputs"]]
  arg_size, res_size = int(info.get("arg_size", len(inputs))), int(info.get("res_size", 1))
  rows = ", ".join(str(x) for x in info["rows"]) or "0"
  cols = ", ".join(str(x) for x in info["cols"]) or "0"
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
  for input_name, size in inputs:
    lines.append(f"static std::array<double, {_array_size(size)}> g_{input_name}{{}};")
  lines += [
    f"static std::array<double, {_array_size(dense_size)}> g_expected_dense{{}};",
    f"static const int check_rows[{_array_size(nnz)}] = {{{rows}}};",
    f"static const int check_cols[{_array_size(nnz)}] = {{{cols}}};",
    "",
    "static bool load_sample_inputs() {",
    "  return "
    + "\n      && ".join(
      [f'load_doubles("{input_paths[k].name}", g_{k})' for k, _ in inputs] + [f'load_doubles("{expected_path.name}", g_expected_dense)']
    )
    + ";",
    "}",
    "",
    "static int correctness_check() {",
    "  if (!load_sample_inputs()) return 1;",
    f"  static std::array<double, {_array_size(nnz)}> compact{{}};",
    f"  static std::array<double, {_array_size(dense_size)}> dense{{}};",
    f"  static std::array<double, {_array_size(w_size)}> w{{}};",
  ]
  if label.startswith("casadi"):
    lines.append(f"  static std::array<int, {_array_size(iw_size)}> iw{{}};")
  lines += [
    f"  std::array<const double*, {_array_size(arg_size)}> arg{{}};",
    *[f"  arg[{i}] = g_{k}.data();" for i, (k, _) in enumerate(inputs)],
    f"  std::array<double*, {_array_size(res_size)}> res{{}};",
    "  res[0] = compact.data();",
    f"  int rc = {name}(arg.data(), res.data(), " + ("iw.data(), w.data(), 0);" if label.startswith("casadi") else "nullptr, w.data(), nullptr);"),
    f'  if (rc != 0) {{ std::fprintf(stderr, "{label} returned %d\\n", rc); return 1; }}',
    "  std::memset(dense.data(), 0, sizeof(dense));",
    f"  for (std::size_t i = 0; i < {nnz}; ++i) dense[check_rows[i] * {n_cols} + check_cols[i]] = compact[i];",
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
    f'  std::printf("correctness ok: {label} dense Jacobian matches reference ({n_rows}x{n_cols})\\n");',
    "  return 0;",
    "}",
    "",
    f"static void {info['benchmark']}(benchmark::State& state) {{",
    f"  static std::array<double, {_array_size(nnz)}> out{{}};",
    f"  static std::array<double, {_array_size(w_size)}> w{{}};",
  ]
  if label.startswith("casadi"):
    lines.append(f"  static std::array<int, {_array_size(iw_size)}> iw{{}};")
  lines += [
    f"  std::array<const double*, {_array_size(arg_size)}> arg{{}};",
    *[f"  arg[{i}] = g_{k}.data();" for i, (k, _) in enumerate(inputs)],
    f"  std::array<double*, {_array_size(res_size)}> res{{}};",
    "  res[0] = out.data();",
    "  for (auto _ : state) {",
    f"    int rc = {name}(arg.data(), res.data(), " + ("iw.data(), w.data(), 0);" if label.startswith("casadi") else "nullptr, w.data(), nullptr);"),
    "    benchmark::DoNotOptimize(rc);",
    "    benchmark::DoNotOptimize(out.data());",
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
  exe = out_dir / "benchmark"
  cmd = [
    compiler(),
    "-O3",
    "-std=c++17",
    "-I",
    str(out_dir),
    "benchmark.cpp",
    str(info["source"]),
    "-o",
    str(exe),
    *pkg_config("benchmark", "--cflags"),
    *pkg_config("benchmark", "--libs"),
    "-lm",
  ]
  started = time.perf_counter()
  try:
    proc = subprocess.Popen(cmd, cwd=out_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
  except OSError as e:
    (out_dir / "compile.log").write_text(f"$ {shlex.join(cmd)}\n{e}\n")
    return "compile_error", None, str(e)
  try:
    stdout, stderr = proc.communicate(timeout=timeout)
  except subprocess.TimeoutExpired:
    try:
      os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
      pass
    stdout, stderr = proc.communicate()
    (out_dir / "compile.log").write_text(f"$ {shlex.join(cmd)}\n{stdout}{stderr}")
    return "timeout", None, f"compile > {timeout:.0f}s"
  elapsed = (time.perf_counter() - started) * 1000
  (out_dir / "compile.log").write_text(f"$ {shlex.join(cmd)}\n{stdout}{stderr}")
  if proc.returncode:
    return "compile_error", elapsed, (stderr or stdout)[-400:].strip().replace("\n", " ")
  return "ok", elapsed, ""


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
