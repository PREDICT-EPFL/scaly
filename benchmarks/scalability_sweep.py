"""Scalability sweep across tracking horizons and unbumpercars car counts.

Runs each (workload, size, backend) cell independently:
  1. Generate the C source via Python (record codegen_ms).
  2. Skip the cell if the generated source exceeds ``--max-source-mb`` (default 50 MB).
  3. Compile a focused Google Benchmark binary that contains Alloy + the selected
     backend, with a per-compile timeout (default 180 s).
  4. Run the binary's correctness check + benchmark; parse runtime from the
     ``--benchmark_format=json`` output.
  5. Append one CSV row per cell with codegen / compile / runtime / size metrics.

Each cell's correctness check uses the existing benchmark harness machinery
(scatter compact -> dense, compare with Python Alloy reference) so a bad
codegen breaks the sweep immediately.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


class CodegenTimeout(RuntimeError):
  pass


@contextlib.contextmanager
def _codegen_deadline(seconds: float):
  """Raise CodegenTimeout if Python codegen takes longer than ``seconds``."""

  def _handler(signum, frame):  # noqa: ARG001 - signal handler signature
    raise CodegenTimeout(f"codegen exceeded {seconds:.0f}s")

  if seconds <= 0:
    yield
    return
  prev = signal.signal(signal.SIGALRM, _handler)
  signal.setitimer(signal.ITIMER_REAL, seconds)
  try:
    yield
  finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, prev)


ROOT = Path(__file__).resolve().parents[1]
TRACKING_PATH = ROOT / "benchmarks" / "alloy_tracking_eq_jac_benchmark.py"
UNBUMPERCARS_PATH = ROOT / "benchmarks" / "alloy_unbumpercars_ineq_jac_benchmark.py"
DEFAULT_BUILD_DIR = ROOT / "benchmarks" / "gen" / "scalability_sweep"


def _load_module(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load {path}")
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


def _parse_runtime_ns(stdout: str, bm_name_prefix: str) -> float | None:
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
    if entry.get("name", "").startswith(bm_name_prefix) and entry.get("run_type") == "iteration":
      return float(entry.get("cpu_time", entry.get("real_time", 0.0)))
  return None


def _row(**kwargs):
  defaults = {
    "workload": "",
    "size": 0,
    "backend": "",
    "codegen_ms": "",
    "source_bytes": "",
    "source_lines": "",
    "workspace": "",
    "nnz": "",
    "compile_ms": "",
    "compile_status": "",
    "runtime_ns": "",
    "runtime_status": "",
    "note": "",
  }
  defaults.update(kwargs)
  return defaults


def _run_bench_cell(
  workload: str,
  size: int,
  backend: str,
  build_dir: Path,
  module,
  generate_alloy,
  generate_casadi,
  write_sample_inputs,
  write_benchmark_cpp,
  bm_name_prefix: str,
  compile_timeout: float,
  max_source_mb: float,
  codegen_timeout: float,
  benchmark_min_time: str,
) -> dict:
  build_dir.mkdir(parents=True, exist_ok=True)
  for f in build_dir.glob("benchmark.cpp"):
    f.unlink()

  try:
    t0 = time.perf_counter()
    with _codegen_deadline(codegen_timeout):
      alloy_info = generate_alloy(size, build_dir)
    alloy_info.setdefault("backend", "alloy")
    alloy_codegen_ms = (time.perf_counter() - t0) * 1000
  except CodegenTimeout as e:
    if backend == "alloy":
      return _row(
        workload=workload,
        size=size,
        backend=backend,
        compile_status="codegen_timeout",
        runtime_status="skipped",
        note=str(e),
      )
    raise  # bug: non-alloy backend asking for alloy codegen, surface it

  if backend == "alloy":
    backend_info = alloy_info
    backend_codegen_ms = alloy_codegen_ms
  else:
    kind = backend.split("_", 1)[1]
    try:
      t0 = time.perf_counter()
      with _codegen_deadline(codegen_timeout):
        backend_info = generate_casadi(size, build_dir, kind)
      backend_info.setdefault("backend", f"casadi_{kind}")
      backend_codegen_ms = (time.perf_counter() - t0) * 1000
    except CodegenTimeout as e:
      return _row(
        workload=workload,
        size=size,
        backend=backend,
        compile_status="codegen_timeout",
        runtime_status="skipped",
        note=str(e),
      )

  src_mb = backend_info["source_bytes"] / (1024 * 1024)
  if src_mb > max_source_mb:
    return _row(
      workload=workload,
      size=size,
      backend=backend,
      codegen_ms=f"{backend_codegen_ms:.1f}",
      source_bytes=backend_info["source_bytes"],
      source_lines=backend_info["source_lines"],
      workspace=backend_info.get("w_size", 0),
      nnz=backend_info["nnz"],
      compile_status="skipped_size",
      runtime_status="skipped",
      note=f"source {src_mb:.1f} MB > {max_source_mb:.1f} MB",
    )

  casadi_infos = [backend_info] if backend != "alloy" else []
  sample_paths = write_sample_inputs(size, build_dir, alloy_info)
  write_benchmark_cpp(alloy_info, casadi_infos, build_dir, sample_paths)

  exe = build_dir / f"{workload}_n{size}_{backend}"
  sources = [
    "benchmark.cpp",
    str(alloy_info["source"]),
    *(str(info["source"]) for info in casadi_infos),
  ]
  cmd = [
    module._compiler(),
    "-O3",
    "-std=c++17",
    "-I",
    str(build_dir),
    *sources,
    "-o",
    str(exe),
    *module._pkg_config("benchmark", "--cflags"),
    *module._pkg_config("benchmark", "--libs"),
    "-lm",
  ]
  t0 = time.perf_counter()
  # start_new_session=True puts the compiler driver in its own process group, so we can
  # kill the whole group (driver + cc1 + ld) on timeout instead of orphaning cc1.
  proc = subprocess.Popen(cmd, cwd=build_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
  try:
    stdout, stderr = proc.communicate(timeout=compile_timeout)
    if proc.returncode != 0:
      raise subprocess.CalledProcessError(proc.returncode, cmd, stdout, stderr)
    compile_status = "ok"
    compile_ms = (time.perf_counter() - t0) * 1000
  except subprocess.TimeoutExpired:
    import os

    try:
      os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
      pass
    proc.wait()
    return _row(
      workload=workload,
      size=size,
      backend=backend,
      codegen_ms=f"{backend_codegen_ms:.1f}",
      source_bytes=backend_info["source_bytes"],
      source_lines=backend_info["source_lines"],
      workspace=backend_info.get("w_size", 0),
      nnz=backend_info["nnz"],
      compile_ms=f">{compile_timeout * 1000:.0f}",
      compile_status="timeout",
      runtime_status="skipped",
      note=f"compile > {compile_timeout:.0f}s",
    )
  except subprocess.CalledProcessError as e:
    msg = (e.stderr or "")[-300:].strip().replace("\n", " ")
    return _row(
      workload=workload,
      size=size,
      backend=backend,
      codegen_ms=f"{backend_codegen_ms:.1f}",
      source_bytes=backend_info["source_bytes"],
      source_lines=backend_info["source_lines"],
      workspace=backend_info.get("w_size", 0),
      nnz=backend_info["nnz"],
      compile_status="compile_error",
      runtime_status="skipped",
      note=msg,
    )

  run_cmd = [str(exe), f"--benchmark_min_time={benchmark_min_time}", "--benchmark_format=json", f"--benchmark_filter={bm_name_prefix}"]
  try:
    result = subprocess.run(run_cmd, cwd=build_dir, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
  except subprocess.CalledProcessError as e:
    msg = ((e.stderr or "") + (e.stdout or ""))[-400:].strip().replace("\n", " ")
    return _row(
      workload=workload,
      size=size,
      backend=backend,
      codegen_ms=f"{backend_codegen_ms:.1f}",
      source_bytes=backend_info["source_bytes"],
      source_lines=backend_info["source_lines"],
      workspace=backend_info.get("w_size", 0),
      nnz=backend_info["nnz"],
      compile_ms=f"{compile_ms:.1f}",
      compile_status=compile_status,
      runtime_status="correctness_fail" if "correctness" in msg.lower() else "runtime_error",
      note=msg,
    )

  runtime_ns = _parse_runtime_ns(result.stdout, bm_name_prefix)
  return _row(
    workload=workload,
    size=size,
    backend=backend,
    codegen_ms=f"{backend_codegen_ms:.1f}",
    source_bytes=backend_info["source_bytes"],
    source_lines=backend_info["source_lines"],
    workspace=backend_info.get("w_size", 0),
    nnz=backend_info["nnz"],
    compile_ms=f"{compile_ms:.1f}",
    compile_status=compile_status,
    runtime_ns=f"{runtime_ns:.1f}" if runtime_ns is not None else "",
    runtime_status="ok" if runtime_ns is not None else "parse_fail",
  )


def _tracking_callbacks(variant: str = "map_structured"):
  track = _load_module("alloy_tracking_bench", TRACKING_PATH)
  fixture = track._load_tracking_fixture()
  variant_tag = "" if variant == "unrolled" else variant.title()

  def generate_alloy(size, build_dir):
    return track._write_alloy(fixture, size, build_dir, variant=variant)

  def generate_casadi(size, build_dir, kind):
    return track._write_casadi(fixture, size, build_dir, kind)

  def write_sample_inputs(size, build_dir, alloy_info):
    return track._write_tracking_sample_inputs(fixture, size, build_dir, alloy_info)

  def bm_prefix(backend):
    return {
      "alloy": f"BM_Alloy{variant_tag}TrackingEqJacN",
      "casadi_sx": "BM_CasadiSxTrackingEqJacN",
      "casadi_mx": "BM_CasadiMxTrackingEqJacN",
    }[backend]

  return track, generate_alloy, generate_casadi, write_sample_inputs, track._write_benchmark_cpp, bm_prefix


def _unbumpercars_callbacks():
  ub = _load_module("alloy_unbumpercars_bench", UNBUMPERCARS_PATH)
  fixture = ub._load_fixture()

  def generate_alloy(size, build_dir):
    return ub._write_alloy(fixture, size, build_dir)

  def generate_casadi(size, build_dir, kind):
    return ub._write_casadi(fixture, size, build_dir, kind)

  def write_sample_inputs(size, build_dir, alloy_info):
    return ub._write_sample_inputs(fixture, size, build_dir, alloy_info)

  def bm_prefix(backend):
    return {"alloy": "BM_AlloyUnbumpercarsIneqJacN", "casadi_sx": "BM_CasadiSxUnbumpercarsIneqJacN", "casadi_mx": "BM_CasadiMxUnbumpercarsIneqJacN"}[
      backend
    ]

  return ub, generate_alloy, generate_casadi, write_sample_inputs, ub._write_benchmark_cpp, bm_prefix


def main():
  parser = argparse.ArgumentParser(description="Scalability sweep across tracking horizons and unbumpercars car counts.")
  parser.add_argument("--tracking-horizons", nargs="+", type=int, default=[1, 5, 10, 25, 50, 100, 200, 500, 1000])
  parser.add_argument("--unbumpercars-cars", nargs="+", type=int, default=[2, 4, 8, 16, 32])
  parser.add_argument("--backends", nargs="+", default=["alloy", "casadi_sx", "casadi_mx"])
  parser.add_argument("--compile-timeout", type=float, default=180.0, help="per-cell compile timeout (s)")
  parser.add_argument("--codegen-timeout", type=float, default=300.0, help="per-cell Python codegen timeout (s); 0 disables")
  parser.add_argument("--max-source-mb", type=float, default=50.0, help="skip cell if generated source exceeds this size")
  parser.add_argument("--benchmark-min-time", default="0.1s")
  parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
  parser.add_argument("--csv", type=Path, default=ROOT / "benchmarks" / "scalability_results.csv")
  parser.add_argument("--workloads", nargs="+", choices=("tracking", "unbumpercars"), default=("tracking", "unbumpercars"))
  args = parser.parse_args()

  fields = [
    "workload",
    "size",
    "backend",
    "codegen_ms",
    "source_bytes",
    "source_lines",
    "workspace",
    "nnz",
    "compile_ms",
    "compile_status",
    "runtime_ns",
    "runtime_status",
    "note",
  ]
  args.csv.parent.mkdir(parents=True, exist_ok=True)
  with args.csv.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()

    def run_cells(workload, sizes, callbacks):
      module, gen_alloy, gen_casadi, gen_samples, write_bench, bm_prefix = callbacks
      build_dir = args.build_dir / workload
      # If a backend already gave up at a smaller size (compile timeout / OOM / size cap),
      # bigger sizes won't fix it: source and compile both grow monotonically.
      gave_up: dict[str, tuple[int, str]] = {}
      for size in sorted(sizes):
        for backend in args.backends:
          print(f"[{workload}] size={size} backend={backend} ... ", end="", flush=True)
          if backend in gave_up:
            failed_size, failed_status = gave_up[backend]
            row = _row(
              workload=workload,
              size=size,
              backend=backend,
              compile_status="skipped_after_failure",
              runtime_status="skipped",
              note=f"{backend} {failed_status} at size={failed_size}; larger sizes won't fit",
            )
            writer.writerow(row)
            fp.flush()
            print(f"skipped_after_failure: {row['note']}")
            continue
          row = _run_bench_cell(
            workload=workload,
            size=size,
            backend=backend,
            build_dir=build_dir,
            module=module,
            generate_alloy=gen_alloy,
            generate_casadi=gen_casadi,
            write_sample_inputs=gen_samples,
            write_benchmark_cpp=write_bench,
            bm_name_prefix=bm_prefix(backend),
            compile_timeout=args.compile_timeout,
            max_source_mb=args.max_source_mb,
            codegen_timeout=args.codegen_timeout,
            benchmark_min_time=args.benchmark_min_time,
          )
          status = row["runtime_status"] or row["compile_status"]
          if row["runtime_ns"]:
            print(f"{status}, runtime={float(row['runtime_ns']) / 1000:.2f} us, source={int(row['source_bytes']) / 1024:.1f} KB")
          else:
            print(f"{status}: {row['note']}")
          writer.writerow(row)
          fp.flush()
          # Mark this backend as given up if compile/run failed in a way that won't recover.
          # Generated source and compile time are monotonically increasing in size, so a
          # compile timeout or size cap at N implies the same at all N' > N.
          if row["compile_status"] in {"timeout", "skipped_size", "compile_error", "codegen_timeout"}:
            gave_up[backend] = (size, row["compile_status"])

    if "tracking" in args.workloads:
      run_cells("tracking", args.tracking_horizons, _tracking_callbacks())
    if "unbumpercars" in args.workloads:
      run_cells("unbumpercars", args.unbumpercars_cars, _unbumpercars_callbacks())

  print(f"\nResults written to {args.csv}")


if __name__ == "__main__":
  main()
