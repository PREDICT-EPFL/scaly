from __future__ import annotations

import contextlib
import csv
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np

import alloy as al
from alloy.codegen.c import _workspace_size, render_c_module
from benchmarks.harness import gbench
from benchmarks.harness.correctness import check_dense_reference, write_samples
from benchmarks.harness.provenance import collect, write
from benchmarks.problems import chain_of_masses, tracking_nmpc, unbumpercars

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"
DEFAULT_SIZES = {
  "chain": [3, 5, 9, 17, 33, 65],
  "tracking": [1, 5, 10, 25, 50, 100, 200, 500],
  "unbumpercars": [2, 4, 8, 16, 32],
}
BACKENDS = ("alloy", "casadi_sx", "casadi_mx")
FIELDS = [
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
  "build_ms",
  "render_ms",
]


class CodegenTimeout(RuntimeError):
  pass


@contextlib.contextmanager
def codegen_deadline(seconds: float):
  def handler(signum, frame):  # noqa: ARG001
    raise CodegenTimeout(f"codegen exceeded {seconds:.0f}s")

  if seconds <= 0:
    yield
    return
  previous = signal.signal(signal.SIGALRM, handler)
  signal.setitimer(signal.ITIMER_REAL, seconds)
  try:
    yield
  finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous)


def row(**values) -> dict[str, object]:
  out: dict[str, object] = {field: "" for field in FIELDS}
  out.update({"size": 0, "compile_status": "", "runtime_status": "", **values})
  return out


def _module_info(name: str, backend: str, module, inputs, sparsity, shape, build_ms: float, render_ms: float, benchmark: str, **extra) -> dict:
  source_path = Path(module.source_name)
  return {
    "name": name,
    "backend": backend,
    "source": source_path,
    "header": module.header_name,
    "inputs": inputs,
    "nnz": sparsity.nnz,
    "n_rows": shape[0],
    "n_cols": shape[1],
    "rows": tuple(int(x) for x in sparsity.rows),
    "cols": tuple(int(x) for x in sparsity.cols),
    "w_size": extra.pop("w_size"),
    "iw_size": extra.pop("iw_size", 0),
    "source_bytes": len(module.source),
    "source_lines": module.source.count("\n") + 1,
    "build_ms": build_ms,
    "render_ms": render_ms,
    "benchmark": benchmark,
    **extra,
  }


def _render_alloy(fun: al.Function, name: str, out_dir: Path):
  started = time.perf_counter()
  module = render_c_module(fun, header_name=f"{name}.h", source_name=f"{name}.c", typed_buffers=False)
  (out_dir / module.header_name).write_text(module.header)
  (out_dir / module.source_name).write_text(module.source)
  return module, (time.perf_counter() - started) * 1000


def _tracking_alloy(size: int, out_dir: Path) -> dict:
  started = time.perf_counter()
  fn = tracking_nmpc.tracking_eq_function_map(size)
  sj = al.sparse_jacobian(fn.outputs[0], fn.inputs[0])
  name = f"alloy_tracking_eq_jac_N{size}"
  spjf = al.Function(name, fn.inputs, [sj.values], fn.input_names, ["spjac_eq_z"], [sj.sparsity])
  build_ms = (time.perf_counter() - started) * 1000
  module, render_ms = _render_alloy(spjf, name, out_dir)
  return _module_info(
    name,
    "alloy",
    module,
    [("z", tracking_nmpc.NZ * (size + 1)), ("p", tracking_nmpc.n_param(size))],
    sj.sparsity,
    (fn.outputs[0].shape[0], fn.inputs[0].shape[0]),
    build_ms,
    render_ms,
    f"BM_AlloyTrackingEqJacN{size}",
    w_size=_workspace_size(spjf),
    callable=spjf,
  )


def _chain_alloy(size: int, out_dir: Path) -> dict:
  horizon = chain_of_masses.HORIZON
  started = time.perf_counter()
  fn = chain_of_masses.chain_eq_function(size, horizon)
  name = f"alloy_chain_eq_jac_M{size}"
  spjf = fn.factory(name, ["z", "p"], ["spjac:eq:z"])
  sparsity = spjf.output_sparsities[0]
  assert sparsity is not None
  build_ms = (time.perf_counter() - started) * 1000
  module, render_ms = _render_alloy(spjf, name, out_dir)
  return _module_info(
    name,
    "alloy",
    module,
    [("z", chain_of_masses.n_dec(size, horizon)), ("p", chain_of_masses.n_param(size))],
    sparsity,
    (fn.outputs[0].shape[0], fn.inputs[0].shape[0]),
    build_ms,
    render_ms,
    f"BM_AlloyChainEqJacM{size}",
    w_size=_workspace_size(spjf),
    callable=spjf,
  )


def _unbumpercars_alloy(size: int, out_dir: Path) -> dict:
  started = time.perf_counter()
  fn = unbumpercars.unbumpercars_ineq_function(size)
  name = f"alloy_unbumpercars_ineq_jac_N{size}"
  spjf = fn.factory(name, ["u", "p"], ["spjac:ineq:u"])
  sparsity = spjf.output_sparsities[0]
  assert sparsity is not None
  build_ms = (time.perf_counter() - started) * 1000
  module, render_ms = _render_alloy(spjf, name, out_dir)
  return _module_info(
    name,
    "alloy",
    module,
    [("u", unbumpercars.n_dec(size)), ("p", unbumpercars.n_param(size))],
    sparsity,
    (fn.outputs[0].shape[0], fn.inputs[0].shape[0]),
    build_ms,
    render_ms,
    f"BM_AlloyUnbumpercarsIneqJacN{size}",
    w_size=_workspace_size(spjf),
    callable=spjf,
  )


def _casadi(workload: str, size: int, backend: str, out_dir: Path) -> dict:
  import casadi as ca

  kind = backend.removeprefix("casadi_")
  sym_t = ca.SX if kind == "sx" else ca.MX
  stem = {"chain": "chain_eq", "tracking": "tracking_eq", "unbumpercars": "unbumpercars_ineq"}[workload]
  name = f"casadi_{kind}_{stem}_jac_{'M' if workload == 'chain' else 'N'}{size}"
  started = time.perf_counter()
  if workload == "chain":
    horizon = chain_of_masses.HORIZON
    fn = chain_of_masses.ca_chain_eq_jac(size, horizon, sym_t=sym_t, name=name, map_stages=True)
    inputs = [("z", chain_of_masses.n_dec(size, horizon)), ("p", chain_of_masses.n_param(size))]
    benchmark = f"BM_Casadi{kind.title()}ChainEqJacM{size}"
  elif workload == "tracking":
    fn = tracking_nmpc.ca_tracking_eq_jac(size, name=name, sym_t=sym_t)
    inputs = [("z", tracking_nmpc.NZ * (size + 1)), ("p", tracking_nmpc.n_param(size))]
    benchmark = f"BM_Casadi{kind.title()}TrackingEqJacN{size}"
  else:
    fn = unbumpercars.ca_unbumpercars_ineq_jac(size, sym_t=sym_t, name=name)
    inputs = [("u", unbumpercars.n_dec(size)), ("p", unbumpercars.n_param(size))]
    benchmark = f"BM_Casadi{kind.title()}UnbumpercarsIneqJacN{size}"
  build_ms = (time.perf_counter() - started) * 1000
  started = time.perf_counter()
  cwd = Path.cwd()
  os.chdir(out_dir)
  try:
    generator = ca.CodeGenerator(f"{name}.c", {"with_header": True, "casadi_int": "int"})
    generator.add(fn)
    generator.generate()
  finally:
    os.chdir(cwd)
  render_ms = (time.perf_counter() - started) * 1000
  source = (out_dir / f"{name}.c").read_text()
  rows, cols = fn.sparsity_out(0).get_triplet()
  return {
    "name": name,
    "backend": backend,
    "source": Path(f"{name}.c"),
    "header": f"{name}.h",
    "inputs": inputs,
    "nnz": fn.sparsity_out(0).nnz(),
    "n_rows": fn.size_out(0)[0],
    "n_cols": fn.size_out(0)[1],
    "rows": tuple(int(x) for x in rows),
    "cols": tuple(int(x) for x in cols),
    "w_size": fn.sz_w(),
    "iw_size": fn.sz_iw(),
    "arg_size": fn.sz_arg(),
    "res_size": fn.sz_res(),
    "source_bytes": len(source),
    "source_lines": source.count("\n") + 1,
    "build_ms": build_ms,
    "render_ms": render_ms,
    "benchmark": benchmark,
    "callable": fn,
  }


def build_kernel(workload: str, size: int, backend: str, out_dir: Path) -> dict:
  if backend == "alloy":
    return {"chain": _chain_alloy, "tracking": _tracking_alloy, "unbumpercars": _unbumpercars_alloy}[workload](size, out_dir)
  return _casadi(workload, size, backend, out_dir)


def _samples(workload: str, size: int, info: dict, out_dir: Path, weights: np.ndarray | None = None):
  if workload == "chain":
    horizon = chain_of_masses.HORIZON
    zv, pv = chain_of_masses.sample_inputs(size, horizon)
    expected = chain_of_masses.chain_eq_jac_dense_reference(size, horizon, zv, pv).reshape(-1)
    values = {"z": zv, "p": pv}
  elif workload == "tracking":
    rng = np.random.default_rng(7)
    zv = rng.normal(scale=0.4, size=tracking_nmpc.NZ * (size + 1))
    pv = np.concatenate([rng.normal(scale=0.4, size=tracking_nmpc.NX * (size + 1)), tracking_nmpc.TrackingParams().array()])
    ref = tracking_nmpc.tracking_eq_function(size).factory(f"tracking_dense_ref_N{size}", ["z", "p"], ["jac:eq:z"])
    expected = np.asarray(ref(zv, pv), dtype=np.float64).reshape(-1)
    values = {"z": zv, "p": pv}
  else:
    uv, pv = unbumpercars.sample_inputs(size, weights=weights)
    ref = unbumpercars.unbumpercars_ineq_function(size).factory(f"unbumpercars_dense_ref_N{size}", ["u", "p"], ["jac:ineq:u"])
    expected = np.asarray(ref(uv, pv), dtype=np.float64).reshape(-1)
    values = {"u": uv, "p": pv}
  args = [values[name] for name, _ in info["inputs"]]
  result = info["callable"](*args)
  compact = np.asarray(result.nonzeros() if info["backend"].startswith("casadi") else result, dtype=np.float64).reshape(-1)
  check_dense_reference(compact, info["rows"], info["cols"], expected, (info["n_rows"], info["n_cols"]), label=info["backend"])
  return write_samples(out_dir, {name: values[name] for name, _ in info["inputs"]}, expected)


def run_cell(
  workload: str,
  size: int,
  backend: str,
  out_dir: Path,
  *,
  codegen_timeout: float,
  compile_timeout: float,
  max_source_mb: float,
  benchmark_min_time: str,
  weights: np.ndarray | None = None,
) -> tuple[dict[str, object], dict | None]:
  if workload == "unbumpercars" and weights is None and not unbumpercars.MODEL_PATH.exists():
    return row(workload=workload, size=size, backend=backend, compile_status="skipped", runtime_status="skipped", note="missing checkpoint"), None
  if out_dir.exists():
    shutil.rmtree(out_dir)
  out_dir.mkdir(parents=True)
  try:
    with codegen_deadline(codegen_timeout):
      info = build_kernel(workload, size, backend, out_dir)
  except CodegenTimeout as e:
    return row(workload=workload, size=size, backend=backend, compile_status="codegen_timeout", runtime_status="skipped", note=str(e)), None
  except Exception as e:
    return row(
      workload=workload, size=size, backend=backend, compile_status="codegen_error", runtime_status="skipped", note=f"{type(e).__name__}: {e}"
    ), None
  codegen_ms = info["build_ms"] + info["render_ms"]
  base = dict(
    workload=workload,
    size=size,
    backend=backend,
    codegen_ms=f"{codegen_ms:.1f}",
    build_ms=f"{info['build_ms']:.1f}",
    render_ms=f"{info['render_ms']:.1f}",
    source_bytes=info["source_bytes"],
    source_lines=info["source_lines"],
    workspace=info["w_size"],
    nnz=info["nnz"],
  )
  source_mb = info["source_bytes"] / (1024 * 1024)
  if source_mb > max_source_mb:
    return row(**base, compile_status="skipped_size", runtime_status="skipped", note=f"source {source_mb:.1f} MB > {max_source_mb:.1f} MB"), info
  try:
    input_paths, expected_path = _samples(workload, size, info, out_dir, weights)
  except Exception as e:
    return row(**base, compile_status="not_run", runtime_status="correctness_fail", note=str(e)), info
  gbench.write_cpp(info, out_dir, input_paths, expected_path)
  compile_status, compile_ms, note = gbench.compile_kernel(info, out_dir, compile_timeout)
  if compile_status != "ok":
    return row(
      **base,
      compile_ms=f"{compile_ms:.1f}" if compile_ms is not None else f">{compile_timeout * 1000:.0f}",
      compile_status=compile_status,
      runtime_status="skipped",
      note=note,
    ), info
  runtime_status, runtime_ns, note = gbench.run_kernel(info, out_dir, benchmark_min_time)
  return row(
    **base,
    compile_ms=f"{compile_ms:.1f}",
    compile_status="ok",
    runtime_ns=f"{runtime_ns:.1f}" if runtime_ns is not None else "",
    runtime_status=runtime_status,
    note=note,
  ), info


def run_sweep(args, cli_args: list[str]) -> bool:
  args.out.parent.mkdir(parents=True, exist_ok=True)
  write(args.out, collect(ROOT, gbench.compiler(), cli_args))
  ok = True
  with args.out.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=FIELDS)
    writer.writeheader()
    for workload in args.workloads:
      gave_up: dict[str, tuple[int, str]] = {}
      sizes = args.sizes or DEFAULT_SIZES[workload]
      for size in sorted(sizes):
        for backend in args.backends:
          print(f"[{workload}] size={size} backend={backend} ... ", end="", flush=True)
          if backend in gave_up:
            failed_size, status = gave_up[backend]
            result = row(
              workload=workload,
              size=size,
              backend=backend,
              compile_status="skipped_after_failure",
              runtime_status="skipped",
              note=f"{backend} {status} at size={failed_size}; larger sizes won't fit",
            )
          else:
            result, _ = run_cell(
              workload,
              size,
              backend,
              RESULTS / "gen" / workload / f"{backend}_N{size}",
              codegen_timeout=args.codegen_timeout,
              compile_timeout=args.compile_timeout,
              max_source_mb=args.max_source_mb,
              benchmark_min_time=args.benchmark_min_time,
            )
          writer.writerow(result)
          fp.flush()
          status = result["runtime_status"] or result["compile_status"]
          print(f"{status}" + (f": {result['note']}" if result["note"] else ""))
          if result["compile_status"] in {"timeout", "skipped_size", "compile_error", "codegen_timeout", "codegen_error"}:
            gave_up[backend] = (size, str(result["compile_status"]))
          # timeout/skipped_size/codegen_timeout mark the expected end of a backend's scaling range; only genuine errors fail the sweep
          if result["runtime_status"] in {"runtime_error", "runtime_timeout", "correctness_fail", "parse_fail"} or result["compile_status"] in {
            "compile_error",
            "codegen_error",
          }:
            ok = False
  print(f"Results written to {args.out}")
  return ok
