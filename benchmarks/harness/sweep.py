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
from alloy.codegen.aot import render_c_module
from benchmarks.harness import ROOT, RESULTS, gbench
from benchmarks.harness.correctness import check_dense_reference, write_samples
from benchmarks.harness.provenance import collect, write
from benchmarks.problems import chain, npmpc, race_cars

DEFAULT_SIZES = {
  "chain": [3, 5, 9, 17, 33, 65],
  "race_cars": [1, 5, 10, 25, 40, 50, 100, 200, 500],
  "unbumpercars": [2, 4, 8],
  "npmpc": [6, 12, 25, 50, 100, 200],
  "npmpc_decoder": [16, 32, 64, 128, 256],
  "npmpc_hess": [6, 12, 25, 50, 100, 200],
  "npmpc_decoder_hess": [16, 32, 64, 128, 256],
}
BACKENDS = ("alloy", "casadi_sx", "casadi_mx")
CELL_AXES = {"chain": "M", "race_cars": "N", "unbumpercars": "C", "npmpc": "N", "npmpc_decoder": "W", "npmpc_hess": "N", "npmpc_decoder_hess": "W"}
NPMPC_WORKLOADS = ("npmpc", "npmpc_decoder", "npmpc_hess", "npmpc_decoder_hess")
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


def _race_cars_alloy(size: int, out_dir: Path) -> dict:
  started = time.perf_counter()
  fn = race_cars.race_car_eq_function_map(size)
  sj = al.sparse_jacobian(fn.outputs[0], fn.inputs[0])
  name = f"alloy_race_car_eq_jac_N{size}"
  spjf = al.Function(name, fn.inputs, [sj.values], fn.input_names, ["spjac_eq_z"], [sj.sparsity])
  build_ms = (time.perf_counter() - started) * 1000
  module, render_ms = _render_alloy(spjf, name, out_dir)
  return _module_info(
    name,
    "alloy",
    module,
    [("z", race_cars.NZ * (size + 1)), ("p", race_cars.n_param(size))],
    sj.sparsity,
    (fn.outputs[0].shape[0], fn.inputs[0].shape[0]),
    build_ms,
    render_ms,
    f"BM_AlloyRaceCarEqJacN{size}",
    w_size=module.workspace_size,
    callable=spjf,
  )


def _chain_alloy(size: int, out_dir: Path) -> dict:
  horizon = chain.HORIZON
  started = time.perf_counter()
  fn = chain.chain_eq_function(size, horizon)
  name = f"alloy_chain_eq_jac_M{size}"
  spjf = fn.factory(name, ["z", "p"], [al.spjac("eq", "z")])
  sparsity = spjf.output_sparsities[0]
  assert sparsity is not None
  build_ms = (time.perf_counter() - started) * 1000
  module, render_ms = _render_alloy(spjf, name, out_dir)
  return _module_info(
    name,
    "alloy",
    module,
    [("z", chain.n_dec(size, horizon)), ("p", chain.n_param(size))],
    sparsity,
    (fn.outputs[0].shape[0], fn.inputs[0].shape[0]),
    build_ms,
    render_ms,
    f"BM_AlloyChainEqJacM{size}",
    w_size=module.workspace_size,
    callable=spjf,
  )


def _unbumpercars_alloy(size: int, out_dir: Path) -> dict:
  from benchmarks.problems.unbumpercars.common import ClosedLoopConfig, FilterConfig, NCTRL, NSTATE, N_PHYSICS, N_PW_DT
  from benchmarks.problems.unbumpercars.filters import build_alloy_oracle

  started = time.perf_counter()
  cfg = ClosedLoopConfig(ncars=size)
  oracle = build_alloy_oracle(cfg, FilterConfig(model="dt"))
  name = f"alloy_unbumpercars_lag_hess_C{size}"
  sphf = oracle.factory(
    name,
    ["z", "lam:cost", "lam:g", "bar_x", "u_des", "pw", "physics", "dt"],
    [al.sphess("gamma", "z")],
    aux={"gamma": ["cost", "g"]},
  )
  sparsity = sphf.output_sparsities[0]
  assert sparsity is not None
  build_ms = (time.perf_counter() - started) * 1000
  module, render_ms = _render_alloy(sphf, name, out_dir)
  return _module_info(
    name,
    "alloy",
    module,
    [
      ("z", NCTRL * size + cfg.n_slack),
      ("lam_f", 1),
      ("lam_g", cfg.n_slack),
      ("bar_x", NSTATE * size),
      ("u_des", NCTRL * size),
      ("pw", N_PW_DT),
      ("physics", N_PHYSICS),
      ("dt", 1),
    ],
    sparsity,
    (NCTRL * size + cfg.n_slack, NCTRL * size + cfg.n_slack),
    build_ms,
    render_ms,
    f"BM_AlloyUnbumpercarsLagHessC{size}",
    w_size=module.workspace_size,
    callable=sphf,
  )


def _npmpc_cell(workload: str, size: int) -> tuple[int, npmpc.Decoder, np.ndarray, np.ndarray | None]:
  """Resolve a cell on any npmpc axis into (horizon, decoder, weights, terminal weight).

  The horizon axes carry the trained decoder the paper deployed, with the Riccati terminal weight
  its linearization gives. The width axes are untrained at every width, including 32, so the axis
  stays homogeneous -- kernel timing and code size depend on the graph's shape, not on the numbers
  in it (see `npmpc.random_decoder_weights`). An untrained model has no reason to be stabilizable
  upright, so those cells take the plain terminal weight instead of solving the Riccati equation;
  the terminal matrix is a constant either way, so this cannot move a timing.
  """
  if workload in ("npmpc", "npmpc_hess"):
    decoder = npmpc.Decoder()
    weights = npmpc.load_decoder_weights(decoder)
    return size, decoder, weights, npmpc.terminal_P(decoder, npmpc.pack_params(decoder, weights))
  decoder = npmpc.Decoder((size, size))
  return npmpc.HORIZON, decoder, npmpc.random_decoder_weights(decoder), None


def _npmpc_alloy(workload: str, size: int, out_dir: Path) -> dict:
  horizon, decoder, _, terminal = _npmpc_cell(workload, size)
  axis = CELL_AXES[workload]
  n_z = npmpc.n_dec(horizon)
  started = time.perf_counter()
  if workload.endswith("_hess"):
    fn = npmpc.npmpc_lag_function(horizon, decoder, terminal)
    name = f"alloy_npmpc_lag_hess_{axis}{size}"
    built = fn.factory(name, ["z", "lam:cost", "lam:eq", "p"], [al.sphess("gamma", "z")], aux={"gamma": ["cost", "eq"]})
    inputs = [("z", n_z), ("lam_f", 1), ("lam_g", npmpc.NX * horizon), ("p", decoder.n_pw)]
    shape, benchmark = (n_z, n_z), f"BM_AlloyNpmpcLagHess{axis}{size}"
  else:
    fn = npmpc.npmpc_eq_function(horizon, decoder)
    name = f"alloy_npmpc_eq_jac_{axis}{size}"
    built = fn.factory(name, ["z", "p"], [al.spjac("eq", "z")])
    inputs = [("z", n_z), ("p", decoder.n_pw)]
    shape, benchmark = (npmpc.NX * horizon, n_z), f"BM_AlloyNpmpcEqJac{axis}{size}"
  sparsity = built.output_sparsities[0]
  assert sparsity is not None
  build_ms = (time.perf_counter() - started) * 1000
  module, render_ms = _render_alloy(built, name, out_dir)
  return _module_info(name, "alloy", module, inputs, sparsity, shape, build_ms, render_ms, benchmark, w_size=module.workspace_size, callable=built)


def _casadi(workload: str, size: int, backend: str, out_dir: Path) -> dict:
  import casadi as ca

  kind = backend.removeprefix("casadi_")
  sym_t = ca.SX if kind == "sx" else ca.MX
  stem = {
    "chain": "chain_eq",
    "race_cars": "race_car_eq",
    "unbumpercars": "unbumpercars_lag",
    "npmpc": "npmpc_eq",
    "npmpc_decoder": "npmpc_eq",
    "npmpc_hess": "npmpc_lag",
    "npmpc_decoder_hess": "npmpc_lag",
  }[workload]
  axis = CELL_AXES[workload]
  kernel = "hess" if workload == "unbumpercars" or workload.endswith("_hess") else "jac"
  name = f"casadi_{kind}_{stem}_{kernel}_{axis}{size}"
  started = time.perf_counter()
  if workload == "chain":
    horizon = chain.HORIZON
    fn = chain.ca_chain_eq_jac(size, horizon, sym_t=sym_t, name=name, map_stages=True)
    inputs = [("z", chain.n_dec(size, horizon)), ("p", chain.n_param(size))]
    benchmark = f"BM_Casadi{kind.title()}ChainEqJacM{size}"
  elif workload == "race_cars":
    fn = race_cars.ca_race_car_eq_jac(size, name=name, sym_t=sym_t)
    inputs = [("z", race_cars.NZ * (size + 1)), ("p", race_cars.n_param(size))]
    benchmark = f"BM_Casadi{kind.title()}RaceCarEqJacN{size}"
  elif workload in NPMPC_WORKLOADS:
    horizon, decoder, _, terminal = _npmpc_cell(workload, size)
    if workload.endswith("_hess"):
      fn = npmpc.ca_npmpc_lag_hess(horizon, decoder, name=name, sym_t=sym_t, P=terminal)
      inputs = [("z", npmpc.n_dec(horizon)), ("lam_f", 1), ("lam_g", npmpc.NX * horizon), ("p", decoder.n_pw)]
      benchmark = f"BM_Casadi{kind.title()}NpmpcLagHess{axis}{size}"
    else:
      fn = npmpc.ca_npmpc_eq_jac(horizon, decoder, name=name, sym_t=sym_t)
      inputs = [("z", npmpc.n_dec(horizon)), ("p", decoder.n_pw)]
      benchmark = f"BM_Casadi{kind.title()}NpmpcEqJac{axis}{size}"
  else:
    from benchmarks.problems.unbumpercars.common import ClosedLoopConfig, FilterConfig, load_dt_mlp_weights
    from benchmarks.problems.unbumpercars.filters import build_casadi_hessian

    fn = build_casadi_hessian(ClosedLoopConfig(ncars=size), FilterConfig(model="dt"), load_dt_mlp_weights(), name, sym_t)
    inputs = [("z", fn.size1_in(0)), ("lam_f", fn.size1_in(1)), ("lam_g", fn.size1_in(2)), ("p", fn.size1_in(3))]
    benchmark = f"BM_Casadi{kind.title()}UnbumpercarsLagHessC{size}"
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
    if workload in NPMPC_WORKLOADS:
      return _npmpc_alloy(workload, size, out_dir)
    return {"chain": _chain_alloy, "race_cars": _race_cars_alloy, "unbumpercars": _unbumpercars_alloy}[workload](size, out_dir)
  return _casadi(workload, size, backend, out_dir)


def _harvested_inputs(workload: str, size: int) -> dict[str, np.ndarray] | None:
  canonical = {
    ("chain", 5): RESULTS / "closed-loop" / "chain" / "ipopt+alloy",
    ("race_cars", 40): RESULTS / "closed-loop" / "race_cars" / "ipopt+alloy",
    ("unbumpercars", 8): RESULTS / "closed-loop" / "unbumpercars" / "ipopt+alloy",
    ("npmpc", npmpc.HORIZON): RESULTS / "closed-loop" / "npmpc" / "ipopt+alloy",
    ("npmpc_hess", npmpc.HORIZON): RESULTS / "closed-loop" / "npmpc" / "ipopt+alloy",
  }.get((workload, size))
  if canonical is None:
    return None
  path = canonical / "representative_fe_inputs.npz"
  if not path.exists():
    return None
  with np.load(path) as data:
    return {name: np.asarray(data[name], dtype=np.float64) for name in data.files}


def _unbumpercars_hessian_inputs(size: int, harvested: dict[str, np.ndarray] | None) -> tuple[dict[str, np.ndarray], np.ndarray]:
  import casadi as ca

  from benchmarks.problems.unbumpercars.common import (
    NCTRL,
    NSTATE,
    N_PHYSICS,
    N_PW_DT,
    ClosedLoopConfig,
    FilterConfig,
    load_dt_mlp_weights,
    sample_initial_states,
  )
  from benchmarks.problems.unbumpercars.filters import build_casadi_hessian

  cfg, filt_cfg, weights = ClosedLoopConfig(ncars=size), FilterConfig(model="dt"), load_dt_mlp_weights()
  if harvested is None:
    bar_x = sample_initial_states(cfg).reshape(-1)
    u_des = np.tile([cfg.nominal_speed, 0.0], size)
    pieces = {
      "z": np.concatenate([u_des, np.zeros(cfg.n_slack)]),
      "lam_f": np.array(1.0),
      "lam_g": np.linspace(0.1, 1.0, cfg.n_slack),
      "bar_x": bar_x,
      "u_des": u_des,
      "pw": weights.packed,
      "physics": cfg.physics.array(),
      "dt": np.array([cfg.dt]),
    }
  else:
    pieces = harvested
  expected_shapes = {
    "z": (NCTRL * size + cfg.n_slack,),
    "lam_f": (),
    "lam_g": (cfg.n_slack,),
    "bar_x": (NSTATE * size,),
    "u_des": (NCTRL * size,),
    "pw": (N_PW_DT,),
    "physics": (N_PHYSICS,),
    "dt": (1,),
  }
  missing = sorted(set(expected_shapes) - set(pieces))
  extra = sorted(set(pieces) - set(expected_shapes))
  if missing or extra:
    raise ValueError(f"harvested unbumpercars fields do not match C={size}: missing={missing}, extra={extra}")
  for name, shape in expected_shapes.items():
    if pieces[name].shape != shape:
      raise ValueError(f"harvested unbumpercars field {name!r} does not match C={size}: {pieces[name].shape}, expected {shape}")
    if not np.all(np.isfinite(pieces[name])):
      raise ValueError(f"harvested unbumpercars field {name!r} contains non-finite values")
  p = np.concatenate([pieces[name] for name in ("bar_x", "u_des", "pw", "physics", "dt")])
  ref = build_casadi_hessian(cfg, filt_cfg, weights, f"unbumpercars_hess_dense_ref_C{size}", ca.MX)
  expected = np.asarray(ref(pieces["z"], pieces["lam_f"], pieces["lam_g"], p), dtype=np.float64).reshape(-1)
  return pieces, expected


def _samples(
  workload: str,
  size: int,
  info: dict,
  out_dir: Path,
  harvested: dict[str, np.ndarray] | None = None,
  *,
  load_harvested: bool = True,
):
  harvested = _harvested_inputs(workload, size) if harvested is None and load_harvested else harvested
  if workload == "chain":
    horizon = chain.HORIZON
    if harvested is None:
      zv, pv = chain.sample_inputs(size, horizon)
    else:
      zv, pv = harvested["z"], harvested["p"]
    if zv.shape != (chain.n_dec(size, horizon),) or pv.shape != (chain.n_param(size),):
      raise ValueError(f"harvested chain input shapes do not match M={size}, N={horizon}: {zv.shape}, {pv.shape}")
    expected = chain.chain_eq_jac_dense_reference(size, horizon, zv, pv).reshape(-1)
    values = {"z": zv, "p": pv}
  elif workload == "race_cars":
    if harvested is None:
      rng = np.random.default_rng(7)
      zv = rng.normal(scale=0.4, size=race_cars.NZ * (size + 1))
      pv = np.concatenate([rng.normal(scale=0.4, size=race_cars.NX * (size + 1)), race_cars.RaceCarParams().array()])
    else:
      zv, pv = harvested["z"], harvested["p"]
    if zv.shape != (race_cars.NZ * (size + 1),) or pv.shape != (race_cars.n_param(size),):
      raise ValueError(f"harvested race_cars input shapes do not match N={size}: {zv.shape}, {pv.shape}")
    ref = race_cars.race_car_eq_function(size).factory(f"race_car_dense_ref_N{size}", ["z", "p"], [al.jac("eq", "z")])
    expected = np.asarray(ref(zv, pv), dtype=np.float64).reshape(-1)
    values = {"z": zv, "p": pv}
  elif workload in NPMPC_WORKLOADS:
    import casadi as ca

    horizon, decoder, weights, terminal = _npmpc_cell(workload, size)
    if harvested is None:
      zv, pv = npmpc.sample_inputs(horizon, decoder, weights)
    else:
      zv, pv = harvested["z"], harvested["p"]
    if zv.shape != (npmpc.n_dec(horizon),) or pv.shape != (decoder.n_pw,):
      raise ValueError(f"harvested npmpc input shapes do not match {CELL_AXES[workload]}={size}: {zv.shape}, {pv.shape}")
    if workload.endswith("_hess"):
      # Multipliers of mixed sign, so the Hessian is not dominated by the objective block alone.
      lam_g = np.linspace(-0.75, 0.75, npmpc.NX * horizon)
      ref = npmpc.ca_npmpc_lag_hess(horizon, decoder, f"npmpc_lag_hess_dense_ref_{CELL_AXES[workload]}{size}", ca.MX, terminal)
      expected = np.asarray(ca.densify(ref(zv, 1.0, lam_g, pv)), dtype=np.float64).reshape(-1)
      values = {"z": zv, "lam_f": np.array(1.0), "lam_g": lam_g, "p": pv}
    else:
      expected = npmpc.npmpc_eq_jac_dense_reference(horizon, zv, pv, decoder)
      values = {"z": zv, "p": pv}
  else:
    pieces, expected = _unbumpercars_hessian_inputs(size, harvested)
    values = (
      pieces
      if info["backend"] == "alloy"
      else {
        "z": pieces["z"],
        "lam_f": pieces["lam_f"],
        "lam_g": pieces["lam_g"],
        "p": np.concatenate([pieces[name] for name in ("bar_x", "u_des", "pw", "physics", "dt")]),
      }
    )
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
  load_harvested: bool = True,
) -> tuple[dict[str, object], dict | None]:
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
    input_paths, expected_path = _samples(workload, size, info, out_dir, load_harvested=load_harvested)
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
              args.out.parent / workload / f"{backend}_{CELL_AXES[workload]}{size}",
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
