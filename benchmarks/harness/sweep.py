from __future__ import annotations

import contextlib
import csv
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import random
import statistics
import os
from dataclasses import replace
from pathlib import Path
import re
import shutil
import signal
import time

import numpy as np

import alloy as al
from alloy.codegen.abi import c_ident
from alloy.codegen.aot import render_c_module
from alloy.ir.expr import ExprOp, topo
from alloy.ir.program import ProgramNode, ProgramOp
from benchmarks.harness import ROOT, RESULTS, gbench
from benchmarks.harness.correctness import check_dense_reference, write_samples
from benchmarks.harness.provenance import collect, write, require_headline_settings
from benchmarks.problems import chain, npmpc, race_cars

DEFAULT_SIZES = {
  "chain": [3, 5, 9, 17, 33, 65],
  "chain_jac": [3, 5, 9, 17, 33, 65],
  "race_cars": [1, 5, 10, 25, 40, 50, 100, 200, 500],
  "race_cars_jac": [1, 5, 10, 25, 40, 50, 100, 200, 500],
  "unbumpercars": [2, 4, 8],
  "npmpc": [6, 12, 25, 50, 100, 200],
  "npmpc_jac": [6, 12, 25, 50, 100, 200],
  "npmpc_decoder": [16, 32, 64, 128, 256],
  "npmpc_decoder_jac": [16, 32, 64, 128, 256],
}
STAGE_BACKENDS = ("alloy", "casadi_sx", "casadi_mx", "casadi_call_mx", "casadi_map_sx")
# The gemm encodings batch every repetition's dense network layer into one matrix-matrix product,
# so they exist only for the two problems with a network in the stage.
GEMM_BACKENDS = ("casadi_mx_gemm", "casadi_mx_gemm_classic", "casadi_mx_gemm_blasfeo")
BACKENDS = (*STAGE_BACKENDS, *GEMM_BACKENDS)
NPMPC_WORKLOADS = ("npmpc", "npmpc_jac", "npmpc_decoder", "npmpc_decoder_jac")
DEFAULT_BACKENDS = {
  workload: ("alloy", "casadi_sx", "casadi_mx", *GEMM_BACKENDS)
  if workload == "unbumpercars"
  else (BACKENDS if workload in NPMPC_WORKLOADS else STAGE_BACKENDS)
  for workload in DEFAULT_SIZES
}
CELL_AXES = {
  "chain": "M",
  "chain_jac": "M",
  "race_cars": "N",
  "race_cars_jac": "N",
  "unbumpercars": "C",
  "npmpc": "N",
  "npmpc_jac": "N",
  "npmpc_decoder": "W",
  "npmpc_decoder_jac": "W",
}


def _kernel_kind(workload: str) -> str:
  return "jac" if workload.endswith("_jac") else "hess"


def _alloy_inputs(kernel: al.Function) -> list[tuple[str, int]]:
  return [(c_ident(name), expr.size) for name, expr in zip(kernel.input_names, kernel.inputs, strict=True)]


FIELDS = [
  "process_id",
  "repetition",
  "backend_order",
  "workload",
  "size",
  "backend",
  "layout",
  "codegen_ms",
  "source_bytes",
  "artifact_bytes",
  "executable_bytes",
  "static_metadata_bytes",
  "source_lines",
  "workspace",
  "integer_workspace",
  "argument_pointers",
  "result_pointers",
  "dispatch_trip_count",
  "dispatch_workspace",
  "dispatch_arithmetic",
  "coloring_width",
  "nnz",
  "compile_ms",
  "kernel_compile_ms",
  "wrapper_compile_ms",
  "link_ms",
  "compile_status",
  "runtime_ns",
  "runtime_status",
  "note",
  "build_ms",
  "render_ms",
]

_STATIC_CONST = re.compile(rb"(?ms)^[ \t]*static const\b.*?;\n")
_ARITHMETIC_OPS = frozenset(
  {
    ProgramOp.ADD,
    ProgramOp.SUB,
    ProgramOp.MUL,
    ProgramOp.DIV,
    ProgramOp.MOD,
    ProgramOp.NEG,
    ProgramOp.SIN,
    ProgramOp.COS,
    ProgramOp.TAN,
    ProgramOp.ASIN,
    ProgramOp.ACOS,
    ProgramOp.ATAN,
    ProgramOp.SINH,
    ProgramOp.COSH,
    ProgramOp.TANH,
    ProgramOp.ERF,
    ProgramOp.EXP,
    ProgramOp.LOG,
    ProgramOp.SQRT,
    ProgramOp.ABS,
    ProgramOp.FLOOR,
    ProgramOp.CEIL,
    ProgramOp.POW,
    ProgramOp.ATAN2,
    ProgramOp.MINIMUM,
    ProgramOp.MAXIMUM,
  }
)


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


def _artifact_sizes(source: str, header: str) -> tuple[int, int, int]:
  """Return total artifact, executable source, and static metadata bytes.

  The executable part is the translation unit with ``static const`` declarations removed. The
  metadata part is those declarations plus the public header, which contains Alloy's sparse tables.
  """
  source_bytes, header_bytes = source.encode(), header.encode()
  static_source = sum(len(match.group()) for match in _STATIC_CONST.finditer(source_bytes))
  return len(source_bytes) + len(header_bytes), len(source_bytes) - static_source, static_source + len(header_bytes)


def _static_trip_count(rng: ProgramNode) -> int | None:
  start, stop, step = rng.args
  if any(node.op != ProgramOp.CONST_INT for node in (start, stop, step)) or int(step.attrs["value"]) <= 0:
    return None
  span = int(stop.attrs["value"]) - int(start.attrs["value"])
  return max(0, -(-span // int(step.attrs["value"])))


def _dispatch_metrics(fun: al.Function, prog: ProgramNode) -> tuple[int | str, int | str, int | str]:
  """Return the retained VMAP trip count, callee workspace, and arithmetic per iteration.

  A function mapped over several axes (race-car stages ``N`` and ``N+1``, unbumpercars cars and
  pairs) groups its dispatch loops by trip count and reports the family whose trip count times
  per-iteration arithmetic is largest; the arithmetic column is that family's per-iteration figure
  and the workspace is the maximum over every dispatch.
  """
  vmaps = [node for node in topo(fun.outputs) if node.op == ExprOp.VMAP]
  trip_counts = {int(node.attrs["length"]) for node in vmaps}
  if not vmaps:
    return "", "", ""
  mapped_callees = {str(node.attrs["callee"].name) for node in vmaps}
  proc_count = int(prog.attrs.get("proc_count", 0))
  procs = {proc.attrs["name"]: proc for proc in prog.args[:proc_count]}
  arithmetic_cache: dict[str, int] = {}
  workspace_cache: dict[str, int] = {}

  def arithmetic(node: ProgramNode) -> int:
    if node.op == ProgramOp.FOR:
      count = _static_trip_count(node.args[0])
      if count is None:
        raise LookupError("dynamic loop has no static arithmetic count")
      return count * sum(arithmetic(child) for child in node.args[1:])
    if node.op == ProgramOp.CALL:
      callee = str(node.attrs["callee"])
      if callee not in procs:
        raise LookupError(f"callee {callee!r} has no Program IR procedure")
      return proc_arithmetic(callee)
    return (1 if node.op in _ARITHMETIC_OPS and node.dtype.is_floating else 0) + sum(arithmetic(child) for child in node.args)

  def proc_arithmetic(name: str) -> int:
    if name not in arithmetic_cache:
      proc = procs[name]
      param_count = int(proc.attrs["param_count"])
      arithmetic_cache[name] = 0
      arithmetic_cache[name] = sum(arithmetic(stmt) for stmt in proc.args[param_count:])
    return arithmetic_cache[name]

  def calls(node: ProgramNode) -> set[str]:
    found = {str(node.attrs["callee"])} if node.op == ProgramOp.CALL else set()
    for child in node.args:
      found.update(calls(child))
    return found

  def proc_workspace(name: str) -> int:
    if name not in workspace_cache:
      proc = procs[name]
      param_count = int(proc.attrs["param_count"])
      body = proc.args[param_count:]
      own = 0
      callees: set[str] = set()
      for stmt in body:
        callees.update(calls(stmt))
        if stmt.op != ProgramOp.BUFFER or stmt.attrs.get("address_space") != "private" or "alias_of" in stmt.attrs:
          continue
        if stmt.dtype.is_floating:
          size = 1
          for dim in stmt.attrs["shape"]:
            size *= int(dim)
          own += size or 1
      if callees - procs.keys():
        raise LookupError("callee has no Program IR procedure")
      workspace_cache[name] = own
      workspace_cache[name] = own + max((proc_workspace(callee) for callee in callees), default=0)
    return workspace_cache[name]

  dispatches: dict[int, list[str]] = {}
  root = procs[fun.name]
  for stmt in root.args[int(root.attrs["param_count"]) :]:
    call, count = None, 1
    if stmt.op == ProgramOp.FOR and stmt.args[-1].op == ProgramOp.CALL and all(node.op == ProgramOp.ASSIGN for node in stmt.args[1:-1]):
      call, count = stmt.args[-1], _static_trip_count(stmt.args[0])
    elif stmt.op == ProgramOp.CALL:
      call = stmt
    if call is not None and count in trip_counts:
      callee = str(call.attrs["callee"])
      origin = procs[callee].attrs.get("hoisted_from", callee) if callee in procs else callee
      if origin in mapped_callees:
        dispatches.setdefault(count, []).append(callee)
  if not dispatches:
    return "", "", ""
  try:
    work = {count: sum(proc_arithmetic(callee) for callee in callees) for count, callees in dispatches.items()}
    workspace = max(proc_workspace(callee) for callees in dispatches.values() for callee in callees)
  except LookupError:
    return "", "", ""
  trip_count = max(work, key=lambda count: count * work[count])
  return trip_count, workspace, work[trip_count]


def _module_info(
  name: str,
  backend: str,
  module,
  inputs,
  sparsity,
  shape,
  build_ms: float,
  render_ms: float,
  benchmark: str,
  coloring_width: int | None = None,
  layout: str = "full",
  **extra,
) -> dict:
  source_path = Path(module.source_name)
  artifact_bytes, executable_bytes, static_metadata_bytes = _artifact_sizes(module.source, module.header)
  dispatch_trip_count, dispatch_workspace, dispatch_arithmetic = _dispatch_metrics(extra["callable"], module.program)
  return {
    "name": name,
    "backend": backend,
    "layout": layout,
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
    "source_bytes": len(module.source.encode()),
    "artifact_bytes": artifact_bytes,
    "executable_bytes": executable_bytes,
    "static_metadata_bytes": static_metadata_bytes,
    "source_lines": module.source.count("\n") + 1,
    "arg_size": len(module.fun.inputs),
    "res_size": len(module.fun.outputs),
    "dispatch_trip_count": dispatch_trip_count,
    "dispatch_workspace": dispatch_workspace,
    "dispatch_arithmetic": dispatch_arithmetic,
    "coloring_width": coloring_width,
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


def _descriptor_kernel(solver: al.Function, kind: str):
  descriptor = solver.descriptor
  function = getattr(descriptor, kind)
  sparsity = getattr(descriptor, f"{kind}_sparsity")
  if not isinstance(function, al.Function) or sparsity is None:
    raise TypeError(f"{descriptor.name} has no Alloy {kind} kernel")
  assert function.output_sparsities[0] == sparsity
  return function, sparsity, function.output_coloring_widths[0]


_CASADI_ORACLE_NAMES = {"hess": "nlp_hess_l", "jac": "nlp_jac_g"}
_CASADI_OUTPUT_INDEX = {"hess": 0, "jac": 1}
# The generated IPOPT callback requests only jac_g_x. CasADi's nlp_jac_g function exposes g as
# output 0 for the separate eval_g callback, then passes NULL for that slot in eval_jac_g.
_CASADI_REQUESTED_OUTPUTS = {"hess": (0,), "jac": (1,)}


def _casadi_descriptor_kernel(ca, name: str, z, p, cost, constraints, kind: str, *, expand: bool, cse: bool = False):
  """Build the CasADi IPOPT oracle and return the function used by its IPOPT callback.

  ``nlpsol`` owns the complete oracle set, so construction cost covers the same base, gradient,
  Jacobian, and Hessian construction that the optimizer sees. CasADi does not accept ``cse`` as an
  ``nlpsol`` option; applying ``ca.cse`` to the NLP expressions first is the reachable equivalent.
  ``nlp_jac_g`` deliberately remains a two-output function: IPOPT requests output 1 and leaves
  output 0 null, which the benchmark records explicitly.
  """
  if kind not in _CASADI_ORACLE_NAMES:
    raise ValueError(f"unsupported CasADi oracle kind {kind!r}")
  if cse:
    cost, constraints = ca.cse([cost, constraints])
  options = {"expand": expand, "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False}
  solver = ca.nlpsol(name, "ipopt", {"x": z, "p": p, "f": cost, "g": constraints}, options)
  return solver.get_function(_CASADI_ORACLE_NAMES[kind])


def _casadi_output_metadata(fn, kind: str) -> dict[str, object]:
  """Return output-indexed sparsity and pointer metadata for a CasADi oracle function."""
  output_index = _CASADI_OUTPUT_INDEX[kind]
  requested_outputs = _CASADI_REQUESTED_OUTPUTS[kind]
  if output_index >= fn.n_out() or any(index >= fn.n_out() for index in requested_outputs):
    raise RuntimeError(f"CasADi {fn.name()!r} has {fn.n_out()} outputs, expected {requested_outputs}")
  output_sparsities = tuple(fn.sparsity_out(index) for index in range(fn.n_out()))
  selected = output_sparsities[output_index]
  rows, cols = selected.get_triplet()
  layout = "upper" if kind == "hess" else "full"
  if layout == "upper" and any(row > col for row, col in zip(rows, cols, strict=True)):
    raise RuntimeError(f"CasADi {fn.name()!r} Hessian output is not upper triangular")
  output_nnz = tuple(int(sp.nnz()) for sp in output_sparsities)
  return {
    "output_index": output_index,
    "requested_output_indices": requested_outputs,
    "output_names": tuple(fn.name_out(index) for index in range(fn.n_out())),
    "layout": layout,
    "output_nnz": output_nnz,
    "rows": tuple(int(value) for value in rows),
    "cols": tuple(int(value) for value in cols),
    "nnz": int(selected.nnz()),
    "n_rows": int(fn.size_out(output_index)[0]),
    "n_cols": int(fn.size_out(output_index)[1]),
    "arg_size": int(fn.sz_arg()),
    "res_size": int(fn.sz_res()),
  }


def _mixed_sign_multipliers(count: int) -> np.ndarray:
  """Multipliers of mixed sign, so the Hessian is not dominated by the objective block alone."""
  return np.linspace(-0.75, 0.75, count)


def _repeat_element(ca, element, count: int, args: tuple, *, mapped: bool):
  if mapped:
    return element.map(count, "serial")(*args)
  return ca.horzcat(*(element(*(arg[:, i] for arg in args)) for i in range(count)))


def _race_cars_repeated_pieces(config, stage_sym, *, mapped: bool):
  from benchmarks.problems.race_cars.casadi_nlp import build_casadi_race_car_nlp

  import casadi as ca

  pieces = build_casadi_race_car_nlp(config, ca.MX, dynamics=False)
  one = build_casadi_race_car_nlp(replace(config, horizon=1), stage_sym)
  stage = ca.Function("race_car_stage", [one["z"], one["p"]], [one["h_eq"][race_cars.NX :]])
  z, p = pieces["z"], pieces["p"]
  stages = ca.reshape(z, race_cars.NZ, config.horizon + 1)
  refs = ca.reshape(p[: race_cars.NX * (config.horizon + 1)], race_cars.NX, config.horizon + 1)
  params = p[race_cars.NX * (config.horizon + 1) :]
  stage_z = ca.vertcat(stages[:, : config.horizon], stages[:, 1:])
  stage_p = ca.vertcat(refs[:, : config.horizon], refs[:, 1:], ca.repmat(params, 1, config.horizon))
  rows = _repeat_element(ca, stage, config.horizon, (stage_z, stage_p), mapped=mapped)
  h_eq = ca.vertcat(z[: race_cars.NX] - p[: race_cars.NX], ca.reshape(rows, race_cars.NX * config.horizon, 1))
  return {**pieces, "h_eq": h_eq}


def _npmpc_repeated_pieces(horizon: int, decoder, stage_sym, *, mapped: bool):
  import casadi as ca

  pieces = npmpc._ca_npmpc_joint_parameter_pieces(horizon, decoder, ca.MX, dynamics=False)
  one = npmpc.ca_npmpc_pieces(1, decoder, stage_sym, cost=False)
  stage = ca.Function("npmpc_stage", [one["z"], one["pw"], one["dt"]], [one["h_eq"]])
  z, p = pieces["z"], pieces["p"]
  offset = npmpc.NX * (horizon + 1)
  states = ca.reshape(z[:offset], npmpc.NX, horizon + 1)
  controls = ca.reshape(z[offset : offset + npmpc.NU * horizon], npmpc.NU, horizon)
  stage_z = ca.vertcat(states[:, :horizon], states[:, 1:], controls, ca.repmat(z[-1], 1, horizon))
  pw = p[npmpc.NX : npmpc.NX + decoder.n_pw]
  dt = p[npmpc.NX + decoder.n_pw]
  rows = _repeat_element(ca, stage, horizon, (stage_z, ca.repmat(pw, 1, horizon), ca.repmat(dt, 1, horizon)), mapped=mapped)
  return {**pieces, "h_eq": ca.reshape(rows, npmpc.NX * horizon, 1)}


def _race_cars_alloy(workload: str, size: int, out_dir: Path) -> dict:
  from benchmarks.problems.race_cars.closed_loop import EpisodeConfig, _race_car_nlp

  kind = _kernel_kind(workload)
  started = time.perf_counter()
  kernel, sparsity, coloring_width = _descriptor_kernel(_race_car_nlp(EpisodeConfig(horizon=size)), kind)
  name = kernel.name
  build_ms = (time.perf_counter() - started) * 1000
  module, render_ms = _render_alloy(kernel, name, out_dir)
  inputs = _alloy_inputs(kernel)
  benchmark = f"BM_AlloyRaceCar{'ConstraintJac' if kind == 'jac' else 'LagHess'}N{size}"
  return _module_info(
    name,
    "alloy",
    module,
    inputs,
    sparsity,
    sparsity.shape,
    build_ms,
    render_ms,
    benchmark,
    coloring_width=coloring_width,
    layout="lower" if _kernel_kind(workload) == "hess" else "full",
    w_size=module.workspace_size,
    callable=kernel,
  )


def _chain_alloy(workload: str, size: int, out_dir: Path) -> dict:
  horizon = chain.HORIZON
  kind = _kernel_kind(workload)
  benchmark = f"BM_AlloyChain{'EqJac' if kind == 'jac' else 'LagHess'}M{size}"
  started = time.perf_counter()
  kernel, sparsity, coloring_width = _descriptor_kernel(chain.chain_nlp(size, horizon), kind)
  inputs = _alloy_inputs(kernel)
  name = kernel.name
  build_ms = (time.perf_counter() - started) * 1000
  module, render_ms = _render_alloy(kernel, name, out_dir)
  return _module_info(
    name,
    "alloy",
    module,
    inputs,
    sparsity,
    sparsity.shape,
    build_ms,
    render_ms,
    benchmark,
    coloring_width=coloring_width,
    layout="lower" if _kernel_kind(workload) == "hess" else "full",
    w_size=module.workspace_size,
    callable=kernel,
  )


def _unbumpercars_alloy(size: int, out_dir: Path) -> dict:
  from benchmarks.problems.unbumpercars.common import ClosedLoopConfig, FilterConfig
  from benchmarks.problems.unbumpercars.filters import build_alloy_nlp

  started = time.perf_counter()
  cfg = ClosedLoopConfig(ncars=size)
  solver = build_alloy_nlp(cfg, FilterConfig(model="dt"))
  kernel, sparsity, coloring_width = _descriptor_kernel(solver, "hess")
  name = kernel.name
  build_ms = (time.perf_counter() - started) * 1000
  module, render_ms = _render_alloy(kernel, name, out_dir)
  return _module_info(
    name,
    "alloy",
    module,
    _alloy_inputs(kernel),
    sparsity,
    sparsity.shape,
    build_ms,
    render_ms,
    f"BM_AlloyUnbumpercarsLagHessC{size}",
    coloring_width=coloring_width,
    layout="lower",
    w_size=module.workspace_size,
    callable=kernel,
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
  if workload in ("npmpc", "npmpc_jac"):
    decoder = npmpc.Decoder()
    weights = npmpc.load_decoder_weights(decoder)
    return size, decoder, weights, npmpc.terminal_P(decoder, npmpc.pack_params(decoder, weights))
  decoder = npmpc.Decoder((size, size))
  return npmpc.HORIZON, decoder, npmpc.random_decoder_weights(decoder), None


def _npmpc_alloy(workload: str, size: int, out_dir: Path) -> dict:
  horizon, decoder, _, _ = _npmpc_cell(workload, size)
  axis = CELL_AXES[workload]
  started = time.perf_counter()
  kind = _kernel_kind(workload)
  built, sparsity, coloring_width = _descriptor_kernel(npmpc.npmpc_nlp(horizon, decoder), kind)
  name = built.name
  if kind == "hess":
    benchmark = f"BM_AlloyNpmpcLagHess{axis}{size}"
  else:
    benchmark = f"BM_AlloyNpmpcConstraintJac{axis}{size}"
  inputs = _alloy_inputs(built)
  build_ms = (time.perf_counter() - started) * 1000
  module, render_ms = _render_alloy(built, name, out_dir)
  return _module_info(
    name,
    "alloy",
    module,
    inputs,
    sparsity,
    sparsity.shape,
    build_ms,
    render_ms,
    benchmark,
    coloring_width=coloring_width,
    layout="lower" if _kernel_kind(workload) == "hess" else "full",
    w_size=module.workspace_size,
    callable=built,
  )


def _casadi(workload: str, size: int, backend: str, out_dir: Path, *, transform: bool = True) -> dict:
  import casadi as ca

  kind = backend.removeprefix("casadi_")
  if backend not in BACKENDS or kind == "alloy":
    raise ValueError(f"unsupported CasADi backend {backend!r}")
  repeated = kind in {"call_mx", "map_sx"}
  mapped = kind == "map_sx"
  gemm = backend in GEMM_BACKENDS
  mtimes = (kind.removeprefix("mx_gemm").removeprefix("_") or None) if gemm else None
  sym_t = ca.SX if kind in {"sx", "map_sx"} else ca.MX
  label = {"sx": "Sx", "mx": "Mx", "call_mx": "CallMx", "map_sx": "MapSx"}.get(kind, "MxGemm" + (mtimes or "").capitalize())
  extra_libs: list[str] = []
  if mtimes in ("classic", "blasfeo"):
    casadi_dir = Path(ca.__file__).parent
    extra_libs = [str(casadi_dir / ("libcasadi-tp-openblas.so.0" if mtimes == "classic" else "libblasfeo.so.0")), f"-Wl,-rpath,{casadi_dir}"]
  stem = {
    "chain": "chain_lag",
    "chain_jac": "chain_eq",
    "race_cars": "race_car_lag",
    "race_cars_jac": "race_car_constraints",
    "unbumpercars": "unbumpercars_lag",
    "npmpc": "npmpc_lag",
    "npmpc_jac": "npmpc_constraints",
    "npmpc_decoder": "npmpc_lag",
    "npmpc_decoder_jac": "npmpc_constraints",
  }[workload]
  axis = CELL_AXES[workload]
  kernel = _kernel_kind(workload)
  name = f"casadi_{kind}_{stem}_{kernel}_{axis}{size}"
  # SX is already the scalar encoding. Expanding an MX outer graph would silently turn the
  # call/map encodings into a different benchmark, so only the literal SX cell asks for expand.
  expand = kind == "sx"
  cse = workload in ("chain", "chain_jac")
  started = time.perf_counter()
  if gemm and workload not in ("unbumpercars", *NPMPC_WORKLOADS):
    raise ValueError(f"{backend} is not defined for {workload}")
  if workload in ("chain", "chain_jac"):
    horizon = chain.HORIZON
    z, p, cost, constraints = chain._ca_nlp_pieces(size, horizon, sym_t, map_stages=mapped, call_stages=repeated)
    fn = _casadi_descriptor_kernel(ca, name, z, p, cost, constraints, kernel, expand=expand, cse=cse)
    benchmark = f"BM_Casadi{label}Chain{'EqJac' if kernel == 'jac' else 'LagHess'}M{size}"
  elif workload in ("race_cars", "race_cars_jac"):
    from benchmarks.problems.race_cars.casadi_nlp import build_casadi_race_car_nlp
    from benchmarks.problems.race_cars.closed_loop import EpisodeConfig

    config = EpisodeConfig(horizon=size)
    pieces = _race_cars_repeated_pieces(config, sym_t, mapped=mapped) if repeated else build_casadi_race_car_nlp(config, sym_t)
    constraints = ca.vertcat(pieces["h_eq"], pieces["g_ineq"])
    fn = _casadi_descriptor_kernel(ca, name, pieces["z"], pieces["p"], pieces["f"], constraints, kernel, expand=expand)
    benchmark = f"BM_Casadi{label}RaceCar{'ConstraintJac' if kernel == 'jac' else 'LagHess'}N{size}"
  elif workload in NPMPC_WORKLOADS:
    horizon, decoder, _, _ = _npmpc_cell(workload, size)
    pieces = (
      _npmpc_repeated_pieces(horizon, decoder, sym_t, mapped=mapped)
      if repeated
      else npmpc._ca_npmpc_joint_parameter_pieces(horizon, decoder, sym_t, batched=gemm, mtimes=mtimes)
    )
    constraints = ca.vertcat(pieces["h_eq"], pieces["g_ineq"])
    fn = _casadi_descriptor_kernel(ca, name, pieces["z"], pieces["p"], pieces["f"], constraints, kernel, expand=expand)
    benchmark = f"BM_Casadi{label}NpmpcLagHess{axis}{size}" if kernel == "hess" else f"BM_Casadi{label}NpmpcConstraintJac{axis}{size}"
  else:
    from benchmarks.problems.unbumpercars.common import ClosedLoopConfig, FilterConfig, load_dt_mlp_weights
    from benchmarks.problems.unbumpercars.filters import CasadiDTCBFSafetyFilter

    controller = CasadiDTCBFSafetyFilter(
      ClosedLoopConfig(ncars=size),
      FilterConfig(model="dt"),
      load_dt_mlp_weights(),
      _build_solver=False,
      _sym_t=sym_t,
      _mlp="batched" if gemm else "percar",
      _mtimes=mtimes,
    )
    fn = _casadi_descriptor_kernel(
      ca,
      name,
      controller.z_expr,
      controller.p_expr,
      controller.cost_expr,
      controller.g_expr,
      kernel,
      expand=expand,
    )
    benchmark = f"BM_Casadi{label}UnbumpercarsLagHessC{size}"
  if transform:
    fn = fn.transform({})
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
  header = (out_dir / f"{name}.h").read_text()
  artifact_bytes, executable_bytes, static_metadata_bytes = _artifact_sizes(source, header)
  details = _casadi_output_metadata(fn, kernel)
  return {
    "name": name,
    "symbol": fn.name(),
    "backend": backend,
    "source": Path(f"{name}.c"),
    "header": f"{name}.h",
    "inputs": [(fn.name_in(index), int(fn.numel_in(index))) for index in range(fn.n_in())],
    "w_size": fn.sz_w(),
    "iw_size": fn.sz_iw(),
    "source_bytes": len(source.encode()),
    "artifact_bytes": artifact_bytes,
    "executable_bytes": executable_bytes,
    "static_metadata_bytes": static_metadata_bytes,
    "source_lines": source.count("\n") + 1,
    "dispatch_trip_count": "",
    "dispatch_workspace": "",
    "dispatch_arithmetic": "",
    "coloring_width": None,
    "expand": expand,
    "cse": cse,
    "build_ms": build_ms,
    "render_ms": render_ms,
    "benchmark": benchmark,
    "callable": fn,
    "extra_libs": extra_libs,
    **details,
  }


def build_kernel(workload: str, size: int, backend: str, out_dir: Path, *, casadi_transform: bool = True) -> dict:
  if backend == "alloy":
    if workload in NPMPC_WORKLOADS:
      return _npmpc_alloy(workload, size, out_dir)
    if workload in ("chain", "chain_jac"):
      return _chain_alloy(workload, size, out_dir)
    if workload in ("race_cars", "race_cars_jac"):
      return _race_cars_alloy(workload, size, out_dir)
    return _unbumpercars_alloy(size, out_dir)
  return _casadi(workload, size, backend, out_dir, transform=casadi_transform)


def _harvested_inputs(workload: str, size: int) -> dict[str, np.ndarray] | None:
  canonical = {
    ("chain", 5): RESULTS / "closed-loop" / "chain" / "ipopt+alloy",
    ("chain_jac", 5): RESULTS / "closed-loop" / "chain" / "ipopt+alloy",
    ("race_cars", 40): RESULTS / "closed-loop" / "race_cars" / "ipopt+alloy",
    ("race_cars_jac", 40): RESULTS / "closed-loop" / "race_cars" / "ipopt+alloy",
    ("unbumpercars", 8): RESULTS / "closed-loop" / "unbumpercars" / "ipopt+alloy",
    ("npmpc", npmpc.HORIZON): RESULTS / "closed-loop" / "npmpc" / "ipopt+alloy",
    ("npmpc_jac", npmpc.HORIZON): RESULTS / "closed-loop" / "npmpc" / "ipopt+alloy",
  }.get((workload, size))
  if canonical is None:
    return None
  path = canonical / "representative_fe_inputs.npz"
  if not path.exists():
    return None
  with np.load(path) as data:
    return {name: np.asarray(data[name], dtype=np.float64) for name in data.files}


def _unbumpercars_hessian_inputs(size: int, harvested: dict[str, np.ndarray] | None) -> tuple[dict[str, np.ndarray], np.ndarray]:
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
  from benchmarks.problems.unbumpercars.filters import CasadiDTCBFSafetyFilter

  cfg, filt_cfg, weights = ClosedLoopConfig(ncars=size), FilterConfig(model="dt"), load_dt_mlp_weights()
  if harvested is None:
    bar_x = sample_initial_states(cfg, collision_free=False).reshape(-1)
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
  reference = CasadiDTCBFSafetyFilter(cfg, filt_cfg, weights, _build_solver=False)
  assert reference.hess_fn is not None
  expected = np.asarray(reference.hess_fn(pieces["z"], p, pieces["lam_f"], pieces["lam_g"]), dtype=np.float64).reshape(-1)
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
  kind = _kernel_kind(workload)
  if workload in ("chain", "chain_jac"):
    horizon = chain.HORIZON
    if harvested is None:
      zv, pv = chain.sample_inputs(size, horizon)
    else:
      zv, pv = harvested["z"], harvested["p"]
    if zv.shape != (chain.n_dec(size, horizon),) or pv.shape != (chain.n_param(size),):
      raise ValueError(f"harvested chain input shapes do not match M={size}, N={horizon}: {zv.shape}, {pv.shape}")
    if kind == "jac":
      expected = chain.chain_eq_jac_dense_reference(size, horizon, zv, pv).reshape(-1)
      values = {"z": zv, "p": pv}
    else:
      lam_g = _mixed_sign_multipliers(chain.n_state(size) * (horizon + 1))
      expected = chain.chain_lag_hess_dense_reference(size, horizon, zv, pv, 1.0, lam_g).reshape(-1)
      values = {"z": zv, "lam_f": np.array(1.0), "lam_g": lam_g, "p": pv}
  elif workload in ("race_cars", "race_cars_jac"):
    if harvested is None:
      rng = np.random.default_rng(7)
      zv = rng.normal(scale=0.4, size=race_cars.NZ * (size + 1))
      pv = np.concatenate([rng.normal(scale=0.4, size=race_cars.NX * (size + 1)), race_cars.RaceCarParams().array()])
    else:
      zv, pv = harvested["z"], harvested["p"]
    if zv.shape != (race_cars.NZ * (size + 1),) or pv.shape != (race_cars.n_param(size),):
      raise ValueError(f"harvested race_cars input shapes do not match N={size}: {zv.shape}, {pv.shape}")
    if kind == "jac":
      expected = race_cars.race_car_constraint_jac_dense_reference(size, zv, pv).reshape(-1)
      values = {"z": zv, "p": pv}
    else:
      from benchmarks.problems.race_cars.closed_loop import EpisodeConfig, race_car_lag_hess_dense_reference

      lam_g = _mixed_sign_multipliers(race_cars.NX * (size + 1) + 2 * size)
      expected = race_car_lag_hess_dense_reference(EpisodeConfig(horizon=size), zv, pv, 1.0, lam_g).reshape(-1)
      values = {"z": zv, "lam_f": np.array(1.0), "lam_g": lam_g, "p": pv}
  elif workload in NPMPC_WORKLOADS:
    horizon, decoder, weights, terminal = _npmpc_cell(workload, size)
    if harvested is None:
      zv, pw = npmpc.sample_inputs(horizon, decoder, weights)
      terminal = np.diag(npmpc.CostWeights().x_end) if terminal is None else terminal
      pv = npmpc.pack_nlp_params(decoder, zv[: npmpc.NX], pw, terminal)
    else:
      zv, pv = harvested["z"], harvested["p"]
    if zv.shape != (npmpc.n_dec(horizon),) or pv.shape != (npmpc.n_param(decoder),):
      raise ValueError(f"harvested npmpc input shapes do not match {CELL_AXES[workload]}={size}: {zv.shape}, {pv.shape}")
    if kind == "hess":
      lam_g = _mixed_sign_multipliers(sum(npmpc.constraint_counts(horizon)))
      expected = npmpc.npmpc_lag_hess_dense_reference(horizon, zv, pv, 1.0, lam_g, decoder).reshape(-1)
      values = {"z": zv, "lam_f": np.array(1.0), "lam_g": lam_g, "p": pv}
    else:
      expected = npmpc.npmpc_constraint_jac_dense_reference(horizon, zv, pv, decoder)
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

  def sample_value(name: str) -> np.ndarray:
    if name in values:
      return values[name]
    if name in {"x", info["inputs"][0][0]} and "z" in values:
      return values["z"]
    raise KeyError(f"no sample value for {name!r}; available values are {sorted(values)}")

  # CasADi's nlpsol oracle names are the ABI: x, p, lam_f, lam_g. Alloy keeps its own named
  # parameter inputs, so the x/z alias is resolved only at this boundary.
  sample_values = {name: sample_value(name) for name, _ in info["inputs"]}
  args = list(sample_values.values())
  kernel = info["callable"]
  # CasADi takes flat positional leaves; an alloy Function takes its declared tree, so rebuild it.
  result = kernel(*args) if info["backend"].startswith("casadi") else kernel(kernel.input_tree.unflatten(tuple(args)))
  if info["backend"].startswith("casadi"):
    outputs = result if isinstance(result, (tuple, list)) else (result,)
    selected = outputs[int(info["output_index"])]
    compact = np.asarray(selected.nonzeros(), dtype=np.float64).reshape(-1)
  else:
    compact = np.asarray(result, dtype=np.float64).reshape(-1)
  check_dense_reference(
    compact,
    info["rows"],
    info["cols"],
    expected,
    (info["n_rows"], info["n_cols"]),
    label=info["backend"],
    layout=str(info.get("layout", "full")),
  )
  return write_samples(out_dir, sample_values, expected)


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
  casadi_transform: bool = True,
) -> tuple[dict[str, object], dict | None]:
  if out_dir.exists():
    shutil.rmtree(out_dir)
  out_dir.mkdir(parents=True)
  try:
    with codegen_deadline(codegen_timeout):
      info = build_kernel(workload, size, backend, out_dir, casadi_transform=casadi_transform)
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
    layout=info["layout"],
    codegen_ms=f"{codegen_ms:.1f}",
    build_ms=f"{info['build_ms']:.1f}",
    render_ms=f"{info['render_ms']:.1f}",
    source_bytes=info["source_bytes"],
    artifact_bytes=info["artifact_bytes"],
    executable_bytes=info["executable_bytes"],
    static_metadata_bytes=info["static_metadata_bytes"],
    source_lines=info["source_lines"],
    workspace=info["w_size"],
    integer_workspace=info["iw_size"],
    argument_pointers=info["arg_size"],
    result_pointers=info["res_size"],
    dispatch_trip_count=info["dispatch_trip_count"],
    dispatch_workspace=info["dispatch_workspace"],
    dispatch_arithmetic=info["dispatch_arithmetic"],
    coloring_width=info["coloring_width"],
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
    **{key: f"{value:.1f}" for key, value in info["compile_timings"].items()},
    runtime_ns=f"{runtime_ns:.1f}" if runtime_ns is not None else "",
    runtime_status=runtime_status,
    note=note,
  ), info


def _fresh_cell(workload, size, backend, out_dir, options):
  result = run_cell(workload, size, backend, out_dir, **options)[0]
  result["process_id"] = os.getpid()
  return result


def summarize_runs(rows: list[dict], out: Path) -> None:
  """Report dispersion across fresh processes while retaining every raw row."""
  groups = {}
  for result in rows:
    groups.setdefault((result["workload"], result["size"], result["backend"]), []).append(result)
  metrics = ("runtime_ns", "build_ms", "render_ms", "kernel_compile_ms", "wrapper_compile_ms", "link_ms")
  fields = [
    "workload",
    "size",
    "backend",
    "attempts",
    "successful",
    *[f"{metric}_{stat}" for metric in metrics for stat in ("mean", "median", "stdev", "cv", "min", "max")],
  ]
  with out.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    for (workload, size, backend), attempts in groups.items():
      successful = [result for result in attempts if result["runtime_status"] == "ok"]
      summary = dict(workload=workload, size=size, backend=backend, attempts=len(attempts), successful=len(successful))
      for metric in metrics:
        values = [float(result[metric]) for result in successful if result.get(metric, "") != ""]
        if not values:
          continue
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else None
        for stat, value in dict(
          mean=mean,
          median=statistics.median(values),
          stdev=stdev,
          cv=stdev / mean if stdev is not None and mean else None,
          min=min(values),
          max=max(values),
        ).items():
          summary[f"{metric}_{stat}"] = "" if value is None else value
      writer.writerow(summary)


def run_sweep(args, cli_args: list[str]) -> bool:
  args.out = args.out.resolve()
  repetitions = args.repetitions
  if args.headline:
    require_headline_settings(args.boost == "on")
  args.out.parent.mkdir(parents=True, exist_ok=True)
  write(args.out, collect(ROOT, gbench.compiler(), cli_args))
  options = dict(
    codegen_timeout=args.codegen_timeout,
    compile_timeout=args.compile_timeout,
    max_source_mb=args.max_source_mb,
    benchmark_min_time=args.benchmark_min_time,
    casadi_transform=args.casadi_transform,
  )
  ok, rows = True, []
  rng = random.Random(args.order_seed)
  orders = {}
  for workload in args.workloads:
    orders[workload] = list(args.backends or DEFAULT_BACKENDS[workload])
    rng.shuffle(orders[workload])
  gave_up: dict[tuple[str, str], tuple[int, str]] = {}
  with args.out.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=FIELDS)
    writer.writeheader()
    for repetition in range(repetitions):
      for workload in args.workloads:
        initial = orders[workload]
        offset = repetition % len(initial)
        backends = initial[offset:] + initial[:offset]
        for size in sorted(args.sizes or DEFAULT_SIZES[workload]):
          for order, backend in enumerate(backends):
            if args.headline:
              require_headline_settings(args.boost == "on")
            print(f"[repeat {repetition + 1}/{repetitions} {workload}] size={size} backend={backend} ... ", end="", flush=True)
            if backend not in DEFAULT_BACKENDS[workload]:
              result = row(
                workload=workload,
                size=size,
                backend=backend,
                compile_status="not_applicable",
                runtime_status="skipped",
                note=f"{backend} has no repeated element for {workload}",
              )
            elif (workload, backend) in gave_up and size >= gave_up[workload, backend][0]:
              failed_size, status = gave_up[workload, backend]
              result = row(
                workload=workload,
                size=size,
                backend=backend,
                compile_status="skipped_after_failure",
                runtime_status="skipped",
                note=f"{backend} {status} at size={failed_size}; this size and larger won't fit",
              )
            else:
              out_dir = args.out.parent / f"repeat_{repetition + 1}" / workload / f"{backend}_{CELL_AXES[workload]}{size}"
              with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
                result = pool.submit(_fresh_cell, workload, size, backend, out_dir, options).result()
            if args.headline:
              require_headline_settings(args.boost == "on")
            result.update(repetition=repetition + 1, backend_order=order + 1)
            writer.writerow(result)
            fp.flush()
            rows.append(result)
            status = result["runtime_status"] or result["compile_status"]
            print(f"{status}" + (f": {result['note']}" if result["note"] else ""))
            if result["compile_status"] in {"timeout", "skipped_size", "compile_error", "codegen_timeout", "codegen_error"}:
              gave_up[workload, backend] = (size, str(result["compile_status"]))
            if result["runtime_status"] in {"runtime_error", "runtime_timeout", "correctness_fail", "parse_fail"} or result["compile_status"] in {
              "compile_error",
              "codegen_error",
            }:
              ok = False
  summarize_runs(rows, args.out.with_suffix(".summary.csv"))
  print(f"Results written to {args.out}")
  return ok
