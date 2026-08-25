from __future__ import annotations

import contextlib
import csv
import os
from dataclasses import replace
from pathlib import Path
import re
import shutil
import signal
import time

import numpy as np

import alloy as al
from alloy.codegen.aot import render_c_module
from alloy.ir.expr import ExprOp, topo
from alloy.ir.program import ProgramNode, ProgramOp
from benchmarks.harness import ROOT, RESULTS, gbench
from benchmarks.harness.correctness import check_dense_reference, write_samples
from benchmarks.harness.provenance import collect, write
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
BACKENDS = ("alloy", "casadi_sx", "casadi_mx", "casadi_call_mx", "casadi_map_sx")
UNBUMPERCARS_BACKENDS = ("alloy", "casadi_sx", "casadi_mx")
DEFAULT_BACKENDS = {workload: UNBUMPERCARS_BACKENDS if workload == "unbumpercars" else BACKENDS for workload in DEFAULT_SIZES}
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
NPMPC_WORKLOADS = ("npmpc", "npmpc_jac", "npmpc_decoder", "npmpc_decoder_jac")


def _kernel_kind(workload: str) -> str:
  return "jac" if workload.endswith("_jac") else "hess"


FIELDS = [
  "workload",
  "size",
  "backend",
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
  """Return the retained MAP trip count, callee workspace, and arithmetic per iteration."""
  maps = [node for node in topo(fun.outputs) if node.op == ExprOp.MAP]
  trip_counts = {int(node.attrs["length"]) for node in maps}
  if not maps or len(trip_counts) != 1:
    return "", "", ""
  mapped_callees = {str(node.attrs["callee"].name) for node in maps}
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

  trip_count = trip_counts.pop()
  dispatches: list[ProgramNode] = []
  root = procs[fun.name]
  for stmt in root.args[int(root.attrs["param_count"]) :]:
    call = None
    if stmt.op == ProgramOp.FOR and len(stmt.args) == 2 and stmt.args[1].op == ProgramOp.CALL:
      if _static_trip_count(stmt.args[0]) == trip_count:
        call = stmt.args[1]
    elif trip_count == 1 and stmt.op == ProgramOp.CALL:
      call = stmt
    if call is not None and str(call.attrs["callee"]) in mapped_callees:
      dispatches.append(call)
  if not dispatches:
    return "", "", ""
  try:
    workspace = max(proc_workspace(str(call.attrs["callee"])) for call in dispatches)
    work = sum(proc_arithmetic(str(call.attrs["callee"])) for call in dispatches)
  except LookupError:
    return "", "", ""
  return trip_count, workspace, work


def _module_info(name: str, backend: str, module, inputs, sparsity, shape, build_ms: float, render_ms: float, benchmark: str, **extra) -> dict:
  source_path = Path(module.source_name)
  artifact_bytes, executable_bytes, static_metadata_bytes = _artifact_sizes(module.source, module.header)
  dispatch_trip_count, dispatch_workspace, dispatch_arithmetic = _dispatch_metrics(extra["callable"], module.program)
  colors = al.column_coloring(sparsity)
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
    "coloring_width": max(colors) + 1 if colors else 0,
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


def _descriptor_kernel(solver: al.SolverFunction, kind: str):
  descriptor = solver.descriptor
  function = getattr(descriptor, kind)
  sparsity = getattr(descriptor, f"{kind}_sparsity")
  if not isinstance(function, al.Function) or sparsity is None:
    raise TypeError(f"{descriptor.name} has no Alloy {kind} kernel")
  assert function.output_sparsities[0] == sparsity
  return function, sparsity


def _casadi_descriptor_kernel(ca, name: str, z, p, cost, constraints, kind: str, *, cse: bool = False):
  """Build the full CasADi oracle set and return the selected descriptor-equivalent kernel."""
  options = {"cse": True} if cse else {}
  ca.Function(f"{name}_base", [z, p], [cost, constraints], options)
  ca.Function(f"{name}_grad", [z, p], [ca.gradient(cost, z)], options)
  jac = ca.Function(name if kind == "jac" else f"{name}_descriptor_jac", [z, p], [ca.jacobian(constraints, z)], options)
  lam_f, lam_g = type(z).sym("lam_f"), type(z).sym("lam_g", int(constraints.shape[0]))
  hess = ca.Function(
    name if kind == "hess" else f"{name}_descriptor_hess",
    [z, lam_f, lam_g, p],
    [ca.hessian(lam_f * cost + ca.dot(lam_g, constraints), z)[0]],
    options,
  )
  return hess if kind == "hess" else jac


def _lag_hess_reference(ca, name: str, z, p, cost, constraints, zv, pv):
  lam_g = np.linspace(-0.75, 0.75, int(constraints.shape[0]))
  ref = _casadi_descriptor_kernel(ca, name, z, p, cost, constraints, "hess")
  expected = np.asarray(ca.densify(ref(zv, 1.0, lam_g, pv)), dtype=np.float64).reshape(-1)
  return lam_g, expected


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


def _npmpc_repeated_pieces(horizon: int, decoder, terminal, stage_sym, *, mapped: bool):
  import casadi as ca

  pieces = npmpc._ca_npmpc_joint_parameter_pieces(horizon, decoder, ca.MX, P=terminal, dynamics=False)
  one = npmpc.ca_npmpc_pieces(1, decoder, stage_sym, cost=False)
  stage = ca.Function("npmpc_stage", [one["z"], one["pw"]], [one["h_eq"]])
  z, p = pieces["z"], pieces["p"]
  offset = npmpc.NX * (horizon + 1)
  states = ca.reshape(z[:offset], npmpc.NX, horizon + 1)
  controls = ca.reshape(z[offset : offset + npmpc.NU * horizon], npmpc.NU, horizon)
  stage_z = ca.vertcat(states[:, :horizon], states[:, 1:], controls, ca.repmat(z[-1], 1, horizon))
  rows = _repeat_element(ca, stage, horizon, (stage_z, ca.repmat(p[npmpc.NX :], 1, horizon)), mapped=mapped)
  return {**pieces, "h_eq": ca.reshape(rows, npmpc.NX * horizon, 1)}


def _race_cars_alloy(workload: str, size: int, out_dir: Path) -> dict:
  from benchmarks.problems.race_cars.closed_loop import EpisodeConfig, _race_car_nlp

  kind = _kernel_kind(workload)
  started = time.perf_counter()
  kernel, sparsity = _descriptor_kernel(_race_car_nlp(EpisodeConfig(horizon=size)), kind)
  name = kernel.name
  build_ms = (time.perf_counter() - started) * 1000
  module, render_ms = _render_alloy(kernel, name, out_dir)
  n_constraints = race_cars.NX * (size + 1) + 2 * size
  inputs = (
    [("z", race_cars.NZ * (size + 1)), ("p", race_cars.n_param(size))]
    if kind == "jac"
    else [("z", race_cars.NZ * (size + 1)), ("lam_f", 1), ("lam_g", n_constraints), ("p", race_cars.n_param(size))]
  )
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
    w_size=module.workspace_size,
    callable=kernel,
  )


def _chain_alloy(workload: str, size: int, out_dir: Path) -> dict:
  horizon = chain.HORIZON
  kind = _kernel_kind(workload)
  inputs = (
    [("z", chain.n_dec(size, horizon)), ("p", chain.n_param(size))]
    if kind == "jac"
    else [("z", chain.n_dec(size, horizon)), ("lam_f", 1), ("lam_g", chain.n_state(size) * (horizon + 1)), ("p", chain.n_param(size))]
  )
  benchmark = f"BM_AlloyChain{'EqJac' if kind == 'jac' else 'LagHess'}M{size}"
  started = time.perf_counter()
  kernel, sparsity = _descriptor_kernel(chain.chain_nlp(size, horizon), kind)
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
    w_size=module.workspace_size,
    callable=kernel,
  )


def _unbumpercars_alloy(size: int, out_dir: Path) -> dict:
  from benchmarks.problems.unbumpercars.common import ClosedLoopConfig, FilterConfig, NCTRL, NSTATE, N_PHYSICS, N_PW_DT
  from benchmarks.problems.unbumpercars.filters import build_alloy_nlp

  started = time.perf_counter()
  cfg = ClosedLoopConfig(ncars=size)
  solver = build_alloy_nlp(cfg, FilterConfig(model="dt"))
  kernel, sparsity = _descriptor_kernel(solver, "hess")
  name = kernel.name
  build_ms = (time.perf_counter() - started) * 1000
  module, render_ms = _render_alloy(kernel, name, out_dir)
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
    sparsity.shape,
    build_ms,
    render_ms,
    f"BM_AlloyUnbumpercarsLagHessC{size}",
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
  horizon, decoder, _, terminal = _npmpc_cell(workload, size)
  axis = CELL_AXES[workload]
  n_z = npmpc.n_dec(horizon)
  started = time.perf_counter()
  terminal = np.diag(npmpc.CostWeights().x_end) if terminal is None else terminal
  kind = _kernel_kind(workload)
  built, sparsity = _descriptor_kernel(npmpc.npmpc_nlp(terminal, horizon, decoder), kind)
  name = built.name
  if kind == "hess":
    inputs = [("z", n_z), ("lam_f", 1), ("lam_g", sum(npmpc.constraint_counts(horizon))), ("p", npmpc.n_param(decoder))]
    benchmark = f"BM_AlloyNpmpcLagHess{axis}{size}"
  else:
    inputs = [("z", n_z), ("p", npmpc.n_param(decoder))]
    benchmark = f"BM_AlloyNpmpcConstraintJac{axis}{size}"
  build_ms = (time.perf_counter() - started) * 1000
  module, render_ms = _render_alloy(built, name, out_dir)
  return _module_info(
    name, "alloy", module, inputs, sparsity, sparsity.shape, build_ms, render_ms, benchmark, w_size=module.workspace_size, callable=built
  )


def _casadi(workload: str, size: int, backend: str, out_dir: Path) -> dict:
  import casadi as ca

  kind = backend.removeprefix("casadi_")
  repeated = kind in {"call_mx", "map_sx"}
  mapped = kind == "map_sx"
  sym_t = ca.SX if kind in {"sx", "map_sx"} else ca.MX
  label = {"sx": "Sx", "mx": "Mx", "call_mx": "CallMx", "map_sx": "MapSx"}[kind]
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
  started = time.perf_counter()
  if workload in ("chain", "chain_jac"):
    horizon = chain.HORIZON
    z, p, cost, constraints = chain._ca_nlp_pieces(size, horizon, sym_t, map_stages=mapped, call_stages=repeated)
    fn = _casadi_descriptor_kernel(ca, name, z, p, cost, constraints, kernel, cse=True)
    inputs = (
      [("z", chain.n_dec(size, horizon)), ("p", chain.n_param(size))]
      if kernel == "jac"
      else [("z", chain.n_dec(size, horizon)), ("lam_f", 1), ("lam_g", chain.n_state(size) * (horizon + 1)), ("p", chain.n_param(size))]
    )
    benchmark = f"BM_Casadi{label}Chain{'EqJac' if kernel == 'jac' else 'LagHess'}M{size}"
  elif workload in ("race_cars", "race_cars_jac"):
    from benchmarks.problems.race_cars.casadi_nlp import build_casadi_race_car_nlp
    from benchmarks.problems.race_cars.closed_loop import EpisodeConfig

    config = EpisodeConfig(horizon=size)
    pieces = _race_cars_repeated_pieces(config, sym_t, mapped=mapped) if repeated else build_casadi_race_car_nlp(config, sym_t)
    constraints = ca.vertcat(pieces["h_eq"], pieces["g_ineq"])
    fn = _casadi_descriptor_kernel(ca, name, pieces["z"], pieces["p"], pieces["f"], constraints, kernel)
    inputs = (
      [("z", race_cars.NZ * (size + 1)), ("p", race_cars.n_param(size))]
      if kernel == "jac"
      else [("z", race_cars.NZ * (size + 1)), ("lam_f", 1), ("lam_g", int(constraints.shape[0])), ("p", race_cars.n_param(size))]
    )
    benchmark = f"BM_Casadi{label}RaceCar{'ConstraintJac' if kernel == 'jac' else 'LagHess'}N{size}"
  elif workload in NPMPC_WORKLOADS:
    horizon, decoder, _, terminal = _npmpc_cell(workload, size)
    terminal = np.diag(npmpc.CostWeights().x_end) if terminal is None else terminal
    pieces = (
      _npmpc_repeated_pieces(horizon, decoder, terminal, sym_t, mapped=mapped)
      if repeated
      else npmpc._ca_npmpc_joint_parameter_pieces(horizon, decoder, sym_t, P=terminal)
    )
    constraints = ca.vertcat(pieces["h_eq"], pieces["g_ineq"])
    if kernel == "hess":
      fn = _casadi_descriptor_kernel(ca, name, pieces["z"], pieces["p"], pieces["f"], constraints, "hess")
      inputs = [("z", npmpc.n_dec(horizon)), ("lam_f", 1), ("lam_g", sum(npmpc.constraint_counts(horizon))), ("p", npmpc.n_param(decoder))]
      benchmark = f"BM_Casadi{label}NpmpcLagHess{axis}{size}"
    else:
      fn = _casadi_descriptor_kernel(ca, name, pieces["z"], pieces["p"], pieces["f"], constraints, "jac")
      inputs = [("z", npmpc.n_dec(horizon)), ("p", npmpc.n_param(decoder))]
      benchmark = f"BM_Casadi{label}NpmpcConstraintJac{axis}{size}"
  else:
    if repeated:
      raise ValueError(f"{backend} is not defined for {workload}")

    from benchmarks.problems.unbumpercars.common import ClosedLoopConfig, FilterConfig, load_dt_mlp_weights
    from benchmarks.problems.unbumpercars.filters import build_casadi_hessian

    fn = build_casadi_hessian(ClosedLoopConfig(ncars=size), FilterConfig(model="dt"), load_dt_mlp_weights(), name, sym_t)
    inputs = [("z", fn.size1_in(0)), ("lam_f", fn.size1_in(1)), ("lam_g", fn.size1_in(2)), ("p", fn.size1_in(3))]
    benchmark = f"BM_Casadi{label}UnbumpercarsLagHessC{size}"
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
  rows, cols = fn.sparsity_out(0).get_triplet()
  colors = al.column_coloring(al.SparsityType((fn.size_out(0)[0], fn.size_out(0)[1]), tuple(int(x) for x in rows), tuple(int(x) for x in cols)))
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
    "source_bytes": len(source.encode()),
    "artifact_bytes": artifact_bytes,
    "executable_bytes": executable_bytes,
    "static_metadata_bytes": static_metadata_bytes,
    "dispatch_trip_count": "",
    "dispatch_workspace": "",
    "dispatch_arithmetic": "",
    "coloring_width": max(colors) + 1 if colors else 0,
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
    if workload in ("chain", "chain_jac"):
      return _chain_alloy(workload, size, out_dir)
    if workload in ("race_cars", "race_cars_jac"):
      return _race_cars_alloy(workload, size, out_dir)
    return _unbumpercars_alloy(size, out_dir)
  return _casadi(workload, size, backend, out_dir)


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
      import casadi as ca

      z, p, cost, constraints = chain._ca_nlp_pieces(size, horizon, ca.MX)
      lam_g, expected = _lag_hess_reference(ca, f"chain_lag_hess_dense_ref_M{size}", z, p, cost, constraints, zv, pv)
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
      import casadi as ca

      from benchmarks.problems.race_cars.casadi_nlp import build_casadi_race_car_nlp
      from benchmarks.problems.race_cars.closed_loop import EpisodeConfig

      pieces = build_casadi_race_car_nlp(EpisodeConfig(horizon=size), ca.MX)
      constraints = ca.vertcat(pieces["h_eq"], pieces["g_ineq"])
      lam_g, expected = _lag_hess_reference(ca, f"race_car_lag_hess_dense_ref_N{size}", pieces["z"], pieces["p"], pieces["f"], constraints, zv, pv)
      values = {"z": zv, "lam_f": np.array(1.0), "lam_g": lam_g, "p": pv}
  elif workload in NPMPC_WORKLOADS:
    import casadi as ca

    horizon, decoder, weights, terminal = _npmpc_cell(workload, size)
    if harvested is None:
      zv, pw = npmpc.sample_inputs(horizon, decoder, weights)
      pv = np.concatenate([zv[: npmpc.NX], pw])
    else:
      zv, pv = harvested["z"], harvested["p"]
    if zv.shape != (npmpc.n_dec(horizon),) or pv.shape != (npmpc.n_param(decoder),):
      raise ValueError(f"harvested npmpc input shapes do not match {CELL_AXES[workload]}={size}: {zv.shape}, {pv.shape}")
    if kind == "hess":
      # Multipliers of mixed sign, so the Hessian is not dominated by the objective block alone.
      lam_g = np.linspace(-0.75, 0.75, sum(npmpc.constraint_counts(horizon)))
      ref = npmpc.ca_npmpc_lag_hess(horizon, decoder, f"npmpc_lag_hess_dense_ref_{CELL_AXES[workload]}{size}", ca.MX, terminal)
      expected = np.asarray(ca.densify(ref(zv, 1.0, lam_g, pv)), dtype=np.float64).reshape(-1)
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
        for backend in args.backends or DEFAULT_BACKENDS[workload]:
          print(f"[{workload}] size={size} backend={backend} ... ", end="", flush=True)
          if backend not in DEFAULT_BACKENDS[workload]:
            result = row(
              workload=workload,
              size=size,
              backend=backend,
              compile_status="not_applicable",
              runtime_status="skipped",
              note=f"{backend} has no repeated element for {workload}",
            )
          elif backend in gave_up:
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
