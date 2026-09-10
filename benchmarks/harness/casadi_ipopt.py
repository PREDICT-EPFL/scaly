"""Fresh-process CasADi IPOPT code generation for controlled benchmark columns."""

from __future__ import annotations

import argparse
import ctypes
from contextvars import ContextVar
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np

from alloy.codegen.jit import _load_library, opt_flag
from alloy.codegen.toolchain import cache_root, find_c_compiler
from alloy.solvers.paths import backend_compile_flags, solver_paths
from alloy.solvers.stats import ALLOY_SOLVER_STATS_VERSION, AlloySolveStatus, SolverStats, SolverStatus, stats_c_timing_defs
from alloy.utils.env import shared_lib_ext, shared_lib_flag
from benchmarks.harness import NATIVE_CFLAGS

INTERPRETED: ContextVar[bool] = ContextVar("casadi_interpreted", default=False)

_CACHE = Path(os.environ.get("ALLOY_CASADI_IPOPT_CACHE", cache_root() / "casadi-ipopt"))
_CACHE_VERSION = 4
_ORACLE_NAMES = ("f", "g", "grad_f", "jac_g", "hess_l")
_DP = ctypes.POINTER(ctypes.c_double)


class _DlInfo(ctypes.Structure):
  _fields_ = [("dli_fname", ctypes.c_char_p), ("dli_fbase", ctypes.c_void_p), ("dli_sname", ctypes.c_char_p), ("dli_saddr", ctypes.c_void_p)]


_INSTRUMENTATION = (
  "\n#include <time.h>\n"
  + "\n".join(stats_c_timing_defs())
  + r"""
static double alloy_bench_fe_s = 0.0;
static double alloy_bench_ipopt_s = 0.0;
static ipindex alloy_bench_iter = 0;
static int alloy_bench_n_f = 0;
static int alloy_bench_n_g = 0;
static int alloy_bench_n_grad_f = 0;
static int alloy_bench_n_jac_g = 0;
static int alloy_bench_n_hess_l = 0;
static bool alloy_bench_intermediate(
    ipindex alg_mod, ipindex iter_count, ipnumber obj_value, ipnumber inf_pr,
    ipnumber inf_du, ipnumber mu, ipnumber d_norm, ipnumber regularization_size,
    ipnumber alpha_du, ipnumber alpha_pr, ipindex ls_trials, UserDataPtr user_data) {
  (void)alg_mod; (void)obj_value; (void)inf_pr; (void)inf_du; (void)mu;
  (void)d_norm; (void)regularization_size; (void)alpha_du; (void)alpha_pr;
  (void)ls_trials; (void)user_data;
  alloy_bench_iter = iter_count;
  return true;
}
"""
)

_SHIM = r"""
static double alloy_bench_total_s = 0.0;
int alloy_bench_solve(const double** arg, double** res, long long* iw, double* w, int mem) {
  alloy_bench_fe_s = 0.0;
  alloy_bench_ipopt_s = 0.0;
  alloy_bench_iter = 0;
  alloy_bench_n_f = alloy_bench_n_g = alloy_bench_n_grad_f = 0;
  alloy_bench_n_jac_g = alloy_bench_n_hess_l = 0;
  double t0 = alloy_clock_s();
  int rc = {name}(arg, res, (casadi_int*)iw, w, mem);
  alloy_bench_total_s = alloy_clock_s() - t0;
  return rc;
}
int alloy_bench_checkout(void) {{ return {name}_checkout(); }}
void alloy_bench_release(int mem) {{ {name}_release(mem); }}
int alloy_bench_native_status(int mem) {{ return (int)casadi_f0_mem[mem].status; }}
int alloy_bench_status(int mem) {{
  switch (casadi_f0_mem[mem].status) {{
    case Solve_Succeeded: return 0;
    case Solved_To_Acceptable_Level: return 1;
    case Feasible_Point_Found: return 1;
    case Maximum_Iterations_Exceeded: return 2;
    case Maximum_CpuTime_Exceeded: return 2;
    case Maximum_WallTime_Exceeded: return 2;
    case Infeasible_Problem_Detected: return 3;
    case Diverging_Iterates: return 5;
    case User_Requested_Stop: return 6;
    case Search_Direction_Becomes_Too_Small: return 5;
    case Restoration_Failed: return 5;
    case Error_In_Step_Computation: return 5;
    case Invalid_Number_Detected: return 5;
    default: return 7;
  }}
}}
long long alloy_bench_sizes(long long* out) {{
  casadi_int sz_arg, sz_res, sz_iw, sz_w;
  {name}_work(&sz_arg, &sz_res, &sz_iw, &sz_w);
  out[0] = sz_arg; out[1] = sz_res; out[2] = sz_iw; out[3] = sz_w;
  return 0;
}}
double alloy_bench_total(void) {{ return alloy_bench_total_s; }}
double alloy_bench_fe(void) {{ return alloy_bench_fe_s; }}
double alloy_bench_ipopt(void) {{ return alloy_bench_ipopt_s; }}
long long alloy_bench_iter_count(void) {{ return alloy_bench_iter; }}
long long alloy_bench_eval_count(int which) {{
  switch (which) {{
    case 0: return alloy_bench_n_f;
    case 1: return alloy_bench_n_g;
    case 2: return alloy_bench_n_grad_f;
    case 3: return alloy_bench_n_jac_g;
    case 4: return alloy_bench_n_hess_l;
    default: return 0;
  }}
}}
"""


def _instrument(source: str, name: str, *, exact_hessian: bool) -> str:
  include = "#include <coin-or/IpStdCInterface.h>\n"
  if source.count(include) != 1:
    raise RuntimeError("CasADi IPOPT code no longer has one IpStdCInterface include")
  source = source.replace(include, include + _INSTRUMENTATION, 1)
  presolve = "  casadi_ipopt_presolve(d);\n"
  if source.count(presolve) != 1:
    raise RuntimeError("CasADi IPOPT code no longer has one presolve call")
  callback = "  if (!SetIntermediateCallback(d->ipopt, alloy_bench_intermediate)) { FreeIpoptProblem(d->ipopt); return 1; }\n"
  source = source.replace(presolve, presolve + callback, 1)

  call_pattern = re.compile(r"if \((casadi_f\d+\(d->arg, d->res, d->iw, d->w, 0\))\) return false;")

  def timed(match: re.Match[str]) -> str:
    call = match.group(1)
    return f"{{ double t0 = alloy_clock_s(); int rc = {call}; alloy_bench_fe_s += alloy_clock_s() - t0; if (rc) return false; }}"

  source, count = call_pattern.subn(timed, source)
  expected_callbacks = len(_ORACLE_NAMES) if exact_hessian else len(_ORACLE_NAMES) - 1
  if count != expected_callbacks:
    raise RuntimeError(f"expected {expected_callbacks} generated IPOPT oracle calls, found {count}")
  for oracle in _ORACLE_NAMES:
    if oracle in ("jac_g", "hess_l"):
      signature = re.compile(rf"(bool casadi_nlp_{oracle}\d*\([^{{]+\) \{{\n(?:.*\n)*?  if \(values\) \{{\n)")
    else:
      signature = re.compile(rf"(bool casadi_nlp_{oracle}\d*\([^{{]+\) \{{\n)")
    source, count = signature.subn(rf"\1  ++alloy_bench_n_{oracle};\n", source, count=1)
    expected = int(oracle != "hess_l" or exact_hessian)
    if count != expected:
      raise RuntimeError(f"CasADi IPOPT code no longer has one {oracle} callback")
  adjoint_pattern = re.compile(
    r"(?P<block>  d->arg\[0\] = d_nlp\.z;\n"
    r"(?:  .*\n)*?"
    r"  if \(casadi_f\d+\(d->arg, d->res, d->iw, d->w, 0\)\) return 1;\n"
    r"  casadi_scal\(\d+, -1\.0, d_nlp\.lam_p\);\n)"
  )

  def skip_unused_adjoint(match: re.Match[str]) -> str:
    block = match.group("block")
    return "  if (d_nlp.lam_p) {\n" + "".join(f"  {line}" for line in block.splitlines(keepends=True)) + "  }\n"

  source, count = adjoint_pattern.subn(skip_unused_adjoint, source)
  if count != 1:
    raise RuntimeError(f"expected one generated post-solve parameter adjoint, found {count}")
  solve = "  casadi_ipopt_solve(d);\n"
  if source.count(solve) != 1:
    raise RuntimeError("CasADi IPOPT code no longer has one solve call")
  source = source.replace(
    solve,
    "  { double t0 = alloy_clock_s(); casadi_ipopt_solve(d); alloy_bench_ipopt_s = alloy_clock_s() - t0; }\n",
    1,
  )
  shim = _SHIM.replace("{name}", name).replace("{{", "{").replace("}}", "}")
  return source + shim


def _transformed_nlpsol(name: str, problem: dict, options: dict):
  import casadi as ca

  solver = ca.nlpsol(name, "ipopt", problem, options)
  cache = {oracle_name: solver.get_function(oracle_name).transform({}) for oracle_name in solver.get_function()}
  return ca.nlpsol(name, "ipopt", problem, {**options, "cache": {**options.get("cache", {}), **cache}})


def _build(spec_path: Path) -> None:
  import casadi as ca

  spec = json.loads(spec_path.read_text())
  nlp = ca.Function.deserialize(spec["nlp"])
  x = ca.MX.sym("x", nlp.sparsity_in(0))
  p = ca.MX.sym("p", nlp.sparsity_in(1))
  f, g = nlp.call([x, p], True, False)
  solver = _transformed_nlpsol(spec["name"], {"x": x, "p": p, "f": f, "g": g}, spec["options"])
  generator = ca.CodeGenerator(f"{spec['name']}.c", {"with_header": True, "casadi_int": "long long int"})
  generator.add(solver)
  exact_hessian = spec["options"].get("ipopt.hessian_approximation") != "limited-memory"
  source = _instrument(generator.dump(), spec["name"], exact_hessian=exact_hessian)
  work = Path(spec["work"])
  work.mkdir(parents=True, exist_ok=True)
  source_path = work / f"{spec['name']}.c"
  suffix = f".{os.getpid()}.tmp"
  tmp_source = source_path.with_suffix(source_path.suffix + suffix)
  tmp_source.write_text(source)
  tmp_source.replace(source_path)
  library = work / f"lib{spec['name']}{shared_lib_ext()}"
  tmp_library = library.with_suffix(library.suffix + suffix)
  command = [
    spec["compiler"],
    spec["opt"],
    *NATIVE_CFLAGS,
    "-fPIC",
    shared_lib_flag(),
    str(source_path),
    *backend_compile_flags(("ipopt",)),
    "-lm",
    "-o",
    str(tmp_library),
  ]
  started = time.perf_counter()
  proc = subprocess.run(command, text=True, capture_output=True)
  compile_s = time.perf_counter() - started
  log = work / "compile.log"
  tmp_log = log.with_suffix(log.suffix + suffix)
  tmp_log.write_text(f"$ {shlex.join(command)}\n{proc.stdout}{proc.stderr}")
  tmp_log.replace(log)
  if proc.returncode:
    tmp_library.unlink(missing_ok=True)
    raise RuntimeError(f"compiling the generated CasADi nlpsol failed:\n{proc.stderr[-3000:]}")
  metadata = work / "metadata.json"
  tmp_metadata = metadata.with_suffix(metadata.suffix + suffix)
  tmp_metadata.write_text(
    json.dumps({"compile_s": compile_s, "source_bytes": len(source.encode()), "source_lines": len(source.splitlines())}, indent=2) + "\n"
  )
  tmp_metadata.replace(metadata)
  tmp_library.replace(library)


def _symbol_library(library: ctypes.CDLL, name: str) -> Path:
  symbol = getattr(library, name)
  info = _DlInfo()
  process = ctypes.CDLL(None)
  process.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
  process.dladdr.restype = ctypes.c_int
  if not process.dladdr(ctypes.cast(symbol, ctypes.c_void_p), ctypes.byref(info)) or info.dli_fname is None:
    raise RuntimeError(f"could not resolve the library that provides {name}")
  return Path(os.fsdecode(info.dli_fname)).resolve()


def _library_digest(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class CompiledCasadiIpopt:
  """A code-generated CasADi `nlpsol` linked to Alloy's IPOPT library."""

  compiled = True

  def __init__(self, name: str, nlp, options: dict[str, Any], *, opt: str | None = None):
    self.options = dict(options)
    self.expand = bool(options.get("expand", False))
    opt = opt or opt_flag()
    compiler_info = find_c_compiler()
    compiler = compiler_info.cc if compiler_info is not None else "cc"
    self.configured_ipopt_library = Path(solver_paths(required=True).loads["ipopt"] or "").resolve()
    solver_digest = _library_digest(self.configured_ipopt_library)
    payload = json.dumps({"nlp": nlp.serialize(), "options": options}, sort_keys=True)
    cache_inputs = (
      payload
      + str(_CACHE_VERSION)
      + _INSTRUMENTATION
      + _SHIM
      + importlib.metadata.version("casadi")
      + solver_digest
      + compiler
      + opt
      + " ".join(NATIVE_CFLAGS)
      + " ".join(backend_compile_flags(("ipopt",)))  # the rpath baked into the library must match this checkout
    )
    key = hashlib.sha256(cache_inputs.encode()).hexdigest()[:20]
    work = _CACHE / key
    library = work / f"lib{name}{shared_lib_ext()}"
    started = time.perf_counter()
    metadata_path = work / "metadata.json"
    if not library.exists() or not metadata_path.exists():
      work.mkdir(parents=True, exist_ok=True)
      spec = work / "spec.json"
      tmp_spec = spec.with_suffix(spec.suffix + f".{os.getpid()}.tmp")
      tmp_spec.write_text(json.dumps({"name": name, "nlp": nlp.serialize(), "options": options, "compiler": compiler, "opt": opt, "work": str(work)}))
      tmp_spec.replace(spec)
      root = Path(__file__).resolve().parents[2]
      proc = subprocess.run([sys.executable, "-m", "benchmarks.harness.casadi_ipopt", "build", str(spec)], cwd=root, text=True, capture_output=True)
      if proc.returncode:
        raise RuntimeError(f"CasADi IPOPT child build failed:\n{proc.stdout}{proc.stderr}")
    self.build_ms = (time.perf_counter() - started) * 1000.0
    self.metadata = json.loads(metadata_path.read_text())
    self.library = library
    self._dll = _load_library(library, isolated=True)
    self.resolved_ipopt_library = _symbol_library(self._dll, "CreateIpoptProblem")
    if _library_digest(self.resolved_ipopt_library) != _library_digest(self.configured_ipopt_library):
      raise RuntimeError(
        f"generated CasADi solver resolved {self.resolved_ipopt_library}, expected {self.configured_ipopt_library}; "
        "the loaded IPOPT build does not match the benchmark configuration"
      )
    self.ipopt_library = self.resolved_ipopt_library
    self._solve = self._dll.alloy_bench_solve
    self._solve.argtypes = [ctypes.POINTER(_DP), ctypes.POINTER(_DP), ctypes.POINTER(ctypes.c_longlong), _DP, ctypes.c_int]
    self._solve.restype = ctypes.c_int
    self._dll.alloy_bench_checkout.restype = ctypes.c_int
    self._dll.alloy_bench_release.argtypes = [ctypes.c_int]
    self._mem = int(self._dll.alloy_bench_checkout())
    if self._mem < 0:
      raise RuntimeError("all generated CasADi nlpsol memory slots are busy; close the existing solver instance")
    sizes = (ctypes.c_longlong * 4)()
    self._dll.alloy_bench_sizes(sizes)
    sz_arg, sz_res, sz_iw, sz_w = (int(value) for value in sizes)
    self._argp = (_DP * max(sz_arg, 1))()
    self._resp = (_DP * max(sz_res, 1))()
    self._iw = (ctypes.c_longlong * max(sz_iw, 1))()
    self._w = (ctypes.c_double * max(sz_w, 1))()
    n_x, n_p, n_g = int(nlp.numel_in(0)), int(nlp.numel_in(1)), int(nlp.numel_out(1))
    self._input_sizes = (n_x, n_p, n_x, n_x, n_g, n_g, n_x, n_g)
    self._out_sizes = [n_x, 1, n_g, n_x, n_g, 0]
    self._outs = [np.empty(max(size, 1), dtype=np.float64) for size in self._out_sizes]
    self._out_ptrs = [out.ctypes.data_as(_DP) if size else _DP() for out, size in zip(self._outs, self._out_sizes, strict=True)]
    self.last_stats: SolverStats | None = None
    self.last_status: SolverStatus | None = None
    for function in ("alloy_bench_total", "alloy_bench_fe", "alloy_bench_ipopt"):
      getattr(self._dll, function).restype = ctypes.c_double
    self._dll.alloy_bench_iter_count.restype = ctypes.c_longlong
    self._dll.alloy_bench_eval_count.argtypes = [ctypes.c_int]
    self._dll.alloy_bench_eval_count.restype = ctypes.c_longlong
    self._dll.alloy_bench_native_status.argtypes = [ctypes.c_int]
    self._dll.alloy_bench_native_status.restype = ctypes.c_int
    self._dll.alloy_bench_status.argtypes = [ctypes.c_int]
    self._dll.alloy_bench_status.restype = ctypes.c_int

  def close(self) -> None:
    if getattr(self, "_mem", -1) >= 0:
      self._dll.alloy_bench_release(self._mem)
      self._mem = -1

  def __del__(self) -> None:
    self.close()

  def __call__(self, *values: np.ndarray) -> list[np.ndarray]:
    if len(values) != 8:
      raise ValueError(f"CasADi nlpsol expects 8 inputs, got {len(values)}")
    for index, pointer in enumerate(self._out_ptrs):
      self._resp[index] = pointer
    held = [np.ascontiguousarray(value, dtype=np.float64) for value in values]
    for index, (value, expected) in enumerate(zip(held, self._input_sizes, strict=True)):
      if value.size != expected:
        raise ValueError(f"CasADi nlpsol input {index} has {value.size} values, expected {expected}")
    for index, value in enumerate(held):
      self._argp[index] = value.ctypes.data_as(_DP)
    rc = int(self._solve(self._argp, self._resp, self._iw, self._w, self._mem))
    if rc:
      raise RuntimeError(f"generated CasADi nlpsol returned {rc}")
    total = float(self._dll.alloy_bench_total())
    fe = float(self._dll.alloy_bench_fe())
    ipopt = float(self._dll.alloy_bench_ipopt())
    native = int(self._dll.alloy_bench_native_status(self._mem))
    status = AlloySolveStatus(int(self._dll.alloy_bench_status(self._mem)))
    counts = [int(self._dll.alloy_bench_eval_count(index)) for index in range(len(_ORACLE_NAMES))]
    self.last_stats = SolverStats(
      version=ALLOY_SOLVER_STATS_VERSION,
      status=status,
      native_status=native,
      iter=int(self._dll.alloy_bench_iter_count()),
      obj=float(self._outs[1][0]),
      t_total=total,
      t_fe=fe,
      t_solver=max(ipopt - fe, 0.0),
      t_qp=0.0,
      t_globalization=0.0,
      t_glue=max(total - ipopt, 0.0),
      n_eval_f=counts[0],
      n_eval_g=counts[1],
      n_eval_grad_f=counts[2],
      n_eval_jac_g=counts[3],
      n_eval_h=counts[4],
    )
    self.last_status = self.last_stats.to_solver_status()
    return [out[:size].copy() for out, size in zip(self._outs, self._out_sizes, strict=True)]


class InterpretedCasadiIpopt:
  """CasADi's Python nlpsol path, reported separately from the controlled C comparison."""

  compiled = False

  def __init__(self, name, nlp, options):
    import casadi as ca

    started = time.perf_counter()
    x, p = ca.MX.sym("x", nlp.sparsity_in(0)), ca.MX.sym("p", nlp.sparsity_in(1))
    f, g = nlp.call([x, p], True, False)
    self.solver = _transformed_nlpsol(name, {"x": x, "p": p, "f": f, "g": g}, {**options, "record_time": True})
    self.build_ms = (time.perf_counter() - started) * 1000.0
    self.ipopt_library = str(Path(ca.__file__).parent / f"libipopt{shared_lib_ext()}")
    self.configured_ipopt_library = self.resolved_ipopt_library = self.ipopt_library
    self.last_stats = None
    self.last_status = None

  def __call__(self, *values):
    started = time.perf_counter()
    outputs = self.solver(*values)
    total = time.perf_counter() - started
    raw = self.solver.stats()
    native = str(raw["return_status"])
    status, native_code = {
      "Solve_Succeeded": (AlloySolveStatus.OK, 0),
      "Solved_To_Acceptable_Level": (AlloySolveStatus.ACCEPTABLE, 1),
      "Feasible_Point_Found": (AlloySolveStatus.ACCEPTABLE, 6),
      "Maximum_Iterations_Exceeded": (AlloySolveStatus.MAX_ITER, -1),
      "Maximum_CpuTime_Exceeded": (AlloySolveStatus.MAX_ITER, -4),
      "Maximum_WallTime_Exceeded": (AlloySolveStatus.MAX_ITER, -5),
      "Infeasible_Problem_Detected": (AlloySolveStatus.PRIMAL_INFEASIBLE, 2),
      "Diverging_Iterates": (AlloySolveStatus.NUMERICS, 4),
      "Search_Direction_Becomes_Too_Small": (AlloySolveStatus.NUMERICS, 3),
      "Restoration_Failed": (AlloySolveStatus.NUMERICS, -2),
      "Error_In_Step_Computation": (AlloySolveStatus.NUMERICS, -3),
      "Invalid_Number_Detected": (AlloySolveStatus.NUMERICS, -13),
      "User_Requested_Stop": (AlloySolveStatus.USER_STOP, 5),
      "Not_Enough_Degrees_Of_Freedom": (AlloySolveStatus.ERROR, -10),
      "Invalid_Problem_Definition": (AlloySolveStatus.ERROR, -11),
      "Invalid_Option": (AlloySolveStatus.ERROR, -12),
      "Unrecoverable_Exception": (AlloySolveStatus.ERROR, -100),
      "NonIpopt_Exception_Thrown": (AlloySolveStatus.ERROR, -101),
      "Insufficient_Memory": (AlloySolveStatus.ERROR, -102),
      "Internal_Error": (AlloySolveStatus.ERROR, -199),
    }.get(native, (AlloySolveStatus.ERROR, -199))
    fe = sum(float(raw.get(f"t_wall_nlp_{name}", 0.0)) for name in _ORACLE_NAMES)
    self.last_stats = SolverStats(
      version=ALLOY_SOLVER_STATS_VERSION,
      status=status,
      native_status=native_code,
      iter=int(raw["iter_count"]),
      obj=float(outputs[1]),
      t_total=total,
      t_fe=fe,
      t_solver=max(total - fe, 0.0),
      t_qp=0.0,
      t_globalization=0.0,
      t_glue=0.0,
      **{f"n_eval_{'h' if name == 'hess_l' else name}": int(raw.get(f"n_call_nlp_{name}", 0)) for name in _ORACLE_NAMES},
    )
    self.last_status = self.last_stats.to_solver_status()
    return [np.asarray(value, dtype=np.float64).reshape(-1) for value in outputs]


def make_casadi_ipopt(name, nlp, options):
  """Build the CasADi mode selected by the closed-loop runner."""
  return (InterpretedCasadiIpopt if INTERPRETED.get() else CompiledCasadiIpopt)(name, nlp, options)


class CasadiIpoptSolver:
  """`al.Function`-shaped adapter for a compiled CasADi IPOPT NLP."""

  compiled = True

  def __init__(
    self,
    name: str,
    z,
    p,
    cost,
    h_eq,
    g_ineq,
    *,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    l_ineq: np.ndarray,
    u_ineq: np.ndarray,
    options: dict[str, Any],
    expand: bool,
  ):
    import casadi as ca

    self.n_eq, self.n_ineq = int(h_eq.shape[0]), int(g_ineq.shape[0])
    constraints = ca.vertcat(h_eq, g_ineq)
    self.base = ca.Function(f"{name}_base", [z, p], [cost, h_eq, g_ineq])
    nlp = ca.Function(f"{name}_nlp", [z, p], [cost, constraints])
    self._bounds = (
      np.asarray(x_lb, dtype=np.float64),
      np.asarray(x_ub, dtype=np.float64),
      np.concatenate([np.zeros(self.n_eq), np.asarray(l_ineq, dtype=np.float64)]),
      np.concatenate([np.zeros(self.n_eq), np.asarray(u_ineq, dtype=np.float64)]),
    )
    self._compiled = make_casadi_ipopt(
      name,
      nlp,
      {
        "print_time": False,
        "expand": expand,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.warm_start_init_point": "yes",
        **options,
      },
    )
    self.compiled = self._compiled.compiled
    self.build_ms = self._compiled.build_ms
    self.ipopt_library = self._compiled.ipopt_library
    self.configured_ipopt_library = self._compiled.configured_ipopt_library
    self.resolved_ipopt_library = self._compiled.resolved_ipopt_library
    self.expand = expand
    self.last_stats: SolverStats | None = None
    self.last_status: SolverStatus | None = None

  def __call__(self, z0, lam_eq0, lam_ineq0, lam_box0, p) -> dict[str, np.ndarray]:
    lam_g0 = np.concatenate([np.asarray(lam_eq0, dtype=np.float64), np.asarray(lam_ineq0, dtype=np.float64)])
    solution = self._compiled(z0, p, *self._bounds, lam_box0, lam_g0)
    self.last_stats = self._compiled.last_stats
    self.last_status = self._compiled.last_status
    x, f, _, lam_x, lam_g, _ = solution
    _, h_eq, g_ineq = self.base(x, p)
    return {
      "x": x,
      "f": np.asarray(f, dtype=np.float64).reshape(()),
      "h_eq": np.asarray(h_eq, dtype=np.float64).reshape(-1),
      "g_ineq": np.asarray(g_ineq, dtype=np.float64).reshape(-1),
      "lam_eq": lam_g[: self.n_eq],
      "lam_ineq": lam_g[self.n_eq :],
      "lam_box": lam_x,
    }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("command", choices=["build"])
  parser.add_argument("spec", type=Path)
  args = parser.parse_args()
  _build(args.spec)


if __name__ == "__main__":
  main()
