"""Benchmark generation, correctness, and sweep helpers."""

from scaly.function.concrete import ConcreteFunction

from scaly.function.model import as_concrete


from pathlib import Path
from functools import lru_cache
import os
import subprocess
from scaly.codegen.jit import HOST_CFLAGS
from scaly.codegen.toolchain import find_c_compiler, native_recipe

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"
CLOSED_LOOP_RESULTS = RESULTS / "closed-loop"
SMOKE_RESULTS = RESULTS / "smoke"
SWEEP_RESULTS = RESULTS / "sweep"
# Every kernel the harness compiles targets the machine that runs it, with the JIT's own flags. Only
# distributed binaries (the solver plugin wheels) stay at the portable x86-64 baseline.
NATIVE_CFLAGS = HOST_CFLAGS


def vector_libm() -> str:
  """Resolve the benchmark math policy, retaining an explicit scalar override."""
  policy = os.environ.get("SCALY_VECTOR_LIBM")
  if policy is None:
    compiler = find_c_compiler()
    return native_recipe(compiler.command).vector_libm if compiler is not None else "none"
  if policy not in {"none", "glibc"}:
    raise ValueError(f"SCALY_VECTOR_LIBM must be 'none' or 'glibc', got {policy!r}")
  return policy


@lru_cache
def compiler_version(*command: str) -> str:
  """Read the compiler identity used to select supported math flags."""
  return subprocess.run([*command, "--version"], check=True, text=True, capture_output=True).stdout.splitlines()[0]


def math_flags(*command: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
  """Return the supported vector-library selection flag and required libraries."""
  if vector_libm() == "none":
    return (), ()
  flags = ("-fveclib=libmvec",) if "clang" in compiler_version(*command).lower() else ()
  return flags, ("-lmvec",)


def configure_math_policy(*, measured_jit: bool = True) -> None:
  """Apply the benchmark math policy to JIT children and reject unequal compiler flags."""
  os.environ.setdefault("SCALY_VECTOR_LIBM", vector_libm())
  compiler = find_c_compiler()
  if compiler is not None and vector_libm() == "glibc" and native_recipe(compiler.command).vector_libm != "glibc":
    raise ValueError("SCALY_VECTOR_LIBM=glibc requires a glibc x86-64 host with vector math support")
  if measured_jit and compiler is not None and math_flags(*compiler.command)[0]:
    raise ValueError("vector-libm closed-loop builds require GCC: Scaly JIT does not pass Clang's -fveclib=libmvec; set SCALY_CC=gcc")


def vector_symbols(path: Path) -> list[str]:
  """Read actual vector-math references from a compiled object or shared library."""
  result = subprocess.run(["nm", "-u", str(path)], check=True, text=True, capture_output=True)
  return sorted({line.split()[-1] for line in result.stdout.splitlines() if "_ZGV" in line})


CLOSED_LOOP_PAIRS: dict[str, tuple[tuple[str, str | None], ...]] = {
  "chain": (("ipopt", "scaly"), ("sqp", "scaly"), ("sqp", "casadi")),
  "race_cars": (("ipopt", "scaly"), ("ipopt", "casadi"), ("sqp", "scaly"), ("sqp", "casadi")),
  "unbumpercars": (("ipopt", "scaly"), ("ipopt", "casadi"), ("sqp", "scaly"), ("sqp", "casadi"), ("none", None)),
  "npmpc": (("ipopt", "scaly"), ("ipopt", "casadi"), ("sqp", "scaly"), ("sqp", "casadi")),
}


def closed_loop_results_root(*, smoke: bool, out_dir: Path | None = None) -> Path:
  if out_dir is not None:
    return out_dir
  return SMOKE_RESULTS / "closed-loop" if smoke else CLOSED_LOOP_RESULTS


def solver_oracle_name(solver: str, oracle: str | None) -> str:
  if solver == "none":
    return solver
  if oracle is None:
    raise ValueError(f"solver {solver!r} requires an oracle")
  return f"{solver}+{oracle}"


def solve_problem(solver, x0, lam_eq, lam_ineq, lam_box, params):
  """Run either a typed Scaly ``Solver`` or the benchmark CasADi adapter."""
  import numpy as np
  import scaly as sc

  if not isinstance(solver, sc.Solver):
    return solver(x0, lam_eq, lam_ineq, lam_box, params)

  x, lam_box, lam_eq, lam_ineq = solver(params, warm=(x0, lam_box, lam_eq, lam_ineq))
  descriptor = as_concrete(solver.function).descriptor
  base = descriptor.base
  if isinstance(base, ConcreteFunction):
    values = base.numerical_call(np.asarray(x).reshape(-1), params)
  else:
    evaluator = getattr(solver.function, "_benchmark_base", None)
    if evaluator is None:
      raise TypeError(f"solver {solver.function.name!r} has no numerical benchmark oracle")
    values = evaluator(np.asarray(x).reshape(-1), params)
  if isinstance(values, tuple):
    cost, constraints = values
  else:
    cost, constraints = values, np.zeros(0)
  constraints = np.asarray(constraints).reshape(-1)
  return {
    "x": x,
    "f": np.asarray(cost).reshape(()),
    "h_eq": constraints[: descriptor.n_eq],
    "g_ineq": constraints[descriptor.n_eq :],
    "lam_eq": lam_eq,
    "lam_ineq": lam_ineq,
    "lam_box": lam_box,
  }


def problem_stats(solver):
  """Return stats from either a typed Scaly ``Solver`` or the CasADi adapter."""
  import scaly as sc

  return solver.stats() if isinstance(solver, sc.Solver) else solver.last_stats
