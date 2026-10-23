"""Benchmark generation, correctness, and sweep helpers."""

from scaly.function.concrete import ConcreteFunction

from scaly.function.model import as_concrete


from pathlib import Path
from functools import lru_cache
import os
import re
import struct
import subprocess
import sys
from weakref import WeakKeyDictionary
from scaly.codegen.jit import HOST_CFLAGS
from scaly.codegen.toolchain import VectorLibm, find_c_compiler, native_recipe

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"
CLOSED_LOOP_RESULTS = RESULTS / "closed-loop"
SMOKE_RESULTS = RESULTS / "smoke"
SWEEP_RESULTS = RESULTS / "sweep"
# Every kernel the harness compiles targets the machine that runs it, with the JIT's own flags. Only
# distributed binaries (the solver plugin wheels) stay at the portable x86-64 baseline.
NATIVE_CFLAGS = HOST_CFLAGS


def vector_libm() -> VectorLibm:
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
  if sys.platform == "win32":
    _, symbols = _coff_object(path)
    return sorted({name for name, section, _ in symbols if section == 0 and "_ZGV" in name})
  result = subprocess.run(["nm", "-u", str(path)], check=True, text=True, capture_output=True)
  return sorted({line.split()[-1] for line in result.stdout.splitlines() if "_ZGV" in line})


# Mach-O keeps read-only data in these __TEXT sections, and Apple's `size` and `nm` lack `-A` and `-S`.
_MACHO_RODATA = re.compile(r"Section \(__TEXT, __(?:const|literal\d+|cstring)\): (\d+)")
_MACHO_RODATA_SYMBOL = re.compile(r"\(__TEXT,__(?:const|literal\d+|cstring)\) .* _\w+$")


def _coff_object(path: Path) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
  data = path.read_bytes()
  machine, count, _, symbol_offset, symbol_count, optional_size, _ = struct.unpack_from("<HHIIIHH", data)
  if machine != 0x8664 or optional_size:
    raise ValueError(f"expected an x86-64 COFF object: {path}")
  strings = data[symbol_offset + 18 * symbol_count :]

  def string(offset: int) -> str:
    return strings[offset : strings.index(b"\0", offset)].decode()

  sections = []
  for index in range(count):
    offset = 20 + index * 40
    raw = data[offset : offset + 8].rstrip(b"\0")
    name = string(int(raw[1:])) if raw.startswith(b"/") else raw.decode()
    size = struct.unpack_from("<I", data, offset + 16)[0]
    flags = struct.unpack_from("<I", data, offset + 36)[0]
    sections.append((name, size, flags))
  symbols = []
  index = 0
  while index < symbol_count:
    offset = symbol_offset + index * 18
    raw, _, section, _, storage, auxiliaries = struct.unpack_from("<8sIhHBB", data, offset)
    name = string(struct.unpack_from("<I", raw, 4)[0]) if raw[:4] == b"\0" * 4 else raw.rstrip(b"\0").decode()
    # Section-definition symbols describe storage, not the source objects it contains.
    if not (storage == 3 and auxiliaries and name.startswith(".")):
      symbols.append((name, section, storage))
    index += 1 + auxiliaries
  return sections, symbols


def object_bytes(path: Path) -> tuple[int, int]:
  """The total `size` reports for a compiled object, which counts machine code, constants and static
  data, and the part of it in read-only data sections."""
  if sys.platform == "win32":
    sections, _ = _coff_object(path)
    return (
      sum(size for _, size, flags in sections if flags & 0xE0 and not flags & 0x02000000),
      sum(size for name, size, _ in sections if name.startswith(".rdata")),
    )
  header, values = subprocess.run(["size", str(path)], check=True, text=True, capture_output=True).stdout.splitlines()[:2]
  total = int(values.split()[header.split().index("dec")])
  if sys.platform == "darwin":
    sections = subprocess.run(["size", "-m", str(path)], check=True, text=True, capture_output=True).stdout
    return total, sum(int(n) for n in _MACHO_RODATA.findall(sections))
  sections = subprocess.run(["size", "-A", str(path)], check=True, text=True, capture_output=True).stdout.splitlines()
  return total, sum(int(line.split()[1]) for line in sections if line.startswith(".rodata"))


_STATIC = re.compile(rb'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|#(?:\\\n|[^\n])*|\bstatic\b', re.S)
_TOKEN = re.compile(rb'\s+|/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\\n])*"|\w+|.', re.S)
_BRACE = re.compile(rb'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|[{};]', re.S)


def constant_tables(source: bytes) -> list[int]:
  """The length of every `static const` array definition with an initializer, at file or function scope,
  from `static` through the closing `;`. Comments, literals and preprocessor lines are skipped, so line
  breaks and spacing inside a definition do not matter."""
  lengths, pos = [], 0
  while m := _STATIC.search(source, pos):
    pos = m.end()
    if m[0] != b"static":
      continue
    words, depth = set(), 0
    for t in _TOKEN.finditer(source, pos):
      if depth == 0 and t[0] in (b"=", b";", b"{"):
        break
      depth += (t[0] in (b"(", b"[")) - (t[0] in (b")", b"]"))
      words.add(t[0])
    else:
      raise ValueError(f"unterminated declaration at byte {m.start()}")
    pos = t.end()
    if t[0] != b"=" or b"[" not in words or b"const" not in words:
      continue
    depth = 0
    for t in _BRACE.finditer(source, pos):
      if depth == 0 and t[0] == b";":
        break
      depth += (t[0] == b"{") - (t[0] == b"}")
    else:
      raise ValueError(f"unterminated initializer at byte {m.start()}")
    lengths.append(t.end() - m.start())
    pos = t.end()
  return lengths


def object_constant_symbols(path: Path) -> int:
  """The named read-only data objects in a compiled object. Each is a source table the compiler kept, so
  the count can never exceed the number of tables `constant_tables` finds in the source."""
  if sys.platform == "win32":
    sections, symbols = _coff_object(path)
    return sum(
      1
      for name, section, storage in symbols
      if section > 0 and sections[section - 1][0].startswith(".rdata") and storage in (2, 3) and not name.startswith((".L", "__real@", "__xmm@"))
    )
  if sys.platform == "darwin":
    lines = subprocess.run(["nm", "-m", str(path)], check=True, text=True, capture_output=True).stdout.splitlines()
    return sum(1 for line in lines if _MACHO_RODATA_SYMBOL.search(line))
  lines = subprocess.run(["nm", "-S", str(path)], check=True, text=True, capture_output=True).stdout.splitlines()
  return sum(1 for p in map(str.split, lines) if len(p) == 4 and p[2] in "rR" and not p[3].startswith(".L"))


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


_BASES: WeakKeyDictionary = WeakKeyDictionary()


def register_base(function, base) -> None:
  """Record the numerical cost and constraint evaluator that result checks use for a solver
  Function whose descriptor holds no Scaly base, such as one driving CasADi oracles."""
  _BASES[function] = base


def registered_base(function):
  """The evaluator ``register_base`` recorded for ``function``, or ``None``."""
  return _BASES.get(function)


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
    evaluator = registered_base(solver.function)
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


class _Captured(Exception):
  pass


def _casadi_encoding(fn, *, expand: bool = False) -> str:
  import casadi as ca

  if expand or fn.is_a("SXFunction"):
    return "sx"
  return "call_mx" if any(fn.instruction_id(k) == ca.OP_CALL for k in range(fn.n_instructions())) else "mx"


def check_fastest_casadi_encoding(workload: str, size: int, *builds) -> None:
  """Each closed-loop CasADi controller in ``builds`` uses the encoding the published sweep found
  fastest for ``workload`` at ``size``. A build stops when it hands its functions to a code generator."""
  import csv
  from unittest import mock

  import scaly_sqp.casadi

  from benchmarks.harness import casadi_ipopt

  rows = [
    row
    for row in csv.DictReader((ROOT / "docs/assets/benchmarks/sweep.summary.csv").open())
    if row["workload"] == workload and row["size"] == str(size) and row["backend"].startswith("casadi_") and row["runtime_ns_mean"]
  ]
  fastest = min(rows, key=lambda row: float(row["runtime_ns_mean"]))["backend"].removeprefix("casadi_")

  for build in builds:
    seen: set[str] = set()

    def sqp(**kwargs):
      seen.update(_casadi_encoding(kwargs[key]) for key in ("base", "grad", "jac", "hess"))
      raise _Captured

    def ipopt(name, nlp, options):
      seen.add(_casadi_encoding(nlp, expand=bool(options.get("expand"))))
      raise _Captured

    with mock.patch.object(scaly_sqp.casadi, "build_casadi_external_sqp", sqp), mock.patch.object(casadi_ipopt, "make_casadi_ipopt", ipopt):
      try:
        build()
      except _Captured:
        pass
      else:
        raise AssertionError(f"{build} never reached CasADi code generation")
    assert seen == {fastest}, f"{workload} at size {size} uses CasADi {sorted(seen)}, but the sweep's fastest is {fastest}"
