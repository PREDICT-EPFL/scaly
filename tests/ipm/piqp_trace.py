"""Vendored PIQP as a trace oracle: build the C driver, run it on a problem, parse what it prints.

``run(qp)`` returns PIQP's per-iteration table (its verbose output, printed with 4 to 6 significant
digits) and its final result at full precision. PIQP 0.6.2 is deterministic and starts from
scratch on every solve, so ``run(qp, max_iter=k)`` returns the state after exactly ``k``
iterations of the full run: ``exact_trace`` uses that to read every iteration at full precision,
the step parameter ``sigma`` included, which the table leaves out.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse

from scaly.codegen.jit import compile_flags
from scaly.codegen.toolchain import cache_root, find_c_compiler
from scaly.solvers.paths import backend_compile_flags
from tests.ipm.problems import QP

SOURCE = Path(__file__).with_name("piqp_trace.c")
COLUMNS = ("iter", "primal_obj", "dual_obj", "duality_gap", "primal_res", "dual_res", "rho", "delta", "mu", "primal_step", "dual_step")
_ROW = re.compile(r"^\s*\d+(\s+[-+]?\d\.\d+(e[-+]\d+)?){10}\s*$")


@dataclass
class Trace:
  status: int
  backend: str  # "dense" or "sparse", as PIQP reports it
  table: np.ndarray  # one row per printed iteration, columns as ``COLUMNS``
  info: dict[str, float] = field(default_factory=dict)
  vectors: dict[str, np.ndarray] = field(default_factory=dict)

  def column(self, name: str) -> np.ndarray:
    return self.table[:, COLUMNS.index(name)]


def driver() -> Path:
  """The compiled driver, built once per source and toolchain."""
  compiler = find_c_compiler()
  if compiler is None:
    raise RuntimeError("no C compiler for the PIQP trace driver")
  flags = [*compile_flags(), *backend_compile_flags(["piqp"])]
  key = hashlib.sha256((SOURCE.read_text() + "\0" + compiler.cc + "\0" + " ".join(flags)).encode()).hexdigest()[:16]
  exe = cache_root() / "piqp_trace" / key / "piqp_trace"
  if not exe.exists():
    exe.parent.mkdir(parents=True, exist_ok=True)
    tmp = exe.with_suffix(f".{id(exe)}.tmp")
    subprocess.run([compiler.cc, str(SOURCE), *flags, "-lm", "-o", str(tmp)], check=True, capture_output=True, text=True)
    tmp.replace(exe)
  return exe


def _csc(a: sparse.csc_array, rows: int, cols: int) -> list[np.ndarray]:
  a = sparse.csc_array(a, shape=(rows, cols))
  a.sum_duplicates()
  a.sort_indices()
  return [a.indptr.astype(np.int32), a.indices.astype(np.int32), a.data.astype(np.float64)]


def write_problem(qp: QP, path: Path, *, dense: bool = False, max_iter: int | None = None) -> None:
  n, p, m = qp.n, qp.A.shape[0], qp.G.shape[0]
  parts = [_csc(sparse.triu(qp.P, format="csc"), n, n), [qp.c], _csc(qp.A, p, n), [qp.b], _csc(qp.G, m, n), [qp.h_l, qp.h_u, qp.x_l, qp.x_u]]
  nnz = [parts[0][2].size, parts[2][2].size, parts[4][2].size]
  header = np.array([n, p, m, *nnz, int(dense), -1 if max_iter is None else int(max_iter)], dtype=np.int64)
  with open(path, "wb") as f:
    f.write(header.tobytes())
    for group in parts:
      for arr in group:
        f.write(np.ascontiguousarray(arr, dtype=arr.dtype if arr.dtype == np.int32 else np.float64).tobytes())


def parse(output: str) -> Trace:
  head, _, tail = output.partition("SCALY_TRACE_RESULT\n")
  rows = [[float(v) for v in line.split()] for line in head.splitlines() if _ROW.match(line)]
  backend = next((line.split()[0] for line in head.splitlines() if line.endswith(")") and " backend (" in line), "")
  trace = Trace(status=0, backend=backend, table=np.array(rows, dtype=np.float64).reshape(-1, len(COLUMNS)))
  for line in tail.splitlines():
    kind, name, *values = line.split()
    if kind == "info":
      trace.info[name] = float(values[0])
    elif kind == "vec":
      trace.vectors[name] = np.array([float(v) for v in values], dtype=np.float64)
  trace.status = int(trace.info["status"])
  return trace


def run(qp: QP, *, dense: bool = False, max_iter: int | None = None, workdir: Path | None = None) -> Trace:
  workdir = workdir or cache_root() / "piqp_trace" / "problems"
  workdir.mkdir(parents=True, exist_ok=True)
  path = workdir / f"{qp.name}_{'dense' if dense else 'sparse'}_{max_iter}.bin"
  write_problem(qp, path, dense=dense, max_iter=max_iter)
  out = subprocess.run([str(driver()), str(path)], check=True, capture_output=True, text=True)
  return parse(out.stdout)


def exact_trace(qp: QP, *, dense: bool = False, fields: tuple[str, ...] = ("rho", "delta", "mu", "sigma", "primal_step", "dual_step")) -> np.ndarray:
  """``fields`` after each iteration ``1 .. iter`` of the full run, at full precision: one row per iteration."""
  full = run(qp, dense=dense)
  return np.array([[run(qp, dense=dense, max_iter=k).info[f] for f in fields] for k in range(1, int(full.info["iter"]) + 1)])
