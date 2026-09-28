"""Shared by the interp benchmarks: build a Scaly Function or a CasADi Function into a shared library
with the JIT's flags, and time its universal entry in C with `../perf_2026_09_27_integrators/time_entry.c`."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD = HERE / "build"
DRIVER_SRC = HERE.parent / "perf_2026_09_27_integrators" / "time_entry.c"


def blob(flat: list[np.ndarray], out_sizes: list[int], workspace: int) -> bytes:
  head = np.array([len(flat), len(out_sizes), workspace, *(a.size for a in flat), *out_sizes], dtype="<i8").tobytes()
  return head + b"".join(np.ascontiguousarray(a, dtype="<f8").tobytes() for a in flat)


def _compile(source: Path, out: Path) -> float:
  from scaly.codegen.jit import compile_flags
  from scaly.codegen.toolchain import find_c_compiler

  t0 = time.perf_counter()
  subprocess.run([find_c_compiler().cc, *compile_flags(), "-fPIC", "-shared", str(source), "-lm", "-o", str(out)], check=True)
  return time.perf_counter() - t0


def build_scaly(fn, args: list[np.ndarray], out: Path) -> dict:
  from scaly.codegen import render_c_module
  from scaly.codegen.abi import c_ident

  t0 = time.perf_counter()
  module = render_c_module(fn)
  generate = time.perf_counter() - t0
  out.mkdir(parents=True, exist_ok=True)
  (out / "f.c").write_text(module.body)
  compile_s = _compile(out / "f.c", out / "lib.so")
  shaped = [a.reshape(i.shape) for a, i in zip(args, fn.inputs, strict=True)]
  values = fn(shaped[0]) if len(shaped) == 1 else fn(tuple(shaped))  # a traced Function takes its leaves as one tuple
  values = values if isinstance(values, tuple) else (values,)
  (out / "inputs.bin").write_bytes(blob(args, [np.size(v) for v in values], int(module.workspace_size)))
  np.save(out / "expected.npy", np.concatenate([np.ravel(v) for v in values]))
  body = module.body
  return {"symbol": c_ident(fn.name), "generate_s": generate, "compile_s": compile_s, "c_lines": sum(1 for line in body.splitlines() if line.strip()), "c_bytes": len(body)}


def build_casadi(fn, args: list[np.ndarray], out: Path) -> dict:
  import casadi as ca

  out.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  gen = ca.CodeGenerator("f.c", {"casadi_int": "long long", "casadi_real": "double"})
  gen.add(fn)
  gen.generate(str(out) + "/")
  generate = time.perf_counter() - t0
  compile_s = _compile(out / "f.c", out / "lib.so")
  # CasADi uses the argument and result arrays past n_in and n_out as pointer scratch for nested
  # calls: empty dummy slots up to sz_arg and sz_res give it room.
  sizes = [fn.nnz_out(i) for i in range(fn.n_out())] + [0] * (fn.sz_res() - fn.n_out())
  padded = [*args, *(np.zeros(0) for _ in range(fn.sz_arg() - fn.n_in()))]
  assert fn.sz_iw() <= 1 << 16, fn.sz_iw()
  (out / "inputs.bin").write_bytes(blob(padded, sizes, fn.sz_w()))
  body = (out / "f.c").read_text()
  return {"symbol": fn.name(), "generate_s": generate, "compile_s": compile_s, "c_lines": sum(1 for line in body.splitlines() if line.strip()), "c_bytes": len(body)}


def driver() -> Path:
  exe = BUILD / "time_entry"
  if not exe.exists() or exe.stat().st_mtime < DRIVER_SRC.stat().st_mtime:
    BUILD.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cc", "-O2", "-o", str(exe), str(DRIVER_SRC)], check=True)
  return exe


def time_cell(cell: Path, meta: dict, reps: int, inner: int) -> tuple[float, np.ndarray]:
  """The fastest of ``reps`` samples of ``inner`` back-to-back calls, per call, in ns; and the outputs."""
  cmd = [str(driver()), str(cell / "lib.so"), meta["symbol"], str(cell / "inputs.bin"), str(reps), str(cell / "outputs.bin"), str(inner)]
  out = subprocess.run(cmd, capture_output=True, text=True, check=True)
  return float(out.stdout.split()[0]), np.fromfile(cell / "outputs.bin", dtype="<f8")
