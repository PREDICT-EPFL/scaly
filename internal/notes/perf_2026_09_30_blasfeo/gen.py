"""Generate and compile Scaly's dense kernels (gemm ``a @ b``, ``cholesky``, lower ``solve_triangular``
with n right-hand sides) for the BLASFEO comparison, one per order ``n``.

    PYTHONPATH=src python internal/notes/perf_2026_09_30_blasfeo/gen.py [n ...]

Per op and n: the Function over n x n inputs (row-major, Scaly's ABI), its C (``render_c_module``),
and two shared libraries: the JIT's own flags (``compile_flags()``) and ``-O3 -mcpu=native
-fno-math-errno``. Writes ``build/<op>_<n>/{kernel.c, jit.so, o3.so, meta.json}``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
O3 = ("-O3", "-mcpu=native", "-fno-math-errno")


def text_bytes(lib: Path) -> int:
  out = subprocess.run(["size", "-A", str(lib)], capture_output=True, text=True, check=True).stdout
  return sum(int(line.split()[1]) for line in out.splitlines() if line.startswith(".text"))


def one(op: str, n: int) -> dict:
  import scaly as sc
  from scaly.codegen import render_c_module
  from scaly.codegen.jit import compile_flags
  from scaly.codegen.toolchain import find_c_compiler
  from scaly.linalg import cholesky, solve_triangular

  t0 = time.perf_counter()
  a = sc.sym("a", (n, n))
  b = sc.sym("b", (n, n))
  if op == "gemm":
    ins, names, out = [a, b], ["a", "b"], a @ b
  elif op == "potrf":
    ins, names, out = [a], ["a"], cholesky(a)
  elif op == "trsm":
    ins, names, out = [a, b], ["a", "b"], solve_triangular(a, b, lower=True)
  else:
    raise ValueError(op)
  name = f"bf_{op}_{n}"
  fn = sc.Function.from_exprs(name, ins, [out], names, ["o"])
  build_s = time.perf_counter() - t0
  t0 = time.perf_counter()
  module = render_c_module(fn)
  generate_s = time.perf_counter() - t0
  d = BUILD / f"{op}_{n}"
  d.mkdir(parents=True, exist_ok=True)
  src = d / "kernel.c"
  src.write_text(module.body)
  cc = find_c_compiler().cc
  meta = {"op": op, "n": n, "symbol": name, "build_s": build_s, "generate_s": generate_s, "workspace": int(module.workspace_size),
          "c_bytes": len(module.body.encode()), "c_lines": sum(1 for line in module.body.splitlines() if line.strip()),
          "cc": cc, "jit_flags": list(compile_flags()), "o3_flags": list(O3)}
  for tag, flags in (("jit", tuple(compile_flags())), ("o3", O3)):
    lib = d / f"{tag}.so"
    t0 = time.perf_counter()
    subprocess.run([cc, *flags, "-fPIC", "-shared", str(src), *module.link_flags, "-lm", "-o", str(lib)], check=True)
    meta[f"{tag}_compile_s"] = time.perf_counter() - t0
    meta[f"{tag}_text_bytes"] = text_bytes(lib)
  (d / "meta.json").write_text(json.dumps(meta, indent=1))
  return meta


def main() -> None:
  ns = [int(x) for x in sys.argv[1:]] or [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
  for n in ns:
    for op in ("gemm", "potrf", "trsm"):
      m = one(op, n)
      print(f"{op:5} n={n:4d} build {m['build_s']*1e3:7.1f} ms gen {m['generate_s']*1e3:7.1f} ms cc(jit) {m['jit_compile_s']*1e3:7.1f} ms "
            f"cc(o3) {m['o3_compile_s']*1e3:7.1f} ms  C {m['c_bytes']:8d} B  text jit {m['jit_text_bytes']:7d} o3 {m['o3_text_bytes']:7d}  ws {m['workspace']}", flush=True)


if __name__ == "__main__":
  main()
