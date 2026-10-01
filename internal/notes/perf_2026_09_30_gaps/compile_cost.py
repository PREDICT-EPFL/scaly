"""What the blocked solves cost to compile: C bytes, compile time and machine code, against a base tree.

    uv run internal/notes/perf_2026_09_30_gaps/compile_cost.py --base <worktree>/src

Each case is rendered from this checkout and from the base (its ``src`` first on the path),
compiled with the JIT's flags, and reported as bytes of C, seconds of ``cc`` (best of three) and
bytes of text. Found by the Tier 5 review: a blocked triangular solve is tens of kilobytes of C,
and derivative code through solves, which holds several, compiles several times slower (C-220).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

CASES = {
  "solve_35_11": "a, b = sc.sym('a', (35, 35)), sc.sym('b', (35, 11)); outs = [solve_triangular(a, b, lower=True)]; ins = [a, b]",
  "grad_cho_solve_50_9": "a, b = sc.sym('a', (50, 50)), sc.sym('b', (50, 9)); f = sc.sumsqr(cho_solve(cholesky(a), b)); outs = [sc.gradient(f, a)]; ins = [a, b]",
  "solve_67_19_transposed": "a, b = sc.sym('a', (67, 67)), sc.sym('b', (67, 19)); outs = [solve_triangular(a, b, lower=True, trans=True)]; ins = [a, b]",
}
RENDER = """
import sys, numpy as np
import scaly as sc
from scaly.codegen import render_c_source
from scaly.linalg import cholesky, cho_solve, solve_triangular
{case}
fn = sc.Function.from_exprs("cc_case", ins, outs, [f"i{{k}}" for k in range(len(ins))], [f"o{{k}}" for k in range(len(outs))])
open(sys.argv[1], "w").write(render_c_source(fn))
"""


def measure(case: str, src: Path | None, flags: list[str], tmp: Path, tag: str) -> tuple[int, float, int]:
  c = tmp / f"{tag}.c"
  env = {**os.environ, **({"PYTHONPATH": str(src)} if src else {})}
  subprocess.run([sys.executable, "-c", RENDER.format(case=case), str(c)], check=True, env=env, cwd=ROOT, capture_output=True)
  best = float("inf")
  for _ in range(3):
    t = time.perf_counter()
    subprocess.run(["cc", *flags, "-c", str(c), "-o", str(tmp / f"{tag}.o")], check=True)
    best = min(best, time.perf_counter() - t)
  size = subprocess.run(["size", str(tmp / f"{tag}.o")], capture_output=True, text=True, check=True).stdout.splitlines()[1].split()[0]
  return c.stat().st_size, best, int(size)


def main() -> None:
  sys.path.insert(0, str(ROOT))
  from scaly.codegen.jit import compile_flags

  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--base", type=Path, required=True)
  args = parser.parse_args()
  with tempfile.TemporaryDirectory() as d:
    for name, case in CASES.items():
      base = measure(case, args.base, compile_flags(), Path(d), f"{name}_base")
      new = measure(case, None, compile_flags(), Path(d), f"{name}_new")
      print(
        f"{name:26s} C {base[0] / 1e3:7.1f} -> {new[0] / 1e3:7.1f} KB   cc {base[1]:5.2f} -> {new[1]:5.2f} s   text {base[2] / 1e3:6.1f} -> {new[2] / 1e3:6.1f} KB",
        flush=True,
      )


if __name__ == "__main__":
  main()
