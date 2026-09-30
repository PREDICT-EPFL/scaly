"""The chain's stage Hessian and the instruction cache, natively (A1's measurement).

    uv run internal/notes/perf_2026_09_30_gaps/chain_cliff.py [M ...]

For each number of masses M (default 3 5 7 9): the benchmark's Lagrangian Hessian (lower triangle,
horizon 40) rendered for the host, compiled at the JIT's flags and at ``-Os``, and timed from C
with ``../perf_2026_09_30_codegen/time_entry.c`` (fastest of 9 samples, after a 200 ms warm-up).
Prints the colouring width, the multiplications in the largest procedure's C (the stage body), that
procedure's machine code (address differences in the object's symbol table), the time and the time
per 1 000 stage multiplications. Writes ``results/chain_cliff.json``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "perf_2026_09_30_codegen"))
BUILD = HERE / "build" / "chain"


def hessian(m: int):
  import scaly as sc
  from bench.problems import chain as ch
  from scaly.function import ConcreteFunction
  from scaly.opt.nlp import nlp_oracles

  n = ch.HORIZON

  @sc.opt.problem(vars=sc.L("z", ch.n_dec(m, n)), params=sc.L("p", ch.n_param(m)), name=f"chain_M{m}")
  def problem(z, p):
    return sc.opt.ProblemSpec(minimize=ch._objective(z, m, n), eq=(ch._chain_eq_expr(z, p, m, n),))

  o = nlp_oracles(problem)
  h = o.hess.triangle("lower")
  fn = ConcreteFunction.from_exprs(f"chain_hess_M{m}", o.hess_inputs, (h.values,), tuple(f"in{i}" for i in range(len(o.hess_inputs))), ("h",))
  rng = np.random.default_rng(0)
  args = [rng.standard_normal(e.shape) * 0.1 for e in o.hess_inputs]
  args[1][-5:] = [0.033, 1.0, 0.033, -9.81, 0.2]  # mass, spring, rest length, gravity, step
  args[0] = args[0] + 0.1
  return fn, args, h.coloring_width


def procedures(src: str) -> dict[str, str]:
  heads = [(m.start(), m.group(1)) for m in re.finditer(r"^(?:static )?(?:inline )?(?:__attribute__\(\([^)]*\)\) )*(?:void|int) (\w+)\([^;]*\{\s*$", src, re.M)]
  heads.append((len(src), ""))
  return {name: src[a:b] for (a, name), (b, _) in zip(heads, heads[1:], strict=False)}


def code_sizes(obj: Path) -> dict[str, int]:
  """Bytes of machine code per function symbol, from the sorted addresses of the text section."""
  out = subprocess.run(["nm", "-n", "-U", str(obj)], capture_output=True, text=True, check=True).stdout
  fields = (line.split() for line in out.splitlines())
  syms = [(int(a, 16), name.lstrip("_")) for a, kind, name in (f for f in fields if len(f) == 3) if kind in "Tt" and not name.startswith("ltmp")]
  sections = subprocess.run(["size", "-m", str(obj)], capture_output=True, text=True, check=True).stdout
  end = int(re.search(r"\(__TEXT, __text\): (\d+)", sections).group(1))
  bounds = [a for a, _ in syms] + [end]
  return {name: bounds[i + 1] - a for i, (a, name) in enumerate(syms)}


def main() -> None:
  from corpus import blob
  from scaly.codegen import render_c_module
  from scaly.codegen.jit import compile_flags

  ms = [int(v) for v in sys.argv[1:]] or [3, 5, 7, 9]
  BUILD.mkdir(parents=True, exist_ok=True)
  driver = HERE / "build" / "time_entry"
  if not driver.exists():
    subprocess.run(["cc", "-O2", str(HERE.parent / "perf_2026_09_30_codegen" / "time_entry.c"), "-o", str(driver)], check=True)
  rows = []
  for m in ms:
    fn, args, width = hessian(m)
    module = render_c_module(fn)
    c = BUILD / f"chain_{m}.c"
    c.write_text(module.body)
    procs = procedures(module.body)
    stage = max(procs, key=lambda name: len(procs[name]))
    mults = procs[stage].count("*")
    (BUILD / f"chain_{m}.bin").write_bytes(blob([np.ravel(a) for a in args], [int(fn.outputs[0].size)], int(module.workspace_size)))
    row = {"M": m, "colours": width, "stage": stage, "stage_mults": mults}
    for tag, flags in (("O2", list(compile_flags())), ("Os", ["-Os", *[f for f in compile_flags() if not f.startswith("-O")]])):
      obj, lib = BUILD / f"chain_{m}_{tag}.o", BUILD / f"chain_{m}_{tag}.so"
      subprocess.run(["cc", *flags, "-c", str(c), "-o", str(obj)], check=True)
      subprocess.run(["cc", *flags, "-fPIC", "-shared", str(c), "-lm", "-o", str(lib)], check=True)
      sizes = code_sizes(obj)
      row[f"stage_code_{tag}"] = next((v for k, v in sizes.items() if k.startswith(stage)), None)
      t = time.perf_counter()
      while time.perf_counter() - t < 0.2:
        subprocess.run([str(driver), str(lib), fn.name.replace(":", "_"), str(BUILD / f"chain_{m}.bin"), "3", "/dev/null"], capture_output=True, check=True)
      best = min(
        float(subprocess.run([str(driver), str(lib), fn.name, str(BUILD / f"chain_{m}.bin"), "5", "/dev/null"], capture_output=True, text=True, check=True).stdout.split()[0])
        for _ in range(9)
      )
      row[f"time_us_{tag}"] = best / 1e3
      row[f"us_per_1000_mults_{tag}"] = best / 1e3 / (mults / 1000)
    rows.append(row)
    print(
      f"M={m} colours {width:3d} stage mults {mults:6d} code O2 {row['stage_code_O2'] / 1024:6.0f} KiB Os {row['stage_code_Os'] / 1024:6.0f} KiB  "
      f"time O2 {row['time_us_O2']:8.1f} us Os {row['time_us_Os']:8.1f} us  per 1000 mults O2 {row['us_per_1000_mults_O2']:5.1f} Os {row['us_per_1000_mults_Os']:5.1f}",
      flush=True,
    )
  (HERE / "results").mkdir(exist_ok=True)
  (HERE / "results" / "chain_cliff.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
  main()
