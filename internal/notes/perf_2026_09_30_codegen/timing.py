"""Time the kernels ``corpus.py`` built, from C, interleaving cells round by round.

    uv run internal/notes/perf_2026_09_30_codegen/timing.py --cells base/jit,base/O3,new/jit [--kernels tiny] [--rounds 5]

A cell is ``<corpus variant>/<compile variant>``: the generated C of one checkout (``build/<corpus
variant>/``) compiled one way. The compile variants are diagnostics of where time goes:

- ``jit``: the JIT's own flags (``scaly.codegen.jit.compile_flags()``), the reference;
- ``O3``: ``-O3`` in place of ``-O2``;
- ``fast``: the JIT's flags plus ``-ffast-math`` (reassociation, reciprocals: an upper bound on
  what value-changing arithmetic rewrites could buy, never a candidate itself);
- ``contract``, ``assoc``, ``recip``, ``finite``, ``nosz``: one part of ``-ffast-math`` each
  (``-ffp-contract=fast``; ``-fassociative-math`` with what it needs; ``-freciprocal-math``;
  ``-ffinite-math-only``; ``-fno-signed-zeros``), to say which property a kernel's gain needs;
  ``noerrno_contract_off``: ``-ffp-contract=off``, no contraction at all;
- ``inline``: every ``noinline`` callee made ``static inline``;
- ``fmaximum``: the NaN-propagating max/min selects over plain names as clang's
  ``__builtin_elementwise_maximum``/``minimum`` (one ``fmax``/``fmin`` instruction on AArch64);
- ``restrict``: every pointer parameter of a callee ``restrict``-qualified (only safe where no
  call site aliases; the output check says whether these inputs caught one);
- ``erestrict``: the entry's ABI pointers (``arg[i]``, ``res[i]``, ``w``) read once into
  ``restrict`` locals, which is what an ABI rule that no two buffers overlap would allow;
- ``arestrict``: both; ``arestrict_inline``: both, and every callee inline.

Each round calls every cell of every kernel through ``time_entry.c`` for about ``--budget``
seconds, each sample the mean of a batch of back-to-back calls sized to take about 20 us; a
cell's time is its fastest sample over all rounds. Prints a Markdown table with each cell's time
as a ratio of the first cell's and its largest relative difference from ``expected.bin`` (the
JIT's own outputs); writes ``build/timing_<cells>.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
BUILD = HERE / "build"
DRIVER_SRC = HERE / "time_entry.c"


def driver() -> Path:
  exe = BUILD / "time_entry"
  if not exe.exists() or exe.stat().st_mtime < DRIVER_SRC.stat().st_mtime:
    BUILD.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cc", "-O2", "-o", str(exe), str(DRIVER_SRC)], check=True)
  return exe


def _inline(src: str) -> str:
  return src.replace("static __attribute__((noinline)) void", "static inline void")


def _restrict(src: str) -> str:
  def qualify(m: re.Match) -> str:
    params = re.sub(r"(double|float|int64_t|int32_t|bool)\* (\w+)", r"\1* restrict \2", m.group(2))
    return f"{m.group(1)}({params}) {{"

  return re.sub(r"^(static [^(]*_raw)\(([^)]*)\) \{", qualify, src, flags=re.M)


def _entry_restrict(src: str) -> str:
  """The entry's ABI pointers as ``restrict`` locals: ``arg[i]``, ``res[i]`` and ``w``."""
  lines = src.split("\n")
  start = next(
    i for i, line in enumerate(lines) if re.match(r"^int \w+\(const double\*\* arg, double\*\* res, int\* iw, double\* w, int mem\) \{", line)
  )
  end = next(i for i in range(start, len(lines)) if lines[i] == "}")
  body = lines[start + 1 : end]
  last_check = max(i for i, line in enumerate(body) if line.startswith("  if (!") or line.startswith("  (void)"))
  text = "\n".join(body[last_check + 1 :])
  n_arg = 1 + max((int(m) for m in re.findall(r"arg\[(\d+)\]", "\n".join(body))), default=-1)
  n_res = 1 + max((int(m) for m in re.findall(r"res\[(\d+)\]", "\n".join(body))), default=-1)
  text = re.sub(r"\barg\[(\d+)\]", r"scaly_a\1", text)
  text = re.sub(r"\bres\[(\d+)\]", r"scaly_r\1", text)
  uses_w = re.search(r"\bw\b", text) is not None
  text = re.sub(r"\bw\b", "scaly_w", text)
  decls = [f"  const double* restrict scaly_a{i} = arg[{i}];" for i in range(n_arg)]
  decls += [f"  double* restrict scaly_r{i} = res[{i}];" for i in range(n_res)]
  if uses_w:
    decls.append("  double* restrict scaly_w = w;")
  new_body = body[: last_check + 1] + decls + text.split("\n")
  return "\n".join(lines[: start + 1] + new_body + lines[end:])


_NAN_MAX = re.compile(r"\(\(\((\w+) < (\w+)\) \|\| \(\2 != \2\)\) \? \2 : \1\)")
_NAN_MIN = re.compile(r"\(\(\((\w+) < (\w+)\) \|\| \(\1 != \1\)\) \? \1 : \2\)")


def _fmaximum(src: str) -> str:
  """NaN-propagating max and min selects over plain names as clang's one-instruction builtins."""
  src = _NAN_MAX.sub(r"__builtin_elementwise_maximum(\1, \2)", src)
  return _NAN_MIN.sub(r"__builtin_elementwise_minimum(\1, \2)", src)


def _all_restrict(src: str) -> str:
  return _entry_restrict(_restrict(src))


def compile_variants() -> dict[str, tuple[tuple[str, ...], object]]:
  from scaly.codegen.jit import compile_flags

  jit = tuple(compile_flags())
  return {
    "jit": (jit, None),
    "O3": (tuple("-O3" if f == "-O2" else f for f in jit), None),
    "fast": ((*jit, "-ffast-math"), None),
    "contract": ((*jit, "-ffp-contract=fast"), None),
    "assoc": ((*jit, "-fassociative-math", "-fno-signed-zeros", "-fno-trapping-math"), None),
    "recip": ((*jit, "-freciprocal-math"), None),
    "finite": ((*jit, "-ffinite-math-only"), None),
    "nosz": ((*jit, "-fno-signed-zeros"), None),
    "noerrno_contract_off": ((*jit, "-ffp-contract=off"), None),
    "inline": (jit, _inline),
    "restrict": (jit, _restrict),
    "fmaximum": (jit, _fmaximum),
    "erestrict": (jit, _entry_restrict),
    "arestrict": (jit, _all_restrict),
    "arestrict_inline": (jit, lambda src: _inline(_all_restrict(src))),
  }


def build_cell(kernel_dir: Path, variant: str) -> Path:
  flags, transform = compile_variants()[variant]
  meta = json.loads((kernel_dir / "meta.json").read_text())
  src = (kernel_dir / "kernel.c").read_text()
  if transform is not None:
    src = transform(src)  # ty: ignore[call-non-callable]
  path = kernel_dir / f"kernel_{variant}.c"
  lib = kernel_dir / f"lib_{variant}.so"
  if not lib.exists() or not path.exists() or path.read_text() != src or lib.stat().st_mtime < path.stat().st_mtime:
    path.write_text(src)
    subprocess.run(["cc", *flags, "-fPIC", "-shared", str(path), *meta["link_flags"], "-lm", "-o", str(lib)], check=True)
  return lib


def call(kernel_dir: Path, lib: Path, reps: int, batch: int) -> tuple[float, float, np.ndarray]:
  meta = json.loads((kernel_dir / "meta.json").read_text())
  out_path = lib.with_suffix(".out")
  out = subprocess.run(
    [str(driver()), str(lib), meta["symbol"], str(kernel_dir / "inputs.bin"), str(reps), str(out_path), str(batch)],
    capture_output=True,
    text=True,
    check=True,
  )
  best, median = (float(v) for v in out.stdout.split())
  return best, median, np.fromfile(out_path, dtype="<f8")


def rel_err(got: np.ndarray, want: np.ndarray) -> float:
  if got.shape != want.shape:
    return math.inf
  both = np.isfinite(got) & np.isfinite(want)
  if not np.array_equal(np.isfinite(got), np.isfinite(want)):
    return math.inf
  if not both.any():
    return 0.0
  return float(np.max(np.abs(got[both] - want[both]) / np.maximum(1.0, np.abs(want[both]))))


def main() -> None:
  from corpus import KERNELS, TIERS

  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--cells", default="base/jit,base/O3,base/fast,base/inline,base/restrict")
  parser.add_argument("--kernels", help="comma-separated names or tiers; default all")
  parser.add_argument("--rounds", type=int, default=5)
  parser.add_argument("--budget", type=float, default=0.08)
  parser.add_argument("--tag", default="")
  args = parser.parse_args()
  names: list[str] = []
  for item in (args.kernels or ",".join(KERNELS)).split(","):
    names += list(TIERS.get(item, (item,)))
  cells = [tuple(c.split("/")) for c in args.cells.split(",")]
  libs: dict[tuple[str, str, str], Path] = {}
  for name in names:
    for corpus_variant, variant in cells:
      kdir = BUILD / corpus_variant / name
      if (kdir / "kernel.c").exists():
        libs[(name, corpus_variant, variant)] = build_cell(kdir, variant)
  plan: dict[tuple[str, str, str], tuple[int, int]] = {}
  for key, lib in libs.items():
    kdir = BUILD / key[1] / key[0]
    # The Mac's clock ticks every 41.7 ns: grow the batch until one sample is well above that.
    batch = 1
    while True:
      best, _, _ = call(kdir, lib, 3, batch)
      if best * batch >= 20_000.0 or batch >= 10**6:
        break
      batch *= 10 if best * batch < 2_000.0 else 4
    reps = max(5, int(args.budget * 1e9 / (best * batch)))
    plan[key] = (reps, batch)
  best_time: dict[tuple[str, str, str], float] = {}
  median_time: dict[tuple[str, str, str], float] = {}
  errors: dict[tuple[str, str, str], float] = {}
  for _ in range(args.rounds):
    for key, lib in libs.items():
      kdir = BUILD / key[1] / key[0]
      reps, batch = plan[key]
      best, median, out = call(kdir, lib, reps, batch)
      best_time[key] = min(best, best_time.get(key, math.inf))
      median_time[key] = min(median, median_time.get(key, math.inf))
      errors[key] = rel_err(out, np.fromfile(kdir / "expected.bin", dtype="<f8"))
  heads = [f"{c}/{v}" for c, v in cells]
  print("| kernel | " + " | ".join(heads) + " |")
  print("| --- | " + " | ".join("---" for _ in heads) + " |")
  ratios: dict[str, list[float]] = {h: [] for h in heads}
  for name in names:
    first = best_time.get((name, *cells[0]))
    row = [name]
    for (c, v), h in zip(cells, heads, strict=True):
      t = best_time.get((name, c, v))
      if t is None:
        row.append("—")
        continue
      err = errors[(name, c, v)]
      flag = "" if err < 1e-9 else f" (err {err:.0e})"
      if first is None or (c, v) == cells[0]:
        row.append(f"{t / 1000:.3f} µs{flag}" if t >= 1000 else f"{t:.1f} ns{flag}")
      else:
        ratios[h].append(t / first)
        row.append(f"{t / first:.3f}{flag}")
    print("| " + " | ".join(row) + " |")
  print()
  for h in heads[1:]:
    if ratios[h]:
      print(f"geomean {h} / {heads[0]}: {math.exp(np.mean(np.log(ratios[h]))):.3f}")
  tag = args.tag or "_".join(h.replace("/", "-") for h in heads)
  out = {
    "cells": heads,
    "kernels": names,
    "best_ns": {"/".join(k): v for k, v in best_time.items()},
    "median_ns": {"/".join(k): v for k, v in median_time.items()},
    "err": {"/".join(k): v for k, v in errors.items()},
  }
  (BUILD / f"timing_{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
