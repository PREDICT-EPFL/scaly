"""Where a corpus kernel spends its time: self samples per generated procedure, from macOS ``sample``.

    uv run internal/notes/perf_2026_09_30_gaps/prof_kernel.py internal/notes/perf_2026_09_30_codegen/build/<variant>/<kernel> [--seconds 4]

The corpus counterpart of ``perf_2026_09_27_ipm_speed/prof.py``: recompiles the cell's ``kernel.c``
with the JIT's flags plus ``-g`` and every procedure out of line, runs it through the corpus's
``time_entry`` driver under ``sample``, and prints each procedure's share. It leaves ``prof.c`` and
``sample.txt`` in the cell, which ``step_lines.py <cell> <procedure>`` reads for the shares by loop.

Out of line a small stage can cost several times what it does inlined into its loop (the race
cars': 4x). ``--inlined`` keeps the build as the JIT makes it and attributes each sample to the
procedure whose source lines it falls in, from the debug line table: the shares of the real build,
by where the code was written.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "perf_2026_09_30_codegen"))


def by_source(text: str, src: list[str]) -> Counter[str]:
  """Self samples by the procedure whose source lines they fall in (a library function by its name)."""
  starts = [
    (i + 1, m.group(1)) for i, line in enumerate(src) if line.startswith(("static ", "int ")) and (m := re.search(r"(\w+)\([^()]*\) \{$", line))
  ]

  def owner(line: int) -> str:
    return next((name for first, name in reversed(starts) if first <= line), "?")

  graph = text.split("Call graph:")[1].split("Total number in stack")[0]
  nodes = []
  for row in graph.splitlines():
    m = re.match(r"^(\s+[+!:| ]*?)(\d+) (\S+)\s+\(in ([^)]+)\).*?(?:prof\.c:(\d+))?\s*$", row)
    if m:
      nodes.append((len(m.group(1)), int(m.group(2)), m.group(3), m.group(4), int(m.group(5)) if m.group(5) else 0))
  counts: Counter[str] = Counter()
  for i, (depth, count, symbol, image, line) in enumerate(nodes):
    j, child = i + 1, None
    while j < len(nodes) and nodes[j][0] > depth:
      child = nodes[j][0] if child is None else min(child, nodes[j][0])
      j += 1
    own = count - sum(c for d, c, *_ in nodes[i + 1 : j] if d == child)
    if own > 0 and (image.startswith("prof") or image.startswith("libsystem")):
      counts[owner(line) if image.startswith("prof") and line else symbol] += own
  return counts


def main() -> None:
  from timing import driver

  from scaly.codegen.jit import compile_flags

  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("cell", type=Path)
  parser.add_argument("--seconds", type=float, default=4.0)
  parser.add_argument("--inlined", action="store_true", help="the JIT's own inlining; samples attributed by source line")
  args = parser.parse_args()
  cell = args.cell.resolve()
  meta = json.loads((cell / "meta.json").read_text())
  src = (cell / "kernel.c").read_text()
  if not args.inlined:
    src = re.sub(r"static inline void (\w+)\(", r"static __attribute__((noinline)) void \1(", src)
  (cell / "prof.c").write_text(src)
  lib = cell / "prof.so"
  subprocess.run(["cc", *compile_flags(), "-g", "-fPIC", "-shared", str(cell / "prof.c"), "-lm", "-o", str(lib)], check=True)
  run = [str(driver()), str(lib), meta["symbol"], str(cell / "inputs.bin")]
  cold = float(subprocess.run([*run, "5", "/dev/null", "10"], capture_output=True, text=True, check=True).stdout.split()[0])
  batch = max(1, int(1e6 / cold))  # a millisecond a sample
  per_ns = float(subprocess.run([*run, "400", "/dev/null", str(batch)], capture_output=True, text=True, check=True).stdout.split()[0])
  proc = subprocess.Popen([*run, str(int((args.seconds + 3.0) * 1e3)), "/dev/null", str(batch)])
  time.sleep(0.5)
  out = cell / "sample.txt"
  subprocess.run(["sample", str(proc.pid), str(int(args.seconds)), "1", "-file", str(out)], capture_output=True, check=True)
  proc.kill()
  text = out.read_text()
  counts: Counter[str] = Counter()
  if args.inlined:
    counts = by_source(text, src.splitlines())
  else:
    section = text.split("Sort by top of stack, same collapsed")[1] if "Sort by top of stack" in text else ""
    for line in section.splitlines():
      m = re.match(r"\s+(.+?)\s+\(in ([^)]+)\)\s+(\d+)\s*$", line)
      if m:
        counts[m.group(1)] += int(m.group(3))
  total = sum(counts.values())
  print(f"{meta['kernel']}: {per_ns / 1e3:.1f} us per call {'as built' if args.inlined else 'out of line'}, {total} samples")
  for name, k in counts.most_common(14):
    print(f"  {100.0 * k / total:5.1f}%  {name}")


if __name__ == "__main__":
  main()
