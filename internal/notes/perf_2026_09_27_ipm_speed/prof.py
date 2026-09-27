"""Where a generated solver spends its time: self samples per generated procedure, from macOS ``sample``.

    uv run internal/notes/perf_2026_09_27_ipm_speed/prof.py build/<variant>/<problem>_<backend> [--seconds 4]

Recompiles the cell's ``solver.c`` with the JIT's flags plus ``-g``, marking every procedure
``noinline`` except the loop bodies a ``scan`` calls once per step (their names end in
``_inplace_raw`` or their arguments slice a table), so each procedure's own work is
attributed to it while the per-step bodies stay inlined into their loops as in the real build. Runs
the C driver in a loop and prints each procedure's share of the samples.
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
sys.path.insert(0, str(HERE))


def main() -> None:
  from scaly.codegen.jit import compile_flags
  from timing import driver

  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("cell", type=Path)
  parser.add_argument("--seconds", type=float, default=4.0)
  parser.add_argument("--inline-all", action="store_true", help="keep the procedures inline (only the entry is attributed)")
  args = parser.parse_args()
  cell = args.cell if args.cell.is_absolute() else (Path.cwd() / args.cell)
  meta = json.loads((cell / "meta.json").read_text())
  src = (cell / "solver.c").read_text()
  # Procedures a scan calls per step (their arguments slice the loop's tables): keep those inline.
  called_in_loops = set(re.findall(r"(\w+_raw)\([^;]*\(k\d+ \+ ", src))
  if not args.inline_all:

    def mark(m: re.Match) -> str:
      name = m.group(1)
      if name.endswith("_inplace_raw") or name in called_in_loops:
        return m.group(0)
      return f"static __attribute__((noinline)) void {name}("

    src = re.sub(r"static inline void (\w+)\(", mark, src)
  prof_c = cell / "prof.c"
  prof_c.write_text(src)
  lib = cell / "prof.so"
  subprocess.run(["cc", *compile_flags(), "-g", "-fPIC", "-shared", str(prof_c), "-lm", "-o", str(lib)], check=True)
  b = subprocess.run([str(driver()), str(lib), meta["symbol"], str(cell / "inputs.bin"), "5", "/dev/null"], capture_output=True, text=True, check=True)
  per = float(b.stdout.split()[0]) * 1e-9
  reps = int((args.seconds + 2.0) / per)
  proc = subprocess.Popen([str(driver()), str(lib), meta["symbol"], str(cell / "inputs.bin"), str(reps), "/dev/null"])
  time.sleep(0.5)
  out = cell / "sample.txt"
  subprocess.run(["sample", str(proc.pid), str(int(args.seconds)), "1", "-file", str(out)], capture_output=True, check=True)
  proc.kill()
  text = out.read_text()
  # "Sort by top of stack" section: "        name  (in lib)        count"
  section = text.split("Sort by top of stack, same collapsed")[1] if "Sort by top of stack" in text else ""
  counts: Counter[str] = Counter()
  for line in section.splitlines():
    m = re.match(r"\s+(.+?)\s+\(in ([^)]+)\)\s+(\d+)\s*$", line)
    if m:
      counts[m.group(1)] += int(m.group(3))
  total = sum(counts.values())
  print(f"{meta['problem']} {meta['backend']}: {per * 1e6:.1f} us per solve, {meta['iter']} iterations, {total} samples")
  for name, k in counts.most_common(25):
    print(f"  {100.0 * k / total:5.1f}%  {name.replace(meta['symbol'] + '_', '').replace('g_' + meta['problem'] + '_', '')}")


if __name__ == "__main__":
  main()
