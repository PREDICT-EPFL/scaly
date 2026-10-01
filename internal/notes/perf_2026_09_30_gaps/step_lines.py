"""Where one generated procedure spends its own time, loop by loop, and what the C compiler did with each loop.

    uv run internal/notes/perf_2026_09_27_ipm_speed/prof.py <cell>          # samples the cell first
    uv run internal/notes/perf_2026_09_30_gaps/step_lines.py <cell> <procedure> [rows]

``prof.py`` leaves ``prof.c`` (the cell's solver with its procedures out of line) and ``sample.txt``
in the cell. This reads the call graph's source lines, takes the procedure's self samples, assigns
them to its top-level loops, and compiles ``prof.c`` again with clang's vectorization remarks to say
which of those loops were vectorized. C-217 came from it: 42% of the IPM step's samples sat in loops
left scalar, most of them a select around a load (``results/step_lines_dualc8.txt``).
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def self_samples(text: str, proc: str) -> Counter[int]:
  """Self samples of ``proc`` per source line: each node's count less its direct children's."""
  graph = text.split("Call graph:")[1].split("Total number in stack")[0]
  nodes = []
  for line in graph.splitlines():
    m = re.match(r"^(\s+[+!:| ]*?)(\d+) (\S+)\s+\(in ([^)]+)\).*?(?:prof\.c:(\d+))?\s*$", line)
    if m:
      nodes.append((len(m.group(1)), int(m.group(2)), m.group(3), int(m.group(5)) if m.group(5) else -1))
  out: Counter[int] = Counter()
  for i, (depth, count, symbol, line) in enumerate(nodes):
    if symbol != proc:
      continue
    j, child = i + 1, None
    while j < len(nodes) and nodes[j][0] > depth:
      child = nodes[j][0] if child is None else min(child, nodes[j][0])
      j += 1
    out[line] += count - sum(c for d, c, _, _ in nodes[i + 1 : j] if d == child)
  return out


def top_loops(src: list[str], start: int) -> list[tuple[int, int, str]]:
  """The procedure's top-level loops as (first line, last line, header), lines counted from 1."""
  loops, depth, first = [], 0, None
  for i in range(start, len(src)):
    line = src[i]
    if line.startswith("}"):
      break
    if depth == 0 and re.match(r"  (for|while) \(", line):
      first = i
    depth += line.count("{") - line.count("}")
    if first is not None and depth == 0:
      loops.append((first + 1, i + 1, src[first].strip()))
      first = None
  return loops


def remarks(prof_c: Path) -> dict[int, list[str]]:
  """Clang's vectorization remarks per source line, compiling with the JIT's flags."""
  from scaly.codegen.jit import compile_flags

  cmd = ["cc", *compile_flags(), "-c", str(prof_c), "-o", "/dev/null", "-Rpass=loop-vectorize", "-Rpass-missed=loop-vectorize"]
  out: dict[int, list[str]] = {}
  for line in subprocess.run(cmd, capture_output=True, text=True, check=False).stderr.splitlines():
    m = re.match(r"^[^:]+:(\d+):\d+: remark: ([^(\[]+)", line)
    if m:
      out.setdefault(int(m.group(1)), []).append(m.group(2).strip())
  return out


def main() -> None:
  cell, proc = Path(sys.argv[1]), sys.argv[2]
  rows = int(sys.argv[3]) if len(sys.argv) > 3 else 25
  src = (cell / "prof.c").read_text().splitlines()
  start = next(i for i, line in enumerate(src) if "static" in line and re.search(rf"\b{proc}\(", line)) + 1
  by_line = self_samples((cell / "sample.txt").read_text(), proc)
  total = sum(by_line.values())
  loops = top_loops(src, start)
  by_loop: Counter[int] = Counter()
  for line, count in by_line.items():
    hit = next((k for k, (a, b, _) in enumerate(loops) if a <= line <= b), None)
    if hit is not None:
      by_loop[hit] += count
  print(
    f"{proc}: {total} self samples, {100 * (total - sum(by_loop.values())) / total:.1f}% outside loops; {len(loops)} loops, {len(by_loop)} sampled"
  )
  said = remarks(cell / "prof.c")
  verdicts: Counter[str] = Counter()
  table = []
  for k, count in by_loop.most_common():
    a, b, header = loops[k]
    notes = [note for line in range(a, b + 1) for note in said.get(line, [])]
    verdict = (
      "vectorized"
      if "vectorized loop" in notes
      else "interleaved"
      if "interleaved loop" in notes
      else f"scalar: {notes[0] if notes else 'no remark'}"
    )
    verdicts[verdict] += count
    table.append((count, a - start, verdict, header, " ".join(x.strip() for x in src[a : min(b - 1, a + 2)])))
  print("\nthe procedure's samples by the compiler's verdict on the loop:")
  for verdict, count in verdicts.most_common():
    print(f"  {100 * count / total:5.1f}%  {verdict}")
  print("\nthe loops, by samples:")
  for count, line, verdict, header, body in table[:rows]:
    print(f"  {100 * count / total:5.1f}%  L{line:<5d} {verdict[:44]:44s}  {header[:58]:58s}  {body[:130]}")


if __name__ == "__main__":
  main()
