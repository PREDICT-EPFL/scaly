"""B5 (Tier 7): how much of a generated kernel's machine code moves values to and from the stack.

Disassembles a codegen-corpus kernel as the JIT built it and, per function, counts the
instructions and those that load from or store to the stack frame (AArch64: an ``sp``- or
frame-relative ``ldr``/``str``/``ldp``/``stp``/``ldur``/``stur``). The proposal's figure for the
chain's stage was 60%, taken before C-211 split the stage into seed groups.

  uv run internal/notes/perf_2026_09_30_gaps/stack_traffic.py t6final chain_9 chain_5 race_cars_40 [--top 8]
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

BUILD = Path(__file__).resolve().parents[1] / "perf_2026_09_30_codegen" / "build"
MOVE = re.compile(r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2} ?){4}\s*\t?(ldr|str|ldp|stp|ldur|stur)\s+.*\[(sp|x29)[,\]]")
INSN = re.compile(r"^\s*[0-9a-f]+:\s")
FUNC = re.compile(r"^[0-9a-f]+ <(.+)>:$")


def functions(lib: Path) -> dict[str, tuple[int, int]]:
  out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(lib)], capture_output=True, text=True, check=True).stdout
  found: dict[str, tuple[int, int]] = {}
  name = None
  for line in out.splitlines():
    match = FUNC.match(line)
    if match:
      name = match.group(1)
      found[name] = (0, 0)
    elif name is not None and INSN.match(line):
      total, moves = found[name]
      text = line.split("\t", 1)[-1].strip()
      stack = text.split()[0] in ("ldr", "str", "ldp", "stp", "ldur", "stur") and ("[sp" in text or "[x29" in text)
      found[name] = (total + 1, moves + stack)
  return found


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("variant")
  parser.add_argument("kernels", nargs="+")
  parser.add_argument("--top", type=int, default=6)
  args = parser.parse_args()
  for kernel in args.kernels:
    found = functions(BUILD / args.variant / kernel / "lib_jit.so")
    total, moves = (sum(v[k] for v in found.values()) for k in (0, 1))
    print(f"{kernel}: {total} instructions, {moves} to or from the stack ({moves / max(total, 1):.0%}), {len(found)} functions")
    for name, (count, stack) in sorted(found.items(), key=lambda item: -item[1][0])[: args.top]:
      print(f"  {name[:60]:60s} {count:8d} {stack:8d} {stack / max(count, 1):5.0%}")


if __name__ == "__main__":
  main()
