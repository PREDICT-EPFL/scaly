"""B5 (Tier 7): how long the scalar temporaries of a generated straight-line function stay live.

For each function of a kernel's generated C with 500 temporaries or more: how many statements lie
between a temporary's definition and its last use and its first use, and how many temporaries are
live at once, against the 32 registers they compete for.

  uv run internal/notes/perf_2026_09_30_gaps/live_ranges.py internal/notes/perf_2026_09_30_codegen/build/t6final/chain_9/kernel_jit.c
"""

import collections
import re
import sys

src = open(sys.argv[1]).read()
# split into functions
funcs = re.split(r"\n(?=static [^\n]*\(|[a-z_]+ [A-Za-z_0-9]+\([^\n]*\) \{)", src)
DEF = re.compile(r"^\s*(?:const )?double (v\d+) = (.*);$")
USE = re.compile(r"\bv\d+\b")
for f in funcs:
  head = f.split("\n", 1)[0][:90]
  lines = f.split("\n")
  defs, last, first = {}, {}, {}
  for i, line in enumerate(lines):
    m = DEF.match(line)
    rhs = m.group(2) if m else line
    for u in USE.findall(rhs if m else line):
      if u in defs:
        last[u] = i
        first.setdefault(u, i)
    if m:
      defs[m.group(1)] = i
  if len(defs) < 500:
    continue
  spans = sorted(last[v] - defs[v] for v in defs if v in last)
  gaps = sorted(first[v] - defs[v] for v in defs if v in first)
  events = collections.Counter()
  for v in defs:
    if v in last:
      events[defs[v]] += 1
      events[last[v]] -= 1
  live = peak = 0
  hist = []
  for i in range(len(lines)):
    live += events.get(i, 0)
    peak = max(peak, live)
    hist.append(live)
  n = len(spans)
  print(
    f"{head}\n   {len(defs)} temporaries; live span median {spans[n // 2]}, p90 {spans[int(n * 0.9)]}, max {spans[-1]}; def-to-first-use median {gaps[len(gaps) // 2]}, p90 {gaps[int(len(gaps) * 0.9)]}; live values: mean {sum(hist) / len(hist):.0f}, peak {peak}"
  )
