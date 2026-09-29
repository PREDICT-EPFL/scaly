"""Render a ``timing.py`` result as an HTML table for the reports: the first cell in time, the others as ratios.

    uv run internal/notes/perf_2026_09_30_codegen/tables.py build/timing_<tag>.json [--kernels a,b] [--cells x,y]

Ratios under 0.95 are marked as wins, over 1.05 as losses; a cell whose outputs differ from the
JIT's own by more than 1e-9 (relative) carries the difference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fmt_time(ns: float) -> str:
  if ns >= 1e6:
    return f"{ns / 1e6:.3f} ms"
  if ns >= 1e3:
    return f"{ns / 1e3:.2f} µs"
  return f"{ns:.1f} ns"


def table(data: dict, kernels: list[str] | None = None, cells: list[str] | None = None, labels: dict[str, str] | None = None) -> str:
  cells = cells or data["cells"]
  kernels = kernels or data["kernels"]
  labels = labels or {}
  head = "".join(f'<th class="r">{labels.get(c, c)}</th>' for c in cells)
  rows = []
  logs: dict[str, list[float]] = {c: [] for c in cells[1:]}
  for k in kernels:
    first = data["best_ns"].get(f"{k}/{cells[0]}")
    tds = []
    for c in cells:
      t = data["best_ns"].get(f"{k}/{c}")
      err = data["err"].get(f"{k}/{c}", 0.0)
      note = "" if err < 1e-9 else f' <span class="warn">(Δ {err:.0e})</span>'
      if t is None:
        tds.append('<td class="r">—</td>')
      elif c == cells[0] or first is None:
        tds.append(f'<td class="r">{fmt_time(t)}{note}</td>')
      else:
        r = t / first
        logs[c].append(math.log(r))
        cls = ' class="r ok"' if r < 0.95 else ' class="r bad"' if r > 1.05 else ' class="r"'
        tds.append(f"<td{cls}>{r:.3f}{note}</td>")
    rows.append(f"<tr><td>{k}</td>{''.join(tds)}</tr>")
  geo = "".join(f'<td class="r"><b>{math.exp(sum(v) / len(v)):.3f}</b></td>' if v else '<td class="r">—</td>' for v in logs.values())
  rows.append(f'<tr><td><b>geometric mean</b></td><td class="r">—</td>{geo}</tr>')
  return f'<div class="scroll"><table><tr><th>kernel</th>{head}</tr>\n' + "\n".join(rows) + "\n</table></div>"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("json", type=Path)
  parser.add_argument("--kernels")
  parser.add_argument("--cells")
  args = parser.parse_args()
  path = args.json if args.json.is_absolute() else HERE / args.json
  data = json.loads(path.read_text())
  print(table(data, args.kernels.split(",") if args.kernels else None, args.cells.split(",") if args.cells else None))


if __name__ == "__main__":
  main()
