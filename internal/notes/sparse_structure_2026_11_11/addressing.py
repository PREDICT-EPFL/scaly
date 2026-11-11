"""Model exact mixed-product addressing after splitting out regular stored regions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scaly.passes.affine import affine_index_map
from structure import inspect_pattern, runs
from study import ROOT


def model(shape, rows, cols, physical, kind):
  order = np.lexsort((cols, rows))
  rows, cols, physical = rows[order], cols[order], physical[order]
  flat = rows * shape[1] + cols
  entry_form = "table" if any(len(affine_index_map(a).residual) > 1 for a in (cols, physical) if len(a)) else "affine"
  if entry_form == "affine" and all(len(a) < 2 or np.all(np.diff(a) == 1) for a in (cols, physical)):
    entry_form = "dense"
  regions = []
  if kind == "blocks":
    for lo, hi, first, last in inspect_pattern(shape, rows, cols)["blocks"]:
      regions.append(np.searchsorted(flat, (np.arange(lo, hi)[:, None] * shape[1] + np.arange(first, last)).ravel()))
  elif kind == "bands":
    for offset in np.unique(cols - rows):
      slots = np.flatnonzero(cols - rows == offset)
      regions.extend(slots[np.searchsorted(rows[slots], part)] for part in runs(rows[slots], 4))
  elif kind == "supernodes":
    for first, last, _ in inspect_pattern(shape, rows, cols)["super_groups"]:
      regions.append(np.flatnonzero((cols >= first) & (cols < last)))
  tables = set()

  def add_map(array):
    if len(array):
      residual = affine_index_map(array).residual
      if len(residual) > 1:
        tables.add(tuple(map(int, residual)))

  used = np.zeros(len(rows), dtype=bool)
  for slots in regions:
    assert not np.any(used[slots])
    used[slots] = True
    if kind == "bands":
      add_map(physical[slots])
    elif kind == "supernodes":
      add_map(np.unique(rows[slots]))
      add_map(physical[slots])
    else:
      # Either loop orientation is legal. Pick the smaller source-slot residual.
      column_order = np.lexsort((rows[slots], cols[slots]))
      layouts = [physical[slots], physical[slots][column_order]]
      add_map(min(layouts, key=lambda a: len(affine_index_map(a).residual)))
  rest = ~used
  active, counts = np.unique(rows[rest], return_counts=True)
  if len(active):
    if len(active) > 1:
      tables.add(tuple(map(int, np.r_[0, np.cumsum(counts)])))
    for array in (active, cols[rest], physical[rest]):
      add_map(array)
  return {
    "table_entries": sum(map(len, tables)),
    "table_arrays": len(tables),
    "loaded_invocations": len(active) if len(active) > 1 else 0,
    "regular_entries": int(used.sum()),
    "regions": len(regions),
    "entry_form": entry_form,
    "entries": len(rows),
  }


def main():
  import importlib.util

  spec = importlib.util.spec_from_file_location("kkt_data", ROOT / "tests/integration/test_sparse_kkt.py")
  data = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(data)
  baseline = {r["name"]: r for r in map(json.loads, Path(__file__).with_name("kkt.jsonl").read_text().splitlines())}
  output = []
  for name in data.PROBLEMS:
    for label, matrix in zip(("P", "A", "G"), data._problem(name), strict=True):
      coo = matrix.tocoo()
      physical = np.arange(matrix.nnz)
      for transpose in (False, True) if label != "P" else (False,):
        if transpose:
          shape, rows, cols, suffix = matrix.shape[::-1], coo.col, coo.row, "Ty"
        else:
          shape, rows, cols, suffix = matrix.shape, coo.row, coo.col, "x"
        result = {"name": f"{name}/{label}{suffix}"}
        for kind in ("baseline", "blocks", "bands", "supernodes"):
          result[kind] = model(shape, rows, cols, physical, kind)
        result["actual_table_entries"] = baseline[result["name"]]["program"]["table_entries"]
        assert result["baseline"]["table_entries"] == result["actual_table_entries"]
        output.append(result)
  Path(__file__).with_name("addressing.json").write_text(json.dumps(output, indent=2) + "\n")
  print(f"Wrote {len(output)} addressing models")


if __name__ == "__main__":
  main()
