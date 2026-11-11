"""Model rectangle splits of the actual compact-map regions used in KKT assembly."""

from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scaly.ir.types import stored_indices
from study import Observer, kkt_functions, plans, render_c_module, affine_index_map, ROOT
from structure import inspect_pattern


def tables(arrays):
  result = Counter()
  if arrays and len(arrays[0]) > 4:
    for array in arrays:
      if len(array) and array[0] >= 0:
        residual = affine_index_map(array).residual
        if len(residual) > 1:
          result[tuple(map(int, residual))] += 1
  return result


class MapStudy(Observer):
  def __init__(self):
    super().__init__()
    self.before = Counter()
    self.after = {"all": Counter(), "eight": Counter()}
    self.regions_after = Counter()
    self.entries = Counter()

  def emit(self, ctx, node, *args, **kwargs):
    self.current = ctx, node
    return self.original_emit(ctx, node, *args, **kwargs)

  def regions(self, writes, reads):
    result = super().regions(writes, reads)
    ctx, node = self.current
    for region in result:
      arrays = [region.writes, *(r for r in region.reads if len(r) and r[0] >= 0)]
      before = tables(arrays)
      self.before.update(before)
      if not before or node.type.pattern is None or len(node.shape) != 2:
        for key in self.after:
          self.after[key].update(before)
        continue
      inverse = np.argsort(ctx.physical_slots(node, np.arange(node.type.storage_size)))
      logical = stored_indices(node.type)[inverse[region.writes]]
      rows, cols = logical // node.shape[1], logical % node.shape[1]
      blocks = inspect_pattern(node.shape, rows, cols)["blocks"]
      for key, selected in (("all", blocks), ("eight", blocks[:8])):
        used = np.zeros(len(rows), dtype=bool)
        after = Counter()
        for lo, hi, first, last in selected:
          slots = np.flatnonzero((rows >= lo) & (rows < hi) & (cols >= first) & (cols < last))
          used[slots] = True
          orders = (slots[np.lexsort((cols[slots], rows[slots]))], slots[np.lexsort((rows[slots], cols[slots]))])
          choices = [tables([a[order] for a in arrays]) for order in orders]
          after.update(min(choices, key=lambda c: sum(map(len, c))))
        after.update(tables([a[~used] for a in arrays]))
        if sum(map(len, after)) < sum(map(len, before)):
          self.after[key].update(after)
          self.regions_after[key] += len(selected)
          self.entries[key] += int(used.sum())
        else:
          self.after[key].update(before)
    return result


def main():
  output = []
  names = sorted(p.stem for p in (ROOT / "tests/integration/data/maros_meszaros").glob("*.npz") if not p.stem.endswith("_patterns"))
  for name in names:
    for label, fun, _ in kkt_functions(name):
      if not label.endswith(("/kkt", "/upper")):
        continue
      print(label, flush=True)
      with MapStudy() as observer:
        observer.original_emit = plans.emit_map
        with patch.object(plans, "emit_map", observer.emit):
          render_c_module(fun, lanes=1, dialect="c", cpu="generic")
      output.append(
        {
          "name": label,
          "map_table_entries_before": sum(map(len, observer.before)),
          "map_table_entries_after": {k: sum(map(len, c)) for k, c in observer.after.items()},
          "rectangles": dict(observer.regions_after),
          "regular_entries": dict(observer.entries),
        }
      )
  Path(__file__).with_name("maps.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
  main()
