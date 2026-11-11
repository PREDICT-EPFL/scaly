"""Count exact rectangles, diagonal runs and equal-row column groups with NumPy."""

from __future__ import annotations

import numpy as np


def runs(values, minimum):
  values = np.asarray(values, dtype=np.int64)
  return [part for part in np.split(values, np.flatnonzero(np.diff(values) != 1) + 1) if len(part) >= minimum]


def inspect_pattern(shape, rows, cols):
  rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
  order = np.lexsort((cols, rows))
  rows, cols = rows[order], cols[order]
  flat = rows * shape[1] + cols
  assert len(np.unique(flat)) == len(flat)
  by_row = np.split(cols, np.searchsorted(rows, np.arange(1, shape[0])))
  covered = np.zeros(len(rows), dtype=bool)
  rectangles = set()
  for r in range(max(0, shape[0] - 3)):
    common = by_row[r]
    for rr in range(r + 1, r + 4):
      common = np.intersect1d(common, by_row[rr], assume_unique=True)
    for part in runs(common, 4):
      first, last = int(part[0]), int(part[-1]) + 1
      lo, hi = r, r + 4
      while lo and np.all(np.isin(part, by_row[lo - 1], assume_unique=True)):
        lo -= 1
      while hi < shape[0] and np.all(np.isin(part, by_row[hi], assume_unique=True)):
        hi += 1
      rectangles.add((lo, hi, first, last))
      for rr in range(r, r + 4):
        covered[np.searchsorted(flat, rr * shape[1] + part)] = True
  chosen = []
  used = np.zeros(len(rows), dtype=bool)
  for lo, hi, first, last in sorted(rectangles, key=lambda x: (-(x[1] - x[0]) * (x[3] - x[2]), x)):
    positions = np.searchsorted(flat, (np.arange(lo, hi)[:, None] * shape[1] + np.arange(first, last)).ravel())
    if not np.any(used[positions]):
      used[positions] = True
      chosen.append((lo, hi, first, last))
  band = np.zeros(len(rows), dtype=bool)
  diagonals = 0
  for offset in np.unique(cols - rows):
    slots = np.flatnonzero(cols - rows == offset)
    for part in runs(rows[slots], 4):
      band[slots[np.searchsorted(rows[slots], part)]] = True
      diagonals += 1
  col_order = np.lexsort((rows, cols))
  by_col = np.split(rows[col_order], np.searchsorted(cols[col_order], np.arange(1, shape[1])))
  super_mask = np.zeros(len(rows), dtype=bool)
  groups = []
  c = 0
  while c < shape[1]:
    stop = c + 1
    while stop < shape[1] and np.array_equal(by_col[c], by_col[stop]):
      stop += 1
    if stop - c >= 2 and len(by_col[c]):
      groups.append((c, stop, len(by_col[c])))
      super_mask |= (cols >= c) & (cols < stop)
    c = stop
  row_degree = np.bincount(rows, minlength=shape[0])
  return {
    "shape": list(map(int, shape)),
    "nnz": len(rows),
    "block_entries": int(covered.sum()),
    "disjoint_block_entries": int(used.sum()),
    "blocks": chosen,
    "block_rows": sum(hi - lo for lo, hi, _, _ in chosen),
    "band_entries": int(band.sum()),
    "diagonal_runs": diagonals,
    "super_entries": int(super_mask.sum()),
    "super_groups": groups,
    "super_row_indices_saved": sum((stop - start - 1) * length for start, stop, length in groups),
    "union_entries": int((covered | band | super_mask).sum()),
    "row_square_sum": int(row_degree @ row_degree),
    "active_rows": int(np.count_nonzero(row_degree)),
  }


def self_check():
  r, c = np.indices((8, 8))
  dense = inspect_pattern((8, 8), r.ravel(), c.ravel())
  assert dense["block_entries"] == dense["band_entries"] + 12 == dense["super_entries"] == 64
  assert dense["super_row_indices_saved"] == 56
  punctured = np.ones((8, 8), dtype=bool)
  punctured[3, 3] = False
  result = inspect_pattern((8, 8), *np.nonzero(punctured))
  assert result["block_entries"] < 63
  assert result["disjoint_block_entries"] <= result["block_entries"]
  diagonal = inspect_pattern((8, 8), np.arange(8), np.arange(8))
  assert diagonal["band_entries"] == 8 and diagonal["block_entries"] == diagonal["super_entries"] == 0
  assert inspect_pattern((0, 8), (), ())["nnz"] == 0
