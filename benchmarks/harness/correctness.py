from __future__ import annotations

from pathlib import Path

import numpy as np


def check_dense_reference(
  compact,
  rows,
  cols,
  expected,
  shape: tuple[int, int],
  *,
  label: str,
  layout: str = "full",
) -> None:
  """Compare a compact result with a dense reference at the pattern's coordinates.

  ``full`` checks every position in the matrix, which also catches a nonzero reference entry
  omitted by the pattern. A triangular Hessian intentionally omits the reflected half, so
  ``lower`` and ``upper`` compare every position in the selected triangle and treat a missing
  coordinate as zero.
  """
  expected = np.asarray(expected, dtype=np.float64).reshape(-1)
  if expected.size != shape[0] * shape[1]:
    raise RuntimeError(f"{label}: expected dense {shape}, got {expected.shape}")
  n_rows, n_cols = shape
  for row in range(n_rows):
    if np.isnan(expected[row * n_cols : (row + 1) * n_cols]).any():
      raise RuntimeError(f"{label}: reference contains NaN - fix the sample inputs before benchmarking")
  compact = np.asarray(compact, dtype=np.float64).reshape(-1)
  rows = np.asarray(rows, dtype=np.int64).reshape(-1)
  cols = np.asarray(cols, dtype=np.int64).reshape(-1)
  if compact.size != rows.size or rows.size != cols.size:
    raise RuntimeError(f"{label}: compact result has {compact.size} values for {rows.size} coordinates")
  if layout not in {"full", "lower", "upper"}:
    raise ValueError(f"{label}: unsupported dense-reference layout {layout!r}")
  if np.any(rows < 0) or np.any(rows >= shape[0]) or np.any(cols < 0) or np.any(cols >= shape[1]):
    raise RuntimeError(f"{label}: compact pattern contains a coordinate outside {shape}")
  if layout == "lower" and np.any(rows < cols):
    raise RuntimeError(f"{label}: lower layout contains an upper-triangle coordinate")
  if layout == "upper" and np.any(rows > cols):
    raise RuntimeError(f"{label}: upper layout contains a lower-triangle coordinate")
  flat = rows * shape[1] + cols
  if np.unique(flat).size != flat.size:
    raise RuntimeError(f"{label}: compact pattern contains duplicate coordinates")
  entries = sorted(
    ((int(row), int(col), float(value)) for row, col, value in zip(rows, cols, compact, strict=True)),
    key=lambda entry: (entry[0], entry[1]),
  )
  cursor = 0

  def check_value(row: int, col: int, got: float, want: float) -> None:
    matches = np.isclose(got, want, atol=1e-9, rtol=1e-9)
    matches |= got == want  # inf == inf must pass without producing NaN in a diff
    matches |= np.isnan(got) & np.isnan(want)
    if not matches:
      raise RuntimeError(f"{label}: dense[{row},{col}]={got:.17g}, expected {want:.17g}")

  def check_zero_run(row: int, start: int, stop: int) -> None:
    if start >= stop:
      return
    expected_run = expected[row * n_cols + start : row * n_cols + stop]
    matches = np.isclose(expected_run, 0.0, atol=1e-9, rtol=1e-9)
    matches |= expected_run == 0.0  # exact zero also covers signed zero
    if not np.all(matches):
      bad = int(np.flatnonzero(~matches)[0])
      col = start + bad
      raise RuntimeError(f"{label}: dense[{row},{col}]=0, expected {expected[row * n_cols + col]:.17g}")

  for row in range(n_rows):
    if layout == "lower":
      start, stop = 0, min(row + 1, n_cols)
    elif layout == "upper":
      start, stop = row, n_cols
    else:
      start, stop = 0, n_cols
    col = start
    while cursor < len(entries) and entries[cursor][0] == row:
      _, entry_col, entry_value = entries[cursor]
      check_zero_run(row, col, entry_col)
      check_value(row, entry_col, entry_value, expected[row * n_cols + entry_col])
      col = entry_col + 1
      cursor += 1
    check_zero_run(row, col, stop)
  if cursor != len(entries):
    raise RuntimeError(f"{label}: compact pattern contains a coordinate outside the selected {layout} region")


def write_samples(out_dir: Path, inputs: dict[str, np.ndarray], expected: np.ndarray) -> tuple[dict[str, Path], Path]:
  paths: dict[str, Path] = {}
  for name, value in inputs.items():
    paths[name] = out_dir / f"sample_{name}.bin"
    paths[name].write_bytes(np.asarray(value, dtype=np.float64).reshape(-1).tobytes())
  expected_path = out_dir / "expected_dense.bin"
  expected_path.write_bytes(np.asarray(expected, dtype=np.float64).reshape(-1).tobytes())
  return paths, expected_path
