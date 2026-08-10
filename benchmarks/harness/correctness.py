from __future__ import annotations

from pathlib import Path

import numpy as np


def scatter_compact(compact, rows, cols, shape: tuple[int, int]) -> np.ndarray:
  dense = np.zeros(shape[0] * shape[1], dtype=np.float64)
  dense[np.asarray(rows, dtype=np.int64) * shape[1] + np.asarray(cols, dtype=np.int64)] = np.asarray(compact, dtype=np.float64).reshape(-1)
  return dense


def check_dense_reference(compact, rows, cols, expected, shape: tuple[int, int], *, label: str) -> None:
  got = scatter_compact(compact, rows, cols, shape)
  expected = np.asarray(expected, dtype=np.float64).reshape(-1)
  if expected.size != shape[0] * shape[1]:
    raise RuntimeError(f"{label}: expected dense {shape}, got {expected.shape}")
  if np.isnan(expected).any():
    raise RuntimeError(f"{label}: reference Jacobian contains NaN - fix the sample inputs before benchmarking")
  if not np.allclose(got, expected, atol=1e-9, rtol=1e-9):
    bad = np.flatnonzero(~np.isclose(got, expected, atol=1e-9, rtol=1e-9))[0]
    raise RuntimeError(f"{label}: dense[{bad // shape[1]},{bad % shape[1]}]={got[bad]:.17g}, expected {expected[bad]:.17g}")


def write_samples(out_dir: Path, inputs: dict[str, np.ndarray], expected: np.ndarray) -> tuple[dict[str, Path], Path]:
  paths: dict[str, Path] = {}
  for name, value in inputs.items():
    paths[name] = out_dir / f"sample_{name}.bin"
    paths[name].write_bytes(np.asarray(value, dtype=np.float64).reshape(-1).tobytes())
  expected_path = out_dir / "expected_dense.bin"
  expected_path.write_bytes(np.asarray(expected, dtype=np.float64).reshape(-1).tobytes())
  return paths, expected_path
