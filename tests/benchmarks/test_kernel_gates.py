"""Generated-source gates on the benchmark kernels: source size must not grow with the horizon or the
decoder width, and workspace must stay near its recorded baseline. Code generation only, no compile."""

from __future__ import annotations

from pathlib import Path

from benchmarks.harness.sweep import build_kernel


def _build(tmp_path: Path, workload: str, size: int) -> dict:
  out = tmp_path / f"{workload}_{size}"
  out.mkdir()
  return build_kernel(workload, size, "scaly", out)


def test_race_cars_jacobian_preserves_the_horizon_loop(tmp_path: Path) -> None:
  small, large = (_build(tmp_path, "race_cars_jac", size) for size in (5, 50))
  assert large["source_lines"] < 1.2 * small["source_lines"]
  assert int(large["w_size"]) <= 3 * 2100


def test_chain_jacobian_preserves_the_mass_loop(tmp_path: Path) -> None:
  small, large = (_build(tmp_path, "chain_jac", size) for size in (5, 33))
  assert large["source_lines"] < 1.2 * small["source_lines"]


def test_npmpc_jacobian_source_is_independent_of_horizon_and_width(tmp_path: Path) -> None:
  small, large, wide = _build(tmp_path, "npmpc_jac", 6), _build(tmp_path, "npmpc_jac", 100), _build(tmp_path, "npmpc_decoder_jac", 128)
  assert large["source_lines"] < 1.2 * small["source_lines"]
  assert wide["source_lines"] < 1.2 * small["source_lines"]
  # baseline 1600 doubles at N=100 (VMAP buffers scale linearly with the horizon); 3x headroom catches superlinear regressions
  assert int(large["w_size"]) <= 3 * 1600


def test_npmpc_lagrangian_hessian_source_is_independent_of_horizon_and_width(tmp_path: Path) -> None:
  # This is the kernel that caught an unrolled objective: a cost built as a Python loop over stages
  # grows the Hessian source linearly and, past roughly 75 stages, exceeds the Program IR passes'
  # recursion depth. Scanning the cost keeps it constant, so the gate is on the horizon as well as the width.
  small, large, wide = _build(tmp_path, "npmpc", 6), _build(tmp_path, "npmpc", 100), _build(tmp_path, "npmpc_decoder", 128)
  assert large["source_lines"] < 1.2 * small["source_lines"]
  assert small["nnz"] < large["nnz"], "the Hessian pattern must grow with the horizon even though its source does not"
  assert wide["source_lines"] < 1.2 * small["source_lines"]
