"""Quick baseline measurement for the safety-filter Jacobian lowering.

For each (ncars, variant) it builds the ineq vector + dense Jacobian + sparse Jacobian,
records build time and rendered-C line count, prints a small report.

Usage: ``uv run python benchmarks/measure_safety_filter.py``
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import alloy as al
from alloy.codegen.c import render_c_module

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "alloy" / "test_safety_filter_workload.py"


def _load_fixture():
  spec = importlib.util.spec_from_file_location("safety_filter_workload", FIXTURE)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def _render_lines(fn: al.Function) -> int:
  module = render_c_module(fn, header_name=f"{fn.name}.h", source_name=f"{fn.name}.c", typed_buffers=False)
  return module.source.count("\n") + 1


def _measure(ncars: int, build_jac, build_spjac, label: str) -> None:
  t0 = time.perf_counter()
  primal = build_jac.__self__ if hasattr(build_jac, "__self__") else None
  ineq_fn = build_jac(ncars)
  t_primal = time.perf_counter() - t0

  t0 = time.perf_counter()
  jac_fn = build_spjac(ncars, "jac")
  t_jac_build = time.perf_counter() - t0

  t0 = time.perf_counter()
  spjac_fn = build_spjac(ncars, "spjac")
  t_spjac_build = time.perf_counter() - t0

  t0 = time.perf_counter()
  ineq_lines = _render_lines(ineq_fn)
  t_ineq_render = time.perf_counter() - t0
  t0 = time.perf_counter()
  jac_lines = _render_lines(jac_fn)
  t_jac_render = time.perf_counter() - t0
  t0 = time.perf_counter()
  spjac_lines = _render_lines(spjac_fn)
  t_spjac_render = time.perf_counter() - t0

  print(f"{label} ncars={ncars:>2}")
  print(f"  ineq:  build={t_primal * 1000:7.1f}ms  render={t_ineq_render * 1000:7.1f}ms  lines={ineq_lines}")
  print(f"  jac:   build={t_jac_build * 1000:7.1f}ms  render={t_jac_render * 1000:7.1f}ms  lines={jac_lines}")
  print(f"  spjac: build={t_spjac_build * 1000:7.1f}ms  render={t_spjac_render * 1000:7.1f}ms  lines={spjac_lines}")
  _ = primal


def main() -> None:
  fixture = _load_fixture()

  def _build_jac(ncars: int, kind: str) -> al.Function:
    base = fixture.safety_filter_affine_fn(ncars)
    spec = "jac:ineq:u" if kind == "jac" else "spjac:ineq:u"
    return al.Function.factory(base, f"safety_affine_N{ncars}_{spec.replace(':', '_')}", list(base.input_names), [spec])

  for ncars in (2, 16):
    _measure(ncars, fixture.safety_filter_affine_fn, _build_jac, "affine ")


if __name__ == "__main__":
  main()
