from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any

import alloy as al
from alloy.codegen.c import render_c_source
from alloy.viz import clear_recordings, recording_path, recordings, serve, unvisualize_function, visualize_function

ROOT = Path(__file__).resolve().parents[1]
TRACKING_FIXTURE = ROOT / "tests" / "alloy" / "test_tracking_workload.py"
DEFAULT_VIZ_DIR = ROOT / "benchmarks" / "gen" / "alloy_tracking_eq_jac_viz"

_FOR_RE = re.compile(r"prog\.for %[^=]+ = (-?\d+) to (-?\d+) step (-?\d+)")


def _load_tracking_fixture() -> Any:
  spec = importlib.util.spec_from_file_location("alloy_tracking_workload", TRACKING_FIXTURE)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load {TRACKING_FIXTURE}")
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def _trip_count(start: int, stop: int, step: int) -> int | None:
  if step <= 0:
    return None
  return max(0, -(-(stop - start) // step))


def _loop_counts(assembly: str) -> tuple[int, int, int]:
  total = unit = empty = 0
  for match in _FOR_RE.finditer(assembly):
    total += 1
    tc = _trip_count(*(int(g) for g in match.groups()))
    unit += int(tc == 1)
    empty += int(tc == 0)
  return total, unit, empty


def _tracking_structured_spjac(horizon: int) -> al.Function:
  tracking = _load_tracking_fixture()
  fn = tracking.tracking_eq_function_map(horizon)
  sj = al.sparse_jacobian(fn.outputs[0], fn.inputs[0])
  return al.Function(
    f"viz_map_structured_tracking_eq_jac_N{horizon}",
    [fn.inputs[0]],
    [sj.values],
    ["z"],
    ["spjac_eq_z"],
    [sj.sparsity],
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Capture an Alloy viz recording for the tracking NMPC structured eq sparse-Jacobian.")
  parser.add_argument("--horizon", "-N", type=int, default=10)
  parser.add_argument("--viz-dir", type=Path, default=DEFAULT_VIZ_DIR, help="directory that will contain recordings.json")
  parser.add_argument("--append", action="store_true", help="append to existing recordings instead of clearing first")
  parser.add_argument("--serve", action="store_true", help="serve the recording with alloy_viz after capture")
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8000)
  parser.add_argument("--browser", action="store_true", help="open a browser when serving")
  args = parser.parse_args()

  os.environ["ALLOY_VIZ_DIR"] = str(args.viz_dir)
  if not args.append:
    clear_recordings(disk=True)

  fun = _tracking_structured_spjac(args.horizon)
  label = f"tracking eq structured spjac N={args.horizon}"
  visualize_function(fun, label=label)
  try:
    source = render_c_source(fun)
  finally:
    unvisualize_function(fun)

  rec = recordings()[-1]
  print(f"captured: {rec['name']}")
  print(f"recording: {recording_path()}")
  print(f"generated C: {len(source)} bytes, {source.count(chr(10)) + 1} lines")
  print("steps:")
  for step in rec["steps"]:
    if step.get("dialect") != "prog":
      print(f"  {step['name']}")
      continue
    total, unit, empty = _loop_counts(step.get("assembly", ""))
    print(f"  {step['name']}: for={total}, unit={unit}, empty={empty}")

  if args.serve:
    serve(host=args.host, port=args.port, path=recording_path(), open_browser=args.browser)


if __name__ == "__main__":
  main()
