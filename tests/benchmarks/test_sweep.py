from __future__ import annotations

import numpy as np
import alloy as al
from alloy.codegen import render_c_module

from benchmarks.harness.sweep import FIELDS, _artifact_sizes, _dispatch_metrics


def test_artifact_sizes_split_static_data_from_executable_source() -> None:
  executable = "int kernel(void) { return table[0]; }\n"
  static = "static const int table[2] =\n  {1, 2};\n"
  header = "#pragma once\nint kernel(void);\n"

  artifact_bytes, executable_bytes, metadata_bytes = _artifact_sizes(executable + static, header)

  assert artifact_bytes == len((executable + static + header).encode())
  assert executable_bytes == len(executable.encode())
  assert metadata_bytes == len((static + header).encode())
  assert artifact_bytes == executable_bytes + metadata_bytes


def test_dispatch_metrics_count_retained_map_work_per_iteration() -> None:
  x = al.sym("x", 2)
  stage = al.Function("metric_stage", [x], [2.0 * x + x.sin()], ["x"], ["y"])
  z = al.sym("z", 8)
  mapped = al.Function("metric_map", [z], [al.map_(stage, 4, [(z, 0, 2)])], ["z"], ["y"])

  assert _dispatch_metrics(mapped, render_c_module(mapped).program) == (4, 0, 6)


def test_dispatch_metrics_include_stack_scratch_and_exclude_index_arithmetic() -> None:
  x = al.sym("x", 2)
  matrix = al.Expr.const(np.arange(8.0).reshape(4, 2))
  hidden = matrix @ x
  inner = al.Function("metric_workspace_inner", [x], [(hidden * hidden).sum().reshape((1,))], ["x"], ["y"])
  y = al.sym("y", 2)
  outer = al.Function("metric_workspace_outer", [y], [inner.call([y])[0] + 1], ["y"], ["z"])
  z = al.sym("z", 8)
  mapped = al.Function("metric_workspace_map", [z], [al.map_(outer, 4, [(z, 0, 2)])], ["z"], ["y"])

  # Five slots in the inner callee and its caller's one call-output slot coexist. The 25 operations
  # are floating point only: integer multiply/add nodes used in generated subscripts do not count.
  assert _dispatch_metrics(mapped, render_c_module(mapped).program) == (4, 6, 25)


def test_dispatch_metrics_include_spilled_nested_call_output() -> None:
  x = al.sym("x", 1024)
  inner = al.Function("metric_spill_inner", [x], [x * x], ["x"], ["y"])
  y = al.sym("y", 1024)
  called = inner.call([y])[0]
  outer = al.Function("metric_spill_outer", [y], [called + 1], ["y"], ["z"])
  z = al.sym("z", 2048)
  mapped = al.Function("metric_spill_map", [z], [al.map_(outer, 2, [(z, 0, 1024)])], ["z"], ["y"])

  assert _dispatch_metrics(mapped, render_c_module(mapped).program) == (2, 1024, 2048)


def test_dispatch_metrics_handle_unit_and_mixed_trip_counts() -> None:
  x = al.sym("x", 1)
  stage = al.Function("metric_scalar_stage", [x], [x * x], ["x"], ["y"])
  z = al.sym("z", 5)
  unit = al.Function("metric_unit_map", [z], [al.map_(stage, 1, [(z, 0, 1)])], ["z"], ["y"])
  mixed = al.Function(
    "metric_mixed_map",
    [z],
    [al.concat([al.map_(stage, 2, [(z, 0, 1)]), al.map_(stage, 3, [(z, 2, 1)])])],
    ["z"],
    ["y"],
  )

  assert _dispatch_metrics(unit, render_c_module(unit).program) == (1, 0, 1)
  assert _dispatch_metrics(mixed, render_c_module(mixed).program) == ("", "", "")


def test_sweep_csv_has_dispatch_and_artifact_fields() -> None:
  assert {"dispatch_trip_count", "dispatch_workspace", "dispatch_arithmetic", "coloring_width"} <= set(FIELDS)
  assert {"artifact_bytes", "executable_bytes", "static_metadata_bytes"} <= set(FIELDS)
  assert {"integer_workspace", "argument_pointers", "result_pointers"} <= set(FIELDS)
