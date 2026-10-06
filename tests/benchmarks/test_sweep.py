from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import shutil
from typing import cast

from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
import scaly as sc
import numpy as np
import pytest
from scaly.codegen import render_c_module

from benchmarks.harness import gbench
from benchmarks.harness.correctness import check_dense_reference, write_samples
from benchmarks.harness.sweep import (
  FIELDS,
  _artifact_sizes,
  _casadi_descriptor_kernel,
  _casadi_output_metadata,
  _descriptor_kernel,
  _dispatch_metrics,
  _module_info,
)


def test_artifact_sizes_split_static_data_from_executable_source() -> None:
  executable = "int kernel(void) { return table[0]; }\n"
  static = "static const int table[2] =\n  {1, 2};\n"
  header = "#pragma once\nint kernel(void);\n"

  artifact_bytes, executable_bytes, metadata_bytes = _artifact_sizes(executable + static, header)

  assert artifact_bytes == len((executable + static + header).encode())
  assert executable_bytes == len(executable.encode())
  assert metadata_bytes == len((static + header).encode())
  assert artifact_bytes == executable_bytes + metadata_bytes


def test_dispatch_metrics_count_retained_vmap_work_per_iteration() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"), name="metric_stage")
  def stage(x: sc.Expr) -> sc.Expr:
    return 2.0 * x + x.sin()

  @sc.function(sc.arg("z", 8), outputs=sc.arg("y"), name="metric_vmap")
  def mapped(z: sc.Expr) -> sc.Expr:
    return _mapped_call(stage, 4, [z])

  assert _dispatch_metrics(mapped.instantiate(), render_c_module(mapped).program) == (4, 0, 6)


def test_dispatch_metrics_include_stack_scratch_and_exclude_index_arithmetic() -> None:
  matrix = sc.Expr.const(np.arange(8.0).reshape(4, 2))

  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"), name="metric_workspace_inner")
  def inner(x: sc.Expr) -> sc.Expr:
    hidden = matrix @ x
    return (hidden * hidden).sum().reshape((1,)).block()

  @sc.function(sc.arg("y", 2), outputs=sc.arg("z"), name="metric_workspace_outer")
  def outer(y: sc.Expr) -> sc.Expr:
    return inner(y) + 1

  @sc.function(sc.arg("z", 8), outputs=sc.arg("y"), name="metric_workspace_vmap")
  def mapped(z: sc.Expr) -> sc.Expr:
    return _mapped_call(outer, 4, [z])

  # Five slots in the inner callee and its caller's one call-output slot coexist. The 25 operations
  # are floating point only: integer multiply/add nodes used in generated subscripts do not count.
  assert _dispatch_metrics(mapped.instantiate(), render_c_module(mapped).program) == (4, 6, 25)


@pytest.mark.parametrize("stages", [1, 5])
def test_dispatch_metrics_follow_hoisted_callees_and_exclude_the_prologue(stages: int) -> None:
  @sc.function(sc.group(sc.arg("x", 3), sc.arg("w", 9)), outputs=sc.arg("y"), name="metric_hoist_stage")
  def stage(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, w = inputs
    return (w.reshape((3, 3)).exp() @ x).sin().block()

  @sc.function(sc.group(sc.arg("z", 3 * stages), sc.arg("weights", 9)), outputs=sc.arg("y"), name="metric_hoist_map")
  def mapped(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, weights = inputs
    return _mapped_call(stage, stages, {"x": z, "w": (weights, 0, 0)})

  program = render_c_module(mapped).program
  assert any(proc.attrs.get("hoisted_from") == stage.name for proc in program.args)
  assert _dispatch_metrics(mapped.instantiate(), program) == (stages, 3 if stages == 1 else 24, 21)


def test_dispatch_metrics_allow_scheduled_indices_before_a_mapped_call() -> None:
  from scaly.ir.program import ProgramNode, ProgramOp, const_int

  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"), name="metric_scheduled_stage")
  def stage(x: sc.Expr) -> sc.Expr:
    return (2.0 * x + x.sin()).block()

  @sc.function(sc.arg("z", 8), outputs=sc.arg("y"), name="metric_scheduled_map")
  def mapped(z: sc.Expr) -> sc.Expr:
    return _mapped_call(stage, 4, [z])

  program = render_c_module(mapped, lanes=1).program
  proc_count = int(program.attrs["proc_count"])
  procs = list(program.args[:proc_count])
  root = procs[-1]
  param_count = int(root.attrs["param_count"])
  body = list(root.args[param_count:])
  loop_index = next(i for i, stmt in enumerate(body) if stmt.op == ProgramOp.FOR and stmt.args[-1].op == ProgramOp.CALL)
  loop = body[loop_index]
  scheduled = ProgramNode(ProgramOp.ASSIGN, (const_int(0),), {"target": "v0", "declare": True}, const_int(0).dtype)
  body[loop_index] = ProgramNode(loop.op, (loop.args[0], scheduled, *loop.args[1:]), {**loop.attrs, "body_len": 2}, loop.dtype)
  changed_root = ProgramNode(root.op, (*root.args[:param_count], *body), root.attrs, root.dtype)
  changed = ProgramNode(program.op, (*procs[:-1], changed_root, *program.args[proc_count:]), program.attrs, program.dtype)
  assert _dispatch_metrics(mapped.instantiate(), changed) == (4, 0, 6)


def test_dispatch_metrics_include_spilled_nested_call_output() -> None:
  @sc.function(sc.arg("x", 1024), outputs=sc.arg("y"), name="metric_spill_inner")
  def inner(x: sc.Expr) -> sc.Expr:
    return (x * x).block()

  @sc.function(sc.arg("y", 1024), outputs=sc.arg("z"), name="metric_spill_outer")
  def outer(y: sc.Expr) -> sc.Expr:
    return inner(y) + 1

  @sc.function(sc.arg("z", 2048), outputs=sc.arg("y"), name="metric_spill_vmap")
  def mapped(z: sc.Expr) -> sc.Expr:
    return _mapped_call(outer, 2, [z])

  assert _dispatch_metrics(mapped.instantiate(), render_c_module(mapped).program) == (2, 1024, 2048)


def test_dispatch_metrics_count_shared_scalar_arithmetic_once() -> None:
  @sc.function(sc.arg("x", 1), outputs=sc.arg("y"), name="metric_shared_stage")
  def stage(x: sc.Expr) -> sc.Expr:
    shared = x[0].sin()
    return sc.stack([shared * shared, shared + 1])

  @sc.function(sc.arg("z", 4), outputs=sc.arg("y"), name="metric_shared_vmap")
  def mapped(z: sc.Expr) -> sc.Expr:
    return _mapped_call(stage, 4, [z])

  assert _dispatch_metrics(mapped.instantiate(), render_c_module(mapped).program) == (4, 0, 3)


def test_store_pairs_preserve_dispatch_arithmetic() -> None:
  from scaly.ir.program import ProgramOp, walk_program
  from scaly.passes.lowering import lower_function

  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="metric_pair_stage")
  def stage(x: sc.Expr) -> sc.Expr:
    return (x * x).scalar()

  @sc.function(sc.arg("z", 12), outputs=sc.arg("y"), name="metric_pair_map")
  def mapped(z: sc.Expr) -> sc.Expr:
    return _mapped_call(stage, 4, [z]).sum()

  stages = {}
  lower_function(mapped, observe=lambda name, program: stages.__setitem__(name, program))
  before, paired, prepared = (stages[name] for name in ("pass:pack_workspace", "pass:coalesce_stores", "pass:prepare_scalar"))
  assert not any(node.op == ProgramOp.STORE_PAIR for node in walk_program(before))
  assert any(node.op == ProgramOp.STORE_PAIR for node in walk_program(paired))
  assert [_dispatch_metrics(mapped.instantiate(), program) for program in (before, paired, prepared)] == [(4, 0, 3)] * 3


def test_dispatch_metrics_handle_unit_and_mixed_trip_counts() -> None:
  @sc.function(sc.arg("x", 1), outputs=sc.arg("y"), name="metric_scalar_stage")
  def stage(x: sc.Expr) -> sc.Expr:
    return x * x

  @sc.function(sc.arg("z", 5), outputs=sc.arg("y"), name="metric_unit_vmap")
  def unit(z: sc.Expr) -> sc.Expr:
    return _mapped_call(stage, 1, [(z, 0, 1)])

  @sc.function(sc.arg("z", 5), outputs=sc.arg("y"), name="metric_mixed_vmap")
  def mixed(z: sc.Expr) -> sc.Expr:
    return sc.concat([_mapped_call(stage, 2, [(z, 0, 1)]), _mapped_call(stage, 3, [(z, 2, 1)])])

  assert _dispatch_metrics(unit.instantiate(), render_c_module(unit).program) == (1, 0, 1)
  # Fusion shares the first two trips, then peels the third square. The repeated
  # range therefore contains two squares, or the heavy expression and one square.
  assert _dispatch_metrics(mixed.instantiate(), render_c_module(mixed).program) == (2, 0, 2)

  @sc.function(sc.arg("x", 1), outputs=sc.arg("y"), name="metric_heavy_stage")
  def heavy(x: sc.Expr) -> sc.Expr:
    return ((x * x + x) * x).sin()

  @sc.function(sc.arg("z", 5), outputs=sc.arg("y"), name="metric_weighted_vmap")
  def weighted(z: sc.Expr) -> sc.Expr:
    return sc.concat([_mapped_call(heavy, 2, [(z, 0, 1)]), _mapped_call(stage, 3, [(z, 2, 1)])])

  assert _dispatch_metrics(weighted.instantiate(), render_c_module(weighted).program) == (2, 0, 5)


def test_unit_dispatch_excludes_an_unmapped_top_level_call() -> None:
  @sc.function(sc.arg("x", 1), outputs=sc.arg("y"), name="metric_unit_stage")
  def stage(x: sc.Expr) -> sc.Expr:
    return (x * x).block()

  @sc.function(sc.arg("x", 1), outputs=sc.arg("y"), name="metric_unmapped_stage")
  def other(x: sc.Expr) -> sc.Expr:
    return ((x + 1) * (x + 2)).block()

  @sc.function(sc.arg("z", 1), outputs=sc.arg("y"), name="metric_unit_with_call")
  def root(z: sc.Expr) -> sc.Expr:
    return _mapped_call(stage, 1, [z]) + other(z)

  assert _dispatch_metrics(root.instantiate(), render_c_module(root).program) == (1, 0, 1)


def test_sweep_csv_has_dispatch_and_artifact_fields() -> None:
  assert "layout" in FIELDS
  assert {"dispatch_trip_count", "dispatch_workspace", "dispatch_arithmetic", "coloring_width"} <= set(FIELDS)
  assert {"artifact_bytes", "executable_bytes", "static_metadata_bytes"} <= set(FIELDS)
  assert {"integer_workspace", "argument_pointers", "result_pointers"} <= set(FIELDS)


def test_descriptor_kernel_exposes_carried_hessian_coloring_width() -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="sweep_width_fixture")
  def primal(x: sc.Expr) -> sc.Expr:
    return x[0] * x[0] + x[1] * x[2]

  hessian = sc.sparse_hessian(primal, "y", "x").instantiate()
  descriptor = SimpleNamespace(name="sweep_width_fixture", hess=hessian, hess_sparsity=hessian.output_sparsities[0])
  solver = cast(sc.Solver, SimpleNamespace(function=SimpleNamespace(instantiate=lambda: SimpleNamespace(descriptor=descriptor))))

  _, sparsity, coloring_width = _descriptor_kernel(solver, "hess")

  assert sparsity == hessian.output_sparsities[0]
  assert coloring_width == hessian.output_coloring_widths[0]
  assert coloring_width == 2


def test_module_info_records_constructed_local_coloring_width() -> None:
  @sc.function(sc.group(sc.arg("a", 1), sc.arg("b", 1)), outputs=sc.arg("y"), name="module_info_piece")
  def piece(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    a, b = inputs
    return sc.stack([a, b])

  @sc.function(sc.arg("z", 5), outputs=sc.arg("y"), name="module_info_mapped")
  def mapped(z: sc.Expr) -> sc.Expr:
    return _mapped_call(piece, 4, [(z, 0, 1), (z, 1, 1)])

  built = mapped.factory("module_info_spjac", ["z"], [sc.factory.SpJac("y", "z")])
  sparsity = as_concrete(built).output_sparsities[0]
  coloring_width = as_concrete(built).output_coloring_widths[0]
  assert sparsity is not None
  assert coloring_width == 2
  assert max(sc.column_coloring(sparsity), default=-1) + 1 == 1

  module = render_c_module(built, header_name="module_info_spjac.h", source_name="module_info_spjac.c")
  info = _module_info(
    built.name,
    "scaly",
    module,
    [("z", 5)],
    sparsity,
    sparsity.shape,
    0.0,
    0.0,
    "BM_ModuleInfoSpJac",
    coloring_width=coloring_width,
    w_size=module.workspace_size,
    callable=built,
  )

  assert info["coloring_width"] == 2


def test_casadi_descriptor_kernel_uses_nlpsol_and_applies_cse_before_it() -> None:
  class FakeSolver:
    def get_function(self, name: str):
      self.function_name = name
      return self

  class FakeCasadi:
    def __init__(self) -> None:
      self.cse_inputs: list[object] = []
      self.nlpsol_inputs: list[tuple] = []

    def cse(self, expressions):
      self.cse_inputs.append(expressions)
      return ["cse_cost", "cse_constraints"]

    def nlpsol(self, name, plugin, nlp, options):
      self.nlpsol_inputs.append((name, plugin, nlp, options))
      return FakeSolver()

  ca = FakeCasadi()
  fn = _casadi_descriptor_kernel(ca, "sweep_fake", "z", "p", "cost", "constraints", "jac", expand=False, cse=True)

  assert fn.function_name == "nlp_jac_g"
  assert ca.cse_inputs == [["cost", "constraints"]]
  name, plugin, nlp, options = ca.nlpsol_inputs[0]
  assert (name, plugin) == ("sweep_fake", "ipopt")
  assert nlp == {"x": "z", "p": "p", "f": "cse_cost", "g": "cse_constraints"}
  assert options == {"expand": False, "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False}

  no_cse = FakeCasadi()
  fn = _casadi_descriptor_kernel(no_cse, "sweep_fake_hess", "z", "p", "cost", "constraints", "hess", expand=True)
  assert fn.function_name == "nlp_hess_l"
  assert no_cse.cse_inputs == []
  assert no_cse.nlpsol_inputs[0][3]["expand"] is True


def test_casadi_output_metadata_tracks_jacobian_output_and_full_layout() -> None:
  ca = pytest.importorskip("casadi")
  x, p = ca.MX.sym("x", 2), ca.MX.sym("p")
  constraints = ca.vertcat(x[0] + p, x[0] + x[1] * x[1])
  fn = _casadi_descriptor_kernel(
    ca,
    "sweep_metadata_jac",
    x,
    p,
    ca.dot(x, x),
    constraints,
    "jac",
    expand=False,
  )

  metadata = _casadi_output_metadata(fn, "jac")

  assert fn.name_in() == ["x", "p"]
  assert fn.name_out() == ["g", "jac_g_x"]
  assert metadata["output_index"] == 1
  assert metadata["requested_output_indices"] == (1,)
  assert metadata["output_names"] == ("g", "jac_g_x")
  assert metadata["layout"] == "full"
  assert metadata["n_rows"] == 2
  assert metadata["n_cols"] == 2
  assert metadata["nnz"] == 3
  assert metadata["rows"] == (0, 1, 1)
  assert metadata["cols"] == (0, 0, 1)


def test_casadi_hessian_metadata_and_values_use_upper_triangle() -> None:
  ca = pytest.importorskip("casadi")
  x, p = ca.MX.sym("x", 2), ca.MX.sym("p")
  lam_f, lam_g = ca.MX.sym("lam_f"), ca.MX.sym("lam_g", 2)
  cost = x[0] * x[0] + x[0] * x[1] + p * x[1]
  constraints = ca.vertcat(x[0] * x[1], x[1] * x[1])
  fn = _casadi_descriptor_kernel(ca, "sweep_metadata_hess", x, p, cost, constraints, "hess", expand=False)
  metadata = _casadi_output_metadata(fn, "hess")
  dense = ca.Function(
    "sweep_metadata_hess_dense",
    [x, p, lam_f, lam_g],
    [ca.hessian(lam_f * cost + ca.dot(lam_g, constraints), x)[0]],
  )
  values = fn(np.array([0.3, -0.4]), np.array([0.2]), np.array(1.1), np.array([0.7, -0.5]))
  compact = np.asarray(values.nonzeros(), dtype=np.float64)
  expected = np.asarray(dense(0.3, 0.2, 1.1, [0.7, -0.5]), dtype=np.float64)
  rows = cast(tuple[int, ...], metadata["rows"])
  cols = cast(tuple[int, ...], metadata["cols"])
  layout = cast(str, metadata["layout"])

  assert fn.name_in() == ["x", "p", "lam_f", "lam_g"]
  assert fn.name_out() == ["triu_hess_gamma_x_x"]
  assert metadata["output_index"] == 0
  assert metadata["requested_output_indices"] == (0,)
  assert metadata["output_names"] == ("triu_hess_gamma_x_x",)
  assert metadata["layout"] == "upper"
  assert all(row <= col for row, col in zip(rows, cols, strict=True))
  check_dense_reference(
    compact,
    rows,
    cols,
    expected,
    (2, 2),
    label="casadi_hess",
    layout=layout,
  )


def test_dense_reference_checks_coordinates_and_declared_layout() -> None:
  expected = np.array([[1.0, 2.0], [3.0, 4.0]])
  check_dense_reference(
    [1.0, 2.0, 3.0, 4.0],
    (0, 0, 1, 1),
    (0, 1, 0, 1),
    expected,
    expected.shape,
    label="full",
    layout="full",
  )
  check_dense_reference([1.0, 3.0, 4.0], (0, 1, 1), (0, 0, 1), expected, expected.shape, label="lower", layout="lower")
  check_dense_reference([1.0, 2.0, 4.0], (0, 0, 1), (0, 1, 1), expected, expected.shape, label="upper", layout="upper")
  sparse_expected = np.array([[1.0, 0.0], [0.0, 3.0]])
  check_dense_reference([1.0, 3.0], (0, 1), (0, 1), sparse_expected, sparse_expected.shape, label="sparse-lower", layout="lower")
  check_dense_reference([1.0, 3.0], (0, 1), (0, 1), sparse_expected, sparse_expected.shape, label="sparse-upper", layout="upper")

  with pytest.raises(RuntimeError, match=r"full: dense\[1,0\]"):
    check_dense_reference([1.0, 2.0, 9.0, 4.0], (0, 0, 1, 1), (0, 1, 0, 1), expected, expected.shape, label="full")
  with pytest.raises(RuntimeError, match="upper layout contains a lower-triangle coordinate"):
    check_dense_reference([3.0], (1,), (0,), expected, expected.shape, label="upper", layout="upper")
  with pytest.raises(ValueError, match="unsupported dense-reference layout"):
    check_dense_reference([1.0], (0,), (0,), expected, expected.shape, label="bad", layout="diagonal")
  with pytest.raises(RuntimeError, match=r"omitted: dense\[1,0\]"):
    check_dense_reference([1.0, 3.0], (0, 1), (0, 1), np.array([[1.0, 2.0], [2.0, 3.0]]), (2, 2), label="omitted", layout="lower")


def test_dense_reference_streams_sparse_triangle_without_dense_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
  def forbid(*args, **kwargs):
    raise AssertionError("dense result or triangle mask allocation")

  monkeypatch.setattr(np, "zeros", forbid)
  monkeypatch.setattr(np, "ones", forbid)
  monkeypatch.setattr(np, "tri", forbid)
  expected = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
  n_cols = expected.shape[1]
  original_isnan = np.isnan

  def guarded_isnan(value, *args, **kwargs):
    value_array = np.asarray(value)
    if value_array.size > n_cols:
      raise AssertionError("np.isnan received more than one expected row")
    return original_isnan(value, *args, **kwargs)

  monkeypatch.setattr(np, "isnan", guarded_isnan)
  check_dense_reference([1.0, 2.0, 3.0], (0, 1, 2), (0, 1, 2), expected, expected.shape, label="streamed", layout="lower")


def test_gbench_driver_uses_selected_nlpsol_result_and_input_contract(tmp_path) -> None:
  info = {
    "name": "casadi_jac_fixture",
    "symbol": "nlp_jac_g",
    "backend": "casadi_mx",
    "header": "casadi_jac_fixture.h",
    "inputs": [("x", 2), ("p", 1)],
    "n_rows": 2,
    "n_cols": 2,
    "nnz": 3,
    "rows": (0, 1, 1),
    "cols": (0, 0, 1),
    "w_size": 5,
    "iw_size": 2,
    "arg_size": 4,
    "res_size": 3,
    "output_index": 1,
    "requested_output_indices": (1,),
    "output_nnz": (2, 3),
    "layout": "full",
    "benchmark": "BM_CasadiJacFixture",
  }
  input_paths = {name: tmp_path / f"sample_{name}.bin" for name, _ in info["inputs"]}
  expected_path = tmp_path / "expected_dense.bin"
  source = gbench.write_cpp(info, tmp_path, input_paths, expected_path).read_text()

  assert "static std::array<double, 4> dense{};" in source
  assert "static std::array<double, 5> w{};" in source
  assert "static std::array<double, 3> g_out_1{};" in source
  assert "g_out_0" not in source
  assert "std::array<double*, 3> res{};" in source
  assert source.count("res.fill(nullptr);") == 2
  assert source.count("res[1] = g_out_1.data();") == 2
  assert "res[0] =" not in source
  assert "arg[0] = g_x.data();" in source
  assert "arg[1] = g_p.data();" in source
  assert source.count("nlp_jac_g(arg.data()") == 2


def test_gbench_driver_sizes_selected_hessian_output(tmp_path) -> None:
  info = {
    "name": "casadi_hess_fixture",
    "symbol": "nlp_hess_l",
    "backend": "casadi_mx",
    "header": "casadi_hess_fixture.h",
    "inputs": [("x", 2), ("p", 1), ("lam_f", 1), ("lam_g", 1)],
    "n_rows": 2,
    "n_cols": 2,
    "nnz": 3,
    "rows": (0, 0, 1),
    "cols": (0, 1, 1),
    "w_size": 5,
    "iw_size": 2,
    "arg_size": 6,
    "res_size": 2,
    "output_index": 0,
    "requested_output_indices": (0,),
    "output_nnz": (3,),
    "layout": "upper",
    "benchmark": "BM_CasadiHessFixture",
  }
  input_paths = {name: tmp_path / f"sample_{name}.bin" for name, _ in info["inputs"]}
  expected_path = tmp_path / "expected_dense.bin"
  source = gbench.write_cpp(info, tmp_path, input_paths, expected_path).read_text()

  assert "static std::array<double, 3> g_out_0{};" in source
  assert "g_out_1" not in source
  assert "std::array<double*, 2> res{};" in source
  assert source.count("res[0] = g_out_0.data();") == 2
  assert source.count("nlp_hess_l(arg.data()") == 2


def _run_compiled_driver(
  tmp_path: Path, info: dict, source: str, header: str, inputs: dict[str, np.ndarray], expected: np.ndarray
) -> tuple[str, float | None, str]:
  compiler = gbench.compiler()
  if shutil.which(compiler) is None:
    pytest.skip("a C++ compiler is required for the driver smoke test")
  prefix = gbench.ROOT / "benchmarks" / "third_party" / "gbench" / gbench.GBENCH_TAG
  if not (prefix / "include" / "benchmark" / "benchmark.h").exists() or not (prefix / "lib" / "libbenchmark.a").exists():
    pytest.skip("the pinned Google Benchmark build is not available")

  (tmp_path / info["header"]).write_text(header)
  (tmp_path / info["source"]).write_text(source)
  input_paths, expected_path = write_samples(tmp_path, inputs, expected)
  gbench.write_cpp(info, tmp_path, input_paths, expected_path)
  compile_status, _, compile_note = gbench.compile_kernel(info, tmp_path, timeout=30)
  assert compile_status == "ok", compile_note
  return gbench.run_kernel(info, tmp_path, "0.001", timeout=30)


def test_compiled_driver_checks_casadi_result_index_and_kernel_input_order(tmp_path: Path) -> None:
  header = """
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
int nlp_jac_g(const double** arg, double** res, int* iw, double* w, int mem);
#ifdef __cplusplus
}
#endif
"""
  source = """
#include "casadi_jac_fixture.h"
int nlp_jac_g(const double** arg, double** res, int*, double*, int) {
  if (!arg[0] || !arg[1] || arg[0][0] != 2.0 || arg[1][0] != 3.0) return 11;
  if (res[0] != 0 || !res[1]) return 12;
  res[1][0] = arg[0][0];
  res[1][1] = arg[1][0];
  return 0;
}
"""

  class Kernel:
    def n_in(self) -> int:
      return 2

    def name_in(self, index: int) -> str:
      return ("first", "second")[index]

  info = {
    "name": "constructor_label",
    "symbol": "nlp_jac_g",
    "backend": "casadi_mx",
    "source": Path("casadi_jac_fixture.c"),
    "header": "casadi_jac_fixture.h",
    "callable": Kernel(),
    "inputs": [("second", 1), ("first", 1)],
    "n_rows": 1,
    "n_cols": 2,
    "nnz": 2,
    "rows": (0, 0),
    "cols": (0, 1),
    "w_size": 5,
    "iw_size": 2,
    "arg_size": 4,
    "res_size": 3,
    "output_index": 1,
    "requested_output_indices": (1,),
    "output_nnz": (1, 2),
    "layout": "full",
    "benchmark": "BM_CompiledCasadiJacFixture",
  }
  _run_compiled_driver(tmp_path, info, source, header, {"first": np.array([2.0]), "second": np.array([3.0])}, np.array([2.0, 3.0]))


def test_compiled_driver_runs_a_rendered_scaly_kernel_with_sanitized_inputs(tmp_path: Path) -> None:
  @sc.function(sc.group(sc.arg("x", 1), sc.arg("lam:f", 1), sc.arg("lam:g", 1)), outputs=sc.arg("y"), name="rendered_driver_kernel")
  def kernel(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
    x, lam_f, lam_g = inputs
    return x + lam_f + 2.0 * lam_g

  module = render_c_module(kernel, header_name="rendered_driver_kernel.h", source_name="rendered_driver_kernel.c")
  info = {
    "name": "wrong_construction_label",
    "backend": "scaly",
    "source": Path(module.source_name),
    "header": module.header_name,
    "callable": kernel,
    "inputs": [("lam_g", 1), ("x", 1), ("lam_f", 1)],
    "n_rows": 1,
    "n_cols": 1,
    "nnz": 1,
    "rows": (0,),
    "cols": (0,),
    "w_size": module.workspace_size,
    "arg_size": 3,
    "res_size": 1,
    "output_nnz": (1,),
    "layout": "full",
    "benchmark": "BM_CompiledScalyFixture",
  }
  _run_compiled_driver(
    tmp_path,
    info,
    module.source,
    module.header,
    {"x": np.array([2.0]), "lam_f": np.array([3.0]), "lam_g": np.array([5.0])},
    np.array([15.0]),
  )


def test_compiled_driver_rejects_omitted_nonzero_triangle_entry(tmp_path: Path) -> None:
  header = """
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
int sparse_lower_fixture(const double** arg, double** res, const int* iw, double* w, int mem);
#ifdef __cplusplus
}
#endif
"""
  source = """
#include "sparse_lower_fixture.h"
int sparse_lower_fixture(const double**, double** res, const int*, double*, int) {
  if (!res[0]) return 11;
  res[0][0] = 1.0;
  res[0][1] = 3.0;
  return 0;
}
"""
  info = {
    "name": "sparse_lower_fixture",
    "symbol": "sparse_lower_fixture",
    "backend": "scaly",
    "source": Path("sparse_lower_fixture.c"),
    "header": "sparse_lower_fixture.h",
    "inputs": [],
    "n_rows": 2,
    "n_cols": 2,
    "nnz": 2,
    "rows": (0, 1),
    "cols": (0, 1),
    "w_size": 1,
    "arg_size": 1,
    "res_size": 1,
    "output_index": 0,
    "requested_output_indices": (0,),
    "output_nnz": (2,),
    "layout": "lower",
    "benchmark": "BM_CompiledSparseLowerFixture",
  }
  status, _, note = _run_compiled_driver(tmp_path, info, source, header, {}, np.array([1.0, 2.0, 2.0, 3.0]))
  assert status == "correctness_fail", note


def test_dispatch_workspace_counts_promoted_lane_scratch():
  @sc.function(sc.arg("x", 12), outputs=sc.arg("y", (3, 3)), name="metric_lane_stage")
  def stage(x: sc.Expr) -> sc.Expr:
    matrix = x.reshape((3, 4))
    return (matrix @ matrix.T).sin().block()

  @sc.function(sc.arg("z", 60), outputs=sc.arg("y", 45), name="metric_lane_map")
  def mapped(z: sc.Expr) -> sc.Expr:
    return sc.vmap(stage, 5)(z).vec()

  assert _dispatch_metrics(mapped.instantiate(), render_c_module(mapped, lanes=1).program) == (5, 21, 81)
  assert _dispatch_metrics(mapped.instantiate(), render_c_module(mapped, lanes=4).program) == (5, 168, 81)
