from scaly.function.model import as_concrete
import json

from scaly import group, Function, arg, function, render_expr_assembly, render_program_assembly
from scaly.codegen.aot import render_c_source
from scaly.passes.lowering import lower_function
from scaly.viz import clear_recordings, recordings, unvisualize_function, visualize_function


def _fun() -> Function:
  @function(arg("x", (2,)), outputs=arg("y"))
  def square_plus_one(x):
    return x * x + 1.0

  return square_plus_one


def test_expr_and_program_assembly():
  f = _fun()
  expr_asm = render_expr_assembly(f)
  assert "expr.module" in expr_asm
  assert "expr.func @square_plus_one" in expr_asm
  assert "expr.mul" in expr_asm
  assert "expr.add" in expr_asm
  assert "expr.return" in expr_asm
  assert "sem." not in expr_asm

  prog_asm = render_program_assembly(lower_function(f))
  assert "prog.module" in prog_asm
  assert "prog.proc" in prog_asm
  assert "prog.for" in prog_asm
  assert "prog.store" in prog_asm


def test_render_c_source_tracing_is_opt_in(tmp_path, monkeypatch):
  monkeypatch.setenv("SCALY_VIZ_DIR", str(tmp_path))
  f = _fun()
  clear_recordings(disk=True)
  render_c_source(f)
  assert recordings() == []
  assert not (tmp_path / "recordings.json").exists()

  visualize_function(f, label="debug square")
  try:
    src = render_c_source(f)
  finally:
    unvisualize_function(f)
  assert "square_plus_one" in src
  captured = recordings()
  assert len(captured) == 1
  assert captured[0]["name"] == "debug square"
  step_names = [s["name"] for s in captured[0]["steps"]]
  assert step_names[:3] == ["expression", "normalized:square_plus_one", "lowered"]
  assert captured[0]["steps"][0]["phase"] == "expression dialect"
  assert captured[0]["steps"][0]["dialect"] == "expr"
  assert "pass:fuse_elementwise" in step_names
  assert "pass:unroll_unit_loops" in step_names
  assert "pass:pack_workspace" in step_names
  assert step_names[-1] == "generated C"
  assert captured[0]["steps"][0]["graph"]["nodes"]
  assert captured[0]["steps"][-1]["code"] == src

  disk = json.loads((tmp_path / "recordings.json").read_text())
  assert len(disk) == 1
  assert disk[0]["name"] == "debug square"


def test_recording_keeps_original_and_normalized_expressions(tmp_path, monkeypatch):
  monkeypatch.setenv("SCALY_VIZ_DIR", str(tmp_path))

  @function(group(arg("matrix", (2, 3)), arg("vector", 2)), outputs=arg("y"), name="normalized_matmul")
  def fun(inputs):
    matrix, vector = inputs
    return matrix.T @ vector

  (output,) = as_concrete(fun).outputs
  clear_recordings(disk=True)
  visualize_function(fun)
  try:
    render_c_source(fun)
  finally:
    unvisualize_function(fun)
  original, normalized = recordings()[0]["steps"][:2]
  assert original["name"] == "expression" and "expr.transpose" in original["assembly"]
  assert normalized["name"] == "normalized:normalized_matmul" and "expr.transpose" not in normalized["assembly"]
  assert as_concrete(fun).outputs[0] is output


def test_viz_serve_is_not_shadowed_by_its_submodule():
  import scaly.viz
  from scaly.viz.serve import serve

  assert scaly.viz.serve is serve


def test_visualization_registers_existing_and_future_instances_without_tracing(tmp_path, monkeypatch):
  import numpy as np

  monkeypatch.setenv("SCALY_VIZ_DIR", str(tmp_path / "viz"))
  monkeypatch.setenv("SCALY_CACHE_DIR", str(tmp_path / "cache"))
  traces = []

  @function(arg("x"), outputs=arg("y"))
  def template(x):
    traces.append(x.shape)
    return x * x

  unvisualize_function(template)
  assert traces == []
  existing = template.instantiate((2,))
  visualize_function(template, label="all sizes")
  assert traces == [(2,)]
  clear_recordings()
  try:
    render_c_source(existing)
    np.testing.assert_array_equal(template(np.arange(3.0)), [0.0, 1.0, 4.0])
  finally:
    unvisualize_function(template)
  assert traces == [(2,), (3,)]
  assert [recording["name"] for recording in recordings()] == ["all sizes", "all sizes"]
  assert len({recording["function"] for recording in recordings()}) == 2
  render_c_source(template.instantiate((4,)))
  assert len(recordings()) == 2


def test_visualization_instance_labels_and_capture_scope(tmp_path, monkeypatch):
  from scaly.viz import capture

  monkeypatch.setenv("SCALY_VIZ_DIR", str(tmp_path))

  @function(arg("x"), outputs=arg("y"))
  def template(x):
    return x * x

  clear_recordings()
  with capture(template, label="declaration"):
    instance = template.instantiate((2,))
    with capture(instance, label="instance"):
      render_c_source(instance)
    render_c_source(instance)
    other = function(arg("x", 2), outputs=arg("y"), name=template.name)(lambda x: x + 1.0)
    render_c_source(other)
  render_c_source(instance)
  assert [recording["name"] for recording in recordings()] == ["instance", "declaration"]


def test_visualization_registration_does_not_keep_declarations_alive():
  import gc
  import weakref

  from scaly.viz.recording import begin_recording

  template = function(arg("x"), outputs=arg("y"))(lambda x: x * x)
  instance = template.instantiate((2,))
  reference = weakref.ref(template)
  visualize_function(template)
  del template
  gc.collect()
  assert reference() is None
  assert begin_recording(instance) is None


def test_visualization_fixed_wrappers_share_registration_with_cached_instances(tmp_path, monkeypatch):
  from scaly import sparse_jacobian

  monkeypatch.setenv("SCALY_VIZ_DIR", str(tmp_path))
  clear_recordings()
  f = _fun()
  visualize_function(sparse_jacobian(f), label="temporary derivative")
  derivative = sparse_jacobian(f)
  try:
    render_c_source(derivative)
  finally:
    unvisualize_function(derivative.instantiate())
  assert [recording["name"] for recording in recordings()] == ["temporary derivative"]
  render_c_source(derivative)
  assert len(recordings()) == 1
  visualize_function(f.instantiate())
  unvisualize_function(f)
  render_c_source(f)
  assert len(recordings()) == 1
