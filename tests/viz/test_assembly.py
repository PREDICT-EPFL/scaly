from scaly.function.model import as_concrete
import json

from scaly import group, Function, arg, function, render_expr_assembly, render_program_assembly
from scaly.codegen.aot import render_c_source
from scaly.passes.lowering import lower_function
from scaly.viz import clear_recordings, recordings, unvisualize_function, visualize_function


def _fun() -> Function:
  @function(arg("x", (2,)), outputs=arg("y", ...))
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

  @function(group(arg("matrix", (2, 3)), arg("vector", 2)), outputs=arg("y", ...), name="normalized_matmul")
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
