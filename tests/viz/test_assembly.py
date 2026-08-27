import json

from alloy import Function, render_expr_assembly, render_program_assembly, sym
from alloy.codegen.aot import render_c_source
from alloy.passes.lowering import lower_function
from alloy.viz import clear_recordings, recordings, unvisualize_function, visualize_function


def _fun() -> Function:
  x = sym("x", (2,))
  y = x * x + 1.0
  return Function._from_exprs("square_plus_one", [x], [y], output_names=["y"])


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
  monkeypatch.setenv("ALLOY_VIZ_DIR", str(tmp_path))
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
  assert step_names[:2] == ["expression", "lowered"]
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
