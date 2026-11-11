"""Generated symbols and locals stay distinct while preserving Function identity."""

import subprocess

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_module
from scaly.codegen.jit import JitUnavailable
from scaly.codegen.c import _includes, _render_raw_callee
from scaly.ir import program as p
from scaly.ir.program_spec import verify_program
from scaly.ir.spec import VerifyError
from scaly.ir.types import dtypes
from tests.compilers import c_compiler, check_c_syntax, cxx_compiler


@pytest.mark.parametrize(
  "first,second",
  [
    ("exit", "exit"),
    ("exit", "buffer"),
    ("exit", "assign"),
    ("exit", "param"),
    ("buffer", "buffer"),
    ("buffer", "assign"),
    ("buffer", "param"),
    ("assign", "assign"),
    ("assign", "param"),
    ("param", "param"),
  ],
)
@pytest.mark.parametrize("names,collide", [(("i.0", "i_0"), True), (("i_0", "j_0"), False)])
def test_program_declaration_collisions_match_c99(tmp_path, first, second, names, collide):
  params, body = [], []
  for kind, name in zip((first, second), names, strict=True):
    if kind == "param":
      params.append(p.buffer(name, dtypes.float64, (1,)))
    elif kind == "buffer":
      body.append(p.buffer(name, dtypes.float64, (1,)))
    elif kind == "assign":
      body.append(p.assign(name, p.const_float(0), declare=True))
    else:
      body.append(p.for_(p.range_(name, 0, 4, carried=True), [], exit_var=True))
  proc = p.proc("declarations", params, body)
  prog = p.program([proc])
  path = tmp_path / "declarations.c"
  path.write_text("\n".join([*_includes(dialect="c", prog=prog), *_render_raw_callee(proc, dialect="c")]))
  result = check_c_syntax(path)
  assert (result.returncode != 0) == collide, result.stderr
  if collide:
    with pytest.raises(VerifyError, match="duplicate C identifier 'i_0'"):
      verify_program(prog)
  else:
    verify_program(prog)


@pytest.mark.parametrize("nested", [False, True])
def test_scalar_declarations_follow_c99_block_and_loop_scopes(tmp_path, nested):
  first = p.assign("value.0", p.const_float(0), declare=True)
  second = p.assign("value_0", p.const_float(1), declare=True)
  scope = p.for_(p.range_("i", 0, 1), [second]) if nested else p.block(p.block(second))
  proc = p.proc("scopes", [], [p.block(first), scope])
  prog = p.program([proc])
  path = tmp_path / "scopes.c"
  path.write_text("\n".join([*_includes(dialect="c", prog=prog), *_render_raw_callee(proc, dialect="c")]))
  result = check_c_syntax(path)
  assert (result.returncode == 0) == nested, result.stderr
  if nested:
    verify_program(prog)
  else:
    with pytest.raises(VerifyError, match="duplicate C identifier 'value_0'"):
      verify_program(prog)


def test_repeated_local_buffers_and_scalar_updates_are_not_redeclarations(tmp_path):
  buf = p.buffer("value", dtypes.float64, (1,))
  proc = p.proc("repeated", [], [buf, buf, p.assign("scalar", p.const_float(0), declare=True), p.assign("scalar", p.const_float(1))])
  prog = p.program([proc])
  verify_program(prog)
  path = tmp_path / "repeated.c"
  path.write_text("\n".join([*_includes(dialect="c", prog=prog), *_render_raw_callee(proc, dialect="c")]))
  result = check_c_syntax(path)
  assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kind", ["exit", "buffer", "assign"])
@pytest.mark.parametrize("parameter", ["flags", "w"])
def test_local_declarations_cannot_redefine_a_buffer_parameter(tmp_path, kind, parameter):
  params = [p.buffer(parameter, dtypes.bool_, (4,))] if parameter != "w" else []
  if kind == "exit":
    local = p.for_(p.range_(parameter, 0, 4, carried=True), [], exit_var=True)
  elif kind == "buffer":
    local = p.buffer(parameter, dtypes.float64, (1,))
  else:
    local = p.assign(parameter, p.const_float(0), declare=True)
  proc = p.proc("parameter", params, [local])
  prog = p.program([proc])
  path = tmp_path / "parameter.c"
  path.write_text("\n".join([*_includes(dialect="c", prog=prog), *_render_raw_callee(proc, dialect="c")]))
  result = check_c_syntax(path)
  assert result.returncode != 0
  with pytest.raises(VerifyError, match=f"duplicate C identifier '{parameter}'"):
    verify_program(prog)


def test_same_named_callees_on_same_arguments() -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y", 3), name="same")
  def left(x):
    return (x * x).block()

  @sc.function(sc.arg("x", 3), outputs=sc.arg("y", 3), name="same")
  def right(x):
    return (x + 2.0).block()

  @sc.function(sc.arg("x", 3), outputs=sc.arg("y", 3), name="host")
  def host(x):
    return (left(x) + right(x)).block()

  x = np.array([1.0, 3.0, -2.0])
  np.testing.assert_array_equal(host(x), x * x + x + 2)
  source = render_c_module(host, lanes=1).body
  assert "same_raw(" in source
  assert "same_2_raw(" in source


@pytest.mark.parametrize("name", ["log", "logf", "new", "_Upper", "has__reserved", "SCALY_SUCCESS", "double2", "k0"])
def test_exported_reserved_symbol_is_a_user_error(name) -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2), name=name)
  def fun(x):
    return x + 1

  with pytest.raises(ValueError, match=name) as error:
    fun(np.ones(2))
  assert not isinstance(error.value, JitUnavailable)
  assert "codegen does not support" not in str(error.value)


@pytest.mark.parametrize("name", ["i_y", "t0", "new", "default", "log", "k0", "v0"])
def test_input_names_do_not_shadow_generated_names(name) -> None:
  @sc.function(sc.arg(name, 4), outputs=sc.arg("y", 4), name="locals")
  def fun(x):
    return ((x + 1).sin() + x).block()

  @sc.function(sc.arg("x", 4), outputs=sc.arg("result", 4), name="call_locals")
  def host(x):
    return fun(x).block()

  x = np.arange(4.0)
  np.testing.assert_allclose(host(x), np.sin(x + 1) + x)


def test_shared_input_output_names_keep_distinct_buffers() -> None:
  @sc.function(sc.arg("x", 2), sc.arg("y", 2), outputs=sc.group(sc.arg("y", 2), sc.arg("z", 2)), name="shared")
  def fun(x, y):
    return (x * 3).block(), (y + x).block()

  x, y = np.array([1.0, 2.0]), np.array([5.0, 6.0])
  out, z = fun(x, y)
  np.testing.assert_array_equal(out, 3 * x)
  np.testing.assert_array_equal(z, y + x)
  header = render_c_module(fun).header
  assert "y_in" in header and "y_out" in header


def test_cpp_keyword_inputs_and_self_named_sparse_output(tmp_path) -> None:
  @sc.function(sc.arg("new", 2), sc.arg("default", 2), outputs=sc.arg("y", 2), name="sparse_entry")
  def fun(x, y):
    return x * y

  sparse = sc.sparse_jacobian(fun, "y", "new", name="spjac_y_new")
  module = render_c_module(sparse, lang="cpp", lanes=1)
  assert "namespace spjac_y_new_ {" in module.header
  cc, cxx = c_compiler(), cxx_compiler()
  (tmp_path / module.header_name).write_text(module.header)
  (tmp_path / module.source_name).write_text(module.source)
  (tmp_path / "use.cpp").write_text(f'#include "{module.header_name}"\nint main() {{ return 0; }}\n')
  subprocess.run([*cc, "-c", str(tmp_path / module.source_name), "-o", str(tmp_path / "kernel.o")], check=True)
  subprocess.run(
    [*cxx, "-std=c++17", "-pedantic-errors", str(tmp_path / "use.cpp"), str(tmp_path / "kernel.o"), "-lm", "-o", str(tmp_path / "use")], check=True
  )
  subprocess.run([str(tmp_path / "use")], check=True)


def test_internal_raw_symbol_yields_to_exported_entry() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2), name="inner")
  def inner(x):
    return (x + 2).block()

  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2), name="inner_raw")
  def host(x):
    return inner(x).block()

  np.testing.assert_array_equal(host(np.array([1.0, 3.0])), [3.0, 5.0])


def test_scope_suffixes_remain_valid_and_children_see_global_names() -> None:
  from scaly.utils.names import NameScope

  names = NameScope()
  names.claim("same")
  with pytest.raises(ValueError, match="exported C identifier 'same'"):
    names.claim("same")
  child = names.child()
  assert child.allocate("same") == "same_2"
  assert child.allocate("new") == "new_"
  assert child.allocate("new") == "new_2"
  assert child.allocate("has__reserved") == "has_reserved"


@pytest.mark.solver("piqp")
def test_solver_oracles_with_same_name_keep_their_identity() -> None:
  from dataclasses import replace

  from scaly.function.model import as_concrete
  from scaly.function.concrete import ConcreteFunction
  from scaly.solvers.model import descriptor_function
  from tests.solvers.problem_helpers import build_qp

  left = build_qp(P=np.eye(2), c=np.array([1.0, 2.0]), name="left_solver").function
  right = build_qp(P=np.eye(2), c=np.array([3.0, 4.0]), name="right_solver").function
  left_desc, right_desc = as_concrete(left).descriptor, as_concrete(right).descriptor
  # Rebuild the second oracle under the first's name without changing its expressions.
  oracle = right_desc.oracle
  renamed = ConcreteFunction._from_exprs(left_desc.oracle.name, oracle.inputs, oracle.outputs, oracle.input_names, oracle.output_names)
  left = as_concrete(descriptor_function(left_desc))
  right = as_concrete(descriptor_function(replace(right_desc, oracle=renamed)))

  @sc.function(outputs=sc.arg("sum", 2), name="oracle_host")
  def host():
    zero, empty = sc.const(np.zeros(2)), sc.const(np.zeros(0))
    return left(zero, zero, empty, empty)[0] + right(zero, zero, empty, empty)[0]

  np.testing.assert_allclose(host(), [-4.0, -6.0], atol=1e-7)
  module = render_c_module(host, lanes=1)
  assert len(module._libraries) == 2
  sources = ["\n".join(text for _, text in library.sources) for library in module._libraries]
  assert all(f"void {left_desc.oracle.name}_entry_raw(" in source for source in sources)
  assert sources[0] != sources[1]


@pytest.mark.solver("piqp")
def test_solver_bridge_reserves_its_own_raw_symbol() -> None:
  from dataclasses import replace

  from scaly.function.model import as_concrete
  from scaly.solvers.model import descriptor_function
  from tests.solvers.problem_helpers import build_qp

  original = build_qp(P=np.eye(2), c=np.array([1.0, 2.0]))
  descriptor = as_concrete(original.function).descriptor
  solve = descriptor_function(replace(descriptor, name=descriptor.oracle.name))
  zero, empty = np.zeros(2), np.zeros(0)
  np.testing.assert_allclose(solve(zero, zero, empty, empty)[0], [-1.0, -2.0], atol=1e-7)


def test_header_scope_preserves_library_and_table_names() -> None:
  from scaly.function.concrete import ConcreteFunction
  from scaly.codegen.abi import buffer_idents
  from scaly.function.model import as_concrete

  @sc.function(sc.arg("new", 2), sc.arg("time", 2), sc.arg("k1", 2), outputs=sc.arg("log", 2), name="header_names")
  def fun(a, b, c):
    return a * b + c

  assert buffer_idents(as_concrete(fun)) == (["new_2", "time", "k1"], ["log"])
  for lang in ("c", "cpp"):
    header = render_c_module(fun, lang=lang, lanes=1).header
    for name in ("time", "k1", "log", "new_2"):
      assert f"{name}_t" in header
    assert "new__t" not in header

  sparse = as_concrete(sc.sparse_jacobian(fun, "log", "new"))
  named = ConcreteFunction._from_exprs("header_sparse", sparse.inputs, sparse.outputs, sparse.input_names, ["log"], sparse.output_sparsities)
  assert "namespace log {" in render_c_module(named, lang="cpp", lanes=1).header


def test_header_parameters_do_not_shadow_buffer_aliases() -> None:
  from scaly.codegen.abi import buffer_idents
  from scaly.function.model import as_concrete

  @sc.function(sc.arg("x_t", 2), sc.arg("x", 2), outputs=sc.arg("y", 2), name="alias_names")
  def fun(a, b):
    return a + b

  inputs, outputs = buffer_idents(as_concrete(fun))
  assert inputs == ["x_t", "x_2"]
  assert not set((*inputs, *outputs)) & {f"{name}_t" for name in (*inputs, *outputs)}


@pytest.mark.parametrize("input_name", ["x", "workspace_2"])
def test_workspace_parameter_does_not_shadow_exported_entry(tmp_path, input_name) -> None:
  @sc.function(sc.arg(input_name, 2048), outputs=sc.group(sc.arg("sum"), sc.arg("sumsqr")), name="workspace")
  def fun(x):
    value = x.sin()
    return value.sum(), (value * value).sum()

  module = render_c_module(fun, lanes=1, frame_budget=0)
  assert module.workspace_size > 0
  (tmp_path / module.header_name).write_text(module.header)
  source = tmp_path / module.source_name
  source.write_text(module.source)
  main = tmp_path / "main.c"
  exe = tmp_path / "main"
  main.write_text(
    f"""
#include <math.h>
#include "workspace.h"

int main(void) {{
  workspace_{input_name}_t x = {{{{0}}}};
  workspace_sum_t sum = {{{{0}}}};
  workspace_sumsqr_t sumsqr = {{{{0}}}};
  workspace_workspace_t scratch;
  x.data[0] = 0.5;
  int err = workspace_call(&x, &sum, &sumsqr, &scratch);
  if (err) return err;
  if (fabs(sum.data[0] - sin(0.5)) > 1e-12) return 10;
  if (fabs(sumsqr.data[0] - sin(0.5) * sin(0.5)) > 1e-12) return 11;
  return workspace_call(&x, &sum, &sumsqr, NULL) == SCALY_ERR_NULL_WORK ? 0 : 12;
}}
"""
  )
  subprocess.run([*c_compiler(), "-std=c11", "-Wall", "-Wextra", "-Werror", str(main), str(source), "-lm", "-o", str(exe)], check=True)
  subprocess.run([str(exe)], check=True)


@pytest.mark.parametrize(
  "input_name,first_output,second_output",
  [("f_y_t", "y", "z"), ("f_workspace_t", "y", "z"), ("x", "f_z_t", "z"), ("x", "f_workspace_t", "z")],
)
def test_c_wrapper_parameters_do_not_hide_later_typedefs(tmp_path, input_name, first_output, second_output) -> None:
  @sc.function(sc.arg(input_name, 2), outputs=sc.group(sc.arg(first_output), sc.arg(second_output)), name="f")
  def fun(x):
    return x * 2, x + 1

  module = render_c_module(fun, lanes=1)
  assert module.workspace_size == 0
  (tmp_path / module.header_name).write_text(module.header)
  source = tmp_path / module.source_name
  source.write_text(module.source)
  main = tmp_path / "main.c"
  exe = tmp_path / "main"
  main.write_text(
    f"""
#include "f.h"

int main(void) {{
  f_{input_name}_t x = {{{{2, -3}}}};
  f_{first_output}_t first = {{{{0}}}};
  f_{second_output}_t second = {{{{0}}}};
  f_workspace_t scratch;
  int err = f_call(&x, &first, &second, &scratch);
  if (err) return err;
  if (first.data[0] != 4 || first.data[1] != -6) return 10;
  if (second.data[0] != 3 || second.data[1] != -2) return 11;
  return f_call(&x, &first, &second, NULL);
}}
"""
  )
  subprocess.run([*c_compiler(), "-std=c11", "-Wall", "-Wextra", "-Werror", str(main), str(source), "-lm", "-o", str(exe)], check=True)
  subprocess.run([str(exe)], check=True)


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("suffix", ["solve", "stats", "set_options", "default_options"])
def test_host_cannot_use_a_solver_bridge_export_name(suffix: str) -> None:
  from tests.solvers.problem_helpers import build_qp

  solve = build_qp(P=np.eye(2), c=np.array([1.0, 2.0]), name="bridge_clash")

  @sc.function(outputs=sc.arg("x", 2), name=f"bridge_clash_{suffix}")
  def host():
    zero, empty = sc.const(np.zeros(2)), sc.const(np.zeros(0))
    return solve.function(zero, zero, empty, empty, ())[0]

  with pytest.raises(ValueError, match="exported C identifier.*clashes"):
    host()
