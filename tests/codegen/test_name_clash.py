"""Generated symbols and locals stay distinct while preserving Function identity."""

import shutil
import subprocess

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_module
from scaly.codegen.jit import JitUnavailable


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
  cc, cxx = shutil.which("cc"), shutil.which("c++")
  assert cc and cxx
  (tmp_path / module.header_name).write_text(module.header)
  (tmp_path / module.source_name).write_text(module.source)
  (tmp_path / "use.cpp").write_text(f'#include "{module.header_name}"\nint main() {{ return 0; }}\n')
  subprocess.run([cc, "-c", str(tmp_path / module.source_name), "-o", str(tmp_path / "kernel.o")], check=True)
  subprocess.run(
    [cxx, "-std=c++17", "-pedantic-errors", str(tmp_path / "use.cpp"), str(tmp_path / "kernel.o"), "-lm", "-o", str(tmp_path / "use")], check=True
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
  symbols = dict(module.program.attrs["function_symbols"])
  assert symbols[left_desc.oracle] != symbols[renamed]


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
