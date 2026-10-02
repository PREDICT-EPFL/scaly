"""Shape template bindings, concrete graph identity, and lifted derivative instances."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_source, render_c_module
from scaly.passes.lowering import LoweringError


def test_shape_bindings_trace_once_and_share_symbolic_and_numerical_instances() -> None:
  traces = []

  @sc.function(sc.arg("x"), sc.arg("p", ()), outputs=sc.arg("cost", ()))
  def cost(x: sc.Expr, p: sc.Expr) -> sc.Expr:
    traces.append(x.shape)
    return (x * x).sum() * p

  assert not cost.instances
  first = cost.instantiate((3, ()))
  np.testing.assert_array_equal(cost(np.arange(3.0), np.array(2.0)), 10.0)
  assert cost.symbolic_call(sc.sym("actual", 3), sc.const(2.0)).attrs["callee"] is first
  assert cost.instantiate((sc.TensorType((3,)), sc.TensorType(()))) is first
  assert traces == [(3,)]
  second = cost.instantiate((4, ()))
  assert first is not second and first.name != second.name
  assert "__" not in first.name
  assert traces == [(3,), (4,)]
  with pytest.raises(TypeError, match="shape holes"):
    cost.instantiate()
  with pytest.raises(TypeError, match="contradicts"):
    cost.instantiate((3, 1))
  with pytest.raises(ValueError, match="expected shape"):
    cost(np.ones(3), np.ones(1))


def test_fixed_declarations_keep_names_and_fail_at_the_decorator() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2), name="fixed")
  def fixed(x: sc.Expr) -> sc.Expr:
    return x * x

  assert fixed.instantiate().name == "fixed"
  assert fixed.instantiate() is fixed.instantiate((2,))
  with pytest.raises(TypeError, match="expected shape"):
    sc.function(sc.arg("x", 2), outputs=sc.arg("y", 3))(lambda x: x)


def test_bare_structure_is_part_of_the_binding_and_name() -> None:
  @sc.function()
  def first(value):
    return value[0] if isinstance(value, tuple) else value

  flat = first.symbolic_call(sc.sym("flat", 2)).attrs["callee"]
  nested = first.symbolic_call((sc.sym("nested", 2),)).attrs["callee"]
  assert flat is not nested and flat.name != nested.name
  np.testing.assert_array_equal(first(np.arange(2.0)), np.arange(2.0))
  assert len(first.instances) == 2
  with pytest.raises(TypeError, match="bare function leaves"):
    first([1.0, 2.0])
  with pytest.raises(TypeError, match="call it instead"):
    first.instantiate((2,))
  with pytest.raises(ValueError, match="float64"):
    first.symbolic_call(sc.sym("integer", 2, dtype=sc.dtypes.int64, diff=False))


def test_specializations_generate_distinct_procedures() -> None:
  @sc.function(sc.arg("x"), outputs=sc.arg("y"))
  def square(x: sc.Expr) -> sc.Expr:
    return x * x

  @sc.function(sc.arg("x", 5), outputs=sc.arg("y", ()))
  def host(x: sc.Expr) -> sc.Expr:
    return square(x[:2]).sum() + square(x[2:]).sum()

  source = render_c_source(host, lanes=1)
  assert all(instance.name in source for instance in square.instances.values())
  np.testing.assert_array_equal(host(np.arange(5.0)), 30.0)
  with pytest.raises(TypeError, match="shape holes"):
    render_c_module(square)


def test_duplicate_generated_identifiers_fail_before_call_reuse() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2), name="same")
  def left(x: sc.Expr) -> sc.Expr:
    return x * x

  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2), name="same")
  def right(x: sc.Expr) -> sc.Expr:
    return x + 2.0

  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2))
  def host(x: sc.Expr) -> sc.Expr:
    return left(x) + right(x)

  with pytest.raises(LoweringError, match="distinct function instances"):
    render_c_source(host)


@pytest.mark.parametrize("operation", [sc.gradient, sc.jacobian, sc.hessian, sc.sparse_jacobian, sc.sparse_hessian])
def test_unseeded_derivatives_bind_and_keep_declared_output_names(operation) -> None:
  @sc.function(sc.arg("x"), outputs=sc.arg("f", ()))
  def cost(x: sc.Expr) -> sc.Expr:
    return (x * x).sum()

  derivative = operation(cost, name="derived")
  two = derivative.instantiate((2,))
  three = derivative.instantiate((3,))
  assert two.name != three.name
  assert two.output_names == derivative.outputs.names
  assert operation(cost, name="derived").instantiate((2,)) is two
  assert operation(cost).instantiate((2,)).input_tree.types == cost.instantiate((2,)).input_tree.types
  values = derivative(np.array([2.0, 3.0]))
  assert np.all(np.isfinite(values))
  if operation is sc.gradient:
    second = sc.jacobian(derivative, "grad_f_x", "x")
    np.testing.assert_array_equal(second(np.array([2.0, 3.0])), 2.0 * np.eye(2))


def test_seeded_derivative_defaults_preserve_whole_leaf_types() -> None:
  @sc.function(sc.arg("x", sc.TensorType((2,), diff=False)), outputs=sc.arg("y"))
  def square(x: sc.Expr) -> sc.Expr:
    return x * x

  forward = sc.forward(square)
  adjoint = sc.adjoint(square)
  assert sc.forward(square).instantiate() is forward.instantiate()
  assert sc.adjoint(square).instantiate() is adjoint.instantiate()
  assert forward.instantiate().input_tree.types[-1] == square.instantiate().input_tree.types[0]
  assert adjoint.instantiate().input_tree.types[-1] == square.instantiate().output_tree.types[0]
  np.testing.assert_array_equal(forward(np.array([2.0, 3.0]), np.ones(2)), [4.0, 6.0])


def test_seeded_templates_and_lagrangian_multiplier_trees() -> None:
  @sc.function(sc.arg("x"), outputs=sc.group(sc.arg("f", ()), sc.arg("g")))
  def problem(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return (x * x).sum(), x * x * x

  x = np.array([2.0, 3.0])
  np.testing.assert_array_equal(sc.forward(problem, "g", "x")(x, np.ones(2)), 3.0 * x * x)
  np.testing.assert_array_equal(sc.adjoint(problem, "g", "x")(x, np.ones(2)), 3.0 * x * x)
  for operation in (sc.lagrangian_hessian, sc.sparse_lagrangian_hessian):
    derivative = operation(problem, "x")
    result = derivative(x, (np.array(2.0), np.array([1.0, 2.0])))
    expected = np.diag(4.0 + 6.0 * np.array([1.0, 2.0]) * x)
    instance = next(iter(derivative.instances.values()))
    pattern = instance.output_sparsities[0]
    if pattern is not None:
      expected = expected[np.asarray(pattern.rows), np.asarray(pattern.cols)]
    np.testing.assert_array_equal(result, expected)
    with pytest.raises(ValueError, match="seed or argument types"):
      derivative(x, (np.array(2.0), np.ones(3)))


def test_template_cache_hit_in_another_process(tmp_path: Path) -> None:
  script = tmp_path / "template.py"
  script.write_text(
    "import sys\nimport numpy as np\nimport scaly as sc\nfrom scaly.codegen import jit\n"
    '@sc.function(sc.arg("x"), outputs=sc.arg("y"))\ndef square(x: sc.Expr) -> sc.Expr:\n  return x * x\n'
    'if len(sys.argv) > 1:\n  original = jit.subprocess.run\n  def refuse(command, **kwargs):\n    if "-shared" in command or "-dynamiclib" in command:\n      raise AssertionError("unexpected compilation")\n    return original(command, **kwargs)\n  jit.subprocess.run = refuse\n'
    "np.testing.assert_array_equal(square(np.arange(3.0)), [0.0, 1.0, 4.0])\n"
    "print(square.instantiate((3,))._compile().cache_key)\n"
  )
  env = {**os.environ, "SCALY_CACHE_DIR": str(tmp_path / "cache")}
  first = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, check=True)
  second = subprocess.run([sys.executable, str(script), "cached"], env=env, capture_output=True, text=True, check=True)
  assert first.stdout == second.stdout

  cold = subprocess.run(
    [sys.executable, str(script), "cached"], env={**env, "SCALY_CACHE_DIR": str(tmp_path / "cold")}, capture_output=True, text=True
  )
  assert cold.returncode != 0 and "unexpected compilation" in cold.stderr


@pytest.mark.parametrize("template_first", [False, True])
def test_derivative_names_do_not_depend_on_concrete_or_template_call_history(template_first) -> None:
  @sc.function(sc.arg("x"), outputs=sc.arg("f", ()))
  def cost(x: sc.Expr) -> sc.Expr:
    return (x * x).sum()

  concrete = cost.instantiate((2,))
  if template_first:
    template = sc.gradient(cost, name="derived").instantiate((2,))
    fixed = sc.gradient(concrete, name="derived").instantiate()
  else:
    fixed = sc.gradient(concrete, name="derived").instantiate()
    template = sc.gradient(cost, name="derived").instantiate((2,))
  assert fixed.name == "derived"
  assert template.name.startswith("derived_2_t")
  assert fixed is not template


def test_bare_zero_input_name_has_no_reserved_separator() -> None:
  @sc.function()
  def zero() -> sc.Expr:
    return sc.const(2.0)

  assert zero() == np.array(2.0)
  instance = zero.symbolic_call().attrs["callee"]
  assert instance.name.startswith("zero_t") and "__" not in instance.name


def test_concrete_symbolic_call_checks_dtype() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2))
  def identity(x: sc.Expr) -> sc.Expr:
    return x

  with pytest.raises(ValueError, match="expected dtype"):
    identity.instantiate().symbolic_call(sc.sym("integer", 2, dtype=sc.dtypes.int64, diff=False))


def test_partial_declaration_names_encode_only_holes_and_detect_collisions(monkeypatch) -> None:
  from importlib import import_module

  model = import_module("scaly.function.model")

  def declare():
    return sc.function(sc.arg("x"), sc.arg("p", 4), outputs=sc.arg("y"), name="partial")(lambda x, p: x)

  first, second = declare(), declare()
  name = first.instantiate((3, 4)).name
  second.instantiate((5, 4))
  assert second.instantiate((3, 4)).name == name
  assert name.startswith("partial_3_t")
  monkeypatch.setattr(model, "_mangle", lambda *args: "collision")
  collision = declare()
  collision.instantiate((2, 4))
  with pytest.raises(ValueError, match="instance name collision"):
    collision.instantiate((3, 4))


def test_cli_requires_an_explicit_template_binding(tmp_path, monkeypatch) -> None:
  from scaly.codegen.aot import main

  module = tmp_path / "template_export.py"
  module.write_text(
    'import scaly as sc\n@sc.function(sc.arg("x"), outputs=sc.arg("y"))\ndef square(x: sc.Expr) -> sc.Expr:\n  return x * x\n'
    "bound = square.instantiate((2,))\n"
  )
  monkeypatch.syspath_prepend(str(tmp_path))
  with pytest.raises(TypeError, match="shape holes"):
    main(["template_export:square", "-o", str(tmp_path / "unbound")])
  main(["template_export:bound", "-o", str(tmp_path / "bound")])
  assert len(tuple((tmp_path / "bound").glob("*.c"))) == 1
