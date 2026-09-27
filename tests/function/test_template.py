"""Templates: a declaration with shape holes, instantiated per argument signature, cached under a deterministic name."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import aot, render_c_source
from scaly.function import model
from scaly.passes.lowering import lower_function


def _counted() -> tuple[sc.Function[Any, Any, Any, Any], list[tuple[int, ...]]]:
  traced: list[tuple[int, ...]] = []

  @sc.function(sc.L(), output="y")
  def double(x):
    traced.append(x.shape)
    return 2.0 * x

  return double, traced


def test_a_hole_makes_a_template_and_each_shape_one_instance() -> None:
  double, traced = _counted()
  assert type(double) is sc.Function and not double.is_concrete and double.instances == {}
  np.testing.assert_allclose(double(np.arange(3.0)), [0.0, 2.0, 4.0])
  np.testing.assert_allclose(double(np.ones(3)), [2.0, 2.0, 2.0])
  np.testing.assert_allclose(double(np.ones((2, 2))), 2.0 * np.ones((2, 2)))
  np.testing.assert_allclose(double(1.5), 3.0)
  assert list(double.instances) == ["double__3", "double__2x2", "double__s"]
  assert traced == [(3,), (2, 2), ()]  # one trace per instance, not per call
  # A float, an int, a NumPy scalar and a 0-d array bind the same way.
  for scalar in (2, np.float64(2.0), np.array(2.0)):
    assert double(scalar) == 4.0
  assert len(traced) == 3
  three = double.instances["double__3"]
  assert isinstance(three, sc.ConcreteFunction) and three.input_names == ("x",) and three.output_names == ("y",)
  assert "double__3(" in render_c_source(three) and "double__2x2(" in render_c_source(double.instances["double__2x2"])


def test_a_symbolic_call_instantiates_inside_another_trace() -> None:
  double, traced = _counted()

  @sc.function(3, output="z")
  def caller(x):
    return double(x) + double(x)

  assert caller.is_concrete and list(double.instances) == ["double__3"] and traced == [(3,)]
  a = sc.sym("a", 3)
  assert double(a) is double(a)  # the cache returns one instance, so the call node interns
  call = double(a)
  assert call.op == sc.ExprOp.CALL and call.attrs["callee"] is double.instances["double__3"]
  np.testing.assert_allclose(caller(np.ones(3)), 4.0 * np.ones(3))


def test_two_instances_lower_to_two_procedures() -> None:
  double, _ = _counted()

  @sc.function(3, 4, output="z")
  def both(x, y):
    return double(x).sum() + double(y).sum()

  program = lower_function(both)
  procs = [node.attrs["name"] for node in program.args if node.attrs.get("name")]
  assert "double__3" in procs and "double__4" in procs and procs.count("double__3") == 1
  np.testing.assert_allclose(both(np.ones(3), np.ones(4)), 14.0)


def test_partial_shapes_bind_only_their_holes() -> None:
  @sc.function((2, None), output="y")
  def rows(x):
    return (x * x).sum()

  np.testing.assert_allclose(rows(np.ones((2, 5))), 10.0)
  assert list(rows.instances) == ["rows__5"]
  with pytest.raises(ValueError, match=r"rows: expected shape \(2, None\) for 'x', got \(3, 5\)"):
    rows(np.ones((3, 5)))
  with pytest.raises(ValueError, match=r"expected shape \(2, None\) for 'x', got \(2,\)"):
    rows(sc.sym("v", 2))


def test_a_fully_declared_function_is_its_one_instance() -> None:
  @sc.function(3, output="y")
  def square(x):
    return x * x

  assert type(square) is sc.ConcreteFunction and square.is_concrete
  assert square.concrete is square and square.instances == {"square": square}
  assert square.instantiate(3) is square and square.instantiate(np.zeros(3)) is square
  with pytest.raises(ValueError, match=r"expected shape \(3,\) for 'x', got \(4,\)"):
    square.instantiate(4)


def test_instantiate_takes_one_declaration_per_parameter() -> None:
  @sc.function(sc.L(), sc.G(sc.L(), (2,)), output="y")
  def f(x, p):
    a, b = p
    return x.sum() + a.sum() + b.sum()

  by_shape = f.instantiate(3, ((4, 1), 2))
  assert by_shape.name == "f__3_4x1" and by_shape.input_shapes == ((3,), (4, 1), (2,))
  assert f.instantiate(np.zeros(3), (sc.TensorType((4, 1)), sc.L(2))) is by_shape
  assert f.instantiate(sc.sym("x", 3), p=(np.zeros((4, 1)), np.zeros(2))) is by_shape
  with pytest.raises(ValueError, match="does not have the declared structure"):
    f.instantiate(3, (4, 1))  # a tuple of ints is one shape, not the group's parts
  with pytest.raises(TypeError, match="instantiate takes a shape"):
    f.instantiate(1.5, (1, 2))
  with pytest.raises(TypeError, match=r"f\(\) takes 2 arguments \(x, p\), got 1"):
    f.instantiate(3)


def test_graph_attributes_of_a_template_name_the_fix() -> None:
  double, _ = _counted()
  for attr in ("concrete", "input_names", "inputs", "input_shapes"):
    with pytest.raises(sc.NotConcrete, match=r"double leaves x: any shape to its calls; build an instance with double.instantiate"):
      getattr(double, attr)
  with pytest.raises(sc.NotConcrete):
    double.factory("d", ["x"], ["y"])
  assert not hasattr(double, "no_such_attribute")
  assert repr(double) == "Function('double', (x: any shape) -> ('y',), instances=[])"


def test_loop_builders_take_an_instance_and_while_loop_binds_its_own() -> None:
  double, _ = _counted()
  with pytest.raises(sc.NotConcrete, match="instantiate"):
    sc.vmap(double, 2, [(sc.sym("xs", 6), 0, 3)])
  assert sc.vmap(double.instantiate(3), 2, [(sc.sym("xs", 6), 0, 3)]).shape == (6,)
  with pytest.raises(sc.NotConcrete, match="instantiate"):
    sc.scan(double, sc.sym("c", 2), length=3)

  @sc.function(sc.L(), (), output="go")
  def below(carry, limit):
    return carry.sum() < limit

  @sc.function(sc.L(), sc.L((), dtype="int64"), (), output="next")
  def step(carry, k, limit):
    return carry + 1.0

  @sc.function(2, output="out")
  def loop(x):
    carry, n = sc.while_loop(below, step, x, max_iter=10, index=True, params=[sc.const(6.0)])
    return carry

  np.testing.assert_allclose(loop(np.zeros(2)), [3.0, 3.0])
  assert list(below.instances) == ["below__2"] and list(step.instances) == ["step__2"]


def test_a_dtype_left_open_binds_from_an_expr_and_spells_itself() -> None:
  @sc.function(sc.L(), output="y")
  def ident(k):
    return k

  ident(sc.sym("i", 3, dtype="int64"))
  ident(sc.sym("r", 3))
  ident(np.zeros(3, dtype=np.int64))  # numerical values are coerced to the declared dtype, float64 here
  assert list(ident.instances) == ["ident__3int64", "ident__3"]

  @sc.function(sc.L("k", 3, dtype="int64"), output="y")
  def fixed(k):
    return k

  with pytest.raises(ValueError, match="call argument 'k' has dtype float64, expected int64"):
    fixed(sc.sym("r", 3))

  @sc.function(sc.L(..., dtype="int64"), output="y")
  def open_shape(k):
    return k

  with pytest.raises(ValueError, match="expected dtype int64 for 'k', got float64"):
    open_shape(sc.sym("r", 3))
  assert open_shape(sc.sym("i", 2, dtype="int64")).type.dtype == sc.dtypes.int64


def test_two_bindings_may_never_share_a_name(monkeypatch) -> None:
  double, _ = _counted()
  monkeypatch.setattr(model, "instance_name", lambda name, decls, types: f"{name}__same")
  double(np.ones(2))
  with pytest.raises(RuntimeError, match="two argument signatures would share the instance name 'double__same'"):
    double(np.ones(3))


def test_with_device_and_recompile_apply_to_every_instance() -> None:
  double, _ = _counted()
  placed = double.with_device("host")
  assert type(placed) is sc.Function and placed is not double
  double(np.ones(2))
  double(np.ones(3))
  double.recompile()
  assert all(instance._compiled is None for instance in double.instances.values())


def test_the_cli_renders_an_instance_and_refuses_a_template(tmp_path: Path, monkeypatch, capsys) -> None:
  (tmp_path / "cli_templates.py").write_text(
    textwrap.dedent(
      """
      import scaly as sc

      @sc.function(sc.L(), output="y")
      def double(x):
        return 2.0 * x

      double_3 = double.instantiate(3)

      @sc.function(output="c")
      def constant():
        return sc.const([1.0, 2.0])
      """
    )
  )
  monkeypatch.syspath_prepend(str(tmp_path))
  aot.main(["cli_templates:double_3", "-o", str(tmp_path)])
  aot.main(["cli_templates:constant", "-o", str(tmp_path)])  # a Function, not a factory, even with no arguments
  assert (tmp_path / "double__3.c").exists() and (tmp_path / "constant.c").exists()
  with pytest.raises(SystemExit):
    aot.main(["cli_templates:double", "-o", str(tmp_path)])
  assert "has shape holes; export a concrete instance instead, such as `double_3 = double.instantiate(...)`" in capsys.readouterr().err


def test_an_instance_hits_the_jit_cache_from_a_fresh_process(tmp_path: Path) -> None:
  script = textwrap.dedent(
    """
    import numpy as np
    import scaly as sc

    @sc.function(sc.L(), output="y")
    def double(x):
      return 2.0 * x

    double(np.ones(5))
    instance = double.instances["double__5"]
    print(instance._compiled.cache_key, instance._compiled.lib_path)
    """
  )
  env = {**os.environ, "SCALY_CACHE_DIR": str(tmp_path)}
  runs = [subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True).stdout.split() for _ in range(2)]
  (key_a, lib_a), (key_b, lib_b) = runs
  assert key_a == key_b and lib_a == lib_b
  assert len(list(tmp_path.rglob("*.so")) + list(tmp_path.rglob("*.dylib"))) == 1


def test_leaf_kind_checks_come_before_binding() -> None:
  double, _ = _counted()
  with pytest.raises(TypeError, match="mix Expr and numerical leaves"):
    cast(Any, sc.function(sc.L(), sc.L(), output="y")(lambda a, b: a + b))(sc.sym("a", 2), np.ones(2))
  with pytest.raises(ValueError, match="expected an Expr for 'x', got SparseMatrix"):
    double(sc.SparseMatrix.symbol("A", np.eye(2, dtype=bool)))


def test_every_inspecting_entry_point_refuses_a_template_with_holes() -> None:
  double, _ = _counted()

  @sc.function(3, output="y")
  def square(x):
    return x * x

  for inspect in (render_c_source, lower_function, sc.render_expr_assembly, lambda f: sc.custom_derivative(f)):
    with pytest.raises(sc.NotConcrete, match="double.instantiate"):
      inspect(double)
  with pytest.raises(sc.NotConcrete, match="double.instantiate"):
    sc.custom_derivative(square, jvp=double)
  for derive in (sc.gradient, sc.jacobian, sc.forward, sc.adjoint):
    with pytest.raises(sc.NotConcrete, match="double.instantiate"):
      derive(double, "y", "x")
  assert "@square" in sc.render_expr_assembly(square)


def test_a_repeated_call_skips_binding_but_not_checking(monkeypatch) -> None:
  from scaly.function.tree import L

  @sc.function(sc.G(sc.L(), sc.G(sc.L(), sc.L())), output="y")
  def nested(p):
    a, (b, c) = p
    return a.sum() + b.sum() + c.sum()

  binds = []
  original = L.bind
  monkeypatch.setattr(L, "bind", lambda self, value, what: binds.append(self.names) or original(self, value, what))
  one = np.ones(2)
  assert nested((one, (one, one))) == 6.0
  assert nested((one, (one, one))) == 6.0
  assert len(binds) == 3  # three leaves, bound once
  with pytest.raises(ValueError, match=r"nested__2_2_2.numerical_call: expected shape \(2,\) for 'p_0', got \(2, 2\)"):
    nested(((one, one), one))  # ty: ignore[no-matching-overload]  (the same flat shapes, the wrong nesting)
