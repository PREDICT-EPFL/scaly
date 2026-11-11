from __future__ import annotations

import concurrent.futures
from dataclasses import FrozenInstanceError, fields, is_dataclass
import inspect
import threading
from typing import Any, cast

import numpy as np
import pytest

from scaly.function.model import as_concrete
import scaly as sc


def test_scoped_function_decorator_builds_fresh_named_function() -> None:
  @sc.function(sc.group(sc.arg("x", 3), sc.arg("p", sc.TensorType((3,), diff=False))), outputs=sc.arg("y", 3), name="scoped")
  def scoped(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, p = inputs
    return (x + p).sin()

  assert isinstance(scoped, sc.Function)
  assert scoped.name == "scoped"
  assert as_concrete(scoped).input_names == ("x", "p")
  assert as_concrete(scoped).output_names == ("y",)
  assert as_concrete(scoped).inputs[0].type.diff
  assert not as_concrete(scoped).inputs[1].type.diff

  xv = np.array([0.1, 0.2, 0.3])
  pv = np.array([1.0, 2.0, 3.0])
  np.testing.assert_allclose(scoped((xv, pv)), np.sin(xv + pv))


def test_derivative_names_dispatch_for_expression_and_function_inputs() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"), name="f")
  def fn(x):
    return (x * x).sum()

  (x,) = as_concrete(fn).inputs
  (y,) = as_concrete(fn).outputs

  builders = (sc.jacobian, sc.gradient, sc.hessian, sc.sparse_jacobian, sc.sparse_hessian)
  for build in builders:
    signature = inspect.signature(build)
    assert signature.parameters["wrt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert tuple(signature.parameters)[1] != "of"
    expr_result = build(y, x)
    keyword_expr_result = build(y, wrt=x)
    assert isinstance(expr_result, sc.Expr)
    assert isinstance(keyword_expr_result, sc.Expr)
    assert (expr_result.type.pattern is not None) == (build in (sc.sparse_jacobian, sc.sparse_hessian))
    dynamic_build = cast(Any, build)
    with pytest.raises(TypeError):
      dynamic_build(y, wrt=x, name="expr")
    with pytest.raises(TypeError):
      dynamic_build(y, wrt=x, extra_inputs=("p",))
    assert isinstance(build(fn, "y", "x"), sc.Function)
    assert isinstance(build(fn, of="y", wrt="x"), sc.Function)


def test_expression_sparse_derivatives_reject_non_default_layouts() -> None:
  x = sc.sym("x", 2)
  y = (x * x).sum()
  for build, layout in ((sc.sparse_jacobian, "csr"), (sc.sparse_hessian, "bogus")):
    with pytest.raises(TypeError, match=f"{build.__name__} Expr form does not accept layout"):
      cast(Any, build)(y, x, layout=layout)


def test_factory_specs_are_frozen_and_hessian_names_are_doubled() -> None:
  specs = (sc.factory.Jac, sc.factory.Grad, sc.factory.Hess, sc.factory.SpJac, sc.factory.SpHess, sc.factory.Fwd, sc.factory.Adj)
  for spec_type in specs:
    spec = spec_type("y", "x")
    assert is_dataclass(spec)
    expected_fields = {sc.factory.SpJac: ("of", "wrt", "layout"), sc.factory.SpHess: ("of", "wrt", "triangle", "layout")}.get(
      spec_type, ("of", "wrt")
    )
    assert tuple(field.name for field in fields(spec)) == expected_fields
    with pytest.raises(FrozenInstanceError):
      setattr(spec, "of", "other")

  assert is_dataclass(sc.factory.DerivSpec)
  assert sc.factory.DerivSpec.__dataclass_params__.frozen
  assert sc.factory.Hess("y", "x").output_name == "hess_y_x_x"
  assert sc.factory.SpHess("y", "x").output_name == "sphess_y_x_x"
  with pytest.raises(TypeError):
    getattr(sc.factory, "Hess")("y", "x", "other")
  with pytest.raises(TypeError):
    getattr(sc.factory, "SpHess")("y", "x", "other")


def test_forward_convenience_api_matches_factory() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2))
  def f(x):
    return sc.stack([x[0] * x[1], x[0].sin()])

  fwd_api = sc.forward(f, "y", "x")
  fwd_factory = f.factory("fwd", ["x", "fwd:x"], [sc.factory.Fwd("y", "x")])
  xv = np.array([0.3, 2.0])
  seed = np.array([1.5, -0.25])

  assert as_concrete(fwd_api).input_names == ("x", "fwd:x")
  assert as_concrete(fwd_api).output_names == ("fwd_y_x",)
  np.testing.assert_allclose(fwd_api(*(xv, seed)), fwd_factory(*(xv, seed)))


def test_adjoint_convenience_api_matches_factory() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"))
  def f(x):
    return sc.stack([x[0] * x[1], x[0].sin()])

  adj_api = sc.adjoint(f, "y", "x")
  adj_factory = f.factory("adj", ["x", "lam:y"], [sc.factory.Adj("y", "x")])
  xv = np.array([0.3, 2.0])
  lam = np.array([1.5, -0.25])

  assert as_concrete(adj_api).input_names == ("x", "lam:y")
  assert as_concrete(adj_api).output_names == ("adj_y_x",)
  np.testing.assert_allclose(adj_api(*(xv, lam)), adj_factory(*(xv, lam)))


def test_seeded_factory_outputs_require_seed_inputs() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"))
  def f(x):
    return x * x

  for spec, missing in [(sc.factory.Fwd("y", "x"), "fwd:x"), (sc.factory.Adj("y", "x"), "lam:y")]:
    try:
      _ = f.factory("bad", ["x"], [spec])
    except ValueError as e:
      assert f"undeclared symbolic inputs: ['{missing}']" in str(e)
    else:  # pragma: no cover
      raise AssertionError(f"{spec} without seed input should fail")


def test_factory_unknown_names_report_value_errors() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"))
  def f(x):
    return x * x

  cases = [
    (lambda: f.factory("bad", ["missing"], ["y"]), "unknown factory inputs: ['missing']"),
    (lambda: f.factory("bad", ["x"], ["missing"]), "unknown factory output 'missing'"),
    (lambda: f.factory("bad", ["x"], [sc.factory.Jac("missing", "x")]), "unknown factory output 'missing' in output Jac(of='missing', wrt='x')"),
    (lambda: f.factory("bad", ["x"], [sc.factory.Jac("y", "missing")]), "unknown factory input 'missing' in output Jac(of='y', wrt='missing')"),
    (lambda: f.factory("bad", ["x"], ["gamma"], aux={"gamma": ["missing"]}), "unknown factory aux outputs for 'gamma': ['missing']"),
    (lambda: f.factory("bad", ["x"], ["y"], aux={"y": ["y"]}), "factory aux output 'y' shadows an existing output"),
  ]
  for make, message in cases:
    try:
      _ = make()
    except ValueError as e:
      assert message in str(e)
    else:  # pragma: no cover
      raise AssertionError(f"{message} should fail")


def test_print_unreachable_from_the_outputs_raises_at_trace_time() -> None:
  with pytest.raises(ValueError, match=r"print 'x=\{\}' in function 'f' does not reach its outputs"):

    @sc.function(sc.arg("x", ()), outputs=sc.arg("y"))
    def f(x: sc.Expr) -> sc.Expr:
      sc.print("x={}", x)
      return x * 2.0

  @sc.function(sc.arg("x", ()), outputs=sc.arg("y"))
  def g(x: sc.Expr) -> sc.Expr:
    return sc.print("g={}", x) * 2.0

  @sc.function(sc.arg("x", ()), outputs=sc.arg("y"))
  def h(x: sc.Expr) -> sc.Expr:
    return g.symbolic_call(x) + 1.0


@pytest.mark.parametrize("transform", ["scalar", "block", "simplify", "substitute"])
def test_print_rebuilt_by_a_transform_still_reaches_the_outputs(transform) -> None:
  from scaly.ir.expr import substitute

  def body(x: sc.Expr) -> sc.Expr:
    if transform == "scalar":
      return sc.print("x={}", x).scalar()
    if transform == "block":
      return sc.print("x={}", x).block()
    if transform == "simplify":
      return sc.simplify(sc.print("x={}", x + 0.0))
    return substitute(sc.print("x={}", x), {x: x + 1.0})

  sc.function(sc.arg("x", ()), outputs=sc.arg("y"), name=transform)(body)


def test_prints_are_recorded_per_thread() -> None:
  """While one thread traces, a print built by another belongs to the other's trace."""
  started, built, finished = threading.Event(), threading.Event(), threading.Event()

  def a(x: sc.Expr) -> sc.Expr:
    started.set()
    assert built.wait(5)
    return sc.print("a={}", x)

  def build_b() -> None:
    def b(x: sc.Expr) -> sc.Expr:
      assert started.wait(5)
      y = sc.print("b={}", x)
      built.set()
      assert finished.wait(5)
      return y

    sc.function(sc.arg("x", ()), outputs=sc.arg("y"), name="b")(b)

  def build_a() -> None:
    try:
      sc.function(sc.arg("x", ()), outputs=sc.arg("y"), name="a")(a)
    finally:
      finished.set()

  with concurrent.futures.ThreadPoolExecutor(2) as pool:
    for future in [pool.submit(build_b), pool.submit(build_a)]:
      future.result()
