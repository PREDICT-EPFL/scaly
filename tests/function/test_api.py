from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
import inspect
from typing import Any, cast

import numpy as np
import pytest

import scaly as sc
from scaly.ad.sparse import SparseJacobian


def test_scoped_function_decorator_builds_fresh_named_function() -> None:
  @sc.function(sc.G(sc.L("x", 3), sc.L("p", sc.TensorType((3,), diff=False))), sc.L("y", ...), name="scoped")
  def scoped(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, p = inputs
    return (x + p).sin()

  assert isinstance(scoped, sc.Function)
  assert scoped.name == "scoped"
  assert scoped.input_names == ("x", "p")
  assert scoped.output_names == ("y",)
  assert scoped.inputs[0].type.diff
  assert not scoped.inputs[1].type.diff

  xv = np.array([0.1, 0.2, 0.3])
  pv = np.array([1.0, 2.0, 3.0])
  np.testing.assert_allclose(scoped((xv, pv)), np.sin(xv + pv))


def test_scoped_function_decorator_outputs_default_names() -> None:
  @sc.function(sc.L("x", 2), sc.G(sc.L("out0", ...), sc.L("out1", ...)), name="pair")
  def pair(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return x, x.sum()

  assert pair.output_names == ("out0", "out1")
  y, s = pair(np.array([2.0, 3.0]))
  np.testing.assert_allclose(y, np.array([2.0, 3.0]))
  np.testing.assert_allclose(s, 5.0)


def test_derivative_names_dispatch_for_expression_and_function_inputs() -> None:
  x = sc.sym("x", 2)
  y = (x * x).sum()
  fn = sc.Function._from_exprs("f", [x], [y], ["x"], ["y"])

  builders = (sc.jacobian, sc.gradient, sc.hessian, sc.sparse_jacobian, sc.sparse_hessian)
  for build in builders:
    signature = inspect.signature(build)
    assert signature.parameters["wrt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert tuple(signature.parameters)[1] != "of"
    expr_result = build(y, x)
    keyword_expr_result = build(y, wrt=x)
    if build in (sc.sparse_jacobian, sc.sparse_hessian):
      assert isinstance(expr_result, SparseJacobian)
      assert isinstance(keyword_expr_result, SparseJacobian)
    else:
      assert isinstance(expr_result, sc.Expr)
      assert isinstance(keyword_expr_result, sc.Expr)
    dynamic_build = cast(Any, build)
    with pytest.raises(TypeError):
      dynamic_build(y, wrt=x, name="expr")
    with pytest.raises(TypeError):
      dynamic_build(y, wrt=x, extra_inputs=("p",))
    assert isinstance(build(fn, "y", "x"), sc.Function)
    assert isinstance(build(fn, of="y", wrt="x"), sc.Function)


def test_factory_specs_are_frozen_and_hessian_names_are_doubled() -> None:
  specs = (sc.factory.Jac, sc.factory.Grad, sc.factory.Hess, sc.factory.SpJac, sc.factory.SpHess, sc.factory.Fwd, sc.factory.Adj)
  for spec_type in specs:
    spec = spec_type("y", "x")
    assert is_dataclass(spec)
    expected_fields = ("of", "wrt", "triangle") if spec_type is sc.factory.SpHess else ("of", "wrt")
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


def test_gradient_convenience_api_matches_factory() -> None:
  x = sc.sym("x", 3)
  y = (x.sin() + x * x).sum()
  f = sc.Function._from_exprs("f", [x], [y], ["x"], ["y"])

  g_api = sc.gradient(f, "y", "x")
  g_factory = f.factory("g", ["x"], [sc.factory.Grad("y", "x")])
  xv = np.array([0.1, 0.4, 0.9])

  assert g_api.input_names == ("x",)
  assert g_api.output_names == ("grad_y_x",)
  np.testing.assert_allclose(g_api(xv), g_factory(xv))


def test_forward_convenience_api_matches_factory() -> None:
  x = sc.sym("x", 2)
  y = sc.stack([x[0] * x[1], x[0].sin()])
  f = sc.Function._from_exprs("f", [x], [y], ["x"], ["y"])

  fwd_api = sc.forward(f, "y", "x")
  fwd_factory = f.factory("fwd", ["x", "fwd:x"], [sc.factory.Fwd("y", "x")])
  xv = np.array([0.3, 2.0])
  seed = np.array([1.5, -0.25])

  assert fwd_api.input_names == ("x", "fwd:x")
  assert fwd_api.output_names == ("fwd_y_x",)
  np.testing.assert_allclose(fwd_api((xv, seed)), fwd_factory((xv, seed)))


def test_adjoint_convenience_api_matches_factory() -> None:
  x = sc.sym("x", 2)
  y = sc.stack([x[0] * x[1], x[0].sin()])
  f = sc.Function._from_exprs("f", [x], [y], ["x"], ["y"])

  adj_api = sc.adjoint(f, "y", "x")
  adj_factory = f.factory("adj", ["x", "lam:y"], [sc.factory.Adj("y", "x")])
  xv = np.array([0.3, 2.0])
  lam = np.array([1.5, -0.25])

  assert adj_api.input_names == ("x", "lam:y")
  assert adj_api.output_names == ("adj_y_x",)
  np.testing.assert_allclose(adj_api((xv, lam)), adj_factory((xv, lam)))


def test_seeded_factory_outputs_require_seed_inputs() -> None:
  x = sc.sym("x", 2)
  y = x * x
  f = sc.Function._from_exprs("f", [x], [y], ["x"], ["y"])

  for spec, missing in [(sc.factory.Fwd("y", "x"), "fwd:x"), (sc.factory.Adj("y", "x"), "lam:y")]:
    try:
      _ = f.factory("bad", ["x"], [spec])
    except ValueError as e:
      assert f"undeclared symbolic inputs: ['{missing}']" in str(e)
    else:  # pragma: no cover
      raise AssertionError(f"{spec} without seed input should fail")


def test_factory_unknown_names_report_value_errors() -> None:
  x = sc.sym("x", 2)
  y = x * x
  f = sc.Function._from_exprs("f", [x], [y], ["x"], ["y"])

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


def test_lagrangian_hessian_convenience_api() -> None:
  x = sc.sym("x", 2)
  f_expr = x.sin().sum()
  g_expr = x * x
  nlp = sc.Function._from_exprs("nlp", [x], [f_expr, g_expr], ["x"], ["f", "g"])

  h_api = sc.lagrangian_hessian(nlp, "x")
  h_factory = nlp.factory("h", ["x", "lam:f", "lam:g"], [sc.factory.Hess("gamma", "x")], aux={"gamma": ["f", "g"]})

  xv = np.array([0.2, 0.5])
  lam_f = np.array(1.2)
  lam_g = np.array([0.3, -0.7])
  np.testing.assert_allclose(h_api((xv, (lam_f, lam_g))), h_factory((xv, lam_f, lam_g)))
