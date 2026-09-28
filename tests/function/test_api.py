from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
import inspect
from typing import Any, cast

import numpy as np
import pytest

import scaly as sc
from scaly.ad.sparse import SparseJacobian
from scaly.codegen import render_c_module


def test_scoped_function_decorator_builds_fresh_named_function() -> None:
  @sc.function(sc.G(sc.L("x", 3), sc.L("p", sc.TensorType((3,), diff=False))), output=sc.L("y", ...), name="scoped")
  def scoped(inputs):
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
  @sc.function(sc.L("x", 2), output=sc.G(sc.L("out0", ...), sc.L("out1", ...)), name="pair")
  def pair(x):
    return x, x.sum()

  assert pair.output_names == ("out0", "out1")
  y, s = pair(np.array([2.0, 3.0]))
  np.testing.assert_allclose(y, np.array([2.0, 3.0]))
  np.testing.assert_allclose(s, 5.0)


def test_the_old_two_positional_decorator_is_refused_with_the_migration_hint() -> None:
  with pytest.raises(TypeError, match=r"one declaration per parameter.*did you mean sc.function\(<inputs>, output=<outputs>\)"):
    sc.function(sc.L("x", 2), sc.L("y", ...))(lambda x: x)  # ty: ignore[invalid-argument-type]


def test_decorator_slots_must_match_the_body_parameters() -> None:
  with pytest.raises(TypeError, match=r"declared 1 parameters, the body takes 2 \(x, p\)"):
    sc.function(sc.L("x", 2), output=sc.L("y", ...))(lambda x, p: x)  # ty: ignore[invalid-argument-type]
  with pytest.raises(TypeError, match=r"declared 3 parameters, the body takes 1 \(x\)"):
    sc.function(2, 2, 2)(lambda x: x)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
  "body",
  [lambda x=1.0: x, lambda *xs: xs[0], lambda x, **kw: x, lambda *, x: x],
  ids=["default", "var_positional", "var_keyword", "keyword_only"],
)
def test_decorator_refuses_parameters_that_are_not_plain_positional(body: Any) -> None:
  with pytest.raises(TypeError, match="plain positional parameters"):
    sc.function(sc.L("x", 2), output=sc.L("y", ...))(body)


@sc.function(sc.L("x", 3), (), output="f")
def _cost(x, p):
  return (x * x).sum() * p


def test_a_body_takes_one_argument_per_declared_parameter() -> None:
  xv = np.array([1.0, 2.0, 3.0])
  assert _cost.input_names == ("x", "p")
  assert _cost.output_names == ("f",)
  # Keywords bind at run time; statically a declared function takes its arguments by position.
  np.testing.assert_allclose(_cost(xv, np.array(2.0)), 28.0)
  np.testing.assert_allclose(_cost(xv, p=np.array(2.0)), 28.0)  # ty: ignore[no-matching-overload]
  np.testing.assert_allclose(_cost.numerical_call(p=2.0, x=xv), 28.0)  # ty: ignore[missing-argument, unknown-argument]
  call = _cost(sc.sym("a", 3), sc.sym("b"))
  assert call.op == sc.ExprOp.CALL and call.args[1].name == "b"
  assert _cost.symbolic_call(p=sc.sym("b"), x=sc.sym("a", 3)) is call  # ty: ignore[missing-argument, unknown-argument]

  # Grouping is not part of the C signature: the old one-group form renders the same C.
  @sc.function(sc.G(sc.L("x", 3), sc.L("p", ())), output=sc.L("f", ...), name="_cost")
  def grouped(inputs):
    x, p = inputs
    return (x * x).sum() * p

  assert render_c_module(grouped).source == render_c_module(_cost).source


def test_arity_and_keyword_errors_name_the_parameters() -> None:
  xv = np.zeros(3)
  with pytest.raises(TypeError, match=r"_cost\(\) takes 2 arguments \(x, p\), got 1"):
    _cost(xv)  # ty: ignore[no-matching-overload]
  with pytest.raises(TypeError, match=r"takes 2 arguments \(x, p\), got 3"):
    _cost.symbolic_call(sc.sym("a", 3), sc.sym("b"), sc.sym("c"))  # ty: ignore[too-many-positional-arguments]
  with pytest.raises(TypeError, match="unexpected keyword argument 'q'"):
    _cost(xv, 1.0, q=1.0)  # ty: ignore[no-matching-overload]
  grad = sc.gradient(_cost, "f", "x")
  with pytest.raises(TypeError, match="takes positional arguments only, got x, p"):
    grad(x=xv, p=1.0)  # ty: ignore[no-matching-overload]
  with pytest.raises(TypeError, match=r"_cost_grad_f_x\(\) takes 2 arguments \(x, p\), got 1"):
    grad((xv, 1.0))  # ty: ignore[no-matching-overload]


def test_unnamed_leaves_take_their_parameter_names() -> None:
  @sc.function(3, sc.G(2, sc.L("q", ()), sc.G(1, 1)), output=sc.G((), "extra"))
  def f(x, p):
    y, q, (a, b) = p
    return (x.sum() + y.sum() + q) * (a + b).sum(), x

  assert f.input_names == ("x", "p_0", "q", "p_2_0", "p_2_1")
  assert f.output_names == ("f_0", "extra")
  total, x = f(np.ones(3), (np.ones(2), np.array(1.0), (np.ones(1), np.ones(1))))
  np.testing.assert_allclose(total, 12.0)
  np.testing.assert_allclose(x, np.ones(3))

  with pytest.raises(ValueError, match="duplicate names"):
    sc.function(sc.L("x", 1), 1, output="y")(lambda y, x: x)


def test_zero_and_eight_parameters() -> None:
  @sc.function(output=sc.L("c", 2))
  def constant():
    return sc.const([1.0, 2.0])

  np.testing.assert_allclose(constant(), [1.0, 2.0])
  assert constant.symbolic_call().op == sc.ExprOp.CALL
  with pytest.raises(TypeError, match=r"constant\(\) takes 0 arguments \(\), got 1"):
    constant(())  # ty: ignore[no-matching-overload]

  @sc.function(1, 1, 1, 1, 1, 1, 1, 1, output="total")
  def wide(a, b, c, d, e, f, g, h):
    return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f + 7 * g + 8 * h

  assert wide.input_names == tuple("abcdefgh")
  np.testing.assert_allclose(wide(*(np.ones(1),) * 8), [36.0])


def test_seeded_derivatives_append_one_parameter() -> None:
  xv, seed = np.array([1.0, 2.0, 3.0]), np.array([0.5, -1.0, 2.0])
  fwd = sc.forward(_cost, "f", "x")
  adj = sc.adjoint(_cost, "f", "x")
  assert fwd.input_names == ("x", "p", "fwd:x") and adj.input_names == ("x", "p", "lam:f")
  two, three = np.array(2.0), np.array(3.0)
  np.testing.assert_allclose(fwd(xv, two, seed), 4.0 * xv @ seed)
  np.testing.assert_allclose(adj(xv, two, three), 12.0 * xv)

  @sc.function(2, (), output=sc.G("u", "v"))
  def pair(x, p):
    return p * (x * x).sum(), x * x

  lag = sc.lagrangian_hessian(pair, "x")
  assert lag.input_names == ("x", "p", "lam:u", "lam:v")
  np.testing.assert_allclose(lag(np.ones(2), np.array(3.0), (np.array(2.0), np.array([1.0, 5.0]))), np.diag([14.0, 22.0]))


def test_an_unnamed_leaf_outside_a_decorator_is_refused() -> None:
  with pytest.raises(ValueError, match="unnamed leaf"):
    sc.opt.problem(vars=sc.L(3), params=sc.L("p", ()))(lambda x, p: sc.opt.ProblemSpec(minimize=x.sum()))


def test_derivative_names_dispatch_for_expression_and_function_inputs() -> None:
  x = sc.sym("x", 2)
  y = (x * x).sum()
  fn = sc.Function.from_exprs("f", [x], [y], ["x"], ["y"])

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
  f = sc.Function.from_exprs("f", [x], [y], ["x"], ["y"])

  g_api = sc.gradient(f, "y", "x")
  g_factory = f.factory("g", ["x"], [sc.factory.Grad("y", "x")])
  xv = np.array([0.1, 0.4, 0.9])

  assert g_api.input_names == ("x",)
  assert g_api.output_names == ("grad_y_x",)
  np.testing.assert_allclose(g_api(xv), g_factory(xv))


def test_forward_convenience_api_matches_factory() -> None:
  x = sc.sym("x", 2)
  y = sc.stack([x[0] * x[1], x[0].sin()])
  f = sc.Function.from_exprs("f", [x], [y], ["x"], ["y"])

  fwd_api = sc.forward(f, "y", "x")
  fwd_factory = f.factory("fwd", ["x", "fwd:x"], [sc.factory.Fwd("y", "x")])
  xv = np.array([0.3, 2.0])
  seed = np.array([1.5, -0.25])

  assert fwd_api.input_names == ("x", "fwd:x")
  assert fwd_api.output_names == ("fwd_y_x",)
  np.testing.assert_allclose(fwd_api(xv, seed), fwd_factory((xv, seed)))


def test_adjoint_convenience_api_matches_factory() -> None:
  x = sc.sym("x", 2)
  y = sc.stack([x[0] * x[1], x[0].sin()])
  f = sc.Function.from_exprs("f", [x], [y], ["x"], ["y"])

  adj_api = sc.adjoint(f, "y", "x")
  adj_factory = f.factory("adj", ["x", "lam:y"], [sc.factory.Adj("y", "x")])
  xv = np.array([0.3, 2.0])
  lam = np.array([1.5, -0.25])

  assert adj_api.input_names == ("x", "lam:y")
  assert adj_api.output_names == ("adj_y_x",)
  np.testing.assert_allclose(adj_api(xv, lam), adj_factory((xv, lam)))


def test_seeded_factory_outputs_require_seed_inputs() -> None:
  x = sc.sym("x", 2)
  y = x * x
  f = sc.Function.from_exprs("f", [x], [y], ["x"], ["y"])

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
  f = sc.Function.from_exprs("f", [x], [y], ["x"], ["y"])

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
  nlp = sc.Function.from_exprs("nlp", [x], [f_expr, g_expr], ["x"], ["f", "g"])

  h_api = sc.lagrangian_hessian(nlp, "x")
  h_factory = nlp.factory("h", ["x", "lam:f", "lam:g"], [sc.factory.Hess("gamma", "x")], aux={"gamma": ["f", "g"]})

  xv = np.array([0.2, 0.5])
  lam_f = np.array(1.2)
  lam_g = np.array([0.3, -0.7])
  np.testing.assert_allclose(h_api(xv, (lam_f, lam_g)), h_factory((xv, lam_f, lam_g)))
