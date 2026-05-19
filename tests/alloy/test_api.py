from __future__ import annotations

import numpy as np

import alloy as al


def test_scoped_function_decorator_builds_fresh_named_function() -> None:
  @al.function("scoped", {"x": 3, "p": al.TensorType((3,), diff=False)})
  def scoped(x, p):
    return {"y": (x + p).sin()}

  assert isinstance(scoped, al.Function)
  assert scoped.name == "scoped"
  assert scoped.input_names == ("x", "p")
  assert scoped.output_names == ("y",)
  assert scoped.inputs[0].type.diff
  assert not scoped.inputs[1].type.diff

  xv = np.array([0.1, 0.2, 0.3])
  pv = np.array([1.0, 2.0, 3.0])
  np.testing.assert_allclose(scoped(xv, pv), np.sin(xv + pv))


def test_scoped_function_decorator_outputs_default_names() -> None:
  @al.function("pair", {"x": 2})
  def pair(x):
    return x, x.sum()

  assert pair.output_names == ("out0", "out1")
  y, s = pair(np.array([2.0, 3.0]))
  np.testing.assert_allclose(y, np.array([2.0, 3.0]))
  np.testing.assert_allclose(s, 5.0)


def test_gradient_convenience_api_matches_factory() -> None:
  x = al.sym("x", 3)
  y = (x.sin() + x * x).sum()
  f = al.Function("f", [x], [y], ["x"], ["y"])

  g_api = al.gradient(f, "x", "y")
  g_factory = f.factory("g", ["x"], ["grad:y:x"])
  xv = np.array([0.1, 0.4, 0.9])

  assert g_api.input_names == ("x",)
  assert g_api.output_names == ("grad_y_x",)
  np.testing.assert_allclose(g_api(xv), g_factory(xv))


def test_forward_convenience_api_matches_factory() -> None:
  x = al.sym("x", 2)
  y = al.stack([x[0] * x[1], x[0].sin()])
  f = al.Function("f", [x], [y], ["x"], ["y"])

  fwd_api = al.forward(f, "x", "y")
  fwd_factory = f.factory("fwd", ["x", "fwd:x"], ["fwd:y:x"])
  xv = np.array([0.3, 2.0])
  seed = np.array([1.5, -0.25])

  assert fwd_api.input_names == ("x", "fwd:x")
  assert fwd_api.output_names == ("fwd_y_x",)
  np.testing.assert_allclose(fwd_api(xv, seed), fwd_factory(xv, seed))


def test_adjoint_convenience_api_matches_factory() -> None:
  x = al.sym("x", 2)
  y = al.stack([x[0] * x[1], x[0].sin()])
  f = al.Function("f", [x], [y], ["x"], ["y"])

  adj_api = al.adjoint(f, "x", "y")
  adj_factory = f.factory("adj", ["x", "lam:y"], ["adj:y:x"])
  xv = np.array([0.3, 2.0])
  lam = np.array([1.5, -0.25])

  assert adj_api.input_names == ("x", "lam:y")
  assert adj_api.output_names == ("adj_y_x",)
  np.testing.assert_allclose(adj_api(xv, lam), adj_factory(xv, lam))


def test_seeded_factory_outputs_require_seed_inputs() -> None:
  x = al.sym("x", 2)
  y = x * x
  f = al.Function("f", [x], [y], ["x"], ["y"])

  for spec, missing in [("fwd:y:x", "fwd:x"), ("adj:y:x", "lam:y")]:
    try:
      _ = f.factory("bad", ["x"], [spec])
    except ValueError as e:
      assert f"undeclared symbolic inputs: ['{missing}']" in str(e)
    else:  # pragma: no cover
      raise AssertionError(f"{spec} without seed input should fail")


def test_factory_unknown_names_report_value_errors() -> None:
  x = al.sym("x", 2)
  y = x * x
  f = al.Function("f", [x], [y], ["x"], ["y"])

  cases = [
    (lambda: f.factory("bad", ["missing"], ["y"]), "unknown factory inputs: ['missing']"),
    (lambda: f.factory("bad", ["x"], ["jac:missing:x"]), "unknown factory output 'missing' in output 'jac:missing:x'"),
    (lambda: f.factory("bad", ["x"], ["jac:y:missing"]), "unknown factory input 'missing' in output 'jac:y:missing'"),
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
  x = al.sym("x", 2)
  f_expr = x.sin().sum()
  g_expr = x * x
  nlp = al.Function("nlp", [x], [f_expr, g_expr], ["x"], ["f", "g"])

  h_api = al.lagrangian_hessian(nlp, "x", ["f", "g"])
  h_factory = nlp.factory("h", ["x", "lam:f", "lam:g"], ["hess:gamma:x:x"], aux={"gamma": ["f", "g"]})

  xv = np.array([0.2, 0.5])
  lam_f = np.array(1.2)
  lam_g = np.array([0.3, -0.7])
  np.testing.assert_allclose(h_api(xv, lam_f, lam_g), h_factory(xv, lam_f, lam_g))
