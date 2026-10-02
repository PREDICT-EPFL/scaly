from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import scaly as sc

casadi = pytest.importorskip("casadi")
if TYPE_CHECKING:
  import casadi


def test_gradient_matches_casadi_sx() -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y", ...))
  def f(x):
    return (x.sin() + x * x).sum()

  g = f.factory("g", ["x"], [sc.factory.Grad("y", "x")])

  xv = np.array([0.2, 0.7, 1.1])
  np.testing.assert_allclose(g(xv), np.cos(xv) + 2 * xv)

  cx = casadi.SX.sym("x", 3)
  cy = casadi.sum1(casadi.sin(cx) + cx * cx)
  cg = casadi.Function("g", [cx], [casadi.gradient(cy, cx)])
  np.testing.assert_allclose(g(xv), np.array(cg(xv)).reshape(3), rtol=1e-12, atol=1e-12)


def test_jacobian_matches_casadi_sx() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", ...))
  def f(x):
    return sc.stack([x.sin(), x * x], axis=0).reshape((4,))

  jf = f.factory("J", ["x"], [sc.factory.Jac("y", "x")])

  xv = np.array([0.3, 1.2])
  cx = casadi.SX.sym("x", 2)
  cy = casadi.vertcat(casadi.sin(cx), cx * cx)
  cjf = casadi.Function("J", [cx], [casadi.jacobian(cy, cx)])
  np.testing.assert_allclose(jf(xv), np.array(cjf(xv)), rtol=1e-12, atol=1e-12)


def test_forward_matches_casadi_sx() -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y", ...))
  def f(x):
    return sc.stack([x[0] * x[1], x[2].sin() + x[0]])

  ff = f.factory("fwd", ["x", "fwd:x"], [sc.factory.Fwd("y", "x")])

  xv = np.array([0.3, 1.2, 0.7])
  seed = np.array([1.5, -0.25, 0.4])
  cx = casadi.SX.sym("x", 3)
  cseed = casadi.SX.sym("fwd_x", 3)
  cy = casadi.vertcat(cx[0] * cx[1], casadi.sin(cx[2]) + cx[0])
  cff = casadi.Function("fwd", [cx, cseed], [casadi.mtimes(casadi.jacobian(cy, cx), cseed)])

  np.testing.assert_allclose(ff(*(xv, seed)), np.array(cff(xv, seed)).reshape(2), rtol=1e-12, atol=1e-12)


def test_adjoint_matches_casadi_sx() -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y", ...))
  def f(x):
    return sc.stack([x[0] * x[1], x[2].sin() + x[0]])

  af = f.factory("adj", ["x", "lam:y"], [sc.factory.Adj("y", "x")])

  xv = np.array([0.3, 1.2, 0.7])
  lam = np.array([1.5, -0.25])
  cx = casadi.SX.sym("x", 3)
  clam = casadi.SX.sym("lam_y", 2)
  cy = casadi.vertcat(cx[0] * cx[1], casadi.sin(cx[2]) + cx[0])
  caf = casadi.Function("adj", [cx, clam], [casadi.mtimes(casadi.jacobian(cy, cx).T, clam)])

  np.testing.assert_allclose(af(*(xv, lam)), np.array(caf(xv, lam)).reshape(3), rtol=1e-12, atol=1e-12)


def test_jacobian_through_call_node_matches_casadi_mx() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", ...))
  def inner(x):
    return x.sin() + x * x

  @sc.function(sc.arg("z", 2), outputs=sc.arg("y", ...))
  def outer(z):
    return inner(z * z)

  jf = outer.factory("J", ["z"], [sc.factory.Jac("y", "z")])

  zv = np.array([0.4, 1.2])
  cz = casadi.MX.sym("z", 2)
  cx = casadi.MX.sym("x", 2)
  cinner = casadi.Function("inner", [cx], [casadi.sin(cx) + cx * cx])
  cy = cinner(cz * cz)
  cjf = casadi.Function("J", [cz], [casadi.jacobian(cy, cz)])

  np.testing.assert_allclose(jf(zv), np.array(cjf(zv)), rtol=1e-12, atol=1e-12)


def test_hessian_of_lagrangian_style_aux_matches_casadi_sx() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("f", ...), sc.arg("g", ...)))
  def nlp(x):
    return (x.sin()).sum(), x * x

  hfun = nlp.factory("h", ["x", "lam:f", "lam:g"], [sc.factory.Hess("gamma", "x")], aux={"gamma": ["f", "g"]})

  xv = np.array([0.4, 0.9])
  lam_f = np.array(1.3)
  lam_g = np.array([0.2, -0.4])

  cx = casadi.SX.sym("x", 2)
  clf = casadi.SX.sym("lam_f")
  clg = casadi.SX.sym("lam_g", 2)
  cg = cx * cx
  cgamma = clf * casadi.sum1(casadi.sin(cx)) + casadi.dot(clg, cg)
  ch = casadi.Function("h", [cx, clf, clg], [casadi.hessian(cgamma, cx)[0]])

  np.testing.assert_allclose(hfun(*(xv, lam_f, lam_g)), np.array(ch(xv, lam_f, lam_g)), rtol=1e-12, atol=1e-12)
