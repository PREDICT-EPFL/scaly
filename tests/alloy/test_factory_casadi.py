from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import alloy as al

casadi = pytest.importorskip("casadi")
if TYPE_CHECKING:
  import casadi


def test_gradient_matches_casadi_sx() -> None:
  x = al.sym("x", 3)
  y = (x.sin() + x * x).sum()
  f = al.Function("f", [x], [y], ["x"], ["y"])
  g = f.factory("g", ["x"], ["grad:y:x"])

  xv = np.array([0.2, 0.7, 1.1])
  np.testing.assert_allclose(g(xv), np.cos(xv) + 2 * xv)

  cx = casadi.SX.sym("x", 3)
  cy = casadi.sum1(casadi.sin(cx) + cx * cx)
  cg = casadi.Function("g", [cx], [casadi.gradient(cy, cx)])
  np.testing.assert_allclose(g(xv), np.array(cg(xv)).reshape(3), rtol=1e-12, atol=1e-12)


def test_jacobian_matches_casadi_sx() -> None:
  x = al.sym("x", 2)
  y = al.stack([x.sin(), x * x], axis=0).reshape((4,))
  f = al.Function("f", [x], [y], ["x"], ["y"])
  jf = f.factory("J", ["x"], ["jac:y:x"])

  xv = np.array([0.3, 1.2])
  cx = casadi.SX.sym("x", 2)
  cy = casadi.vertcat(casadi.sin(cx), cx * cx)
  cjf = casadi.Function("J", [cx], [casadi.jacobian(cy, cx)])
  np.testing.assert_allclose(jf(xv), np.array(cjf(xv)), rtol=1e-12, atol=1e-12)


def test_forward_matches_casadi_sx() -> None:
  x = al.sym("x", 3)
  y = al.stack([x[0] * x[1], x[2].sin() + x[0]])
  f = al.Function("f", [x], [y], ["x"], ["y"])
  ff = f.factory("fwd", ["x", "fwd:x"], ["fwd:y:x"])

  xv = np.array([0.3, 1.2, 0.7])
  seed = np.array([1.5, -0.25, 0.4])
  cx = casadi.SX.sym("x", 3)
  cseed = casadi.SX.sym("fwd_x", 3)
  cy = casadi.vertcat(cx[0] * cx[1], casadi.sin(cx[2]) + cx[0])
  cff = casadi.Function("fwd", [cx, cseed], [casadi.mtimes(casadi.jacobian(cy, cx), cseed)])

  np.testing.assert_allclose(ff(xv, seed), np.array(cff(xv, seed)).reshape(2), rtol=1e-12, atol=1e-12)


def test_adjoint_matches_casadi_sx() -> None:
  x = al.sym("x", 3)
  y = al.stack([x[0] * x[1], x[2].sin() + x[0]])
  f = al.Function("f", [x], [y], ["x"], ["y"])
  af = f.factory("adj", ["x", "lam:y"], ["adj:y:x"])

  xv = np.array([0.3, 1.2, 0.7])
  lam = np.array([1.5, -0.25])
  cx = casadi.SX.sym("x", 3)
  clam = casadi.SX.sym("lam_y", 2)
  cy = casadi.vertcat(cx[0] * cx[1], casadi.sin(cx[2]) + cx[0])
  caf = casadi.Function("adj", [cx, clam], [casadi.mtimes(casadi.jacobian(cy, cx).T, clam)])

  np.testing.assert_allclose(af(xv, lam), np.array(caf(xv, lam)).reshape(3), rtol=1e-12, atol=1e-12)


def test_jacobian_through_call_node_matches_casadi_mx() -> None:
  x = al.sym("x", 2)
  inner = al.Function("inner", [x], [x.sin() + x * x], ["x"], ["y"])
  z = al.sym("z", 2)
  (inner_z,) = inner.call([z * z])
  outer = al.Function("outer", [z], [inner_z], ["z"], ["y"])
  jf = outer.factory("J", ["z"], ["jac:y:z"])

  zv = np.array([0.4, 1.2])
  cz = casadi.MX.sym("z", 2)
  cx = casadi.MX.sym("x", 2)
  cinner = casadi.Function("inner", [cx], [casadi.sin(cx) + cx * cx])
  cy = cinner(cz * cz)
  cjf = casadi.Function("J", [cz], [casadi.jacobian(cy, cz)])

  np.testing.assert_allclose(jf(zv), np.array(cjf(zv)), rtol=1e-12, atol=1e-12)


def test_hessian_of_lagrangian_style_aux_matches_casadi_sx() -> None:
  x = al.sym("x", 2)
  f_expr = (x.sin()).sum()
  g_expr = x * x
  nlp = al.Function("nlp", [x], [f_expr, g_expr], ["x"], ["f", "g"])
  hfun = nlp.factory("h", ["x", "lam:f", "lam:g"], ["hess:gamma:x:x"], aux={"gamma": ["f", "g"]})

  xv = np.array([0.4, 0.9])
  lam_f = np.array(1.3)
  lam_g = np.array([0.2, -0.4])

  cx = casadi.SX.sym("x", 2)
  clf = casadi.SX.sym("lam_f")
  clg = casadi.SX.sym("lam_g", 2)
  cg = cx * cx
  cgamma = clf * casadi.sum1(casadi.sin(cx)) + casadi.dot(clg, cg)
  ch = casadi.Function("h", [cx, clf, clg], [casadi.hessian(cgamma, cx)[0]])

  np.testing.assert_allclose(hfun(xv, lam_f, lam_g), np.array(ch(xv, lam_f, lam_g)), rtol=1e-12, atol=1e-12)
