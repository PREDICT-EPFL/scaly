"""End-to-end tests for ``al.qp`` (PIQP) and ``al.nlp`` (IPOPT).

These exercise the universal-ABI / opaque-Function path for solvers: oracle
expressions are built symbolically, derivatives come through Alloy's factory,
and the backends are driven through the vendored shared libraries via ctypes.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.toolchain import solver_diagnostic, solver_loadable

need_piqp = pytest.mark.skipif(not solver_loadable("piqp"), reason=solver_diagnostic("piqpc"))
need_ipopt = pytest.mark.skipif(not solver_loadable("ipopt"), reason=solver_diagnostic("ipopt"))


# ---------------------------------------------------------------------------
# QP (PIQP)
# ---------------------------------------------------------------------------


@need_piqp
def test_qp_equality_constrained_quadratic() -> None:
  """min 0.5 (x-1)^2 + 0.5 (y-2)^2  s.t.  x + y == 3, x, y >= 0.

  Expected x* = (1, 2), lam_eq[0] = 0 (constraint slack at optimum).
  """
  P = np.eye(2)
  c = np.array([-1.0, -2.0])
  A = np.array([[1.0, 1.0]])
  b = np.array([3.0])
  qp = al.qp(P=P, c=c, A_eq=A, b_eq=b, x_lb=np.zeros(2))
  out = qp(x0=np.zeros(2), lam_eq0=np.zeros(1), lam_ineq0=np.zeros(0))
  assert qp.last_status is not None and qp.last_status.ok
  np.testing.assert_allclose(out["x"], [1.0, 2.0], atol=1e-7)
  # 0.5 (x-1)^2 + 0.5 (y-2)^2 in the form 0.5 xPx + cx (constant dropped):
  # PIQP reports 0.5 x^T x + c^T x = 0.5 (1 + 4) + (-1 - 4) = -2.5
  np.testing.assert_allclose(out["cost"], -2.5, atol=1e-7)


@need_piqp
def test_qp_two_sided_inequality_box() -> None:
  """min 0.5 x^T x  s.t.  1 <= x[0] + x[1] <= 2, |x[0]| <= 1.

  Expected: minimum norm subject to x[0]+x[1] >= 1 is x* = (0.5, 0.5);
  this hits the lower side of the two-sided ineq -> negative lam_ineq.
  """
  P = np.eye(2)
  c = np.zeros(2)
  G = np.array([[1.0, 1.0]])
  qp = al.qp(
    P=P,
    c=c,
    G_ineq=G,
    l_ineq=np.array([1.0]),
    u_ineq=np.array([2.0]),
    x_lb=np.array([-1.0, -np.inf]),
    x_ub=np.array([1.0, np.inf]),
  )
  out = qp(np.zeros(2), np.zeros(0), np.zeros(1))
  assert qp.last_status is not None and qp.last_status.ok
  np.testing.assert_allclose(out["x"], [0.5, 0.5], atol=1e-7)
  # lower side active -> z_l > 0, z_u = 0 -> lam_ineq = z_u - z_l < 0.
  assert out["lam_ineq"][0] < 0


@need_piqp
def test_qp_with_symbolic_parameters() -> None:
  """The QP data may be Alloy ``Expr``s of free parameters.

  Solve min 0.5 (x - mu)^T (x - mu) for several ``mu`` values without rebuilding
  the solver; this is the input-affine MPC/CBF pattern.
  """
  mu = al.sym("mu", 2)
  P = al.const(np.eye(2))
  c = -mu  # -mu shifts the quadratic minimum to mu
  qp = al.qp(P=P, c=c)
  for mu_val in [np.array([0.0, 0.0]), np.array([1.5, -0.3]), np.array([-2.0, 4.0])]:
    out = qp(x0=np.zeros(2), lam_eq0=np.zeros(0), lam_ineq0=np.zeros(0), mu=mu_val)
    assert qp.last_status is not None and qp.last_status.ok
    np.testing.assert_allclose(out["x"], mu_val, atol=1e-7)


@need_piqp
def test_qp_against_analytic_kkt_reference() -> None:
  """Cross-check a small equality-only QP against its dense KKT solution."""
  P_np = np.array([[2.0, 0.5], [0.5, 1.0]])
  c_np = np.array([-1.0, -0.5])
  A_np = np.array([[1.0, 1.0]])
  b_np = np.array([1.5])
  # Analytic KKT for an equality-only QP: [[P, A^T], [A, 0]] [x; lam] = [-c; b]
  K = np.block([[P_np, A_np.T], [A_np, np.zeros((1, 1))]])
  rhs = np.concatenate([-c_np, b_np])
  sol = np.linalg.solve(K, rhs)
  x_ref = sol[:2]
  qp = al.qp(P=P_np, c=c_np, A_eq=A_np, b_eq=b_np)
  out = qp(np.zeros(2), np.zeros(1), np.zeros(0))
  np.testing.assert_allclose(out["x"], x_ref, atol=1e-8)


# ---------------------------------------------------------------------------
# NLP (IPOPT)
# ---------------------------------------------------------------------------


@need_ipopt
def test_nlp_equality_constrained_quadratic() -> None:
  """min (x-1)^2 + (y-2)^2  s.t.  x + y == 1.

  Analytic optimum: x* = (0, 1), lam = 2.
  """
  x = al.sym("x", 2)
  f = (x[0] - 1) ** 2 + (x[1] - 2) ** 2
  h_eq = al.stack([x[0] + x[1] - 1.0], axis=0)
  nlp = al.nlp(x=x, f=f, h_eq=h_eq)
  out = nlp(np.array([0.5, 0.5]), np.zeros(1), np.zeros(0))
  assert nlp.last_status is not None and nlp.last_status.ok
  np.testing.assert_allclose(out["x"], [0.0, 1.0], atol=1e-6)
  np.testing.assert_allclose(out["f"], 2.0, atol=1e-6)
  np.testing.assert_allclose(out["lam_eq"], [2.0], atol=1e-6)


@need_ipopt
def test_nlp_box_only_quadratic() -> None:
  """Unconstrained convex objective + box bound that becomes active.

  min (x[0] - 0.5)^2 + (x[1] + 2)^2  s.t.  x[1] >= -1.
  Expected x* = (0.5, -1), lam_box[1] active (>0).
  """
  x = al.sym("x", 2)
  f = (x[0] - 0.5) ** 2 + (x[1] + 2) ** 2
  nlp = al.nlp(x=x, f=f, x_lb=np.array([-np.inf, -1.0]))
  out = nlp(np.array([0.0, 0.0]), np.zeros(0), np.zeros(0))
  assert nlp.last_status is not None and nlp.last_status.ok
  np.testing.assert_allclose(out["x"], [0.5, -1.0], atol=1e-6)
  # lam_box = mult_x_U - mult_x_L: negative <=> lower bound active.
  assert out["lam_box"][1] < 0


@need_ipopt
def test_nlp_two_sided_inequality_and_lagrangian_hessian() -> None:
  """min x[0]^2 + 0.5 x[1]^2 + x[0] x[1]   s.t.   0 <= x[0]^2 + x[1] <= 5.

  This exercises:
    - a nonlinear two-sided general inequality,
    - a non-diagonal Lagrangian Hessian (off-diagonal coupling x[0] x[1]),
    - the sparse Jacobian path (g has cross-term 2 x[0]).

  The optimum is the unconstrained minimizer x=(0,0), which lies on the lower
  nonlinear inequality boundary. This keeps the test independent from a second
  IPOPT implementation loaded from CasADi in the same process.
  """
  x = al.sym("x", 2)
  f = x[0] ** 2 + 0.5 * x[1] ** 2 + x[0] * x[1]
  g = al.stack([x[0] ** 2 + x[1]], axis=0)
  nlp = al.nlp(x=x, f=f, g_ineq=g, l_ineq=np.array([0.0]), u_ineq=np.array([5.0]))
  x0 = np.array([1.0, -0.5])
  out = nlp(x0, np.zeros(0), np.zeros(1))
  assert nlp.last_status is not None and nlp.last_status.ok

  np.testing.assert_allclose(out["x"], [0.0, 0.0], atol=2e-4)
  np.testing.assert_allclose(out["g_ineq"], [0.0], atol=2e-4)


@need_ipopt
def test_nlp_with_symbolic_parameter() -> None:
  """Parameter-aware NLP: solve min (x - mu)^2 across different ``mu`` values."""
  x = al.sym("x", 2)
  mu = al.sym("mu", 2)
  f = ((x - mu) * (x - mu)).sum()
  nlp = al.nlp(x=x, f=f, p=mu)
  for mu_val in [np.zeros(2), np.array([1.5, -2.3])]:
    out = nlp(x0=np.array([0.0, 0.0]), lam_eq0=np.zeros(0), lam_ineq0=np.zeros(0), mu=mu_val)
    assert nlp.last_status is not None and nlp.last_status.ok
    np.testing.assert_allclose(out["x"], mu_val, atol=1e-6)


@need_ipopt
def test_nlp_rosenbrock_equality_constrained() -> None:
  """Classic Rosenbrock, equality-constrained.

  min (1 - x)^2 + 100 (y - x^2)^2  s.t.  x + y == 1.
  """
  x_sym = al.sym("x", 2)
  f = (1 - x_sym[0]) ** 2 + 100 * (x_sym[1] - x_sym[0] ** 2) ** 2
  h_eq = al.stack([x_sym[0] + x_sym[1] - 1.0], axis=0)
  nlp = al.nlp(x=x_sym, f=f, h_eq=h_eq)
  x0 = np.array([0.5, 0.5])
  out = nlp(x0, np.zeros(1), np.zeros(0))
  assert nlp.last_status is not None and nlp.last_status.ok

  # Substitute y=1-x. The stationary points solve
  #   400*x^3 + 600*x^2 - 198*x - 202 = 0
  # and the minimizer is the middle real root.
  np.testing.assert_allclose(out["x"], [0.6187956190750259, 0.3812043809249741], atol=1e-5)


def test_solver_function_signature_errors() -> None:
  """Mis-shaped or missing inputs raise ``TypeError`` / ``ValueError``."""
  P = np.eye(2)
  c = np.zeros(2)
  qp = al.qp(P=P, c=c)
  with pytest.raises(TypeError, match="missing keyword inputs"):
    qp(x0=np.zeros(2))
  with pytest.raises(ValueError, match="cannot reshape"):
    qp(x0=np.zeros(3), lam_eq0=np.zeros(0), lam_ineq0=np.zeros(0))
