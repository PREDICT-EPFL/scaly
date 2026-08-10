"""PIQP plugin solver tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import alloy as al


@pytest.mark.solver("piqp")
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


@pytest.mark.solver("piqp")
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


@pytest.mark.solver("piqp")
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


@pytest.mark.solver("piqp")
def test_generated_qp_satisfies_kkt_over_parameter_sweep() -> None:
  """Fully parameterized QP (P/c/A/b/G/bounds all depend on t): the generated
  solve must satisfy stationarity and primal feasibility at every point."""
  theta = al.sym("theta", 1)
  t = theta[0]
  P = al.stack([al.stack([2.0 + 0.1 * t, 0.05 * t]), al.stack([0.05 * t, 1.5 - 0.1 * t])])
  c = al.stack([-0.4 + 0.2 * t, 0.3 - 0.1 * t])
  A_eq = al.stack([al.stack([1.0 + 0.05 * t, 1.0 - 0.05 * t])])
  b_eq = al.stack([0.2 + 0.1 * t])
  G_ineq = al.stack([al.stack([1.0, 0.1 * t]), al.stack([-0.1 * t, 1.0])])
  l_ineq = al.stack([-0.8 + 0.1 * t, -0.9 - 0.1 * t])
  u_ineq = al.stack([0.9 + 0.1 * t, 1.0 - 0.1 * t])
  x_lb = al.stack([-1.0 + 0.05 * t, -1.1 - 0.05 * t])
  x_ub = al.stack([1.1 + 0.05 * t, 1.2 - 0.05 * t])
  kwargs: dict[str, Any] = dict(P=P, c=c, A_eq=A_eq, b_eq=b_eq, G_ineq=G_ineq, l_ineq=l_ineq, u_ineq=u_ineq, x_lb=x_lb, x_ub=x_ub)
  qp = al.qp(**kwargs, name="qp_kkt_sweep")
  x0 = np.zeros(2)
  for t_value in (-0.6, 0.1, 0.8):
    out = qp(x0, np.zeros(1), np.zeros(2), np.array([t_value]))
    assert qp.last_status is not None and qp.last_status.ok
    P_np = np.array([[2.0 + 0.1 * t_value, 0.05 * t_value], [0.05 * t_value, 1.5 - 0.1 * t_value]])
    c_np = np.array([-0.4 + 0.2 * t_value, 0.3 - 0.1 * t_value])
    A_np = np.array([[1.0 + 0.05 * t_value, 1.0 - 0.05 * t_value]])
    G_np = np.array([[1.0, 0.1 * t_value], [-0.1 * t_value, 1.0]])
    stationarity = P_np @ out["x"] + c_np + A_np.T @ out["lam_eq"] + G_np.T @ out["lam_ineq"] + out["lam_box"]
    np.testing.assert_allclose(stationarity, np.zeros(2), atol=1e-6)
    np.testing.assert_allclose(A_np @ out["x"], [0.2 + 0.1 * t_value], atol=1e-7)
    x0 = out["x"]


@pytest.mark.solver("piqp")
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


@pytest.mark.solver("piqp")
def test_nested_qp_in_alloy_function() -> None:
  """The safety-filter assembly pattern: build QP data symbolically and wrap
  the solve as a node inside a larger ``Function``."""

  @al.function("track_qp", {"mu": (2,)})
  def track_qp(mu):
    # min 0.5 |x - mu|^2  -> solution is mu itself
    qp = al.qp(P=al.const(np.eye(2)), c=-mu)
    out = qp.call([al.const(np.zeros(2)), al.const(np.zeros(0)), al.const(np.zeros(0)), mu])
    return {"x": out[0], "cost": out[1]}

  for mu_val in [np.array([0.5, -1.2]), np.zeros(2), np.array([3.0, 2.0])]:
    x, cost = track_qp(mu_val)
    np.testing.assert_allclose(x, mu_val, atol=1e-7)
    # cost = 0.5 mu^T mu + (-mu)^T mu = -0.5 mu^T mu
    np.testing.assert_allclose(cost, -0.5 * float(np.dot(mu_val, mu_val)), atol=1e-7)


@pytest.mark.solver("piqp")
def test_nested_qp_postprocessed() -> None:
  """Combine solver output with downstream symbolic math."""

  @al.function("squared_norm_via_qp", {"mu": (2,)})
  def sq_norm(mu):
    qp = al.qp(P=al.const(np.eye(2)), c=-mu)
    out = qp.call([al.const(np.zeros(2)), al.const(np.zeros(0)), al.const(np.zeros(0)), mu])
    x_star = out[0]
    return {"y": al.dot(x_star, x_star)}

  mu_val = np.array([1.5, -0.3])
  y = sq_norm(mu_val)
  np.testing.assert_allclose(y, float(np.dot(mu_val, mu_val)), atol=1e-7)


@pytest.mark.solver("piqp")
def test_nested_qp_with_general_inequality() -> None:
  """Two-sided general inequality inside a nested QP."""

  @al.function("constrained_filter", {"u_ref": (2,)})
  def filter_fn(u_ref):
    G = al.const(np.array([[1.0, 1.0]]))
    l_ineq = al.const(np.array([-0.5]))
    u_ineq = al.const(np.array([0.5]))
    qp = al.qp(P=al.const(np.eye(2)), c=-u_ref, G_ineq=G, l_ineq=l_ineq, u_ineq=u_ineq)
    out = qp.call([al.const(np.zeros(2)), al.const(np.zeros(0)), al.const(np.zeros(1)), u_ref])
    return {"u": out[0]}

  # u_ref = (1, 1) is infeasible -> the QP projects onto the band.
  u = filter_fn(np.array([1.0, 1.0]))
  np.testing.assert_allclose(np.sum(u), 0.5, atol=1e-6)
  # u_ref = (-0.1, -0.2) is feasible -> solution is u_ref itself.
  u = filter_fn(np.array([-0.1, -0.2]))
  np.testing.assert_allclose(u, [-0.1, -0.2], atol=1e-7)


@pytest.mark.solver("piqp")
def test_nested_qp_jit_compiles_through_piqp() -> None:
  """JIT path: render C that links against libpiqpc and drives the solve."""

  @al.function("safety_filter", {"x": (2,), "u_ref": (2,)})
  def safety_filter(x, u_ref):
    P = al.const(np.eye(2))
    c = -u_ref
    G = al.stack([al.stack([x[0], x[1]], axis=0)], axis=0)
    l_ineq = al.stack([al.const(-1.0)], axis=0)
    u_ineq = al.stack([al.const(1.0)], axis=0)
    qp = al.qp(P=P, c=c, G_ineq=G, l_ineq=l_ineq, u_ineq=u_ineq)
    # Keyword form is more readable and avoids the alphabetical-sort gotcha.
    out = qp.call(
      x0=al.const(np.zeros(2)),
      lam_eq0=al.const(np.zeros(0)),
      lam_ineq0=al.const(np.zeros(1)),
      x=x,
      u_ref=u_ref,
    )
    return {"u": out[0]}

  u = safety_filter(np.array([1.0, 1.0]), np.array([0.5, 0.5]))
  # Unconstrained min is u_ref=(0.5,0.5); G*u = 1 = upper bound -> on boundary.
  np.testing.assert_allclose(u, [0.5, 0.5], atol=1e-3)

  # Same call again exercises the static-workspace update path inside the
  # compiled solver wrapper.
  u2 = safety_filter(np.array([1.0, -1.0]), np.array([0.0, 0.0]))
  np.testing.assert_allclose(u2, [0.0, 0.0], atol=1e-7)


@pytest.mark.solver("piqp")
def test_nested_qp_call_keyword_form() -> None:
  """``.call(...)`` accepts keyword arguments to bypass alphabetical sort order."""
  u_ref = al.sym("u_ref", 2)
  qp = al.qp(P=al.const(np.eye(2)), c=-u_ref)
  # Even with one param the kwarg form is order-independent and self-documenting.
  out_exprs = qp.call(
    x0=al.const(np.zeros(2)),
    lam_eq0=al.const(np.zeros(0)),
    lam_ineq0=al.const(np.zeros(0)),
    u_ref=u_ref,
  )
  assert len(out_exprs) == len(qp.output_names)
  wrapped = al.Function("wrapped", [u_ref], [out_exprs[0]], ["u_ref"], ["u"])
  result = wrapped(np.array([1.5, -0.3]))
  np.testing.assert_allclose(result, [1.5, -0.3], atol=1e-7)
