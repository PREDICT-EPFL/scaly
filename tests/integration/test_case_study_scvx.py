"""The SCvx case study's paths, small: sensitivities by the variational equation through a hand-written
explicit Runge-Kutta step, and a generated QP solver called inside a `while_loop`.

`examples/case_studies/scvx` discretizes the rocket's dynamics with two Tsit5 steps per interval (a
first-order-hold control, the stages written out) and carries `d x / d(x0, u0, u1)` through every stage
by the variational equation, calling a Function that returns the rates and their Jacobians; each PTR
iteration then solves a parametric QP with Scaly's generated PIQP, called inside the loop's body.
This checks, on a two-state pendulum with RK4: the carried sensitivities against `sc.jacobian` through
the same steps and against central differences; and a loop whose body calls a generated QP against the
same QP called from a Python loop.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import scaly as sc

QP_SOLVERS = Path(__file__).resolve().parents[2] / "examples" / "qp_solvers"
RK4_A = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1.0, 0]])
RK4_B, RK4_C = np.array([1, 2, 2, 1]) / 6, np.array([0, 0.5, 0.5, 1.0])
H, STEPS = 0.1, 2


def rates(x, u):
  return sc.stack([x[1], -np.sin(1.0) * x[0].sin() - 0.2 * x[1] + u[0] * u[1]])  # u[1] a held gain


@sc.function(sc.L("x", 2), sc.L("u", 2), output=sc.G("f", "J_x", "J_u"))
def rates_jac(x, u):
  f = rates(x, u)
  return f, sc.jacobian(f, x), sc.jacobian(f, u)


FOH = np.array([1.0, 0.0])  # u[0] first-order hold, u[1] zero-order hold


def steps(x0, u0, u1, carry_tangent: bool):
  x, Phi = x0, sc.const(np.hstack([np.eye(2), np.zeros((2, 4))]))
  for s in range(STEPS):
    ks, kp = [], []
    for i in range(4):
      xi, pi = x, Phi
      for j in range(i):
        if RK4_A[i, j]:
          xi = xi + H * RK4_A[i, j] * ks[j]
          if carry_tangent:
            pi = pi + H * RK4_A[i, j] * kp[j]
      frac = (s * H + RK4_C[i] * H) / (STEPS * H)
      u = u0 + sc.const(FOH * frac) * (u1 - u0)
      if carry_tangent:
        f, Jx, Ju = rates_jac(xi, u)
        du = sc.const(np.hstack([np.zeros((2, 2)), np.eye(2) - frac * np.diag(FOH), frac * np.diag(FOH)]))
        kp.append(Jx @ pi + Ju @ du)
      else:
        f = rates(xi, u)
      ks.append(f)
    for i in range(4):
      x = x + H * RK4_B[i] * ks[i]
      if carry_tangent:
        Phi = Phi + H * RK4_B[i] * kp[i]
  return x, Phi


@sc.function(sc.L("x0", 2), sc.L("u0", 2), sc.L("u1", 2), output=sc.G("x", "Phi", "Phi_ad"))
def discretize(x0, u0, u1):
  x, Phi = steps(x0, u0, u1, True)
  xa, _ = steps(x0, u0, u1, False)
  return x, Phi, sc.concat([sc.jacobian(xa, x0), sc.jacobian(xa, u0), sc.jacobian(xa, u1)], axis=1)


def test_variational_sensitivities_equal_ad_through_the_steps() -> None:
  x0, u0, u1 = np.array([0.7, -0.3]), np.array([0.5, 1.3]), np.array([-0.4, 0.9])
  x, Phi, Phi_ad = (np.asarray(a) for a in discretize(x0, u0, u1))
  np.testing.assert_allclose(Phi, Phi_ad, rtol=1e-12, atol=1e-14)
  p0 = np.r_[x0, u0, u1]

  def endpoint(p):
    return np.asarray(discretize(p[:2], p[2:4], p[4:])[0])

  fd = np.stack([(endpoint(p0 + 1e-6 * e) - endpoint(p0 - 1e-6 * e)) / 2e-6 for e in np.eye(6)], axis=1)
  np.testing.assert_allclose(Phi, fd, atol=1e-8)


@pytest.fixture(scope="module")
def generated_piqp() -> Iterator[dict]:
  sys.path.insert(0, str(QP_SOLVERS))
  try:
    yield runpy.run_path(str(QP_SOLVERS / "generated_piqp.py"))
  finally:
    sys.path.remove(str(QP_SOLVERS))


def test_generated_qp_inside_a_while_loop(generated_piqp: dict) -> None:
  # min |x - r|^2 + |x - x_prev|^2 subject to sum(x) = 1, 0 <= x <= 0.6: each iterate is the next reference.
  @sc.opt.problem(vars=sc.L("x", 4), params=sc.G(sc.L("r", 4), sc.L("x_prev", 4)), name="loop_qp")
  def qp(x, params):
    r, x_prev = params
    return sc.opt.ProblemSpec(
      minimize=((x - r) ** 2).sum() + ((x - x_prev) ** 2).sum(), eq=(x.sum() - 1.0,), lb=sc.const(np.zeros(4)), ub=sc.const(np.full(4, 0.6))
    )

  solve = generated_piqp["solver"](qp, "sparse", name="loop_qp_generated")
  target = np.array([0.9, 0.5, -0.2, 0.1])

  @sc.function
  def body(carry, index):
    x = solve((sc.const(target), carry[:4]))[0]
    return sc.concat([x, (carry[4] + (x - carry[:4]).abs().sum()).reshape((1,))])

  @sc.function
  def go(carry):
    return sc.less(carry[4], 10.0)  # never stops early: max_iter decides

  @sc.function(sc.L("x0", 4), output="x")
  def loop(x0):
    carry, _ = sc.while_loop(go, body, sc.concat([x0, sc.const(np.zeros(1))]), max_iter=6, index=True)
    return carry[:4]

  x0 = np.full(4, 0.25)
  x_loop = np.asarray(loop(x0))
  x = x0
  for _ in range(6):
    x = np.asarray(solve((target, x))[0])
  np.testing.assert_allclose(x_loop, x, atol=1e-12)
  assert abs(x.sum() - 1) < 1e-8 and x.max() <= 0.6 + 1e-8
