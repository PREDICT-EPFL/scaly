"""An optimal control problem with a bicubic table in its dynamics, solved by IPOPT, against the same
problem with the table written out by hand as a piecewise polynomial selected by comparisons."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.interpolate import NdBSpline, make_interp_spline

import scaly as sc
from scaly import interp, mpc

GV = np.linspace(-2.0, 2.0, 5)
GH = np.linspace(0.0, 1.0, 4)
DRAG = 0.3 * GV[:, None] * np.abs(GV[:, None]) * (1.0 + 0.5 * GH[None, :]) + 0.05 * np.cos(3.0 * GH)[None, :]
DT, HORIZON = 0.1, 12
X0 = np.array([0.1, 0.0])


def hand_drag(v: sc.Expr, h: sc.Expr) -> sc.Expr:
  """The not-a-knot bicubic through ``DRAG`` as a sum over cells of an indicator times the cell's
  polynomial in ``(v - v_i, h - h_j)``, its coefficients the Taylor coefficients SciPy's ``NdBSpline``
  gives at the cell's lower corner."""
  sv, sh = make_interp_spline(GV, DRAG, k=3), make_interp_spline(GH, DRAG.T, k=3)
  c = np.moveaxis(make_interp_spline(GH, sv.c, k=3, axis=1).c, 0, 1)
  table = NdBSpline((sv.t, sh.t), c, 3)
  total = None
  for i in range(GV.size - 1):
    in_v = (v >= GV[i]) & (v < GV[i + 1]) if i < GV.size - 2 else v >= GV[i]
    for j in range(GH.size - 1):
      in_h = (h >= GH[j]) & (h < GH[j + 1]) if j < GH.size - 2 else h >= GH[j]
      corner = np.array([GV[i], GH[j]])
      piece = sum(
        float(table(corner, nu=(a, b))) / (math.factorial(a) * math.factorial(b)) * (v - GV[i]) ** a * (h - GH[j]) ** b
        for a in range(4)
        for b in range(4)
      )
      term = sc.where(in_v & in_h, piece, 0.0)
      total = term if total is None else total + term
  assert total is not None
  return total


def ocp(drag, name: str) -> mpc.OCP:
  @sc.function(2, 1, output="xnext", name=f"{name}_step")
  def step(x, u):
    h, v = x[0], x[1]
    return sc.stack([h + DT * v, v + DT * (u[0] - drag(v, h))])

  return mpc.OCP(
    step=step,
    horizon=HORIZON,
    stage_cost=mpc.Quadratic(np.diag([10.0, 0.1]), 0.01 * np.eye(1), x_ref=np.array([0.8, 0.0])),
    terminal_cost=mpc.Quadratic(np.diag([100.0, 1.0]), x_ref=np.array([0.8, 0.0])),
    x_bounds=(np.array([0.0, -2.0]), np.array([1.0, 2.0])),
    u_bounds=(np.array([-3.0]), np.array([3.0])),
    name=name,
  )


@pytest.mark.solver("ipopt")
def test_a_bicubic_table_in_the_dynamics_solves_as_the_hand_written_one() -> None:
  table = interp.interpolant((GV, GH), DRAG, kind="cubic")
  options = {"tol": 1e-12}
  lib = mpc.MPC(ocp(lambda v, h: table(sc.stack([v, h])), "interp_drag"), "ipopt", options=options).solve(X0)
  hand = mpc.MPC(ocp(hand_drag, "hand_drag"), "ipopt", options=options).solve(X0)
  assert lib.status.ok and hand.status.ok
  np.testing.assert_allclose(lib.xs, hand.xs, rtol=1e-8, atol=1e-8)
  np.testing.assert_allclose(lib.us, hand.us, rtol=1e-8, atol=1e-8)
  assert abs(lib.cost - hand.cost) <= 1e-8 * max(1.0, abs(hand.cost))
  assert np.ptp(lib.xs[:, 1]) > 0.3  # the velocity sweeps the table
