"""The NP-MPC case study's Scaly paths, small and self-contained.

`examples/case_studies/np_mpc` reproduces the authors' conditional neural process and both of their
MPC transcriptions. Three things it leans on are checked here, so the study is not the only thing
exercising them: the encoder (GELU through `erf`, one `vmap` over the context, a masked mean as a
matrix product) against NumPy; initial-state bands that follow a runtime parameter, both as row
bounds (the CasADi Opti mirror) and as variable bounds (the laOPT mirror), across consecutive solves
of one generated solver; and the two mirrors agreeing on a miniature of the study's OCP.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erf

import scaly as sc
from scaly import nn

RNG = np.random.default_rng(7)
ENC = [(RNG.standard_normal((8, 7)) / 2, None), (RNG.standard_normal((4, 8)) / 2, RNG.standard_normal(4))]
DEC = [(RNG.standard_normal((6, 9)) / 2, None), (RNG.standard_normal((2, 6)) / 2, RNG.standard_normal(2) / 10)]
CAPACITY, N, DT, BAND = 6, 4, 0.02, 1e-3


def gelu(x):
  return x * 0.5 * (1.0 + (x / math.sqrt(2.0)).erf())


def gelu_np(x):
  return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


@sc.function(sc.G(sc.L("x", 5), sc.L("y", 2)), name="np_mpc_test_point")
def point(inputs):
  x, y = inputs
  return nn.mlp(sc.concat([x, y]), ENC, gelu)


@sc.function(sc.G(sc.L("x", CAPACITY * 5), sc.L("y", CAPACITY * 2), sc.L("w", CAPACITY)), name="np_mpc_test_encode")
def encode(inputs):
  x, y, w = inputs
  return sc.vmap(point, CAPACITY, [(x, 0, 5), (y, 0, 2)]).reshape((CAPACITY, 4)).T @ w


def test_gelu_encoder_masked_mean_matches_numpy() -> None:
  x, y = RNG.standard_normal((CAPACITY, 5)), RNG.standard_normal((CAPACITY, 2))
  w = np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0])  # four context points, two masked
  r = np.array([ENC[1][0] @ gelu_np(ENC[0][0] @ np.r_[x[i], y[i]]) + ENC[1][1] for i in range(CAPACITY)])
  np.testing.assert_allclose(encode((x.reshape(-1), y.reshape(-1), w)), r.T @ w, rtol=1e-12, atol=1e-12)


@sc.function(sc.G(sc.L("x", 5), sc.L("y", 2)), name="np_mpc_test_point_jac")
def point_jac(inputs):
  x, y = inputs
  return sc.jacobian(nn.mlp(sc.concat([x, y]), ENC, gelu), x).reshape((20,))


def test_gelu_encoder_derivative_matches_finite_differences() -> None:
  x, y, h = RNG.standard_normal(5), RNG.standard_normal(2), 1e-6

  def f(v):
    return ENC[1][0] @ gelu_np(ENC[0][0] @ np.r_[v, y]) + ENC[1][1]

  fd = np.column_stack([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(5)])
  np.testing.assert_allclose(np.asarray(point_jac((x, y))).reshape(4, 5), fd, rtol=1e-6, atol=1e-8)


# A two-state chain whose first state sits in a band around a parameter: minimize
# (x0 - 1)^2 + (x1 - 1)^2 + u^2 with x1 = x0 + u. For a > 1 + BAND the band binds from below,
# x0 = a - BAND, and u = (1 - x0) / 2.
@sc.opt.problem(vars=sc.G(sc.L("x", 2), sc.L("u", 1)), params=sc.L("a", 1), name="np_mpc_test_band_rows")
def band_rows(variables, a):
  x, u = variables
  return sc.opt.ProblemSpec(
    minimize=((x - 1.0) * (x - 1.0)).sum() + u[0] * u[0],
    eq=(x[1:2] - x[0:1] - u,),
    ineq=(sc.opt.bounded(x[0:1], hi=a + BAND), sc.opt.bounded(x[0:1], lo=a - BAND)),
  )


@sc.opt.problem(vars=sc.G(sc.L("x", 2), sc.L("u", 1)), params=sc.L("a", 1), name="np_mpc_test_band_bounds")
def band_bounds(variables, a):
  x, u = variables
  inf = sc.const(np.array([np.inf]))
  return sc.opt.ProblemSpec(
    minimize=((x - 1.0) * (x - 1.0)).sum() + u[0] * u[0],
    eq=(x[1:2] - x[0:1] - u,),
    lb=(sc.concat([a - BAND, -inf]), sc.opt.NO_LB),
    ub=(sc.concat([a + BAND, inf]), sc.opt.NO_UB),
  )


def _solve_twice(problem, method: str) -> None:
  opts, atol = ({"print_level": 0, "tol": 1e-10}, 1e-7) if method == "IPOPT" else ({}, 1e-5)  # the SQP's own tolerances
  solver = sc.opt.solver(problem, getattr(sc.opt, method)(options=opts), name=f"{problem.name}_{method.lower()}")
  lam_box = (np.zeros(2), np.zeros(1))
  for a in (3.0, 2.0, 5.0):  # the band moves between calls of one compiled solver
    (x, u), *_ = solver((np.zeros(2), np.zeros(1)), lam_box, np.zeros(1), np.zeros(problem.n_ineq), np.array([a]))
    assert sc.opt.solver_stats(solver).to_solver_status().ok
    x0 = a - BAND
    np.testing.assert_allclose(x, [x0, x0 + (1 - x0) / 2], atol=atol)
    np.testing.assert_allclose(u, [(1 - x0) / 2], atol=atol)


@pytest.mark.method("opt.ipopt")
@pytest.mark.parametrize("problem", [band_rows, band_bounds], ids=["rows", "bounds"])
def test_parametric_band_follows_the_parameter_under_ipopt(problem) -> None:
  _solve_twice(problem, "IPOPT")


@pytest.mark.method("opt.sqp")
@pytest.mark.parametrize("problem", [band_rows, band_bounds], ids=["rows", "bounds"])
def test_parametric_band_follows_the_parameter_under_sqp(problem) -> None:
  _solve_twice(problem, "SQP")


@sc.function(sc.G(sc.L("x", 4), sc.L("u", 1), sc.L("xn", 4), sc.L("z", 4)), name="np_mpc_test_defect")
def defect(inputs):
  x, u, xn, z = inputs
  y = nn.mlp(sc.concat([sc.stack([x[0].sin(), x[0].cos(), x[2], x[3], u[0]]), z]), DEC, nn.sigmoid)
  return x + sc.concat([DT * (x[2:4] + y / 2.0), y]) - xn


def _mini_ocp(form: str):
  @sc.opt.problem(
    vars=sc.G(sc.L("x", (N + 1) * 4), sc.L("u", N), sc.L("s", 4)),
    params=sc.G(sc.L("x0", 4), sc.L("z", 4)),
    name=f"np_mpc_test_{form}",
  )
  def ocp(variables, params):
    x, u, s = variables
    x0, z = params
    phi = x[1 : (N + 1) * 4 : 4]
    cost = (x[4:] * x[4:]).sum() + (u * u).sum() + 1000.0 * 0.5 * (s[1] * s[1] + s[1])
    eq = (sc.vmap(defect, N, [(x, 0, 4), (u, 0, 1), (x, 4, 4), (z, 0, 0)]),)
    if form == "opti":
      ineq = (
        sc.opt.bounded(x[:4], hi=x0 + BAND),
        sc.opt.bounded(x[:4], lo=x0 - BAND),
        sc.opt.bounded(-0.1 - s[1] - phi, hi=0.0),
        sc.opt.bounded(phi - (0.1 + s[1]), hi=0.0),
        sc.opt.bounded(u, lo=-0.05),
        sc.opt.bounded(u, hi=0.05),
        sc.opt.bounded(s[1:2], lo=0.0),
        sc.opt.bounded(s[1:2], hi=5.0),
      )
      return sc.opt.ProblemSpec(minimize=cost, eq=eq, ineq=ineq)
    free = sc.const(np.full(N * 4, np.inf))
    return sc.opt.ProblemSpec(
      minimize=cost,
      eq=eq,
      ineq=(sc.opt.bounded(phi - s[1], hi=0.1), sc.opt.bounded(phi + s[1], lo=-0.1)),
      lb=(sc.concat([x0 - BAND, -free]), sc.const(np.full(N, -0.05)), sc.const(np.zeros(4))),
      ub=(sc.concat([x0 + BAND, free]), sc.const(np.full(N, 0.05)), sc.const(np.array([0.0, 5.0, 0.0, 0.0]))),
    )

  return ocp


@pytest.mark.method("opt.ipopt")
def test_opti_and_laopt_mirrors_reach_the_same_solution() -> None:
  x0, z = np.array([0.3, 0.12, -1.0, 0.5]), RNG.standard_normal(4)
  sols = []
  for form in ("opti", "laopt"):
    ocp = _mini_ocp(form)
    solver = sc.opt.solver(ocp, sc.opt.IPOPT(options={"print_level": 0, "tol": 1e-10}), name=f"np_mpc_test_{form}_ipopt")
    guess = (np.tile(x0, N + 1), np.zeros(N), np.zeros(4))
    lam_box = (np.zeros((N + 1) * 4), np.zeros(N), np.zeros(4))
    (x, u, s), *_ = solver(guess, lam_box, np.zeros(ocp.n_eq), np.zeros(ocp.n_ineq), (x0, z))
    assert sc.opt.solver_stats(solver).to_solver_status().ok
    assert np.all(np.abs(x[:4] - x0) <= BAND + 1e-8)
    assert s[1] > 1e-3  # the arm-angle row is active and softened
    sols.append((x, u, s[1]))
  (xa, ua, sa), (xb, ub, sb) = sols
  np.testing.assert_allclose(xa, xb, atol=1e-6)
  np.testing.assert_allclose(ua, ub, atol=1e-6)
  np.testing.assert_allclose(sa, sb, atol=1e-6)
