"""The ALTRO case study's solver shape, on a double integrator: augmented Lagrangian around iLQR around a line search.

`examples/case_studies/altro` generates ALTRO's augmented-Lagrangian iLQR as one Function: an outer
`while_loop` (with its index) updating multipliers and the penalty, an iLQR `while_loop` inside it, and a
line-search `while_loop` (with its index) inside that, each running forward and backward `scan`s, one of
them with a single output, and constant per-stage data sliced like a variable. This builds a compact
version of that nest for a double integrator with bounded input and a goal constraint, and checks its
solution against SciPy's SLSQP on the same (convex) problem, and the loop counts and the single-output
scan against what they must be, so the study is not the only thing exercising these paths.

ALTRO's second phase projects onto the constraints with a KKT matrix built from a sparse Jacobian of
`vmap`ped stage defects, factored once and reused through `SparseLDL.solve_with` in a `while_loop` that
receives the factor as a parameter. The second test runs that chord iteration for a pendulum and checks
every iterate against the same iteration done densely in NumPy.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

import scaly as sc
from scaly import linalg

NX, NU, K, DT, UMAX = 2, 1, 15, 0.2, 0.45
X0, XF = np.array([0.0, 0.0]), np.array([1.0, 0.0])
QD, RD, QFD = np.array([0.1, 0.1]), np.array([0.5]), np.array([10.0, 10.0])
A = np.array([[1.0, DT], [0.0, 1.0]])
B = np.array([DT * DT / 2, DT])
MASKS = np.ones((K, 2))
MASKS[0] = 0.0  # the first stage carries no bound: a mask that switches rows off, as the study's does
Q, R, QF, XFC, MC = sc.const(np.diag(QD)), sc.const(np.diag(RD)), sc.const(np.diag(QFD)), sc.const(XF), sc.const(MASKS.reshape(-1))
TOL, OUTER, INNER, LS = 1e-8, 20, 50, 12


def dyn(x, u):
  return sc.const(A) @ x + sc.const(B) * u[0]


def stage(x, u):
  e = x - XFC
  return DT * (0.5 * (e @ Q @ e) + 0.5 * (u @ R @ u))


def al_stage(x, u, lam, mask, mu):
  c = mask * sc.concat([u - UMAX, -UMAX - u])
  pen = sc.where(sc.logical_or(sc.greater_equal(c, 0.0), sc.greater(lam, 0.0)), mu, 0.0)
  return stage(x, u) + (lam * c).sum() + 0.5 * (pen * c * c).sum()


def al_term(x, nu, mu):
  g = x - XFC
  return 0.5 * (g @ QF @ g) + (nu * g).sum() + 0.5 * mu * (g * g).sum()


@sc.function(NX, NU, 2, 2, 1)
def roll(x, u, lam, mask, mu):
  return dyn(x, u), x, al_stage(x, u, lam, mask, mu[0]).reshape((1,))


@sc.function(NX + NX * NX, NX, NU, 2, 2, 1)
def back(v, x, u, lam, mask, mu):
  vx, vxx = v[:NX], v[NX:].reshape((NX, NX))
  cost = al_stage(x, u, lam, mask, mu[0])
  xn = dyn(x, u)
  fx, fu = sc.jacobian(xn, x), sc.jacobian(xn, u)
  lu = sc.gradient(cost, u)
  qx, qu = sc.gradient(cost, x) + fx.T @ vx, lu + fu.T @ vx
  qxx, quu = sc.hessian(cost, x) + fx.T @ vxx @ fx, sc.hessian(cost, u) + fu.T @ vxx @ fu
  qux = sc.jacobian(lu, x) + fu.T @ vxx @ fx
  ch = linalg.cholesky(quu)
  k, gain = -linalg.cho_solve(ch, qu), -linalg.cho_solve(ch, qux)
  vxp = qx + gain.T @ (quu @ k) + gain.T @ qu + qux.T @ k
  vxxp = qxx + gain.T @ quu @ gain + gain.T @ qux + qux.T @ gain
  return sc.concat([vxp, (0.5 * (vxxp + vxxp.T)).reshape((NX * NX,))]), sc.concat([k, gain.reshape((NU * NX,))]), (k @ qu).reshape((1,))


@sc.function(NX, NX, NU, NU + NU * NX, 2, 2, 2)
def policy(x, xn, un, g, lam, mask, am):
  u = un + am[0] * g[:NU] + g[NU:].reshape((NU, NX)) @ (x - xn)
  return dyn(x, u), u, al_stage(x, u, lam, mask, am[1]).reshape((1,))


@sc.function(1, 2)
def count_active(acc, lam):  # a single-output scan: the number of positive multipliers
  return acc + sc.cast(sc.greater(lam, 0.0), "float64").sum()


G = NU + NU * NX


@sc.function
def ls_step(carry, trial, xs, us, gains, jd, lam, nu, mu):
  alpha = sc.const(0.5) ** trial.cast("float64")
  xf, un, costs = sc.scan(
    policy,
    sc.const(X0),
    [(xs, 0, NX), (us, 0, NU), (gains, (K - 1) * G, -G), (lam, 0, 2), (MC, 0, 2), (sc.concat([alpha.reshape((1,)), mu]), 0, 0)],
    length=K,
  )
  cost = costs.sum() + al_term(xf, nu, mu[0])
  ok = sc.less_equal(cost - jd[0], 0.5 * alpha * jd[1])
  return sc.concat([sc.cast(ok, "float64").reshape((1,)), cost.reshape((1,)), un])


@sc.function
def ls_go(carry, xs, us, gains, jd, lam, nu, mu):
  return sc.less(carry[0], 0.5)


@sc.function
def ilqr_step(carry, lam, nu, mu):
  us = carry[: K * NU]
  xf, xs, costs = sc.scan(roll, sc.const(X0), [(us, 0, NU), (lam, 0, 2), (MC, 0, 2), (mu, 0, 0)], length=K)
  term = al_term(xf, nu, mu[0])
  j = costs.sum() + term
  v = sc.concat([sc.gradient(term, xf), sc.hessian(term, xf).reshape((NX * NX,))])
  _, gains, dv = sc.scan(
    back, v, [(xs, (K - 1) * NX, -NX), (us, (K - 1) * NU, -NU), (lam, (K - 1) * 2, -2), (MC, (K - 1) * 2, -2), (mu, 0, 0)], length=K
  )
  jd = sc.stack([j, dv.sum()])
  ls, trials = sc.while_loop(ls_go, ls_step, sc.concat([sc.const(np.zeros(2)), us]), max_iter=LS, index=True, params=(xs, us, gains, jd, lam, nu, mu))
  ok = sc.greater(ls[0], 0.5)
  done = sc.logical_or(sc.logical_not(ok), sc.less(j - ls[1], TOL))
  return sc.concat(
    [
      sc.where(ok, ls[2:], us),
      sc.cast(done, "float64").reshape((1,)),
      (carry[K * NU + 1] + 1.0).reshape((1,)),
      (carry[K * NU + 2] + trials.cast("float64")).reshape((1,)),
    ]
  )


@sc.function
def ilqr_go(carry, lam, nu, mu):
  return sc.less(carry[K * NU], 0.5)


@sc.function(NX, NU, 2)
def viol(x, u, mask):
  return dyn(x, u), mask * sc.concat([u - UMAX, -UMAX - u])


@sc.function
def al_step(carry, outer, _x0):
  us, lam, nu, mu = carry[: K * NU], carry[K * NU : K * NU + 2 * K], carry[K * NU + 2 * K : K * NU + 2 * K + NX], carry[K * NU + 2 * K + NX]
  ic, _ = sc.while_loop(ilqr_go, ilqr_step, sc.concat([us, sc.const(np.zeros(3))]), max_iter=INNER, params=(lam, nu, mu.reshape((1,))))
  us = ic[: K * NU]
  xf, c = sc.scan(viol, sc.const(X0), [(us, 0, NU), (MC, 0, 2)], length=K)
  g = xf - XFC
  v = sc.maximum(sc.maximum(c.max(), 0.0), g.abs().max())
  conv = sc.less(v, 1e-7)
  base = K * NU + 2 * K + NX
  return sc.concat(
    [
      us,
      sc.where(conv, lam, sc.maximum(lam + mu * c, 0.0)),
      sc.where(conv, nu, nu + mu * g),
      sc.where(conv, mu, mu * 10.0).reshape((1,)),
      sc.cast(conv, "float64").reshape((1,)),
      (carry[base + 2] + ic[K * NU + 1]).reshape((1,)),
      (carry[base + 3] + ic[K * NU + 2]).reshape((1,)),
      outer.cast("float64").reshape((1,)),
    ]
  )


@sc.function
def al_go(carry, _x0):
  return sc.less(carry[K * NU + 2 * K + NX + 1], 0.5)


@sc.function(NX, output=sc.G("us", "outer", "inner", "trials", "last_index", "active"))
def al_ilqr(x0):
  start = sc.concat([sc.const(np.zeros(K * NU + 2 * K + NX)), sc.const(np.array([1.0, 0.0, 0.0, 0.0, -1.0]))])
  carry, outer = sc.while_loop(al_go, al_step, start, max_iter=OUTER, index=True, params=(x0,))
  base = K * NU + 2 * K + NX
  (active,) = sc.scan(count_active, sc.const(np.zeros(1)), [(carry[K * NU : K * NU + 2 * K], 0, 2)], length=K)
  return carry[: K * NU], outer, carry[base + 2], carry[base + 3], carry[base + 4], active[0]


def slsqp() -> np.ndarray:
  def rollout(us):
    xs = [X0]
    for u in us:
      xs.append(A @ xs[-1] + B * u)
    return np.array(xs)

  def cost(us):
    xs = rollout(us)
    e = xs[:-1] - XF
    return DT * (0.5 * (e**2 @ QD).sum() + 0.5 * RD[0] * (us**2).sum()) + 0.5 * ((xs[-1] - XF) ** 2 @ QFD)

  bounds = [(None, None)] + [(-UMAX, UMAX)] * (K - 1)
  res = optimize.minimize(
    cost,
    np.zeros(K),
    method="SLSQP",
    bounds=bounds,
    constraints=[{"type": "eq", "fun": lambda us: rollout(us)[-1] - XF}],
    options={"ftol": 1e-14, "maxiter": 500},
  )
  assert res.success
  return res.x


def test_three_nested_loops_reach_the_constrained_optimum() -> None:
  us, outer, inner, trials, last_index, active = al_ilqr(X0)
  ref = slsqp()
  np.testing.assert_allclose(us, ref, atol=2e-5)
  assert np.abs(us[1:]).max() <= UMAX + 1e-7  # the bounds bind
  assert np.abs(us[1:]).max() > UMAX - 1e-6
  assert abs(us[0]) > UMAX  # and the masked first stage leaves u_0 free, beyond them
  # The loop counts are consistent: every outer iteration ran the inner loop at least once, every
  # inner iteration ran at least one line-search trial, and the last outer index is the count less one.
  assert 1 < outer < OUTER and outer <= inner <= trials
  assert last_index == outer - 1
  assert active == int((np.abs(ref[1:]) > UMAX - 1e-6).sum())


# The projection: a pendulum's explicit-Euler defects, z = (x_0, u_0, ..., x_{P-1}, u_{P-1}, x_P).
P, PH = 6, 0.3
PZ, PNP = 3, 6 * 3 + 2
PW = np.linspace(1.0, 2.0, PNP)  # the diagonal of H
REG = 1e-8


@sc.function(3, 2)
def pend_defect(xu, x_next):
  th, om, u = xu[0], xu[1], xu[2]
  return sc.stack([th + PH * om, om + PH * (u - th.sin())]) - x_next


def pend_constraints(z, x0):
  return sc.concat([z[:2] - x0, sc.vmap(pend_defect, P, [(z, 0, PZ), (z, PZ, PZ)])])


PLDL: dict[str, linalg.SparseLDL] = {}


@sc.function
def chord_step(carry, index, factor, x0):
  z = carry[:PNP]
  c = pend_constraints(z, x0)
  step = PLDL["kkt"].solve_with(factor, sc.concat([sc.const(np.zeros(PNP)), -c]))[:PNP]
  return sc.concat(
    [
      z + step,
      sc.norm_inf(pend_constraints(z + step, x0)).reshape((1,)),
      carry[PNP + 1 :] + sc.cast(sc.equal(sc.const(np.arange(4.0)), index.cast("float64")), "float64") * sc.norm_inf(c),
    ]
  )


@sc.function
def chord_go(carry, factor, x0):
  return sc.greater(carry[PNP], 0.0)


@sc.function(PNP, 2, output=sc.G("z", "violations"))
def project(z0, x0):
  c = pend_constraints(z0, x0)
  jac = linalg.SparseMatrix.from_sparse_jacobian(sc.sparse_jacobian(c, z0))
  kkt = linalg.SparseMatrix.block([[linalg.SparseMatrix.diag(sc.const(PW)), None], [jac, linalg.SparseMatrix.identity(2 * P + 2) * (-REG)]])
  PLDL["kkt"] = linalg.SparseLDL(kkt, name="pendulum_kkt")
  start = sc.concat([z0, sc.const(np.array([1.0])), sc.const(np.zeros(4))])
  out, _ = sc.while_loop(chord_go, chord_step, start, max_iter=4, index=True, params=(PLDL["kkt"].values, x0))
  return out[:PNP], out[PNP + 1 :]


def test_projection_reuses_one_factor_across_chord_steps() -> None:
  rng = np.random.default_rng(7)
  z0, x0 = rng.standard_normal(PNP), np.array([0.4, -0.2])

  def cons(z):
    xs = [z[k * PZ : k * PZ + 2] for k in range(P + 1)]
    us = [z[k * PZ + 2] for k in range(P)]
    rows = [xs[0] - x0]
    for k in range(P):
      th, om = xs[k]
      rows.append(np.array([th + PH * om, om + PH * (us[k] - np.sin(th))]) - xs[k + 1])
    return np.concatenate(rows)

  def jac(z, eps=1e-7):
    return np.stack([(cons(z + eps * e) - cons(z - eps * e)) / (2 * eps) for e in np.eye(PNP)], axis=1)

  D = jac(z0)
  kkt = np.block([[np.diag(PW), D.T], [D, -REG * np.eye(2 * P + 2)]])
  z, violations = z0.copy(), []
  for _ in range(4):
    c = cons(z)
    violations.append(np.abs(c).max())
    z = z + np.linalg.solve(kkt, np.concatenate([np.zeros(PNP), -c]))[:PNP]
  z_sc, v_sc = project(z0, x0)
  np.testing.assert_allclose(v_sc, violations, rtol=1e-6)
  np.testing.assert_allclose(z_sc, z, atol=1e-7)
  assert violations[-1] < 1e-3 * violations[1]  # the chord steps converge, with one factor
