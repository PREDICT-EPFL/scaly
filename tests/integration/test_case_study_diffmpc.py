"""The DiffMPC case study's paths, small: an LQ MPC with an implicit reverse rule, batched and rolled out.

`examples/case_studies/diffmpc` differentiates a 50-step episode in which a `vmap` over a batch calls an
LQ MPC (a Riccati `scan`) whose reverse derivative is a `custom_derivative` rule: the adjoint of the LQ
problem, a second `scan` over the Riccati recursion's stacked gains read backwards. Written with the
problem data broadcast, Scaly's `scan` hoists the Riccati recursion out of the episode. This checks, on
a 3-state, 2-input problem: the rule against AD through the recursion for every input, the episode's
gradient against central differences of a NumPy rollout, and that hoisting leaves the episode unchanged.
"""

from __future__ import annotations

import numpy as np

import scaly as sc
from scaly import linalg

NX, NU, T, NB, STEPS = 3, 2, 6, 4, 5
RNG = np.random.default_rng(11)
A = np.eye(NX) + 0.3 * RNG.standard_normal((NX, NX))
B = RNG.standard_normal((NX, NU))
BV = 0.1 * RNG.standard_normal(NX)
M = RNG.standard_normal((NU, NU))
R = M @ M.T + np.eye(NU)
X0 = RNG.standard_normal((NB, NX))
NP = NX + NX * NX + NX * NU + NX + NU * NU
NV = NX * NX + NX + NU * NX + NU
NS = NU * NX + NU + NX * NX + NX
EYE = sc.const(np.eye(NX))


def unpack(params):
  qd, A_, B_ = params[:NX], params[NX : NX + NX * NX].reshape((NX, NX)), params[NX + NX * NX : NX + NX * NX + NX * NU].reshape((NX, NU))
  return qd, A_, B_, params[NX + NX * NX + NX * NU : 2 * NX + NX * NX + NX * NU], params[2 * NX + NX * NX + NX * NU :].reshape((NU, NU))


@sc.function(NV, NP)
def riccati_step(value, params):
  qd, A_, B_, b, R_ = unpack(params)
  P, p = value[: NX * NX].reshape((NX, NX)), value[NX * NX : NX * NX + NX]
  pb = P @ b + p
  quu, qux = R_ + B_.T @ P @ B_, B_.T @ P @ A_
  ch = linalg.cholesky(quu)
  K, k = -linalg.cho_solve(ch, qux), -linalg.cho_solve(ch, B_.T @ pb)
  P1 = A_.T @ P @ A_ + qux.T @ K
  gains = sc.concat([K.reshape((NU * NX,)), k])
  return sc.concat([(0.5 * (P1 + P1.T) + qd.reshape((NX, 1)) * EYE).reshape((NX * NX,)), A_.T @ pb + qux.T @ k, gains]), sc.concat(
    [gains, value[: NX * NX + NX]]
  )


def riccati(qd, params):
  start = sc.concat([(qd.reshape((NX, 1)) * EYE).reshape((NX * NX,)), sc.const(np.zeros(NX + NU * NX + NU))])
  return sc.scan(riccati_step, start, [(params, 0, 0)], length=T - 1)


@sc.function(NX, NX, NX * NX, NX * NU, NX, NU * NU)
def mpc(x0, qd, A_, B_, b, R_):
  value, _ = riccati(qd, sc.concat([qd, A_, B_, b, R_]))
  return value[NX * NX + NX : NX * NX + NX + NU * NX].reshape((NU, NX)) @ x0 + value[NX * NX + NX + NU * NX :]


def accumulate(x, dx, u, du, x1, dx1, lam1, dlam1, acc):
  o = [0, NX, NX + NX * NX, NX + NX * NX + NX * NU, 2 * NX + NX * NX + NX * NU, NP]

  def outer(a, c):
    n, m = a.shape[0], c.shape[0]
    return (a.reshape((n, 1)) * c.reshape((1, m))).reshape((n * m,))

  return sc.concat(
    [x1, dx1, acc[o[0] : o[1]] + dx1 * x1, acc[o[1] : o[2]] + outer(lam1, dx) + outer(dlam1, x), acc[o[2] : o[3]] + outer(lam1, du) + outer(dlam1, u)]
    + [acc[o[3] : o[4]] + dlam1, acc[o[4] : o[5]] + outer(du, u)]
  )


@sc.function(2 * NX + NP, NS, NP)
def adjoint_step(carry, stage, params):
  _, A_, B_, b, _ = unpack(params)
  x, dx = carry[:NX], carry[NX : 2 * NX]
  K, k = stage[: NU * NX].reshape((NU, NX)), stage[NU * NX : NU * NX + NU]
  P1, p1 = stage[NU * NX + NU : NU * NX + NU + NX * NX].reshape((NX, NX)), stage[NU * NX + NU + NX * NX :]
  u, du = K @ x + k, K @ dx
  x1, dx1 = A_ @ x + B_ @ u + b, A_ @ dx + B_ @ du
  return accumulate(x, dx, u, du, x1, dx1, P1 @ x1 + p1, P1 @ dx1, carry[2 * NX :])


@sc.function(NX, NX, NX * NX, NX * NU, NX, NU * NU, NU, NU, output=sc.G("x0bar", "qdbar", "Abar", "Bbar", "bbar", "Rbar"))
def mpc_vjp(x0, qd, A_, B_, b, R_, u0, ubar):
  params = sc.concat([qd, A_, B_, b, R_])
  _, Am, Bm, _, Rm = unpack(params)
  value, stages = riccati(qd, params)
  last = stages[(T - 2) * NS :]
  P1 = last[NU * NX + NU : NU * NX + NU + NX * NX].reshape((NX, NX))
  du0 = -linalg.cho_solve(linalg.cholesky(Rm + Bm.T @ P1 @ Bm), ubar)
  x1, dx1 = Am @ x0 + Bm @ u0 + b, Bm @ du0
  first = accumulate(x0, sc.const(np.zeros(NX)), u0, du0, x1, dx1, P1 @ x1 + last[NU * NX + NU + NX * NX :], P1 @ dx1, sc.const(np.zeros(NP)))
  (final,) = sc.scan(adjoint_step, first, [(stages, (T - 3) * NS, -NS), (params, 0, 0)], length=T - 2)
  acc = final[2 * NX :]
  o = [0, NX, NX + NX * NX, NX + NX * NX + NX * NU, 2 * NX + NX * NX + NX * NU, NP]
  S = acc[o[4] : o[5]].reshape((NU, NU))
  rbar = S * sc.const(np.tril(np.ones((NU, NU)))) + S.T * sc.const(np.tril(np.ones((NU, NU)), -1))
  K0 = value[NX * NX + NX : NX * NX + NX + NU * NX].reshape((NU, NX))
  return K0.T @ ubar, acc[o[0] : o[1]], acc[o[1] : o[2]], acc[o[2] : o[3]], acc[o[3] : o[4]], rbar.reshape((NU * NU,))


mpc_rule = sc.custom_derivative(mpc, vjp=mpc_vjp)
ARGS = (X0[0], 1.0 + RNG.random(NX), A.reshape(-1), B.reshape(-1), BV, R.reshape(-1))


def episode(f, hoist: bool):
  offsets = [0, NX, NX + NX * NX, NX + NX * NX + NX * NU, 2 * NX + NX * NX + NX * NU]
  n_carry = NB * NX + 1 + (0 if hoist else NB * NX)

  @sc.function(n_carry, NP)
  def step(carry, params):
    X = carry[: NB * NX]
    weights = (params, 0, 0) if hoist else (carry[NB * NX + 1 :], 0, NX)
    U = sc.vmap(f, NB, [(X, 0, NX), weights, *[(params, offsets[k], 0) for k in range(1, 5)]])
    _, A_, B_, b, _ = unpack(params)
    Xn = X.reshape((NB, NX)) @ A_.T + U.reshape((NB, NU)) @ B_.T + b
    parts = [Xn.reshape((NB * NX,)), (carry[NB * NX] + (Xn * Xn).sum() + (U * U).sum()).reshape((1,))]
    return sc.concat(parts if hoist else [*parts, carry[NB * NX + 1 :]])

  @sc.function(NB * NX, NX, output=sc.G("cost", "grad"))
  def run(x0s, qd):
    params = sc.concat([qd, *(sc.const(a) for a in ARGS[2:])])
    start = [x0s, sc.const(np.zeros(1))] + ([] if hoist else [sc.concat([qd] * NB)])
    (final,) = sc.scan(step, sc.concat(start), [(params, 0, 0)], length=STEPS)
    return final[NB * NX], sc.gradient(final[NB * NX], qd)

  return run


def test_implicit_rule_matches_ad_through_the_recursion() -> None:
  w = RNG.standard_normal(NU)
  grads = []
  for f in (mpc, mpc_rule):

    @sc.function(NX, NX, NX * NX, NX * NU, NX, NU * NU, output=sc.G("a", "b", "c", "d", "e", "g"))
    def g(x0, qd, A_, B_, b, R_):
      loss = (sc.const(w) * f(x0, qd, A_, B_, b, R_)).sum()  # noqa: B023
      return tuple(sc.gradient(loss, v) for v in (x0, qd, A_, B_, b, R_))

    grads.append([np.asarray(a) for a in g(*ARGS)])
  for ad, rule in zip(*grads):
    np.testing.assert_allclose(rule, ad, rtol=1e-10, atol=1e-12)


def test_episode_gradient_matches_finite_differences_and_hoisting_is_exact() -> None:
  def np_episode(qd):
    Q, P, pv = np.diag(qd), np.diag(qd), np.zeros(NX)
    for _ in range(T - 1):
      pb = P @ BV + pv
      quu, qux = R + B.T @ P @ B, B.T @ P @ A
      K, k = -np.linalg.solve(quu, qux), -np.linalg.solve(quu, B.T @ pb)
      P, pv = Q + A.T @ P @ A + qux.T @ K, A.T @ pb + qux.T @ k
      P = 0.5 * (P + P.T)
    X, cost = X0.copy(), 0.0
    for _ in range(STEPS):
      U = X @ K.T + k
      X = X @ A.T + U @ B.T + BV
      cost += np.sum(X**2) + np.sum(U**2)
    return cost

  qd = ARGS[1]
  fd = np.array([(np_episode(qd + 1e-6 * e) - np_episode(qd - 1e-6 * e)) / 2e-6 for e in np.eye(NX)])
  results = [episode(f, hoist)(X0.reshape(-1), qd) for f in (mpc, mpc_rule) for hoist in (False, True)]
  for cost, grad in results:
    np.testing.assert_allclose(cost, np_episode(qd), rtol=1e-12)
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-7)
  np.testing.assert_allclose(results[1][0], results[0][0], rtol=1e-14)  # hoisted and per-solve agree


def test_the_rule_is_the_stagewise_solves_implicit_derivative_at_the_first_control() -> None:
  """``scaly.linalg.stagewise.Riccati.solve`` carries the general implicit rule; at the first control
  it is DiffMPC's rule. The study keeps its own for the first control alone: its affine pass is free of
  ``x0``, so a batch episode hoists the whole recursion, which ``solve``'s x0-coupled pass does not.
  R's cotangent is the symmetric part's there and the lower triangle's here, the same once folded."""
  from scaly.linalg.stagewise import Riccati

  @sc.function(NX, NX, NX * NX, NX * NU, NX, NU * NU, output="u0")
  def stagewise_u0(x0, qd, A_, B_, b, R_):
    Q = qd.reshape((NX, 1)) * EYE
    _, us, _ = Riccati(A_.reshape((NX, NX)), B_.reshape((NX, NU)), Q, R_.reshape((NU, NU)), Q, N=T - 1).solve(x0, c=b)
    return us[0]

  w = RNG.standard_normal(NU)
  grads = []
  for f in (mpc_rule, stagewise_u0):

    @sc.function(NX, NX, NX * NX, NX * NU, NX, NU * NU, output=sc.G("a", "b", "c", "d", "e", "g"))
    def g(x0, qd, A_, B_, b, R_):
      loss = (sc.const(w) * f(x0, qd, A_, B_, b, R_)).sum()  # noqa: B023
      return tuple(sc.gradient(loss, v) for v in (x0, qd, A_, B_, b, R_))

    grads.append([np.asarray(a) for a in g(*ARGS)])
  rule, library = grads
  for mine, theirs in zip(rule[:5], library[:5], strict=True):
    np.testing.assert_allclose(mine, theirs, rtol=1e-10, atol=1e-12)
  folded = library[5].reshape(NU, NU)
  np.testing.assert_allclose(np.tril(rule[5].reshape(NU, NU)), np.tril(folded + folded.T - np.diag(np.diag(folded))), rtol=1e-10, atol=1e-12)
