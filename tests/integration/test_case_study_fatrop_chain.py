"""The Fatrop chain case study's oracles on a small chain, against a NumPy complex-step reference.

`examples/case_studies/fatrop_chain` feeds Fatrop dense stage Lagrangian Hessians and Jacobians of an
RK4 step whose model maps a link function over the chain and guards a square root with `where`. This
is a self-contained copy of that path at 2 masses in 2D, so the study is not the only thing
exercising it: the Hessian and Jacobian of one stage, and the same stage mapped over a horizon whose
last `xnext` is gathered from a differently strided block.
"""

from __future__ import annotations

import numpy as np

import scaly as sc
from scaly import integrators as si

DIM, MASSES, H = 2, 2, 0.16
NX, NU = DIM * (2 * MASSES + 1), DIM
NZ = NX + NU
D, L_REST, M, G = 1.6, 0.0055, 0.03, 9.81


@sc.function(DIM, DIM)
def link_force(left, right):
  d = right - left
  ss = sc.sumsqr(d)
  return D * (1.0 - L_REST / sc.where(sc.greater(ss, 0.0), ss.sqrt(), 0.0)) * d


@sc.function(sc.L("x", NX), sc.L("u", NU), output="xdot", name="tiny_chain")
def model(x, u):
  positions = sc.concat([sc.const(np.zeros(DIM)), x[: DIM * (MASSES + 1)]])
  forces = sc.vmap(link_force, MASSES + 1, [(positions, 0, DIM), (positions, DIM, DIM)])
  down = np.tile([0.0, -G], MASSES)
  return sc.concat([x[DIM * (MASSES + 1) :], u, (1.0 / M) * (forces[DIM:] - forces[: DIM * MASSES] + M * sc.const(down))])


step = si.rk4(model, dt=H)


def cost(ux):
  return (
    25.0 * sc.sumsqr(ux[NU + DIM * MASSES : NU + DIM * (MASSES + 1)] - sc.const(np.array([1.0, 0.0])))
    + sc.sumsqr(ux[NU + DIM * (MASSES + 1) :])
    + 0.01 * sc.sumsqr(ux[:NU])
  )


@sc.function(sc.L("ux", NZ), sc.L("lam", NX), name="tiny_chain_stage")
def stage(ux, lam):
  f = step(ux[NU:], ux[:NU])
  lag = cost(ux) + (lam * f).sum()
  return sc.concat([sc.hessian(lag, ux).reshape((NZ * NZ,)), sc.jacobian(f, ux).reshape((NX * NZ,)), f])


def rhs_np(x, u):
  pos = np.concatenate([np.zeros(DIM), x[: DIM * (MASSES + 1)]]).reshape(MASSES + 2, DIM)
  forces = []
  for i in range(MASSES + 1):
    d = pos[i + 1] - pos[i]
    forces.append(D * (1.0 - L_REST / np.sqrt((d * d).sum())) * d)
  acc = [(forces[i + 1] - forces[i] + M * np.array([0.0, -G])) / M for i in range(MASSES)]
  return np.concatenate([x[DIM * (MASSES + 1) :], u, *acc])


def step_np(x, u):
  k1 = rhs_np(x, u)
  k2 = rhs_np(x + H / 2 * k1, u)
  k3 = rhs_np(x + H / 2 * k2, u)
  k4 = rhs_np(x + H * k3, u)
  return x + H / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def lag_np(ux, lam):
  u, x = ux[:NU], ux[NU:]
  c = 25.0 * ((x[DIM * MASSES : DIM * (MASSES + 1)] - np.array([1.0, 0.0])) ** 2).sum() + (x[DIM * (MASSES + 1) :] ** 2).sum() + 0.01 * (u**2).sum()
  return c + (lam * step_np(x, u)).sum()


def reference(ux, lam):
  """Complex step in one direction, central differences in the other, for the Hessian; complex step for the Jacobian."""
  eps, d = 1e-20, 1e-5
  hess = np.zeros((NZ, NZ))
  jac = np.zeros((NX, NZ))
  for i in range(NZ):
    ei = np.zeros(NZ, dtype=complex)
    ei[i] = 1j * eps
    jac[:, i] = step_np(ux[NU:] + ei[NU:], ux[:NU] + ei[:NU]).imag / eps
    for j in range(NZ):
      ej = np.zeros(NZ)
      ej[j] = d
      hess[i, j] = (lag_np(ux + ei + ej, lam).imag - lag_np(ux + ei - ej, lam).imag) / (2 * eps * d)
  return hess, jac


def point(seed):
  rng = np.random.default_rng(seed)
  line = np.linspace(0.0, 1.0, MASSES + 2)[1:]
  x = np.concatenate([np.stack([line, 0.1 * rng.standard_normal(MASSES + 1)], axis=1).reshape(-1), 0.3 * rng.standard_normal(DIM * MASSES)])
  return np.concatenate([0.5 * rng.standard_normal(NU), x]), rng.uniform(-1.0, 1.0, NX)


def test_stage_hessian_and_jacobian_match_complex_step():
  for seed in range(3):
    ux, lam = point(seed)
    out = np.asarray(stage(ux, lam))
    hess, jac, f = out[: NZ * NZ].reshape(NZ, NZ), out[NZ * NZ : NZ * NZ + NX * NZ].reshape(NX, NZ), out[NZ * NZ + NX * NZ :]
    hess_ref, jac_ref = reference(ux, lam)
    np.testing.assert_allclose(f, step_np(ux[NU:], ux[:NU]), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(jac, jac_ref, rtol=1e-11, atol=1e-11 * np.abs(jac_ref).max())
    np.testing.assert_allclose(hess, hess_ref, rtol=1e-5, atol=1e-6 * np.abs(hess_ref).max())
    np.testing.assert_allclose(hess, hess.T, rtol=0, atol=1e-10 * np.abs(hess).max())


def test_mapped_stages_match_the_loop():
  """`[u_0 x_0 u_1 x_1 u_2 x_2 x_3]`: the stages read `ux` with one stride, and `xnext` from a gather
  whose last block sits at a different offset, as the study's Fatrop oracles do."""
  horizon = 3
  nw = horizon * NZ + NX

  @sc.function(sc.L("ux", NZ), sc.L("xnext", NX), name="tiny_chain_gap")
  def gap(ux, xnext):
    f = step(ux[NU:], ux[:NU])
    return sc.concat([sc.jacobian(f, ux).reshape((NX * NZ,)), f - xnext])

  @sc.function(sc.L("w", nw), name="tiny_chain_gaps")
  def gaps(w):
    xs = w[: horizon * NZ].reshape((horizon, NZ))[:, NU:]
    xnext = sc.concat([xs[1:].reshape(((horizon - 1) * NX,)), w[horizon * NZ :]])
    return sc.vmap(gap, horizon, [(w, 0, NZ), (xnext, 0, NX)])

  rng = np.random.default_rng(7)
  w = np.concatenate([np.concatenate(point(k)[0:1]) for k in range(horizon)] + [point(9)[0][NU:]]) + 0.01 * rng.standard_normal(nw)
  out = np.asarray(gaps(w)).reshape(horizon, NX * NZ + NX)
  for k in range(horizon):
    ux = w[k * NZ : (k + 1) * NZ]
    xnext = w[(k + 1) * NZ + NU : (k + 2) * NZ] if k < horizon - 1 else w[horizon * NZ :]
    np.testing.assert_allclose(out[k, NX * NZ :], step_np(ux[NU:], ux[:NU]) - xnext, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(out[k, : NX * NZ].reshape(NX, NZ), reference(ux, np.zeros(NX))[1], rtol=1e-11, atol=1e-9)
