"""OpenSCvx's penalized trust region (PTR) for the 6-DoF landing, as one Scaly Function.

Each PTR iteration, as OpenSCvx 0.5.3 runs `examples/rocket/6DoF_pdg.py` (`subproblem_spec.md` of the
study's reference dump, citing `cvxpy_ptr_solver.py`):

1. **Discretize** about the reference `(X, U)`: on each of the `N - 1` intervals, integrate
   `dx/dtau = s_k f(x, u(tau))` from `X_k` over a normalized-time length `1/(N-1)` with two fixed Tsit5
   steps, the thrust a first-order hold between `U_k` and `U_{k+1}` and `s` a zero-order hold; `A_d`,
   `B_d`, `C_d` are the endpoint's derivatives in `X_k`, `U_k`, `U_{k+1}` and `x_prop` the endpoint.
   OpenSCvx takes them by forward-mode JVPs through diffrax; `segment_variational` carries the same
   tangent through every stage by the variational equation, and `interval_ad` (`sc.jacobian` through the
   steps) is kept to check it.
2. **Solve the convex subproblem**, a QP in the scaled variables `x = S_x xh + c_x`, `u = S_u uh + c_u`:
   minimize `-0.01 xh_{N-1, mass} + sum (zh - zh_ref)^2 + 10 sum |nu|` subject to the linearized dynamics
   `xh_{k+1} = S_x^-1 (A_d x_k + B_d u_k + C_d u_{k+1} + x_prop - A_d xbar_k - B_d ubar_k - C_d ubar_{k+1} - c_x) + nu_k`,
   the boundary conditions, the state and control boxes, and `|y_k - y_{k-1}| <= 1e-4` for the penalty
   state. `nu = nu_p - nu_m` with both nonnegative makes the l1 term linear. It is solved by Scaly's
   generated PIQP (`examples/qp_solvers/generated_piqp.py`), called inside the loop.
3. **Accept** the solution as the next reference (OpenSCvx's `ConstantProximalWeight`) and stop when
   `J_tr = sum (zh - zh_ref)^2 < 5e-3` and `J_vc = sum |nu / S_x| < 1e-6` (OpenSCvx divides the already
   scaled `nu` by `S_x` once more; kept), at most 200 iterations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import scaly as sc

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "qp_solvers")]
import model as md  # noqa: E402

N, NX, NU = md.N_NODES, md.NX, md.NU
K = N - 1
FOH = np.array([1.0, 1.0, 1.0, 0.0])
TAB = md.tsit5()
H = 1.0 / K / 2  # two fixed steps per interval


@sc.function(sc.L("x", NX), sc.L("u", NU), output="dx", name="pdg6_rates")
def rates(x, u):
  """`dx/dtau`, one call node per stage: the integrator's derivative calls its derivative, not twelve
  inlined copies."""
  return md.augmented(x, u)


def segment(x0, u0, u1, inline: bool = False):
  """The endpoint of one interval: two Tsit5 steps of `s f(x, u(tau))`, the thrust linear in `tau`."""
  x = x0
  for step in range(2):
    tau0 = step * H
    ks = []
    for i in range(6):
      xi = x
      for j in range(i):
        if TAB.a[i, j] != 0.0:
          xi = xi + (H * TAB.a[i, j]) * ks[j]
      frac = (tau0 + TAB.c[i] * H) * K  # the hold's fraction of the interval
      u = u0 + sc.const(FOH * frac) * (u1 - u0)
      ks.append(md.augmented(xi, u) if inline else rates(xi, u))
    for i in range(6):
      x = x + (H * TAB.b[i]) * ks[i]
  return x


@sc.function(sc.L("x", NX), sc.L("u", NU), output=sc.G("dx", "J_x", "J_u"), name="pdg6_rates_jac")
def rates_jac(x, u):
  """`dx/dtau` and its Jacobians in `x` and `u`, by AD of the rates alone."""
  f = md.augmented(x, u)
  return f, sc.jacobian(f, x), sc.jacobian(f, u)


P = NX + 2 * NU  # the sensitivities are taken in (x_k, u_k, u_{k+1})


def segment_variational(x0, u0, u1):
  """The endpoint and its Jacobian in `(x0, u0, u1)`: the same two Tsit5 steps, with the tangent
  `Phi = d x / d(x0, u0, u1)` carried through each stage by the variational equation. For an explicit
  Runge-Kutta step this is the forward-mode derivative of the step itself, stage for stage."""
  x = x0
  Phi = sc.const(np.hstack([np.eye(NX), np.zeros((NX, 2 * NU))]))
  M = np.diag(FOH)
  for step in range(2):
    tau0 = step * H
    ks, kphis = [], []
    for i in range(6):
      xi, phii = x, Phi
      for j in range(i):
        if TAB.a[i, j] != 0.0:
          xi = xi + (H * TAB.a[i, j]) * ks[j]
          phii = phii + (H * TAB.a[i, j]) * kphis[j]
      frac = (tau0 + TAB.c[i] * H) * K
      u = u0 + sc.const(FOH * frac) * (u1 - u0)
      du = sc.const(np.hstack([np.zeros((NU, NX)), np.eye(NU) - frac * M, frac * M]))  # d u / d(x0, u0, u1)
      f, Jx, Ju = rates_jac(xi, u)
      ks.append(f)
      kphis.append(Jx @ phii + Ju @ du)
    for i in range(6):
      x = x + (H * TAB.b[i]) * ks[i]
      Phi = Phi + (H * TAB.b[i]) * kphis[i]
  return x, Phi


@sc.function(sc.L("x0", NX), sc.L("u0", NU), sc.L("u1", NU), name="pdg6_interval")
def interval(x0, u0, u1):
  x, Phi = segment_variational(x0, u0, u1)
  return sc.concat([x, Phi[:, :NX].reshape((NX * NX,)), Phi[:, NX : NX + NU].reshape((NX * NU,)), Phi[:, NX + NU :].reshape((NX * NU,))])


@sc.function(sc.L("x0", NX), sc.L("u0", NU), sc.L("u1", NU), name="pdg6_interval_ad")
def interval_ad(x0, u0, u1):
  """The same by AD through the steps (`sc.jacobian` of the endpoint), for checking."""
  x = segment(x0, u0, u1)
  return sc.concat([x, sc.jacobian(x, x0).reshape((NX * NX,)), sc.jacobian(x, u0).reshape((NX * NU,)), sc.jacobian(x, u1).reshape((NX * NU,))])


W = NX + NX * NX + 2 * NX * NU  # one interval's output


def discretize(X, U):
  """`(x_prop (K, 16), A_d (K, 16, 16), B_d (K, 16, 4), C_d (K, 16, 4))` about `X` (N x 16), `U` (N x 4), flat."""
  out = sc.vmap(interval, K, [(X, 0, NX), (U, 0, NU), (U, NU, NU)]).reshape((K, W))
  xp = out[:, :NX]
  A = out[:, NX : NX + NX * NX]
  B = out[:, NX + NX * NX : NX + NX * NX + NX * NU]
  C = out[:, NX + NX * NX + NX * NU :]
  return xp, A, B, C


def discretize_function(name: str = "pdg6_discretize") -> sc.Function:
  """`discretize` alone, as a Function of the flat `(X, U)`."""

  @sc.function(sc.L("X", N * NX), sc.L("U", N * NU), output=sc.G("x_prop", "A_d", "B_d", "C_d"), name=name)
  def disc(X, U):
    return discretize(X, U)

  return disc


class Data:
  """The static data of OpenSCvx's problem, as its reference dump records them."""

  def __init__(self, ref: dict):
    self.S_x, self.c_x = np.array(ref["S_x"]), np.array(ref["c_x"])
    self.S_u, self.c_u = np.array(ref["S_u"]), np.array(ref["c_u"])
    self.x_min, self.x_max = np.array(ref["x_min"]), np.array(ref["x_max"])
    self.u_min, self.u_max = np.array(ref["u_min"]), np.array(ref["u_max"])
    b = ref["boundary"]
    self.fixed_initial = np.array(b["fixed_initial_indices"])
    self.fixed_final = np.array(b["fixed_final_indices"])
    self.x_init = np.array([np.nan if v is None else v for v in b["x_init_pin"]])
    self.x_term = np.array([np.nan if v is None else v for v in b["x_term_pin"]])
    w = ref["weights"]
    self.lam_prox, self.lam_vc, self.lam_cost = np.array(w["lam_prox"]), np.array(w["lam_vc"]), float(np.ravel(w["lam_cost"])[0])
    self.licq, self.ep_tr, self.ep_vc, self.k_max = float(w["licq_max"]), float(w["ep_tr"]), float(w["ep_vc"]), int(w["k_max"])
    self.X0, self.U0 = np.array(ref["X0"]), np.array(ref["U0"])


def load_data(path: Path) -> Data:
  return Data(json.loads(Path(path).read_text()))


def subproblem(d: Data) -> sc.opt.NLP:
  """The convex subproblem as a Scaly QP with the discretization and the reference as parameters."""
  Sx, cx, Su, cu = (sc.const(a) for a in (d.S_x, d.c_x, d.S_u, d.c_u))

  @sc.opt.problem(
    vars=sc.G(sc.L("xh", (N, NX)), sc.L("uh", (N, NU)), sc.L("nu_p", (K, NX)), sc.L("nu_m", (K, NX))),
    params=sc.G(
      sc.L("xp", (K, NX)), sc.L("A", (K, NX * NX)), sc.L("B", (K, NX * NU)), sc.L("C", (K, NX * NU)), sc.L("Xr", (N, NX)), sc.L("Ur", (N, NU))
    ),
    name="pdg6_subproblem",
  )
  def qp(variables, params):
    xh, uh, nu_p, nu_m = variables
    xp, A, B, C, Xr, Ur = params
    X = xh * Sx.reshape((1, NX)) + cx.reshape((1, NX))
    Uu = uh * Su.reshape((1, NU)) + cu.reshape((1, NU))
    zr = sc.concat([(Xr - cx.reshape((1, NX))) / Sx.reshape((1, NX)), (Ur - cu.reshape((1, NU))) / Su.reshape((1, NU))], axis=1)
    z = sc.concat([xh, uh], axis=1)
    prox = (sc.const(d.lam_prox) * (z - zr) * (z - zr)).sum()
    cost = -d.lam_cost * xh[N - 1, 0] + prox + (sc.const(d.lam_vc) * (nu_p + nu_m)).sum()
    dyn = []
    for k in range(K):
      Ak, Bk, Ck = A[k].reshape((NX, NX)), B[k].reshape((NX, NU)), C[k].reshape((NX, NU))
      bias = xp[k] - Ak @ Xr[k] - Bk @ Ur[k] - Ck @ Ur[k + 1]
      rhs = (Ak @ X[k] + Bk @ Uu[k] + Ck @ Uu[k + 1] + bias - cx) / Sx + (nu_p[k] - nu_m[k])
      dyn.append(xh[k + 1] - rhs)
    y = X[:, NX - 1]
    return sc.opt.ProblemSpec(
      minimize=cost,
      eq=(
        X[0, 1:4] - sc.const(np.array([7.5, 4.5, 2.5])),  # the example's convex equalities
        X[N - 1, 1:3],
        sc.gather(X[0], d.fixed_initial) - sc.const(d.x_init[d.fixed_initial]),
        sc.gather(X[N - 1], d.fixed_final) - sc.const(d.x_term[d.fixed_final]),
        *dyn,
      ),
      ineq=(sc.opt.bounded(y[1:] - y[:-1], lo=sc.const(np.full(K, -d.licq)), hi=sc.const(np.full(K, d.licq)), name="ctcs_increments"),),
      lb=(
        sc.const(np.tile((d.x_min - d.c_x) / d.S_x, (N, 1))),
        sc.const(np.tile((d.u_min - d.c_u) / d.S_u, (N, 1))),
        sc.const(np.zeros((K, NX))),
        sc.const(np.zeros((K, NX))),
      ),
      ub=(sc.const(np.tile((d.x_max - d.c_x) / d.S_x, (N, 1))), sc.const(np.tile((d.u_max - d.c_u) / d.S_u, (N, 1))), sc.opt.NO_UB, sc.opt.NO_UB),
    )

  return qp


TRACE = 20  # iterations recorded in the output traces


def qp_function(d: Data, qp_settings=None, name: str = "pdg6_qp") -> sc.Function:
  """The subproblem solved by Scaly's generated PIQP: a Function of `(x_prop, A_d, B_d, C_d, X, U)`."""
  import generated_piqp
  from scaly.solvers.ipm import Settings

  return generated_piqp.solver(subproblem(d), "sparse", qp_settings or Settings(eps_abs=1e-9, eps_rel=1e-12), name=name)


def ptr_function(d: Data, qp_settings=None, name: str = "pdg6_ptr") -> sc.Function:
  """`ptr() -> (X, U, iterations, J_tr trace, J_vc trace)`: the whole PTR from OpenSCvx's initial guess,
  discretization, generated QP solve and convergence test in one `while_loop`."""
  qp = qp_function(d, qp_settings, name=f"{name}_qp")
  Sx, cx, Su, cu = (sc.const(a) for a in (d.S_x, d.c_x, d.S_u, d.c_u))
  nX, nU = N * NX, N * NU

  @sc.function(name=f"{name}_iteration")
  def iteration(carry, index):
    X, U = carry[:nX], carry[nX : nX + nU]
    xp, A, B, C = discretize(X, U)
    sol = qp((xp, A, B, C, X.reshape((N, NX)), U.reshape((N, NU))))[0]
    xh, uh = sol[:nX].reshape((N, NX)), sol[nX : nX + nU].reshape((N, NU))
    nu = sol[nX + nU : nX + nU + K * NX] - sol[nX + nU + K * NX :]
    X_new = (xh * Sx.reshape((1, NX)) + cx.reshape((1, NX))).reshape((nX,))
    U_new = (uh * Su.reshape((1, NU)) + cu.reshape((1, NU))).reshape((nU,))
    dx = xh - (X.reshape((N, NX)) - cx.reshape((1, NX))) / Sx.reshape((1, NX))
    du = uh - (U.reshape((N, NU)) - cu.reshape((1, NU))) / Su.reshape((1, NU))
    j_tr = (dx * dx).sum() + (du * du).sum()
    j_vc = (nu.reshape((K, NX)) / Sx.reshape((1, NX))).abs().sum()  # OpenSCvx scales nu a second time
    done = sc.logical_and(sc.less(j_tr, d.ep_tr), sc.less(j_vc, d.ep_vc))
    at = sc.cast(sc.equal(sc.const(np.arange(TRACE, dtype=np.float64)), index.cast("float64")), "float64")
    base = nX + nU + 1
    return sc.concat(
      [X_new, U_new, sc.cast(done, "float64").reshape((1,)), carry[base : base + TRACE] + at * j_tr, carry[base + TRACE :] + at * j_vc]
    )

  @sc.function(name=f"{name}_running")
  def running(carry):
    return sc.less(carry[nX + nU], 0.5)

  @sc.function(sc.L("X0", nX), sc.L("U0", nU), output=sc.G("X", "U", "iterations", "J_tr", "J_vc"), name=name)
  def ptr(X0, U0):
    start = sc.concat([X0, U0, sc.const(np.zeros(1 + 2 * TRACE))])
    carry, count = sc.while_loop(running, iteration, start, max_iter=d.k_max, index=True)
    base = nX + nU + 1
    return carry[:nX].reshape((N, NX)), carry[nX : nX + nU].reshape((N, NU)), count, carry[base : base + TRACE], carry[base + TRACE :]

  return ptr
