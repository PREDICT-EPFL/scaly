from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import alloy as al

NU = 3
N_PARAMS = 5
HORIZON = 40  # laopt instance: N=40 shooting intervals, tf=8.0 -> dt=0.2

# Cost weights: the end-mass position tracks `END_REF`, the intermediate velocities and the control
# are driven to zero. Stage and terminal weights differ, as usual, but they track the same reference.
Q_END, Q_VEL, R_U, Q_END_TERMINAL = 2.5, 25.0, 0.1, 10.0
# Where the actuated end mass is asked to go. This is the one deliberate deviation from the laopt
# instance: laopt writes each end-mass term expanded as `-q*x + 0.5*w*||p||^2`, which is minimal at
# `x = q/w`, but uses the same `q = -7.5` per stage and at the end against `w = 2.5` and `w = 10`
# (`chain_mass_ocp.hpp:38-52`) — so its stage cost pulls towards `x = 3.0` while its terminal cost
# pulls towards `x = 0.75`. Reference implementations track one position throughout (acados'
# `chain_mass` sets `W_e = Q` against the same steady-state `yref`; the ACADO hanging-chain
# benchmark uses one actuator target in every stage), so the split is an oversight there rather
# than an instance to reproduce. We keep the terminal one: `0.75` is within 5% of the `L*(M+1)*6 =
# 0.79` that acados asks this size of chain to stretch to, whereas `3.0` is over 4x further out.
END_REF = (0.75, 0.0, 0.0)


@dataclass(frozen=True)
class ChainParams:
  mass: float = 0.033
  spring_d: float = 1.0
  rest_len: float = 0.033
  gravity: float = -9.81
  dt: float = 0.2

  def array(self) -> np.ndarray:
    return np.array([self.mass, self.spring_d, self.rest_len, self.gravity, self.dt], dtype=np.float64)


def n_state(n_masses: int) -> int:
  if n_masses < 3:
    raise ValueError(f"n_masses must be at least 3, got {n_masses}")
  return 3 * (2 * (n_masses - 2) + 1)


def n_dec(n_masses: int, horizon: int) -> int:
  if horizon < 1:
    raise ValueError(f"horizon must be positive, got {horizon}")
  return horizon * (n_state(n_masses) + NU) + n_state(n_masses)


def n_param(n_masses: int) -> int:
  return n_state(n_masses) + N_PARAMS


@al.function("chain_link_accel", {"dist": 3, "mass": 1, "spring_d": 1, "rest_len": 1})
def chain_link_accel_fn(dist, mass, spring_d, rest_len):  # type: ignore[no-untyped-def]
  return {"accel": (spring_d[0] / mass[0]) * (1.0 - rest_len[0] / al.norm_2(dist)) * dist}


@al.function(
  "chain_mass_accel",
  {"left": 3, "pos": 3, "right": 3, "mass": 1, "spring_d": 1, "rest_len": 1, "gravity": 1},
)
def chain_mass_accel_fn(left, pos, right, mass, spring_d, rest_len, gravity):  # type: ignore[no-untyped-def]
  left_accel = chain_link_accel_fn.call([pos - left, mass, spring_d, rest_len])[0]
  right_accel = chain_link_accel_fn.call([right - pos, mass, spring_d, rest_len])[0]
  return {"accel": right_accel - left_accel + al.stack([0.0, 0.0, gravity[0]])}


def chain_ode_fn(n_masses: int) -> al.Function:
  nx = n_state(n_masses)
  x = al.sym("x", nx)
  u = al.sym("u", NU)
  mass, spring_d, rest_len, gravity, dt = (al.sym(name, 1, diff=False) for name in ("mass", "spring_d", "rest_len", "gravity", "dt"))
  n_intermediate = n_masses - 2
  positions = al.concat([al.const(np.zeros(3)), x[: 3 * (n_masses - 1)]])
  velocities = x[3 * (n_masses - 1) :]
  accel = al.scan(
    chain_mass_accel_fn,
    length=n_intermediate,
    inputs={
      "left": (positions, 0, 3),
      "pos": (positions, 3, 3),
      "right": (positions, 6, 3),
      "mass": (mass, 0, 0),
      "spring_d": (spring_d, 0, 0),
      "rest_len": (rest_len, 0, 0),
      "gravity": (gravity, 0, 0),
    },
  )
  xdot = al.concat([velocities, u, accel])
  return al.Function(
    f"chain_ode_M{n_masses}",
    [x, u, mass, spring_d, rest_len, gravity, dt],
    [xdot],
    ["x", "u", "mass", "spring_d", "rest_len", "gravity", "dt"],
    ["xdot"],
  )


def chain_step_fn(n_masses: int) -> al.Function:
  nx = n_state(n_masses)
  x = al.sym("x", nx)
  u = al.sym("u", NU)
  mass, spring_d, rest_len, gravity, dt = (al.sym(name, 1, diff=False) for name in ("mass", "spring_d", "rest_len", "gravity", "dt"))
  ode = chain_ode_fn(n_masses)

  def rhs(state):  # type: ignore[no-untyped-def]
    return ode.call([state, u, mass, spring_d, rest_len, gravity, dt])[0]

  h = dt[0]
  k1 = rhs(x)
  k2 = rhs(x + 0.5 * h * k1)
  k3 = rhs(x + 0.5 * h * k2)
  k4 = rhs(x + h * k3)
  step = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
  return al.Function(
    f"chain_step_M{n_masses}",
    [x, u, mass, spring_d, rest_len, gravity, dt],
    [step],
    ["x", "u", "mass", "spring_d", "rest_len", "gravity", "dt"],
    ["next"],
  )


def _eq_stage_fn(n_masses: int) -> al.Function:
  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  z = al.sym("z", nz)
  xnext = al.sym("xnext", nx)
  params = al.sym("params", N_PARAMS, diff=False)
  step = chain_step_fn(n_masses).call([z[:nx], z[nx:], *[params[i : i + 1] for i in range(N_PARAMS)]])[0]
  return al.Function(f"chain_eq_stage_M{n_masses}", [z, xnext, params], [step - xnext], ["z", "xnext", "params"], ["eq"])


def chain_eq_function(n_masses: int, horizon: int) -> al.Function:
  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  z = al.sym("z", n_dec(n_masses, horizon))
  p = al.sym("p", n_param(n_masses), diff=False)
  stage = _eq_stage_fn(n_masses)
  mapped = al.scan(stage, length=horizon, inputs={"z": (z, 0, nz), "xnext": (z, nz, nz), "params": (p, nx, 0)})
  return al.Function(f"chain_eq_map_M{n_masses}_N{horizon}", [z, p], [al.concat([z[:nx] - p[:nx], mapped])], ["z", "p"], ["eq"])


def chain_eq_function_unrolled(n_masses: int, horizon: int) -> al.Function:
  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  z = al.sym("z", n_dec(n_masses, horizon))
  p = al.sym("p", n_param(n_masses), diff=False)
  stage = _eq_stage_fn(n_masses)
  parts = [z[:nx] - p[:nx]]
  for i in range(horizon):
    parts.append(stage.call([z[i * nz : (i + 1) * nz], z[(i + 1) * nz : (i + 1) * nz + nx], p[nx:]])[0])
  return al.Function(f"chain_eq_M{n_masses}_N{horizon}", [z, p], [al.concat(parts)], ["z", "p"], ["eq"])


def chain_eq_jac_dense_reference(n_masses: int, horizon: int, z: np.ndarray, p: np.ndarray) -> np.ndarray:
  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  z, p = np.asarray(z, dtype=np.float64), np.asarray(p, dtype=np.float64)
  if z.shape != (n_dec(n_masses, horizon),) or p.shape != (n_param(n_masses),):
    raise ValueError(f"invalid z/p shapes {z.shape} / {p.shape}")
  stage = _eq_stage_fn(n_masses).factory(f"chain_stage_dense_ref_M{n_masses}", ["z", "xnext", "params"], [al.jac("eq", "z"), al.jac("eq", "xnext")])
  dense = np.zeros((nx * (horizon + 1), z.size), dtype=np.float64)
  dense[:nx, :nx] = np.eye(nx)
  for i in range(horizon):
    row, col = nx * (i + 1), nz * i
    jac_z, jac_xnext = stage(z[col : col + nz], z[col + nz : col + nz + nx], p[nx:])
    dense[row : row + nx, col : col + nz] = jac_z
    dense[row : row + nx, col + nz : col + nz + nx] = jac_xnext
  return dense


def _ode_expr(x, u, params, n_masses: int):  # type: ignore[no-untyped-def]
  mass, spring_d, rest_len, gravity = [params[i] for i in range(4)]
  positions = [x[3 * i : 3 * (i + 1)] for i in range(n_masses - 1)]
  velocities = [x[3 * (n_masses - 1 + i) : 3 * (n_masses + i)] for i in range(n_masses - 2)]

  def link(dist):  # type: ignore[no-untyped-def]
    norm = (dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]).sqrt()
    scale = (spring_d / mass) * (1.0 - rest_len / norm)
    return al.stack([scale * dist[i] for i in range(3)])

  accel = []
  for i in range(n_masses - 2):
    left = al.const(np.zeros(3)) if i == 0 else positions[i - 1]
    accel.append(link(positions[i + 1] - positions[i]) - link(positions[i] - left) + al.stack([0.0, 0.0, gravity]))
  return al.concat([*velocities, u, *accel])


def _step_expr(x, u, params, n_masses: int):  # type: ignore[no-untyped-def]
  def scale(value, vector):  # type: ignore[no-untyped-def]
    return al.stack([value * vector[i] for i in range(vector.size)])

  h = params[4]
  k1 = _ode_expr(x, u, params, n_masses)
  k2 = _ode_expr(x + scale(0.5 * h, k1), u, params, n_masses)
  k3 = _ode_expr(x + scale(0.5 * h, k2), u, params, n_masses)
  k4 = _ode_expr(x + scale(h, k3), u, params, n_masses)
  return x + scale(h / 6.0, k1 + k2 + k2 + k3 + k3 + k4)


def _objective(z, n_masses: int, horizon: int):  # type: ignore[no-untyped-def]
  # laopt transcribes on normalized time: each stage cost enters as h*(0.5*|x-xref|^2_P + 0.5*u'Pu) with h=1/N, the Mayer term unscaled.
  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  end = 3 * (n_masses - 2)
  vel = 3 * (n_masses - 1)
  h = 1.0 / horizon
  ref = al.const(np.array(END_REF))
  cost = al.const(0.0)
  for i in range(horizon):
    zi = z[i * nz : (i + 1) * nz]
    cost = cost + h * 0.5 * (Q_END * al.sumsqr(zi[end : end + 3] - ref) + Q_VEL * al.sumsqr(zi[vel:nx]) + R_U * al.sumsqr(zi[nx:]))
  terminal = z[horizon * nz : horizon * nz + nx]
  return cost + 0.5 * Q_END_TERMINAL * al.sumsqr(terminal[end : end + 3] - ref)


def chain_objective_fn(n_masses: int, horizon: int) -> al.Function:
  """The transcribed objective on its own, so its stationary points can be checked directly."""
  z = al.sym("z", n_dec(n_masses, horizon))
  return al.Function(f"chain_obj_M{n_masses}_N{horizon}", [z], [_objective(z, n_masses, horizon)], ["z"], ["f"])


def chain_nlp(n_masses: int, horizon: int, *, solver: str = "ipopt"):
  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  z = al.sym("z", n_dec(n_masses, horizon))
  p = al.sym("p", n_param(n_masses), diff=False)
  parts = [z[:nx] - p[:nx]]
  for i in range(horizon):
    zi = z[i * nz : (i + 1) * nz]
    parts.append(_step_expr(zi[:nx], zi[nx:], p[nx:], n_masses) - z[(i + 1) * nz : (i + 1) * nz + nx])
  eq = al.concat(parts)
  lb = np.full(z.size, -np.inf)
  ub = np.full(z.size, np.inf)
  for i in range(horizon):
    lb[i * nz + nx : (i + 1) * nz] = -1.0
    ub[i * nz + nx : (i + 1) * nz] = 1.0
  return al.nlp(
    x=z,
    f=_objective(z, n_masses, horizon),
    p=p,
    h_eq=eq,
    x_lb=lb,
    x_ub=ub,
    solver=solver,
    name=f"chain_M{n_masses}_N{horizon}_{solver}",
    options={"max_iter": 80, "tol": 1e-6} if solver == "sqp" else None,
  )


def _ca_dynamics(n_masses: int, sym_t):
  import casadi as ca

  nx = n_state(n_masses)
  x, u = sym_t.sym("x", nx), sym_t.sym("u", NU)
  params = sym_t.sym("params", N_PARAMS)
  mass, spring_d, rest_len, gravity, dt = [params[i] for i in range(N_PARAMS)]
  positions = [x[3 * i : 3 * (i + 1)] for i in range(n_masses - 1)]
  velocities = [x[3 * (n_masses - 1 + i) : 3 * (n_masses + i)] for i in range(n_masses - 2)]

  def link(dist):
    return (spring_d / mass) * (1.0 - rest_len / ca.norm_2(dist)) * dist

  accel = []
  for i in range(n_masses - 2):
    left = ca.DM.zeros(3) if i == 0 else positions[i - 1]
    accel.append(link(positions[i + 1] - positions[i]) - link(positions[i] - left) + ca.vertcat(0.0, 0.0, gravity))
  ode = ca.Function(f"ca_chain_ode_M{n_masses}_{sym_t.__name__}", [x, u, params], [ca.vertcat(*velocities, u, *accel)])

  def rhs(state):
    return ode(state, u, params)

  k1 = rhs(x)
  k2 = rhs(x + 0.5 * dt * k1)
  k3 = rhs(x + 0.5 * dt * k2)
  k4 = rhs(x + dt * k3)
  return ca.Function(f"ca_chain_step_M{n_masses}_{sym_t.__name__}", [x, u, params], [x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])


def _ca_eq(n_masses: int, horizon: int, sym_t, *, map_stages: bool = False):
  import casadi as ca

  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  z = (ca.MX if map_stages else sym_t).sym("z", n_dec(n_masses, horizon))
  p = (ca.MX if map_stages else sym_t).sym("p", n_param(n_masses))
  step = _ca_dynamics(n_masses, sym_t)
  parts = [z[:nx] - p[:nx]]
  if map_stages:
    stages = ca.reshape(z[: horizon * nz], nz, horizon)
    xnext = ca.horzcat(*[z[(i + 1) * nz : (i + 1) * nz + nx] for i in range(horizon)])
    mapped = step.map(horizon, "serial")(stages[:nx, :], stages[nx:, :], ca.repmat(p[nx:], 1, horizon))
    parts.append(ca.reshape(mapped - xnext, horizon * nx, 1))
  else:
    for i in range(horizon):
      zi = z[i * nz : (i + 1) * nz]
      parts.append(step(zi[:nx], zi[nx:], p[nx:]) - z[(i + 1) * nz : (i + 1) * nz + nx])
  return z, p, ca.vertcat(*parts)


def ca_chain_eq_jac(n_masses: int, horizon: int, sym_t=None, name: str | None = None, *, map_stages: bool = False):
  import casadi as ca

  sym_t = ca.SX if sym_t is None else sym_t
  z, p, eq = _ca_eq(n_masses, horizon, sym_t, map_stages=map_stages)
  return ca.Function(name or f"ca_chain_eq_jac_M{n_masses}_N{horizon}", [z, p], [ca.jacobian(eq, z)], {"cse": True})


def ca_chain_nlpsol(n_masses: int, horizon: int, *, expand: bool = True, jit: bool = False):
  import casadi as ca

  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  z, p, eq = _ca_eq(n_masses, horizon, ca.MX, map_stages=True)
  f = 0
  end, vel = 3 * (n_masses - 2), 3 * (n_masses - 1)
  h = 1.0 / horizon
  ref = ca.DM(np.asarray(END_REF).reshape(3, 1))
  for i in range(horizon):
    zi = z[i * nz : (i + 1) * nz]
    f += h * 0.5 * (Q_END * ca.sumsqr(zi[end : end + 3] - ref) + Q_VEL * ca.sumsqr(zi[vel:nx]) + R_U * ca.sumsqr(zi[nx:]))
  terminal = z[horizon * nz : horizon * nz + nx]
  f += 0.5 * Q_END_TERMINAL * ca.sumsqr(terminal[end : end + 3] - ref)
  return ca.nlpsol(
    f"ca_chain_M{n_masses}_N{horizon}",
    "ipopt",
    {"x": z, "p": p, "f": f, "g": eq},
    {"expand": expand, "jit": jit, "ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes"},
  )


def ca_chain_sqp(n_masses: int, horizon: int):
  import casadi as ca

  from alloy_sqp.casadi import build_casadi_external_sqp

  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  z, p, eq = _ca_eq(n_masses, horizon, ca.MX, map_stages=True)
  end, vel, h = 3 * (n_masses - 2), 3 * (n_masses - 1), 1.0 / horizon
  ref = ca.DM(np.asarray(END_REF).reshape(3, 1))
  cost = 0
  for i in range(horizon):
    zi = z[i * nz : (i + 1) * nz]
    cost += h * 0.5 * (Q_END * ca.sumsqr(zi[end : end + 3] - ref) + Q_VEL * ca.sumsqr(zi[vel:nx]) + R_U * ca.sumsqr(zi[nx:]))
  terminal = z[horizon * nz : horizon * nz + nx]
  cost += 0.5 * Q_END_TERMINAL * ca.sumsqr(terminal[end : end + 3] - ref)
  lam_f, lam_g = ca.MX.sym("lam_f"), ca.MX.sym("lam_g", int(eq.shape[0]))
  stem = f"ca_chain_sqp_M{n_masses}_N{horizon}"
  base = ca.Function(f"{stem}_base", [z, p], [cost, eq])
  grad = ca.Function(f"{stem}_grad", [z, p], [ca.gradient(cost, z)])
  jac = ca.Function(f"{stem}_jac", [z, p], [ca.jacobian(eq, z)])
  hess = ca.Function(f"{stem}_hess", [z, lam_f, lam_g, p], [ca.hessian(lam_f * cost + ca.dot(lam_g, eq), z)[0]])
  lb, ub = np.full(z.shape[0], -np.inf), np.full(z.shape[0], np.inf)
  for i in range(horizon):
    lb[i * nz + nx : (i + 1) * nz] = -1.0
    ub[i * nz + nx : (i + 1) * nz] = 1.0
  return build_casadi_external_sqp(
    name=stem,
    base=base,
    grad=grad,
    jac=jac,
    hess=hess,
    n_eq=int(eq.shape[0]),
    n_ineq=0,
    x_lb=lb,
    x_ub=ub,
    l_ineq=np.zeros(0),
    u_ineq=np.zeros(0),
    options={"max_iter": 80, "tol": 1e-6},
  )


def chain_ode_np(x: np.ndarray, u: np.ndarray, params: ChainParams = ChainParams()) -> np.ndarray:
  x, u = np.asarray(x, dtype=np.float64), np.asarray(u, dtype=np.float64)
  n_masses = (x.size // 3 + 3) // 2
  nx = n_state(n_masses)
  if x.shape != (nx,) or u.shape != (NU,):
    raise ValueError(f"expected x/u shapes {(nx,)} / {(NU,)}, got {x.shape} / {u.shape}")
  positions = x[: 3 * (n_masses - 1)].reshape(-1, 3)
  velocities = x[3 * (n_masses - 1) :].reshape(-1, 3)

  def link(dist: np.ndarray) -> np.ndarray:
    return (params.spring_d / params.mass) * (1.0 - params.rest_len / np.linalg.norm(dist)) * dist

  accel = []
  for i in range(n_masses - 2):
    left = np.zeros(3) if i == 0 else positions[i - 1]
    accel.append(link(positions[i + 1] - positions[i]) - link(positions[i] - left) + np.array([0.0, 0.0, params.gravity]))
  return np.concatenate([velocities.ravel(), u, np.asarray(accel).ravel()])


def rk4_step_np(x: np.ndarray, u: np.ndarray, params: ChainParams = ChainParams()) -> np.ndarray:
  h = params.dt
  k1 = chain_ode_np(x, u, params)
  k2 = chain_ode_np(x + 0.5 * h * k1, u, params)
  k3 = chain_ode_np(x + 0.5 * h * k2, u, params)
  k4 = chain_ode_np(x + h * k3, u, params)
  return np.asarray(x) + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def initial_state(n_masses: int) -> np.ndarray:
  x = np.zeros(n_state(n_masses), dtype=np.float64)
  x[: 3 * (n_masses - 1) : 3] = 7.0 * np.arange(1, n_masses) / (n_masses - 1)
  return x


def sample_inputs(n_masses: int, horizon: int, *, seed: int = 7, params: ChainParams = ChainParams()) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  z = np.empty(n_dec(n_masses, horizon), dtype=np.float64)
  base = initial_state(n_masses)
  for i in range(horizon):
    z[i * nz : i * nz + nx] = base + rng.normal(scale=0.03, size=nx)
    z[i * nz + nx : (i + 1) * nz] = rng.normal(scale=0.1, size=NU)
  z[horizon * nz :] = base + rng.normal(scale=0.03, size=nx)
  return z, np.concatenate([base, params.array()])
