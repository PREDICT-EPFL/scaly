"""Fixture for the continuous-time CBF safety filter described in ``docs/safety_filter.md``.

This file is consumed by ``benchmarks/alloy_safety_filter_benchmark.py``. It also exposes a small
pytest smoke test verifying that the symbolic IR can be built and evaluated end-to-end. The MLP
weights are random (the docstring says any reasonable network is fine for benchmarking) — the
benchmark and smoke test seed numpy so results are reproducible across runs.
"""

from __future__ import annotations

import numpy as np

import alloy as al

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NSTATE = 7  # (px, py, theta, vf, beta_f, beta_r, delta)
NCTRL = 2  # (u_tr, u_st)
NV = 3  # velocity-block size
NPOSE = 2

LF = 0.54
LR = 0.33
K_DELTA = 4.024 / 2
TAU_DELTA = 0.155

SAFETY_RADIUS = 1.7
WALL_SAFETY_RADIUS = 1.0
ARENA_MIN = (0.0, 0.0)
ARENA_MAX = (10.0, 10.0)

GAMMA1 = 5.0
GAMMA2 = 5.0
GAMMA1_W = 5.0
GAMMA2_W = 5.0

SLACK_PENALTY = 1000.0
Q_DIAG = (1.0, 1.0)

# Network sizing — matches the order of magnitude in docs/safety_filter.md but stays a single
# fully-connected 2-hidden-layer MLP. The benchmark caller can re-import this module after
# overriding ``HIDDEN_1``/``HIDDEN_2`` if a different sweep is desired.
HIDDEN_1 = 256
HIDDEN_2 = 128

# Input-affine: shared body 7 -> 256 -> 128, drift head 128 -> 3, control head 128 -> 6 (flat 3x2).
AFFINE_SHAPES: tuple[tuple[int, ...], ...] = (
  (HIDDEN_1, NSTATE),
  (HIDDEN_1,),
  (HIDDEN_2, HIDDEN_1),
  (HIDDEN_2,),
  (3, HIDDEN_2),
  (3,),
  (6, HIDDEN_2),
  (6,),
)
AFFINE_OFFSETS: tuple[int, ...] = tuple(int(x) for x in np.cumsum([0, *(int(np.prod(s)) for s in AFFINE_SHAPES)]))
N_AFFINE_W = AFFINE_OFFSETS[-1]

# Fully nonlinear: 9 -> 256 -> 128 -> 3.
NONLIN_SHAPES: tuple[tuple[int, ...], ...] = (
  (HIDDEN_1, NSTATE + NCTRL),
  (HIDDEN_1,),
  (HIDDEN_2, HIDDEN_1),
  (HIDDEN_2,),
  (3, HIDDEN_2),
  (3,),
)
NONLIN_OFFSETS: tuple[int, ...] = tuple(int(x) for x in np.cumsum([0, *(int(np.prod(s)) for s in NONLIN_SHAPES)]))
N_NONLIN_W = NONLIN_OFFSETS[-1]

# Per-car NN output size (drift + flat control) for input-affine and (drift only) for nonlinear.
AFFINE_FG = 3 + 6
NONLIN_F = 3


# ---------------------------------------------------------------------------
# Pose kinematics — analytic, the bicycle equations of motion shared by the two variants
# ---------------------------------------------------------------------------


def _silu(x: al.Expr) -> al.Expr:
  return x / (1 + (-x).exp())


def _unpack(pw: al.Expr, offsets: tuple[int, ...], shapes: tuple[tuple[int, ...], ...]) -> list[al.Expr]:
  out: list[al.Expr] = []
  for i, sh in enumerate(shapes):
    chunk = pw[offsets[i] : offsets[i + 1]]
    if len(sh) == 2:
      chunk = chunk.reshape(sh)
    out.append(chunk.block())
  return out


def _kappa_pi_expr(v: al.Expr, theta: al.Expr) -> al.Expr:
  """π̇ = κ_π(v, θ) with v=(vf, βf, βr) and θ a length-1 vector — returns shape (2,)."""
  vf, bf, br = v[0], v[1], v[2]
  omega = vf * (bf - br).sin() / ((LF + LR) * br.cos())
  vx_b = vf * bf.cos()
  vy_b = vf * bf.sin() - LF * omega
  th = theta[0]
  return al.stack(
    [
      vx_b * th.cos() - vy_b * th.sin(),
      vx_b * th.sin() + vy_b * th.cos(),
    ]
  )


def _kappa_theta_scalar(v: al.Expr) -> al.Expr:
  vf, bf, br = v[0], v[1], v[2]
  return vf * (bf - br).sin() / ((LF + LR) * br.cos())


# Promote to alloy Functions so we can build Jacobian factories cleanly.


@al.function("safety_kappa_pi", {"v": NV, "theta": 1})
def kappa_pi_fn(v, theta):  # type: ignore[no-untyped-def]
  return {"out": _kappa_pi_expr(v, theta)}


@al.function("safety_kappa_theta", {"v": NV})
def kappa_theta_fn(v):  # type: ignore[no-untyped-def]
  return {"out": al.stack([_kappa_theta_scalar(v)])}


KAPPA_PI_JAC_V = kappa_pi_fn.factory("safety_kappa_pi_jac_v", ["v", "theta"], ["jac:out:v"])
KAPPA_PI_JAC_T = kappa_pi_fn.factory("safety_kappa_pi_jac_theta", ["v", "theta"], ["jac:out:theta"])


# ---------------------------------------------------------------------------
# Velocity networks (input-affine and fully nonlinear)
# ---------------------------------------------------------------------------


@al.function("safety_affine_mlp", {"x": NSTATE, "pw": N_AFFINE_W})
def affine_mlp_fn(x, pw):  # type: ignore[no-untyped-def]
  w0, b0, w1, b1, wf, bf, wg, bg = _unpack(pw, AFFINE_OFFSETS, AFFINE_SHAPES)
  h = _silu((w0 @ x + b0).block()).block()
  h = _silu((w1 @ h + b1).block()).block()
  f = (wf @ h + bf).block()
  g = (wg @ h + bg).block()
  return {"fg": al.concat([f, g])}  # (9,) packed


@al.function("safety_nonlin_mlp", {"x": NSTATE, "u": NCTRL, "pw": N_NONLIN_W})
def nonlin_mlp_fn(x, u, pw):  # type: ignore[no-untyped-def]
  w0, b0, w1, b1, w2, b2 = _unpack(pw, NONLIN_OFFSETS, NONLIN_SHAPES)
  xu = al.concat([x, u]).block()
  h = _silu((w0 @ xu + b0).block()).block()
  h = _silu((w1 @ h + b1).block()).block()
  return {"vdot": (w2 @ h + b2).block()}


# ---------------------------------------------------------------------------
# Constraint assembly helpers
# ---------------------------------------------------------------------------


_WALL_AXES = (
  (0, +1, ARENA_MIN[0]),
  (0, -1, ARENA_MAX[0]),
  (1, +1, ARENA_MIN[1]),
  (1, -1, ARENA_MAX[1]),
)


def _q_weights_const(ncars: int) -> al.Expr:
  return al.const(np.tile(Q_DIAG, ncars))


def _pair_hocbf(pi_a, pi_b, pi_dot_a, pi_dot_b, pi_ddot_a, pi_ddot_b, slack):
  d_pi = pi_a - pi_b
  d_pi_dot = pi_dot_a - pi_dot_b
  d_pi_ddot = pi_ddot_a - pi_ddot_b
  h = al.dot(d_pi, d_pi) - (SAFETY_RADIUS**2)
  hdot = 2 * al.dot(d_pi, d_pi_dot)
  hddot = 2 * al.dot(d_pi_dot, d_pi_dot) + 2 * al.dot(d_pi, d_pi_ddot)
  return (hddot + (GAMMA1 + GAMMA2) * hdot + GAMMA1 * GAMMA2 * h + slack[0]).scalar()


def _wall_hocbf(pi, pi_dot, pi_ddot, slack, axis, sign, bound):
  pos_a = pi[axis]
  vel_a = pi_dot[axis]
  acc_a = pi_ddot[axis]
  if sign == +1:
    h = pos_a - (bound + WALL_SAFETY_RADIUS)
    hdot = vel_a
    hddot = acc_a
  else:
    h = (bound - WALL_SAFETY_RADIUS) - pos_a
    hdot = -vel_a
    hddot = -acc_a
  return (hddot + (GAMMA1_W + GAMMA2_W) * hdot + GAMMA1_W * GAMMA2_W * h + slack[0]).scalar()


def _quadratic_cost(u, u_des, slack, ncars):
  diff = u - u_des
  weights = _q_weights_const(ncars)
  return (al.dot(diff, weights * diff) + SLACK_PENALTY * al.dot(slack, slack)).scalar()


def n_pairs(ncars: int) -> int:
  return ncars * (ncars - 1) // 2


def n_walls(ncars: int) -> int:
  return 4 * ncars


def n_ineq(ncars: int) -> int:
  return n_pairs(ncars) + n_walls(ncars)


def n_dec(ncars: int) -> int:
  return NCTRL * ncars + 1


def n_state(ncars: int) -> int:
  return NSTATE * ncars


def n_udes(ncars: int) -> int:
  return NCTRL * ncars


# ---------------------------------------------------------------------------
# Input-affine safety filter
# ---------------------------------------------------------------------------


def safety_filter_affine_fn(ncars: int) -> al.Function:
  """Build the input-affine safety filter constraint vector + quadratic cost.

  Inputs:
    u (decision, diff=True) : ncars*NCTRL
    s (decision, diff=True) : 1
    bar_x (parameter)       : ncars*NSTATE
    u_des (parameter)       : ncars*NCTRL
    pw (parameter)          : N_AFFINE_W

  Outputs:
    ineq (ncars*(ncars-1)/2 + 4*ncars,) — pair HOCBF rows then wall HOCBF rows.
    cost () — scalar.
  """
  n_u = NCTRL * ncars
  u = al.sym("u", n_u)
  slack = al.sym("s", 1)
  bar_x = al.sym("bar_x", NSTATE * ncars, diff=False)
  u_des = al.sym("u_des", NCTRL * ncars, diff=False)
  pw = al.sym("pw", N_AFFINE_W, diff=False)

  fg_all = al.map_(affine_mlp_fn, ncars, [(bar_x, 0, NSTATE), (pw, 0, 0)])

  cars: list[dict[str, al.Expr]] = []
  for i in range(ncars):
    xi = bar_x[i * NSTATE : (i + 1) * NSTATE]
    pos = xi[0:2]
    ti = xi[2:3]
    vi = xi[3:6]
    fg_i = fg_all[i * AFFINE_FG : (i + 1) * AFFINE_FG]
    f_i = fg_i[0:3]
    g_i = fg_i[3:9].reshape((3, 2))
    ui = u[i * NCTRL : (i + 1) * NCTRL]

    pi_dot = kappa_pi_fn.call([vi, ti])[0]
    theta_dot = kappa_theta_fn.call([vi])[0]
    v_dot = f_i + g_i @ ui
    J_v = KAPPA_PI_JAC_V.call([vi, ti])[0]
    J_t = KAPPA_PI_JAC_T.call([vi, ti])[0]
    pi_ddot = J_v @ v_dot + J_t @ theta_dot
    cars.append({"pos": pos, "pi_dot": pi_dot, "pi_ddot": pi_ddot})

  rows: list[al.Expr] = []
  for i in range(ncars):
    for j in range(i + 1, ncars):
      rows.append(_pair_hocbf(cars[i]["pos"], cars[j]["pos"], cars[i]["pi_dot"], cars[j]["pi_dot"], cars[i]["pi_ddot"], cars[j]["pi_ddot"], slack))
  for i in range(ncars):
    for axis, sign, bound in _WALL_AXES:
      rows.append(_wall_hocbf(cars[i]["pos"], cars[i]["pi_dot"], cars[i]["pi_ddot"], slack, axis, sign, bound))

  ineq = al.stack(rows).scalar() if rows else al.const(np.zeros((0,)))
  cost = _quadratic_cost(u, u_des, slack, ncars)
  return al.Function(
    f"safety_affine_N{ncars}",
    [u, slack, bar_x, u_des, pw],
    [ineq, cost],
    ["u", "s", "bar_x", "u_des", "pw"],
    ["ineq", "cost"],
  )


# ---------------------------------------------------------------------------
# Fully-nonlinear safety filter
# ---------------------------------------------------------------------------


def safety_filter_nonlin_fn(ncars: int) -> al.Function:
  """Same shape as ``safety_filter_affine_fn`` but uses the fully-nonlinear velocity network."""
  n_u = NCTRL * ncars
  u = al.sym("u", n_u)
  slack = al.sym("s", 1)
  bar_x = al.sym("bar_x", NSTATE * ncars, diff=False)
  u_des = al.sym("u_des", NCTRL * ncars, diff=False)
  pw = al.sym("pw", N_NONLIN_W, diff=False)

  vdot_all = al.map_(nonlin_mlp_fn, ncars, [(bar_x, 0, NSTATE), (u, 0, NCTRL), (pw, 0, 0)])

  cars: list[dict[str, al.Expr]] = []
  for i in range(ncars):
    xi = bar_x[i * NSTATE : (i + 1) * NSTATE]
    pos = xi[0:2]
    ti = xi[2:3]
    vi = xi[3:6]
    v_dot = vdot_all[i * NONLIN_F : (i + 1) * NONLIN_F]
    pi_dot = kappa_pi_fn.call([vi, ti])[0]
    theta_dot = kappa_theta_fn.call([vi])[0]
    J_v = KAPPA_PI_JAC_V.call([vi, ti])[0]
    J_t = KAPPA_PI_JAC_T.call([vi, ti])[0]
    pi_ddot = J_v @ v_dot + J_t @ theta_dot
    cars.append({"pos": pos, "pi_dot": pi_dot, "pi_ddot": pi_ddot})

  rows: list[al.Expr] = []
  for i in range(ncars):
    for j in range(i + 1, ncars):
      rows.append(_pair_hocbf(cars[i]["pos"], cars[j]["pos"], cars[i]["pi_dot"], cars[j]["pi_dot"], cars[i]["pi_ddot"], cars[j]["pi_ddot"], slack))
  for i in range(ncars):
    for axis, sign, bound in _WALL_AXES:
      rows.append(_wall_hocbf(cars[i]["pos"], cars[i]["pi_dot"], cars[i]["pi_ddot"], slack, axis, sign, bound))

  ineq = al.stack(rows).scalar() if rows else al.const(np.zeros((0,)))
  cost = _quadratic_cost(u, u_des, slack, ncars)
  return al.Function(
    f"safety_nonlin_N{ncars}",
    [u, slack, bar_x, u_des, pw],
    [ineq, cost],
    ["u", "s", "bar_x", "u_des", "pw"],
    ["ineq", "cost"],
  )


# ---------------------------------------------------------------------------
# Sample inputs for the benchmark + smoke test
# ---------------------------------------------------------------------------


def sample_inputs(ncars: int, variant: str, seed: int = 11) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(seed)
  n_w = N_AFFINE_W if variant == "affine" else N_NONLIN_W
  pw = rng.normal(scale=0.05, size=n_w)
  bar_x = np.zeros(NSTATE * ncars)
  # spread cars apart and give them small velocities so the safety pair is feasible
  axis_step = 2.5
  for i in range(ncars):
    bar_x[i * NSTATE + 0] = 2.0 + axis_step * (i % 3)
    bar_x[i * NSTATE + 1] = 2.0 + axis_step * (i // 3)
    bar_x[i * NSTATE + 2] = 0.05 * (i + 1)  # theta
    bar_x[i * NSTATE + 3] = 0.6 + 0.05 * i  # vf
    bar_x[i * NSTATE + 4] = 0.02 * i  # beta_f
    bar_x[i * NSTATE + 5] = 0.01 * i  # beta_r
    bar_x[i * NSTATE + 6] = 0.04 * i  # delta
  u_des = rng.normal(scale=0.1, size=NCTRL * ncars)
  u = rng.normal(scale=0.1, size=NCTRL * ncars)
  s = np.array([0.0], dtype=np.float64)
  return {"u": u, "s": s, "bar_x": bar_x, "u_des": u_des, "pw": pw}


# ---------------------------------------------------------------------------
# Pytest smoke tests
# ---------------------------------------------------------------------------


def _check_smoke(fn: al.Function, sample: dict[str, np.ndarray]) -> None:
  outs = fn(*[sample[name] for name in fn.input_names])
  ineq, cost = outs
  assert ineq.shape == (fn.outputs[0].shape[0],)
  assert np.all(np.isfinite(ineq))
  assert np.isfinite(cost)


def test_safety_affine_smoke_runs():
  fn = safety_filter_affine_fn(3)
  sample = sample_inputs(3, "affine")
  _check_smoke(fn, sample)


def test_safety_nonlin_smoke_runs():
  fn = safety_filter_nonlin_fn(3)
  sample = sample_inputs(3, "nonlin")
  _check_smoke(fn, sample)
