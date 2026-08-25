"""CasADi mirror of the race-car NMPC, as a drop-in stand-in for the Alloy solver.

`CasadiRaceCarSolver` presents the same call signature, output keys, and statistics
as the `al.SolverFunction` built by `closed_loop._race_car_nlp`, so one episode loop
drives either oracle provider. The decision-variable layout, parameter layout, cost terms,
equality rows, inequality rows, bounds, and IPOPT options are identical by
construction — the only difference is which tool differentiates and evaluates the
oracles. That is the controlled comparison `internal/paper.md` §5 asks for.

The reference implementation in ``minimal_tracking_nmpc/nmpc.py`` builds the same OCP
through `ca.Opti` with per-stage variables, which it needs for FATROP's structure
detection. Here the variables are one flat interleaved vector matching Alloy's `z`,
so the two columns solve a bit-for-bit identical problem.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmarks.harness.casadi_ipopt import CasadiIpoptSolver
from benchmarks.problems.race_cars import CAR_LENGTH, CAR_WIDTH, DELTA_MAX, N_PARAMS, NU, NX, NZ, T_MAX, n_param


def _continuous_dynamics(ca, x, u, params):
  wheelbase, _, mass, c_m0, c_r0, c_r1, c_r2 = [params[i] for i in range(N_PARAMS)]
  phi, v = x[2], x[3]
  throttle, delta = u[0], u[1]
  beta = 0.5 * delta
  vx = v * ca.cos(beta)
  return ca.vertcat(
    v * ca.cos(phi + beta),
    v * ca.sin(phi + beta),
    v * ca.sin(beta) / (0.5 * wheelbase),
    (c_m0 * throttle - (c_r0 + c_r1 * vx + c_r2 * vx * vx) * ca.tanh(10 * vx)) / mass,
  )


def _rk4(ca, x, u, params):
  dt = params[1]
  k1 = _continuous_dynamics(ca, x, u, params)
  k2 = _continuous_dynamics(ca, x + dt / 2 * k1, u, params)
  k3 = _continuous_dynamics(ca, x + dt / 2 * k2, u, params)
  k4 = _continuous_dynamics(ca, x + dt * k3, u, params)
  return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def build_casadi_race_car_nlp(config, sym_t=None, *, dynamics: bool = True) -> dict[str, Any]:
  """Return the symbolic pieces of the OCP, in Alloy's row and column order.

  Set ``dynamics=False`` only when the caller replaces ``h_eq`` with an equivalent encoding.
  """

  import casadi

  sym_t = casadi.MX if sym_t is None else sym_t
  ca = casadi
  n = config.horizon
  z = sym_t.sym("z", NZ * (n + 1))
  p = sym_t.sym("p", n_param(n))
  params = p[NX * (n + 1) :]
  c_m0, c_r0, c_r1, c_r2 = params[3], params[4], params[5], params[6]
  half_length, half_width = 0.5 * CAR_LENGTH, 0.5 * CAR_WIDTH

  weights: list[float] = []
  residuals = []
  corridor = []
  for i in range(n + 1):
    zi, ref = z[i * NZ : (i + 1) * NZ], p[i * NX : (i + 1) * NX]
    v_ref = ref[3]
    throttle_ref = ca.tanh(10.0 * v_ref) * (c_r0 + c_r1 * v_ref + c_r2 * v_ref * v_ref) / c_m0
    weights.extend([config.r_throttle, config.r_steering])
    residuals.extend([zi[NX] - throttle_ref, zi[NX + 1]])
    if i == 0:
      continue
    cos_ref, sin_ref = ca.cos(ref[2]), ca.sin(ref[2])
    dx, dy = zi[0] - ref[0], zi[1] - ref[1]
    e_lon = cos_ref * dx + sin_ref * dy
    e_lat = -sin_ref * dx + cos_ref * dy
    d_phi, d_v = zi[2] - ref[2], zi[3] - v_ref
    if i == n:
      weights.extend([config.q_lon_f, config.q_lat_f, config.q_phi_f, config.q_v_f])
    else:
      weights.extend([config.q_lon, config.q_lat, config.q_phi, config.q_v])
    residuals.extend([e_lon, e_lat, d_phi, d_v])
    reach = e_lat + half_length * ca.sin(d_phi)
    corridor.extend([reach + half_width * ca.cos(d_phi), reach - half_width * ca.cos(d_phi)])
  stacked = ca.vertcat(*residuals)
  cost = ca.dot(ca.DM(np.array(weights)), stacked * stacked)

  eq = [z[:NX] - p[:NX]]
  for i in range(n) if dynamics else ():
    zi = z[i * NZ : (i + 1) * NZ]
    eq.append(_rk4(ca, zi[:NX], zi[NX : NX + NU], params) - z[(i + 1) * NZ : (i + 1) * NZ + NX])

  lb, ub = np.full(NZ * (n + 1), -np.inf), np.full(NZ * (n + 1), np.inf)
  for i in range(n + 1):
    lb[i * NZ + 3], ub[i * NZ + 3] = 0.0, config.max_speed
    lb[i * NZ + NX : (i + 1) * NZ] = [-T_MAX, -DELTA_MAX]
    ub[i * NZ + NX : (i + 1) * NZ] = [T_MAX, DELTA_MAX]
  return {
    "z": z,
    "p": p,
    "f": cost,
    "h_eq": ca.vertcat(*eq),
    "g_ineq": ca.vertcat(*corridor),
    "n_eq": NX * (n + 1),
    "n_ineq": 2 * n,
    "x_lb": lb,
    "x_ub": ub,
  }


def build_casadi_race_car_sqp(config, *, sqp_options: dict[str, str | int | float] | None = None):
  import casadi as ca

  from alloy_sqp.casadi import build_casadi_external_sqp

  pieces = build_casadi_race_car_nlp(config)
  z, p, cost = pieces["z"], pieces["p"], pieces["f"]
  constraints = ca.vertcat(pieces["h_eq"], pieces["g_ineq"])
  lam_f, lam_g = ca.MX.sym("lam_f"), ca.MX.sym("lam_g", int(constraints.shape[0]))
  stem = f"ca_race_sqp_N{config.horizon}"
  return build_casadi_external_sqp(
    name=stem,
    base=ca.Function(f"{stem}_base", [z, p], [cost, constraints]),
    grad=ca.Function(f"{stem}_grad", [z, p], [ca.gradient(cost, z)]),
    jac=ca.Function(f"{stem}_jac", [z, p], [ca.jacobian(constraints, z)]),
    hess=ca.Function(f"{stem}_hess", [z, lam_f, lam_g, p], [ca.hessian(lam_f * cost + ca.dot(lam_g, constraints), z)[0]]),
    n_eq=pieces["n_eq"],
    n_ineq=pieces["n_ineq"],
    x_lb=pieces["x_lb"],
    x_ub=pieces["x_ub"],
    l_ineq=np.full(pieces["n_ineq"], -config.track_half_width),
    u_ineq=np.full(pieces["n_ineq"], config.track_half_width),
    # Identical SQP settings to closed_loop._race_car_nlp.
    options={"tol": config.ipopt_tol, "max_iter": config.sqp_max_iter, **(sqp_options or {})},
  )


class CasadiRaceCarSolver(CasadiIpoptSolver):
  """Compiled CasADi oracles and IPOPT solve behind the benchmark solver contract."""

  def __init__(self, config, *, expand: bool = True):
    pieces = build_casadi_race_car_nlp(config)
    super().__init__(
      f"race_car_closed_loop_casadi_N{config.horizon}",
      pieces["z"],
      pieces["p"],
      pieces["f"],
      pieces["h_eq"],
      pieces["g_ineq"],
      x_lb=pieces["x_lb"],
      x_ub=pieces["x_ub"],
      l_ineq=np.full(pieces["n_ineq"], -config.track_half_width),
      u_ineq=np.full(pieces["n_ineq"], config.track_half_width),
      options={"ipopt.tol": config.ipopt_tol, "ipopt.max_iter": config.ipopt_max_iter},
      expand=expand,
    )
