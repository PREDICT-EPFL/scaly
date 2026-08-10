"""CasADi mirror of the race-car NMPC, as a drop-in stand-in for the Alloy solver.

`CasadiRaceCarSolver` presents the same call signature, output keys, and statistics
as the `al.SolverFunction` built by `closed_loop._race_car_nlp`, so one episode loop
drives either backend. The decision-variable layout, parameter layout, cost terms,
equality rows, inequality rows, bounds, and IPOPT options are identical by
construction — the only difference is which tool differentiates and evaluates the
oracles. That is the controlled comparison ROADMAP.md §2.3 asks for.

The reference implementation in ``minimal_tracking_nmpc/nmpc.py`` builds the same OCP
through `ca.Opti` with per-stage variables, which it needs for FATROP's structure
detection. Here the variables are one flat interleaved vector matching Alloy's `z`,
so the two columns solve a bit-for-bit identical problem.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from alloy.solvers.solver_function import SolverStatus
from alloy.solvers.stats import ALLOY_SOLVER_STATS_VERSION, AlloySolveStatus, SolverStats
from benchmarks.problems.race_cars import CAR_LENGTH, CAR_WIDTH, DELTA_MAX, N_PARAMS, NU, NX, NZ, T_MAX, n_param

# IPOPT's own termination strings, mapped onto Alloy's solver-agnostic status enum
STATUS_MAP = {
  "Solve_Succeeded": AlloySolveStatus.OK,
  "Solved_To_Acceptable_Level": AlloySolveStatus.ACCEPTABLE,
  "Feasible_Point_Found": AlloySolveStatus.ACCEPTABLE,
  "Maximum_Iterations_Exceeded": AlloySolveStatus.MAX_ITER,
  "Maximum_CpuTime_Exceeded": AlloySolveStatus.MAX_ITER,
  "Infeasible_Problem_Detected": AlloySolveStatus.PRIMAL_INFEASIBLE,
  "Diverging_Iterates": AlloySolveStatus.DUAL_INFEASIBLE,
  "Restoration_Failed": AlloySolveStatus.NUMERICS,
  "Error_In_Step_Computation": AlloySolveStatus.NUMERICS,
  "Search_Direction_Becomes_Too_Small": AlloySolveStatus.NUMERICS,
  "User_Requested_Stop": AlloySolveStatus.USER_STOP,
}
# oracle timings CasADi reports per solve; their sum is the function-evaluation share
FE_TIMERS = ("t_wall_nlp_f", "t_wall_nlp_g", "t_wall_nlp_grad_f", "t_wall_nlp_jac_g", "t_wall_nlp_hess_l")


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


def build_casadi_race_car_nlp(config, sym_t=None) -> dict[str, Any]:
  """Return the symbolic pieces of the OCP, in Alloy's row and column order."""
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
  for i in range(n):
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


class CasadiRaceCarSolver:
  """`al.SolverFunction`-shaped wrapper around `ca.nlpsol("ipopt", ...)`."""

  def __init__(self, config, *, expand: bool = True):
    import casadi as ca

    started = time.perf_counter()
    pieces = build_casadi_race_car_nlp(config)
    self.n_eq, self.n_ineq = pieces["n_eq"], pieces["n_ineq"]
    self._bounds = {
      "lbx": pieces["x_lb"],
      "ubx": pieces["x_ub"],
      "lbg": np.concatenate([np.zeros(self.n_eq), np.full(self.n_ineq, -config.track_half_width)]),
      "ubg": np.concatenate([np.zeros(self.n_eq), np.full(self.n_ineq, config.track_half_width)]),
    }
    self.solver = ca.nlpsol(
      f"race_car_closed_loop_casadi_N{config.horizon}",
      "ipopt",
      {"x": pieces["z"], "p": pieces["p"], "f": pieces["f"], "g": ca.vertcat(pieces["h_eq"], pieces["g_ineq"])},
      {
        "print_time": False,
        "expand": expand,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.tol": config.ipopt_tol,
        "ipopt.max_iter": config.ipopt_max_iter,
        "ipopt.warm_start_init_point": "yes",
      },
    )
    self.build_ms = (time.perf_counter() - started) * 1000.0
    self.last_stats: SolverStats | None = None
    self.last_status: SolverStatus | None = None

  def __call__(self, z0, lam_eq0, lam_ineq0, lam_box0, p) -> dict[str, np.ndarray]:
    started = time.perf_counter()
    solution = self.solver(
      x0=z0,
      p=p,
      lam_g0=np.concatenate([np.asarray(lam_eq0, dtype=np.float64), np.asarray(lam_ineq0, dtype=np.float64)]),
      lam_x0=lam_box0,
      **self._bounds,
    )
    t_total = time.perf_counter() - started
    raw = self.solver.stats()
    t_fe = sum(float(raw.get(name, 0.0)) for name in FE_TIMERS)
    self.last_stats = SolverStats(
      version=ALLOY_SOLVER_STATS_VERSION,
      status=STATUS_MAP.get(str(raw.get("return_status", "")), AlloySolveStatus.ERROR),
      native_status=0,
      iter=int(raw.get("iter_count", 0)),
      obj=float(solution["f"]),
      t_total=t_total,
      t_fe=t_fe,
      t_solver=max(t_total - t_fe, 0.0),
      t_glue=0.0,
      n_eval_f=int(raw.get("n_call_nlp_f", 0)),
      n_eval_grad_f=int(raw.get("n_call_nlp_grad_f", 0)),
      n_eval_g=int(raw.get("n_call_nlp_g", 0)),
      n_eval_jac_g=int(raw.get("n_call_nlp_jac_g", 0)),
      n_eval_h=int(raw.get("n_call_nlp_hess_l", 0)),
    )
    self.last_status = self.last_stats.to_solver_status()
    lam_g = np.asarray(solution["lam_g"], dtype=np.float64).reshape(-1)
    return {
      "x": np.asarray(solution["x"], dtype=np.float64).reshape(-1),
      "lam_eq": lam_g[: self.n_eq],
      "lam_ineq": lam_g[self.n_eq :],
      "lam_box": np.asarray(solution["lam_x"], dtype=np.float64).reshape(-1),
    }
