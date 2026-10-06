from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from scaly.function.model import as_concrete
import scaly as sc
from scaly.codegen import render_c_module
from benchmarks.harness.casadi_ipopt import make_casadi_ipopt
from .common import (
  ClosedLoopConfig,
  CTFullWeights,
  DT_OFFSETS,
  DT_RELU_EPS,
  DT_W0_SHAPE,
  DT_W1_SHAPE,
  DT_W2_SHAPE,
  DTMLPWeights,
  FilterConfig,
  NCTRL,
  NSTATE,
  N_PHYSICS,
  N_PW,
  N_PW_DT,
  OFFSETS,
  W0_SHAPE,
  W1_SHAPE,
  W2_SHAPE,
)


# Shared scaly-sqp settings for this problem. The dual KKT tolerance is the
# solver default: the modified sparse LDL^T convexification removed the 8e-3
# stationarity floor the earlier dense Cholesky-probe regularization put under
# active-barrier steps, and the looser tolerance was what made SQP and IPOPT
# disagree by 3.5e-3 in applied control on the shared-state gate.
SQP_DUAL_TOL = 1e-4
SQP_MAX_ITER = 1000


def filter_n_pw(filt_cfg: FilterConfig) -> int:
  """Size of the weight tail the filter's own model reads out of `p`."""
  return N_PW_DT if filt_cfg.model == "dt" else N_PW


@dataclass(slots=True)
class FilterStats:
  impl: str
  step: int
  success: bool
  status: str
  solver_ms: float
  iterations: int | None
  objective: float
  min_g: float
  slack_l1: float
  tracking_cost: float
  eval_counts: dict[str, int] = field(default_factory=dict)
  eval_ms: dict[str, float] = field(default_factory=dict)
  extra: dict[str, Any] = field(default_factory=dict)


class SafetyFilter(Protocol):
  name: str
  stats_history: list[FilterStats]
  last_solve_wall_ms: float

  def compute_safe_input(self, states: np.ndarray, desired: np.ndarray, step: int = 0) -> np.ndarray: ...


class OpenLoopFilter:
  name = "open_loop"
  last_solve_wall_ms = 0.0

  def __init__(self) -> None:
    self.stats_history: list[FilterStats] = []

  def compute_safe_input(self, states: np.ndarray, desired: np.ndarray, step: int = 0) -> np.ndarray:
    _ = states
    u = np.clip(np.asarray(desired, dtype=np.float64), -1.0, 1.0)
    self.stats_history.append(FilterStats(self.name, step, True, "open", 0.0, None, 0.0, float("nan"), 0.0, 0.0, {}, {}))
    return u


class CasadiDTCBFSafetyFilter:
  name = "casadi_dt_hcbf"

  def __init__(
    self,
    loop_cfg: ClosedLoopConfig,
    filt_cfg: FilterConfig,
    weights: CTFullWeights | DTMLPWeights,
    *,
    _build_solver: bool = True,
    _sym_t=None,
    _mlp: str = "percar",
    _mtimes: str | None = None,
  ):
    import casadi as ca

    self.ca = ca
    self.loop_cfg = loop_cfg
    self.filt_cfg = filt_cfg
    self.weights = weights
    self.ncars = loop_cfg.ncars
    self.n_u = NCTRL * self.ncars
    self.n_s = loop_cfg.n_slack
    self.n_z = self.n_u + self.n_s
    self.n_pw = filter_n_pw(filt_cfg)
    self.n_p = NSTATE * self.ncars + self.n_u + self.n_pw + N_PHYSICS + 1
    self.stats_history: list[FilterStats] = []
    self.last_z: np.ndarray | None = None
    self.last_lam_x: np.ndarray | None = None
    self.last_lam_g: np.ndarray | None = None
    self._build_ms = 0.0
    self._sym_t = ca.MX if _sym_t is None else _sym_t
    # _mlp="batched" evaluates every car's MLP layer as one matrix-matrix product; _mtimes picks
    # CasADi 3.8's kernel ("reference", "classic" BLAS or "blasfeo") for the dense products.
    self._mlp = _mlp
    self._mtimes = _mtimes
    self._build(_build_solver)

  def _mm(self, a, b):
    return self.ca.mtimes(a, b) if self._mtimes is None else self.ca.mtimes(a, b, self._mtimes)

  def _unpack_pw(self, pw):
    ca = self.ca
    x_scale = pw[OFFSETS[0] : OFFSETS[1]]

    def mat(i: int, shape: tuple[int, int]):
      return ca.reshape(pw[OFFSETS[i] : OFFSETS[i + 1]], shape[1], shape[0]).T

    return (
      x_scale,
      mat(1, W0_SHAPE),
      pw[OFFSETS[2] : OFFSETS[3]],
      mat(3, W1_SHAPE),
      pw[OFFSETS[4] : OFFSETS[5]],
      mat(5, W2_SHAPE),
      pw[OFFSETS[6] : OFFSETS[7]],
    )

  def _silu(self, x):
    ca = self.ca
    return x / (1.0 + ca.exp(-x))

  def _world_vel(self, x, physics):
    ca = self.ca
    lf, lr = physics[0], physics[1]
    theta, vf, beta_f, beta_r = x[2], x[3], x[4], x[5]
    omega = vf * ca.sin(beta_f - beta_r) / ((lf + lr) * ca.cos(beta_r))
    vx_b = vf * ca.cos(beta_f)
    vy_b = vf * ca.sin(beta_f) - lf * omega
    return vx_b * ca.cos(theta) - vy_b * ca.sin(theta), vx_b * ca.sin(theta) + vy_b * ca.cos(theta), omega

  def _ode(self, x, u, pw, physics):
    ca = self.ca
    max_delta, steering_time_constant = physics[2], physics[3]
    delta = x[6]
    x_dot, y_dot, omega = self._world_vel(x, physics)
    delta_dot = (u[1] * max_delta - delta) / (steering_time_constant * 3.0)
    x_scale, w0, b0, w1, b1, w2, b2 = self._unpack_pw(pw)
    phi = ca.vertcat(x[3:7] / x_scale, u)
    h = self._silu(w0 @ phi + b0)
    h = self._silu(w1 @ h + b1)
    learned = w2 @ h + b2
    return ca.vertcat(x_dot, y_dot, omega, learned[0], learned[1], learned[2], delta_dot)

  def _rk4(self, x, u, pw, physics, dt):
    k1 = self._ode(x, u, pw, physics)
    k2 = self._ode(x + 0.5 * dt * k1, u, pw, physics)
    k3 = self._ode(x + 0.5 * dt * k2, u, pw, physics)
    k4 = self._ode(x + dt * k3, u, pw, physics)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

  def _unpack_pw_dt(self, pw):
    ca = self.ca

    def mat(i: int, shape: tuple[int, int]):
      return ca.reshape(pw[DT_OFFSETS[i] : DT_OFFSETS[i + 1]], shape[1], shape[0]).T

    return (
      pw[DT_OFFSETS[0] : DT_OFFSETS[1]],
      mat(1, DT_W0_SHAPE),
      pw[DT_OFFSETS[2] : DT_OFFSETS[3]],
      mat(3, DT_W1_SHAPE),
      pw[DT_OFFSETS[4] : DT_OFFSETS[5]],
      mat(5, DT_W2_SHAPE),
      pw[DT_OFFSETS[6] : DT_OFFSETS[7]],
    )

  def _pose_rk4(self, x, physics, dt):
    """RK4 on the pose rows with the velocity block held over the step; theta is left unwrapped."""
    ca = self.ca

    def pose_dot(s):
      x_dot, y_dot, omega = self._world_vel(s, physics)
      return ca.vertcat(x_dot, y_dot, omega, 0.0, 0.0, 0.0, 0.0)

    k1 = pose_dot(x)
    k2 = pose_dot(x + 0.5 * dt * k1)
    k3 = pose_dot(x + 0.5 * dt * k2)
    k4 = pose_dot(x + dt * k3)
    return (x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))[0:3]

  def _dt_step(self, x, u, pw, physics, dt):
    """The discrete MLP's one-step map; see ``common.dt_mlp_step_smooth_np``."""
    ca = self.ca
    max_delta, steering_time_constant = physics[2], physics[3]
    delta = x[6]
    x_scale, w0, b0, w1, b1, w2, b2 = self._unpack_pw_dt(pw)
    phi = ca.vertcat(ca.vertcat(x[3], x[4] - delta, x[5], delta) / x_scale, u[1], u[0])
    smooth_relu = lambda t: 0.5 * (t + ca.sqrt(t * t + DT_RELU_EPS**2))  # noqa: E731
    h = smooth_relu(self._mm(w0, phi) + b0)
    h = smooth_relu(self._mm(w1, h) + b1)
    learned = self._mm(w2, h) + b2
    delta_next = delta + dt * (u[1] * max_delta - delta) / steering_time_constant
    return ca.vertcat(self._pose_rk4(x, physics, dt), learned[0], learned[1] + delta_next, learned[2], delta_next)

  def _dt_step_batched(self, states, us, pw, physics, dt) -> list[Any]:
    """``_dt_step`` for every car at once: the MLP layers become ``W @ [phi_1 ... phi_C]``."""
    ca = self.ca
    max_delta, steering_time_constant = physics[2], physics[3]
    x_scale, w0, b0, w1, b1, w2, b2 = self._unpack_pw_dt(pw)
    n = len(states)
    phis = [ca.vertcat(ca.vertcat(x[3], x[4] - x[6], x[5], x[6]) / x_scale, u[1], u[0]) for x, u in zip(states, us, strict=True)]
    smooth_relu = lambda t: 0.5 * (t + ca.sqrt(t * t + DT_RELU_EPS**2))  # noqa: E731
    h = smooth_relu(self._mm(w0, ca.horzcat(*phis)) + ca.repmat(b0, 1, n))
    h = smooth_relu(self._mm(w1, h) + ca.repmat(b1, 1, n))
    learned = self._mm(w2, h) + ca.repmat(b2, 1, n)
    out = []
    for i, (x, u) in enumerate(zip(states, us, strict=True)):
      delta = x[6]
      delta_next = delta + dt * (u[1] * max_delta - delta) / steering_time_constant
      out.append(ca.vertcat(self._pose_rk4(x, physics, dt), learned[0, i], learned[1, i] + delta_next, learned[2, i], delta_next))
    return out

  def _step(self, x, u, pw, physics, dt):
    return self._dt_step(x, u, pw, physics, dt) if self.filt_cfg.model == "dt" else self._rk4(x, u, pw, physics, dt)

  def _pair_b(self, xi, xj, physics):
    """Order-1 hyperbolic pair barrier; see ``common.HCBFConfig``."""
    ca = self.ca
    hcbf, R = self.loop_cfg.hcbf, self.loop_cfg.safety_radius
    p = xj[0:2] - xi[0:2]
    vxi, vyi, _ = self._world_vel(xi, physics)
    vxj, vyj, _ = self._world_vel(xj, physics)
    v = ca.vertcat(vxj - vxi, vyj - vyi)
    r = ca.sqrt(ca.dot(p, p))
    v_x = (p[0] * v[0] + p[1] * v[1]) / r
    v_y = (p[0] * v[1] - p[1] * v[0]) / r
    d_eps = ca.sqrt((r - R) ** 2 + hcbf.eps**2)
    s = (r - R) / d_eps
    # a_env is the braking envelope V(d) = envelope_c d^envelope_q: the largest closing speed
    # the pair can still stop away within the clearance it has. The shipped constants are one
    # conservative fit covering both vehicle models — see common.HCBFConfig.
    a_env = hcbf.envelope_c * d_eps**hcbf.envelope_q
    q = ca.sqrt(d_eps * (r + R)) / R * v_y
    return v_x + s * ca.sqrt(ca.sqrt(a_env**4 + q**4))

  def _wall_b(self, xi, physics) -> list[Any]:
    ca = self.ca
    m = self.loop_cfg.wall_margin
    x_min, x_max, y_min, y_max = [physics[i] for i in range(4, 8)]
    vx, vy, _ = self._world_vel(xi, physics)
    clearances = [xi[0] - (x_min + m), (x_max - m) - xi[0], xi[1] - (y_min + m), (y_max - m) - xi[1]]
    out = []
    for clearance, v_closing in zip(clearances, (-vx, vx, -vy, vy), strict=True):
      d_eps = ca.sqrt(clearance * clearance + self.loop_cfg.wall_eps**2)
      s = clearance / d_eps
      v_max = self.loop_cfg.hcbf.single_envelope_c * d_eps**self.loop_cfg.hcbf.envelope_q
      out.append(s * v_max - v_closing)
    return out

  def _build(self, build_solver: bool = True) -> None:
    ca = self.ca
    t0 = time.perf_counter()
    z = self._sym_t.sym("z", self.n_z)
    p = self._sym_t.sym("p", self.n_p)
    bar_x = p[: NSTATE * self.ncars]
    u_des = p[NSTATE * self.ncars : NSTATE * self.ncars + self.n_u]
    offset = NSTATE * self.ncars + self.n_u
    pw = p[offset : offset + self.n_pw]
    physics = p[offset + self.n_pw : offset + self.n_pw + N_PHYSICS]
    dt = p[-1]
    u = z[: self.n_u]
    slack = z[self.n_u :]
    states = [bar_x[NSTATE * i : NSTATE * (i + 1)] for i in range(self.ncars)]
    us = [u[NCTRL * i : NCTRL * (i + 1)] for i in range(self.ncars)]
    if self._mlp == "batched":
      assert self.filt_cfg.model == "dt", "the batched MLP is only written for the discrete model"
      states_next = self._dt_step_batched(states, us, pw, physics, dt)
    else:
      states_next = [self._step(states[i], us[i], pw, physics, dt) for i in range(self.ncars)]

    rows = []
    for i in range(self.ncars):
      for j in range(i + 1, self.ncars):
        b_next = self._pair_b(states_next[i], states_next[j], physics)
        rows.append(b_next - (1.0 - self.loop_cfg.pair_gamma) * self._pair_b(states[i], states[j], physics))
    if self.loop_cfg.arena_avoidance:
      for i in range(self.ncars):
        for b_next, b_cur in zip(self._wall_b(states_next[i], physics), self._wall_b(states[i], physics), strict=True):
          rows.append(b_next - (1.0 - self.loop_cfg.wall_gamma) * b_cur)
    assert len(rows) == self.n_s
    g = ca.vertcat(*rows) + slack if rows else self._sym_t.zeros(0)
    weights = np.tile(np.asarray(self.filt_cfg.R, dtype=np.float64), self.ncars)
    du = u - u_des
    cost = ca.dot(du, ca.DM(weights) * du) + self.filt_cfg.slack_weight * ca.sum1(slack)

    self.cost_fn = ca.Function("ctdt_cost", [z, p], [cost])
    self.z_expr = z
    self.p_expr = p
    self.cost_expr = cost
    self.g_expr = g
    self.g_fn = ca.Function("ctdt_g", [z, p], [g])
    self.grad_fn = ca.Function("ctdt_grad", [z, p], [ca.gradient(cost, z)])
    self.jac_fn = ca.Function("ctdt_jac", [z, p], [ca.jacobian(g, z)])
    if not self.filt_cfg.limited_memory_hessian:
      lam = self._sym_t.sym("lam", int(g.size1()))
      sigma = self._sym_t.sym("sigma")
      self.hess_fn = ca.Function("ctdt_hess_lag", [z, p, sigma, lam], [ca.hessian(sigma * cost + ca.dot(lam, g), z)[0]])
    else:
      self.hess_fn = None

    if not build_solver:
      self._build_ms = (time.perf_counter() - t0) * 1000.0
      return
    nlp = ca.Function(f"ctdt_casadi_nlp_C{self.ncars}", [z, p], [cost, g])
    opts: dict[str, Any] = {
      "print_time": False,
      "ipopt.print_level": 0,
      "ipopt.sb": "yes",
      "ipopt.tol": self.filt_cfg.ipopt_tol,
      "ipopt.max_iter": self.filt_cfg.ipopt_max_iter,
      "ipopt.warm_start_init_point": "yes",
      "expand": False,
    }
    if self.filt_cfg.limited_memory_hessian:
      opts["ipopt.hessian_approximation"] = "limited-memory"
    self.solver = make_casadi_ipopt(f"ctdt_casadi_solver_C{self.ncars}", nlp, opts)
    self._build_ms = (time.perf_counter() - t0) * 1000.0

  def _pack_p(self, states: np.ndarray, desired: np.ndarray) -> np.ndarray:
    return np.concatenate([states.reshape(-1), desired.reshape(-1), self.weights.packed, self.loop_cfg.physics.array(), [self.loop_cfg.dt]])

  def _time_eval(self, fn, *args) -> float:
    t0 = time.perf_counter()
    for _ in range(max(1, self.filt_cfg.eval_repeats)):
      fn(*args)
    return (time.perf_counter() - t0) * 1000.0 / max(1, self.filt_cfg.eval_repeats)

  def compute_safe_input(self, states: np.ndarray, desired: np.ndarray, step: int = 0) -> np.ndarray:
    ca = self.ca
    states = np.asarray(states, dtype=np.float64)
    desired = np.clip(np.asarray(desired, dtype=np.float64), -1.0, 1.0)
    p = self._pack_p(states, desired)
    if self.last_z is None:
      z0 = np.concatenate([desired.reshape(-1), np.zeros(self.n_s)])
    else:
      z0 = self.last_z.copy()
      z0[: self.n_u] = np.clip(z0[: self.n_u], -1.0, 1.0)
    lbx = np.concatenate([-np.ones(self.n_u), np.zeros(self.n_s)])
    ubx = np.concatenate([np.ones(self.n_u), np.full(self.n_s, ca.inf)])
    n_g = int(self.g_fn.size1_out(0))
    lam_x0 = np.zeros(self.n_z) if self.last_lam_x is None else self.last_lam_x
    lam_g0 = np.zeros(n_g) if self.last_lam_g is None else self.last_lam_g
    started = time.perf_counter()
    sol = self.solver(z0, p, lbx, ubx, np.zeros(n_g), np.full(n_g, ca.inf), lam_x0, lam_g0)
    self.last_solve_wall_ms = (time.perf_counter() - started) * 1000.0
    stats = self.solver.last_stats
    assert stats is not None
    solver_ms = stats.t_total * 1000.0
    raw_success = stats.status in (sc.ScalySolveStatus.OK, sc.ScalySolveStatus.ACCEPTABLE)
    z_sol, f_sol, g_sol, lam_x, lam_g, _ = sol
    feasible = bool(np.all(np.isfinite(z_sol)) and (not g_sol.size or np.min(g_sol) >= -1e-6))
    success = raw_success and feasible
    if success:
      self.last_z = z_sol
      self.last_lam_x = lam_x
      self.last_lam_g = lam_g
      u_safe = np.clip(z_sol[: self.n_u], -1.0, 1.0).reshape(self.ncars, NCTRL)
    else:
      u_safe = desired.copy()
      u_safe[:, 0] = -1.0
      u_safe[:, 1] = 0.0

    eval_ms = {
      "f": self._time_eval(self.cost_fn, z_sol, p),
      "g": self._time_eval(self.g_fn, z_sol, p),
      "grad_f": self._time_eval(self.grad_fn, z_sol, p),
      "jac_g": self._time_eval(self.jac_fn, z_sol, p),
      "fe_total": stats.t_fe * 1000.0,
      "glue": stats.t_glue * 1000.0,
    }
    if self.hess_fn is not None:
      eval_ms["hess_lag"] = self._time_eval(self.hess_fn, z_sol, p, 1.0, lam_g)
    du = u_safe.reshape(-1) - desired.reshape(-1)
    tracking = float(du @ (np.tile(np.asarray(self.filt_cfg.R), self.ncars) * du))
    self.stats_history.append(
      FilterStats(
        self.name,
        step,
        success,
        stats.status.name.lower(),
        solver_ms,
        stats.iter,
        float(f_sol[0]),
        float(np.min(g_sol)) if g_sol.size else float("inf"),
        float(np.sum(z_sol[self.n_u :])),
        tracking,
        {
          "f": stats.n_eval_f,
          "g": stats.n_eval_g,
          "grad_f": stats.n_eval_grad_f,
          "jac_g": stats.n_eval_jac_g,
          "hess_lag": stats.n_eval_h,
        },
        eval_ms,
        {
          "build_ms": self._build_ms,
          "raw_success": raw_success,
          "native_status": stats.native_status,
          "max_slack": float(np.max(z_sol[self.n_u :], initial=0.0)),
        },
      )
    )
    return u_safe


def build_casadi_sqp(
  loop_cfg: ClosedLoopConfig,
  filt_cfg: FilterConfig,
  weights: CTFullWeights | DTMLPWeights,
  *,
  sqp_options: dict[str, str | int | float] | None = None,
):
  import casadi as ca

  from scaly_sqp.casadi import build_casadi_external_sqp

  controller = CasadiDTCBFSafetyFilter(loop_cfg, filt_cfg, weights, _build_solver=False)
  assert controller.hess_fn is not None
  z, p = ca.MX.sym("z", controller.n_z), ca.MX.sym("p", controller.n_p)
  lam_f, lam_g = ca.MX.sym("lam_f"), ca.MX.sym("lam_g", controller.n_s)
  stem = f"ca_unbumpercars_sqp_C{loop_cfg.ncars}"
  return build_casadi_external_sqp(
    name=stem,
    base=ca.Function(f"{stem}_base", [z, p], [controller.cost_fn(z, p), controller.g_fn(z, p)]),
    grad=ca.Function(f"{stem}_grad", [z, p], [controller.grad_fn(z, p)]),
    jac=ca.Function(f"{stem}_jac", [z, p], [controller.jac_fn(z, p)]),
    hess=ca.Function(f"{stem}_hess", [z, p, lam_f, lam_g], [controller.hess_fn(z, p, lam_f, lam_g)]),
    n_eq=0,
    n_ineq=controller.n_s,
    x_lb=np.concatenate([-np.ones(controller.n_u), np.zeros(controller.n_s)]),
    x_ub=np.concatenate([np.ones(controller.n_u), np.full(controller.n_s, np.inf)]),
    l_ineq=np.zeros(controller.n_s),
    u_ineq=np.full(controller.n_s, np.inf),
    options={"max_iter": SQP_MAX_ITER, "tol": filt_cfg.ipopt_tol, "dual_tol": SQP_DUAL_TOL, **(sqp_options or {})},
  )


# ---------------------------------------------------------------------------
# Scaly oracle. The neural ODE/RK4 step is a per-car Function, and the full
# safety-filter oracle uses sc.vmap(...) to evaluate it across the car axis.
# ---------------------------------------------------------------------------


def _silu_expr(x: sc.Expr) -> sc.Expr:
  return x / (1.0 + (-x).exp())


def _unpack_pw_expr(pw: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
  x_scale = pw[OFFSETS[0] : OFFSETS[1]]
  w0 = pw[OFFSETS[1] : OFFSETS[2]].reshape(W0_SHAPE)
  b0 = pw[OFFSETS[2] : OFFSETS[3]]
  w1 = pw[OFFSETS[3] : OFFSETS[4]].reshape(W1_SHAPE)
  b1 = pw[OFFSETS[4] : OFFSETS[5]]
  w2 = pw[OFFSETS[5] : OFFSETS[6]].reshape(W2_SHAPE)
  b2 = pw[OFFSETS[6] : OFFSETS[7]]
  return x_scale, w0, b0, w1, b1, w2, b2


def _world_vel_expr(state: sc.Expr, physics: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  lf, lr = physics[0], physics[1]
  theta, vf, beta_f, beta_r = state[2], state[3], state[4], state[5]
  omega = vf * (beta_f - beta_r).sin() / ((lf + lr) * beta_r.cos())
  vx_b = vf * beta_f.cos()
  vy_b = vf * beta_f.sin() - lf * omega
  return vx_b * theta.cos() - vy_b * theta.sin(), vx_b * theta.sin() + vy_b * theta.cos(), omega


@sc.function(
  sc.arg("state", NSTATE),
  sc.arg("u", NCTRL),
  sc.arg("pw", N_PW),
  sc.arg("physics", N_PHYSICS),
  outputs=sc.arg("xdot", NSTATE),
  name="ctdt_ctfull_ode",
)
def scaly_ctfull_ode_fn(state: sc.Expr, u: sc.Expr, pw: sc.Expr, physics: sc.Expr) -> sc.Expr:
  max_delta, steering_time_constant = physics[2], physics[3]
  delta = state[6]
  x_dot, y_dot, omega = _world_vel_expr(state, physics)
  delta_dot = (u[1] * max_delta - delta) / (steering_time_constant * 3.0)
  x_scale, w0, b0, w1, b1, w2, b2 = _unpack_pw_expr(pw)
  phi = sc.concat([state[3:7] / x_scale, u])
  h = _silu_expr((w0 @ phi + b0))
  h = _silu_expr((w1 @ h + b1))
  learned = w2 @ h + b2
  return sc.stack([x_dot, y_dot, omega, learned[0], learned[1], learned[2], delta_dot])


@sc.function(
  sc.arg("state", NSTATE),
  sc.arg("u", NCTRL),
  sc.arg("pw", N_PW),
  sc.arg("physics", N_PHYSICS),
  sc.arg("dt", 1),
  outputs=sc.arg("next", NSTATE),
  name="ctdt_ctfull_rk4",
)
def scaly_ctfull_rk4_fn(state: sc.Expr, u: sc.Expr, pw: sc.Expr, physics: sc.Expr, dt: sc.Expr) -> sc.Expr:
  h = dt[0]
  k1 = scaly_ctfull_ode_fn(state, u, pw, physics)
  k2 = scaly_ctfull_ode_fn(state + 0.5 * h * k1, u, pw, physics)
  k3 = scaly_ctfull_ode_fn(state + 0.5 * h * k2, u, pw, physics)
  k4 = scaly_ctfull_ode_fn(state + h * k3, u, pw, physics)
  return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _smooth_relu_expr(x: sc.Expr) -> sc.Expr:
  return 0.5 * (x + (x * x + DT_RELU_EPS**2).sqrt())


def _unpack_pw_dt_expr(pw: sc.Expr) -> tuple[sc.Expr, ...]:
  return (
    pw[DT_OFFSETS[0] : DT_OFFSETS[1]],
    pw[DT_OFFSETS[1] : DT_OFFSETS[2]].reshape(DT_W0_SHAPE),
    pw[DT_OFFSETS[2] : DT_OFFSETS[3]],
    pw[DT_OFFSETS[3] : DT_OFFSETS[4]].reshape(DT_W1_SHAPE),
    pw[DT_OFFSETS[4] : DT_OFFSETS[5]],
    pw[DT_OFFSETS[5] : DT_OFFSETS[6]].reshape(DT_W2_SHAPE),
    pw[DT_OFFSETS[6] : DT_OFFSETS[7]],
  )


@sc.function(sc.arg("state", NSTATE), sc.arg("physics", N_PHYSICS), outputs=sc.arg("posedot", NSTATE), name="ctdt_pose_dot")
def scaly_pose_dot_fn(state: sc.Expr, physics: sc.Expr) -> sc.Expr:
  x_dot, y_dot, omega = _world_vel_expr(state, physics)
  zero = 0.0 * state[3]
  return sc.stack([x_dot, y_dot, omega, zero, zero, zero, zero])


@sc.function(
  sc.arg("state", NSTATE),
  sc.arg("u", NCTRL),
  sc.arg("pw", N_PW_DT),
  sc.arg("physics", N_PHYSICS),
  sc.arg("dt", 1),
  outputs=sc.arg("next", NSTATE),
  name="ctdt_dt_mlp_step",
)
def scaly_dt_mlp_step_fn(state: sc.Expr, u: sc.Expr, pw: sc.Expr, physics: sc.Expr, dt: sc.Expr) -> sc.Expr:
  """The discrete MLP's one-step map; see ``common.dt_mlp_step_smooth_np``."""
  h_dt = dt[0]
  max_delta, steering_time_constant = physics[2], physics[3]
  delta = state[6]
  k1 = scaly_pose_dot_fn(state, physics)
  k2 = scaly_pose_dot_fn(state + 0.5 * h_dt * k1, physics)
  k3 = scaly_pose_dot_fn(state + 0.5 * h_dt * k2, physics)
  k4 = scaly_pose_dot_fn(state + h_dt * k3, physics)
  pose = state + (h_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
  x_scale, w0, b0, w1, b1, w2, b2 = _unpack_pw_dt_expr(pw)
  phi = sc.concat([sc.stack([state[3], state[4] - delta, state[5], delta]) / x_scale, sc.stack([u[1], u[0]])])
  h = _smooth_relu_expr((w0 @ phi + b0))
  h = _smooth_relu_expr((w1 @ h + b1))
  learned = w2 @ h + b2
  delta_next = delta + h_dt * (u[1] * max_delta - delta) / steering_time_constant
  return sc.stack([pose[0], pose[1], pose[2], learned[0], learned[1] + delta_next, learned[2], delta_next])


def build_scaly_oracle(loop_cfg: ClosedLoopConfig, filt_cfg: FilterConfig) -> sc.Function:
  """Build the safety filter's tracking cost and discrete barrier rows."""
  ncars = loop_cfg.ncars
  n_u = NCTRL * ncars
  n_s = loop_cfg.n_slack

  def constant(size: int) -> sc.TensorType:
    return sc.TensorType((size,), diff=False)

  @sc.function(
    sc.arg("z", n_u + n_s),
    sc.arg("bar_x", constant(NSTATE * ncars)),
    sc.arg("u_des", constant(n_u)),
    sc.arg("pw", constant(filter_n_pw(filt_cfg))),
    sc.arg("physics", constant(N_PHYSICS)),
    sc.arg("dt", constant(1)),
    outputs=sc.group(sc.arg("cost", ()), sc.arg("g", n_s)),
    name=f"ctdt_scaly_oracle_N{ncars}_{'walls' if loop_cfg.arena_avoidance else 'pairs'}",
  )
  def oracle(z: sc.Expr, bar_x: sc.Expr, u_des: sc.Expr, pw: sc.Expr, physics: sc.Expr, dt: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return _scaly_oracle_outputs((z, bar_x, u_des, pw, physics, dt), loop_cfg, filt_cfg)

  return oracle


def _scaly_oracle_outputs(
  inputs: tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr], loop_cfg: ClosedLoopConfig, filt_cfg: FilterConfig
) -> tuple[sc.Expr, sc.Expr]:
  """Build the objective and barrier rows traced by ``build_scaly_oracle``."""
  z, bar_x, u_des, pw, physics, dt = inputs
  ncars = loop_cfg.ncars
  n_u = NCTRL * ncars
  n_s = loop_cfg.n_slack
  u = z[:n_u]
  slack = z[n_u:]
  step_fn = scaly_dt_mlp_step_fn if filt_cfg.model == "dt" else scaly_ctfull_rk4_fn
  states_next = sc.vmap(step_fn, ncars)(bar_x, u, sc.broadcast(pw), sc.broadcast(physics), sc.broadcast(dt)).vec()
  hcbf, R = loop_cfg.hcbf, loop_cfg.safety_radius

  def pair_b(xi: sc.Expr, xj: sc.Expr, physics: sc.Expr) -> sc.Expr:
    """Order-1 hyperbolic pair barrier; see ``common.HCBFConfig``."""
    px, py = xj[0] - xi[0], xj[1] - xi[1]
    vxi, vyi, _ = _world_vel_expr(xi, physics)
    vxj, vyj, _ = _world_vel_expr(xj, physics)
    vx, vy = vxj - vxi, vyj - vyi
    r = (px * px + py * py).sqrt()
    v_x = (px * vx + py * vy) / r
    v_y = (px * vy - py * vx) / r
    d_eps = ((r - R) * (r - R) + hcbf.eps**2).sqrt()
    s = (r - R) / d_eps
    # a_env is the braking envelope V(d) = envelope_c d^envelope_q: the largest closing speed
    # the pair can still stop away within the clearance it has. The shipped constants are one
    # conservative fit covering both vehicle models — see common.HCBFConfig.
    a_env = hcbf.envelope_c * d_eps**hcbf.envelope_q
    q = (d_eps * (r + R)).sqrt() / R * v_y
    return v_x + s * (a_env**4 + q**4).sqrt().sqrt()

  m = loop_cfg.wall_margin

  def wall_b(xi: sc.Expr, physics: sc.Expr) -> sc.Expr:
    x_min, x_max, y_min, y_max = [physics[i] for i in range(4, 8)]
    vx, vy, _ = _world_vel_expr(xi, physics)
    clearances = [xi[0] - (x_min + m), (x_max - m) - xi[0], xi[1] - (y_min + m), (y_max - m) - xi[1]]
    out = []
    for clearance, v_closing in zip(clearances, (-vx, vx, -vy, vy), strict=True):
      d_eps = (clearance * clearance + loop_cfg.wall_eps**2).sqrt()
      s = clearance / d_eps
      v_max = hcbf.single_envelope_c * d_eps**hcbf.envelope_q
      out.append((s * v_max - v_closing))
    return sc.stack(out)

  @sc.function(
    sc.arg("xi", NSTATE),
    sc.arg("xj", NSTATE),
    sc.arg("xi_next", NSTATE),
    sc.arg("xj_next", NSTATE),
    sc.arg("physics", N_PHYSICS),
    outputs=sc.arg("g", 1),
    name="pair_hcbf",
  )
  def pair_hcbf(xi: sc.Expr, xj: sc.Expr, xi_next: sc.Expr, xj_next: sc.Expr, physics: sc.Expr) -> sc.Expr:
    return sc.stack([pair_b(xi_next, xj_next, physics) - (1.0 - loop_cfg.pair_gamma) * pair_b(xi, xj, physics)])

  @sc.function(
    sc.arg("state", NSTATE),
    sc.arg("state_next", NSTATE),
    sc.arg("physics", N_PHYSICS),
    outputs=sc.arg("g", 4),
    name="wall_hcbf",
  )
  def wall_hcbf(state: sc.Expr, state_next: sc.Expr, physics: sc.Expr) -> sc.Expr:
    return wall_b(state_next, physics) - (1.0 - loop_cfg.wall_gamma) * wall_b(state, physics)

  rows: list[sc.Expr] = []
  if loop_cfg.n_pairs:
    pairs = np.triu_indices(ncars, k=1)
    idx_i, idx_j = [np.concatenate([np.arange(NSTATE) + k * NSTATE for k in bodies]) for bodies in pairs]
    rows.append(
      sc.vmap(pair_hcbf, loop_cfg.n_pairs)(
        sc.gather(bar_x, idx_i), sc.gather(bar_x, idx_j), sc.gather(states_next, idx_i), sc.gather(states_next, idx_j), sc.broadcast(physics)
      ).vec()
    )
  if loop_cfg.arena_avoidance:
    rows.append(sc.vmap(wall_hcbf, ncars)(bar_x, states_next, sc.broadcast(physics)).vec())
  g = (sc.concat(rows) + slack) if rows else sc.const(np.zeros((0,)))
  assert g.shape == (n_s,)
  diff = u - u_des
  weights = sc.const(np.tile(np.asarray(filt_cfg.R, dtype=np.float64), ncars))
  cost = sc.dot(diff, weights * diff) + filt_cfg.slack_weight * slack.sum()
  return cost, g


def build_scaly_nlp(
  loop_cfg: ClosedLoopConfig,
  filt_cfg: FilterConfig,
  *,
  solver: str = "ipopt",
  options: dict[str, str | int | float] | None = None,
  oracle: sc.Function | None = None,
) -> sc.Solver:
  """Build the safety-filter NLP over car controls and nonnegative barrier slacks."""
  base = build_scaly_oracle(loop_cfg, filt_cfg) if oracle is None else oracle
  n_u, n_s = NCTRL * loop_cfg.ncars, loop_cfg.n_slack
  vars_tree = sc.group(sc.arg("u", (n_u,)), sc.arg("s", (n_s,)))
  params_tree = sc.group(
    sc.arg("bar_x", NSTATE * loop_cfg.ncars),
    sc.arg("u_des", n_u),
    sc.arg("pw", filter_n_pw(filt_cfg)),
    sc.arg("physics", N_PHYSICS),
    sc.arg("dt", 1),
  )
  problem_name = base.name.replace("_oracle", "_problem")

  @sc.problem(vars=vars_tree, params=params_tree, name=problem_name)
  def problem(
    variables: tuple[sc.Expr, sc.Expr], params: tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr]
  ) -> sc.ProblemSpec[tuple[sc.Expr, sc.Expr]]:
    u, s = variables
    cost, constraints = _scaly_oracle_outputs((sc.concat([u, s]), *params), loop_cfg, filt_cfg)
    inequalities = (sc.bounded(constraints, lo=sc.const(np.zeros(n_s)), name="barrier"),) if n_s else ()
    return sc.ProblemSpec(
      minimize=cost,
      ineq=inequalities,
      lb=(sc.const(-np.ones(n_u)), sc.const(np.zeros(n_s))),
      ub=(sc.const(np.ones(n_u)), sc.const(np.full(n_s, np.inf))),
    )

  result = sc.solver(problem, solver, name=base.name.replace("_oracle", f"_{solver}_nlp"), options=options)
  setattr(result.function, "_benchmark_base", base)
  return result


class ScalyDTCBFSafetyFilter:
  """DTCBF filter through a generated typed solver Function.

  IPOPT or SQP can consume Scaly-generated oracles; SQP can also consume
  CasADi-generated C oracles through the same wrapper contract. Warm starts
  carry the primal plus signed constraint and box multipliers between steps.
  """

  name = "scaly_dt_hcbf"

  def __init__(
    self,
    loop_cfg: ClosedLoopConfig,
    filt_cfg: FilterConfig,
    weights: CTFullWeights | DTMLPWeights,
    *,
    solver: str = "ipopt",
    oracle_provider: str = "scaly",
  ):
    if solver != "ipopt" and filt_cfg.limited_memory_hessian:
      raise ValueError("limited-memory Hessians apply only to IPOPT")
    self.loop_cfg = loop_cfg
    self.filt_cfg = filt_cfg
    self.weights = weights
    self.name = "scaly_dt_hcbf" if (solver, oracle_provider) == ("ipopt", "scaly") else f"{solver}_{oracle_provider}_dt_hcbf"
    self.ncars = loop_cfg.ncars
    self.n_u = NCTRL * self.ncars
    self.n_s = loop_cfg.n_slack
    self.n_z = self.n_u + self.n_s
    self.stats_history: list[FilterStats] = []
    self.last_z: np.ndarray | None = None
    self.last_mult_g: np.ndarray | None = None
    self.last_lam_box: np.ndarray | None = None
    self.fallback_nlp: sc.Solver | None = None
    self._packed_params = oracle_provider == "casadi"
    self._compile_ms: dict[str, float] = {}
    t0 = time.perf_counter()
    base = build_scaly_oracle(loop_cfg, filt_cfg)
    g = as_concrete(base).outputs[1]
    self.n_g = g.shape[0]
    options: dict[str, str | int | float] = {
      "print_level": 0,
      "sb": "yes",
      "tol": self.filt_cfg.ipopt_tol,
      "max_iter": self.filt_cfg.ipopt_max_iter,
      "warm_start_init_point": "yes",
    }
    if self.filt_cfg.limited_memory_hessian:
      options["hessian_approximation"] = "limited-memory"
    if oracle_provider == "scaly":
      if solver == "sqp":
        options = {
          "max_iter": SQP_MAX_ITER,
          "tol": self.filt_cfg.ipopt_tol,
          "dual_tol": SQP_DUAL_TOL,
        }
      self.nlp = build_scaly_nlp(loop_cfg, filt_cfg, solver=solver, options=options, oracle=base)
    elif oracle_provider == "casadi" and solver == "sqp":
      self.nlp = build_casadi_sqp(loop_cfg, filt_cfg, weights)
    else:
      raise ValueError(f"unsupported solver/oracle provider {solver!r}/{oracle_provider!r}")
    if solver == "sqp":
      fallback_options = {
        "max_iter": SQP_MAX_ITER,
        "tol": self.filt_cfg.ipopt_tol,
        "dual_tol": SQP_DUAL_TOL,
        "globalization": "l1",
        "watchdog": 5,
      }
      if oracle_provider == "scaly":
        self.fallback_nlp = build_scaly_nlp(loop_cfg, filt_cfg, solver="sqp", options=fallback_options, oracle=base)
      else:
        self.fallback_nlp = build_casadi_sqp(loop_cfg, filt_cfg, weights, sqp_options={"globalization": "l1", "watchdog": 5})
    descriptor = self.nlp.function.instantiate().descriptor
    self.jac_sparsity = descriptor.jac_sparsity
    self.hess_fn = descriptor.hess
    hess_sp = descriptor.hess_sparsity
    assert self.jac_sparsity is not None and hess_sp is not None
    self.hess_rows = np.asarray(hess_sp.rows, dtype=np.int32)
    self.hess_cols = np.asarray(hess_sp.cols, dtype=np.int32)
    self._build_ms = (time.perf_counter() - t0) * 1000.0
    self._warm_compile()

  def _warm_compile(self) -> None:
    from benchmarks.harness.timing import prepare_solver

    t0 = time.perf_counter()
    prepare_solver(self.nlp)
    self._compile_ms["solver"] = (time.perf_counter() - t0) * 1000.0
    if self.fallback_nlp is not None:
      t0 = time.perf_counter()
      prepare_solver(self.fallback_nlp)
      self._compile_ms["fallback_solver"] = (time.perf_counter() - t0) * 1000.0

  def dump_c(self, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nlp = self.nlp.function
    module = render_c_module(nlp, header_name=f"{nlp.name}.h", source_name=f"{nlp.name}.c")
    (out_dir / module.header_name).write_text(module.header)
    (out_dir / module.source_name).write_text(module.source)
    (out_dir / "solver.txt").write_text(f"{module.source_name}: {module.source.count(chr(10)) + 1} lines\n")

  def compute_safe_input(self, states: np.ndarray, desired: np.ndarray, step: int = 0) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    desired = np.clip(np.asarray(desired, dtype=np.float64), -1.0, 1.0)
    bar_x = states.reshape(-1)
    u_des = desired.reshape(-1)
    pw = self.weights.packed
    physics = self.loop_cfg.physics.array()
    dt = np.array([self.loop_cfg.dt])
    if self.last_z is None:
      z0 = np.concatenate([u_des, np.zeros(self.n_s)])
    else:
      z0 = self.last_z.copy()
      z0[: self.n_u] = np.clip(z0[: self.n_u], -1.0, 1.0)
    lam_g0 = self.last_mult_g if self.last_mult_g is not None else np.zeros(self.n_g)
    lam_box0 = self.last_lam_box if self.last_lam_box is not None else np.zeros(self.n_z)

    params = (np.concatenate([bar_x, u_des, pw, physics, dt]),) if self._packed_params else (bar_x, u_des, pw, physics, dt)

    self.last_solve_wall_ms = 0.0

    def solve(active_nlp: sc.Solver):
      descriptor = as_concrete(active_nlp.function).descriptor
      if descriptor.n_var_blocks == 2:
        variables0 = (z0[: self.n_u], z0[self.n_u :])
        box0 = (lam_box0[: self.n_u], lam_box0[self.n_u :])
      else:
        variables0, box0 = z0, lam_box0
      param_values = params[0] if self._packed_params else params
      started = time.perf_counter()
      variables, box, lam_eq, lam_ineq = active_nlp(param_values, warm=(variables0, box0, np.zeros(0), lam_g0))
      self.last_solve_wall_ms += (time.perf_counter() - started) * 1000.0
      if descriptor.n_var_blocks == 2:
        z_sol = np.concatenate(variables)
        box_sol = np.concatenate(box)
      else:
        z_sol, box_sol = variables, box
      evaluator = getattr(active_nlp.function, "_benchmark_base")
      _, constraints = evaluator(z_sol, *params)
      return {
        "x": np.asarray(z_sol),
        "g_ineq": np.asarray(constraints).reshape(-1),
        "lam_eq": lam_eq,
        "lam_ineq": lam_ineq,
        "lam_box": np.asarray(box_sol),
      }, active_nlp.stats()

    active_nlp = self.nlp
    out, stats = solve(active_nlp)
    attempt_stats = [stats]
    if stats.status not in (sc.ScalySolveStatus.OK, sc.ScalySolveStatus.ACCEPTABLE) and self.fallback_nlp is not None:
      active_nlp = self.fallback_nlp
      out, stats = solve(active_nlp)
      attempt_stats.append(stats)
    g_val = np.asarray(out["g_ineq"], dtype=np.float64).reshape(-1)
    raw_success = stats.status in (sc.ScalySolveStatus.OK, sc.ScalySolveStatus.ACCEPTABLE)
    feasible = bool(np.all(np.isfinite(out["x"])) and (not g_val.size or np.min(g_val) >= -1e-6))
    success = raw_success and feasible
    if success:
      self.last_z = out["x"].copy()
      self.last_mult_g = out["lam_ineq"].copy()
      self.last_lam_box = out["lam_box"].copy()
      u_safe = np.clip(out["x"][: self.n_u], -1.0, 1.0).reshape(self.ncars, NCTRL)
    else:
      u_safe = desired.copy()
      u_safe[:, 0] = -1.0
      u_safe[:, 1] = 0.0
    du = u_safe.reshape(-1) - u_des
    tracking = float(du @ (np.tile(np.asarray(self.filt_cfg.R), self.ncars) * du))
    eval_counts = {
      "f": sum(item.n_eval_f for item in attempt_stats),
      "grad_f": sum(item.n_eval_grad_f for item in attempt_stats),
      "g": sum(item.n_eval_g for item in attempt_stats),
      "jac_g": sum(item.n_eval_jac_g for item in attempt_stats),
      "hess_lag": sum(item.n_eval_h for item in attempt_stats),
    }
    eval_ms = {
      "fe_total": sum(item.t_fe for item in attempt_stats) * 1000.0,
      "solver": sum(item.t_solver for item in attempt_stats) * 1000.0,
      "qp": sum(item.t_qp for item in attempt_stats) * 1000.0,
      "globalization": sum(item.t_globalization for item in attempt_stats) * 1000.0,
      "glue": sum(item.t_glue for item in attempt_stats) * 1000.0,
    }
    assert self.jac_sparsity is not None
    self.stats_history.append(
      FilterStats(
        self.name,
        step,
        success,
        stats.status.name.lower(),
        sum(item.t_total for item in attempt_stats) * 1000.0,
        sum(item.iter for item in attempt_stats),
        stats.obj,
        float(np.min(g_val)) if g_val.size else float("inf"),
        float(np.sum(out["x"][self.n_u :])),
        tracking,
        eval_counts,
        eval_ms,
        {
          "max_slack": float(np.max(out["x"][self.n_u :], initial=0.0)),
          "build_ms": self._build_ms,
          "compile_ms": dict(self._compile_ms),
          "jac_nnz": int(self.jac_sparsity.nnz),
          "hess_nnz": int(self.hess_rows.size),
          "native_status": stats.native_status,
          "raw_success": raw_success,
          "retried_with_l1": len(attempt_stats) > 1,
        },
      )
    )
    return u_safe
