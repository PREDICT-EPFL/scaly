from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

import alloy as al
from alloy.codegen.c import render_c_module
from alloy_ipopt._ipopt import IPOPT_INF, solve_ipopt
from .common import ClosedLoopConfig, CTFullWeights, FilterConfig, NCTRL, NSTATE, N_PHYSICS, N_PW, OFFSETS, W0_SHAPE, W1_SHAPE, W2_SHAPE


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
  slack: float
  tracking_cost: float
  eval_counts: dict[str, int] = field(default_factory=dict)
  eval_ms: dict[str, float] = field(default_factory=dict)
  extra: dict[str, Any] = field(default_factory=dict)


class SafetyFilter(Protocol):
  name: str
  stats_history: list[FilterStats]

  def compute_safe_input(self, states: np.ndarray, desired: np.ndarray, step: int = 0) -> np.ndarray: ...


class OpenLoopFilter:
  name = "open_loop"

  def __init__(self) -> None:
    self.stats_history: list[FilterStats] = []

  def compute_safe_input(self, states: np.ndarray, desired: np.ndarray, step: int = 0) -> np.ndarray:
    _ = states
    u = np.clip(np.asarray(desired, dtype=np.float64), -1.0, 1.0)
    self.stats_history.append(FilterStats(self.name, step, True, "open", 0.0, None, 0.0, float("nan"), 0.0, 0.0, {}, {}))
    return u


class CasadiDTCBFSafetyFilter:
  name = "casadi_dt_pos_cbf"

  def __init__(self, loop_cfg: ClosedLoopConfig, filt_cfg: FilterConfig, weights: CTFullWeights):
    import casadi as ca

    self.ca = ca
    self.loop_cfg = loop_cfg
    self.filt_cfg = filt_cfg
    self.weights = weights
    self.ncars = loop_cfg.ncars
    self.n_u = NCTRL * self.ncars
    self.n_z = self.n_u + 1
    self.n_p = NSTATE * self.ncars + self.n_u + N_PW + N_PHYSICS + 1
    self.stats_history: list[FilterStats] = []
    self.last_z: np.ndarray | None = None
    self.last_lam_x: np.ndarray | None = None
    self.last_lam_g: np.ndarray | None = None
    self._build_ms = 0.0
    self._build()

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

  def _ode(self, x, u, pw, physics):
    ca = self.ca
    lf, lr, max_delta, steering_time_constant = [physics[i] for i in range(4)]
    theta, vf, beta_f, beta_r, delta = x[2], x[3], x[4], x[5], x[6]
    omega = vf * ca.sin(beta_f - beta_r) / ((lf + lr) * ca.cos(beta_r))
    vx_b = vf * ca.cos(beta_f)
    vy_b = vf * ca.sin(beta_f) - lf * omega
    x_dot = vx_b * ca.cos(theta) - vy_b * ca.sin(theta)
    y_dot = vx_b * ca.sin(theta) + vy_b * ca.cos(theta)
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

  def _pair_h(self, xi, xj):
    ca = self.ca
    d = xi[0:2] - xj[0:2]
    return ca.dot(d, d) - self.loop_cfg.safety_radius**2

  def _wall_h(self, xi, physics) -> list[Any]:
    m = self.loop_cfg.wall_margin
    x_min, x_max, y_min, y_max = [physics[i] for i in range(4, 8)]
    return [xi[0] - (x_min + m), (x_max - m) - xi[0], xi[1] - (y_min + m), (y_max - m) - xi[1]]

  def _build(self) -> None:
    ca = self.ca
    t0 = time.perf_counter()
    z = ca.MX.sym("z", self.n_z)
    p = ca.MX.sym("p", self.n_p)
    bar_x = p[: NSTATE * self.ncars]
    u_des = p[NSTATE * self.ncars : NSTATE * self.ncars + self.n_u]
    offset = NSTATE * self.ncars + self.n_u
    pw = p[offset : offset + N_PW]
    physics = p[offset + N_PW : offset + N_PW + N_PHYSICS]
    dt = p[-1]
    u = z[: self.n_u]
    slack = z[self.n_u]
    states = [bar_x[NSTATE * i : NSTATE * (i + 1)] for i in range(self.ncars)]
    states_next = [self._rk4(states[i], u[NCTRL * i : NCTRL * (i + 1)], pw, physics, dt) for i in range(self.ncars)]

    rows = []
    for i in range(self.ncars):
      for j in range(i + 1, self.ncars):
        rows.append(self._pair_h(states_next[i], states_next[j]) - (1.0 - self.loop_cfg.pair_gamma) * self._pair_h(states[i], states[j]) + slack)
    if self.loop_cfg.arena_avoidance:
      for i in range(self.ncars):
        for h_next, h_cur in zip(self._wall_h(states_next[i], physics), self._wall_h(states[i], physics), strict=True):
          rows.append(h_next - (1.0 - self.loop_cfg.wall_gamma) * h_cur + slack)
    g = ca.vertcat(*rows) if rows else ca.MX.zeros(0)
    weights = np.tile(np.asarray(self.filt_cfg.R, dtype=np.float64), self.ncars)
    du = u - u_des
    cost = ca.dot(du, ca.DM(weights) * du) + self.filt_cfg.slack_weight * slack * slack

    self.cost_fn = ca.Function("ctdt_cost", [z, p], [cost])
    self.g_fn = ca.Function("ctdt_g", [z, p], [g])
    self.grad_fn = ca.Function("ctdt_grad", [z, p], [ca.gradient(cost, z)])
    self.jac_fn = ca.Function("ctdt_jac", [z, p], [ca.jacobian(g, z)])
    if not self.filt_cfg.limited_memory_hessian:
      lam = ca.MX.sym("lam", int(g.size1()))
      sigma = ca.MX.sym("sigma")
      self.hess_fn = ca.Function("ctdt_hess_lag", [z, p, sigma, lam], [ca.hessian(sigma * cost + ca.dot(lam, g), z)[0]])
    else:
      self.hess_fn = None

    nlp = {"x": z, "p": p, "f": cost, "g": g}
    opts: dict[str, Any] = {
      "print_time": False,
      "ipopt.print_level": 0,
      "ipopt.sb": "yes",
      "ipopt.tol": self.filt_cfg.ipopt_tol,
      "ipopt.max_iter": self.filt_cfg.ipopt_max_iter,
      "expand": self.filt_cfg.casadi_expand,
    }
    if self.filt_cfg.limited_memory_hessian:
      opts["ipopt.hessian_approximation"] = "limited-memory"
    self.solver = ca.nlpsol("ctdt_casadi_solver", "ipopt", nlp, opts)
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
      z0 = np.concatenate([desired.reshape(-1), np.zeros(1)])
    else:
      z0 = self.last_z.copy()
      z0[: self.n_u] = np.clip(z0[: self.n_u], -1.0, 1.0)
    lbx = np.concatenate([-np.ones(self.n_u), np.zeros(1)])
    ubx = np.concatenate([np.ones(self.n_u), np.full(1, ca.inf)])
    n_g = int(self.g_fn.size1_out(0))
    args: dict[str, Any] = {"x0": z0, "p": p, "lbx": lbx, "ubx": ubx, "lbg": np.zeros(n_g), "ubg": np.full(n_g, ca.inf)}
    if self.last_lam_x is not None and self.last_lam_g is not None:
      args["lam_x0"] = self.last_lam_x
      args["lam_g0"] = self.last_lam_g
    t0 = time.perf_counter()
    sol = self.solver(**args)
    solver_ms = (time.perf_counter() - t0) * 1000.0
    stats = self.solver.stats()
    raw_success = bool(stats.get("success", False))
    z_sol = np.asarray(sol["x"], dtype=np.float64).reshape(-1)
    g_sol = np.asarray(sol["g"], dtype=np.float64).reshape(-1)
    feasible = bool(np.all(np.isfinite(z_sol)) and (not g_sol.size or np.min(g_sol) >= -1e-6))
    success = raw_success or feasible
    if success:
      self.last_z = z_sol
      self.last_lam_x = np.asarray(sol["lam_x"], dtype=np.float64).reshape(-1)
      self.last_lam_g = np.asarray(sol["lam_g"], dtype=np.float64).reshape(-1)
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
    }
    if self.hess_fn is not None:
      lam_g = np.asarray(sol["lam_g"], dtype=np.float64).reshape(-1)
      eval_ms["hess_lag"] = self._time_eval(self.hess_fn, z_sol, p, 1.0, lam_g)
    du = u_safe.reshape(-1) - desired.reshape(-1)
    tracking = float(du @ (np.tile(np.asarray(self.filt_cfg.R), self.ncars) * du))
    self.stats_history.append(
      FilterStats(
        self.name,
        step,
        success,
        str(stats.get("return_status", "unknown")) + (" (accepted feasible)" if success and not raw_success else ""),
        solver_ms,
        int(stats["iter_count"]) if "iter_count" in stats else None,
        float(sol["f"]),
        float(np.min(g_sol)) if g_sol.size else float("inf"),
        float(z_sol[-1]),
        tracking,
        {k: int(stats[k]) for k in stats if k.startswith("n_call_") and isinstance(stats[k], int)},
        eval_ms,
        {"build_ms": self._build_ms, "raw_success": raw_success},
      )
    )
    return u_safe


# ---------------------------------------------------------------------------
# Alloy oracle. The neural ODE/RK4 step is a per-car Function, and the full
# safety-filter oracle uses al.map_(...) to evaluate it across the car axis.
# ---------------------------------------------------------------------------


def _silu_expr(x: al.Expr) -> al.Expr:
  return x / (1.0 + (-x).exp())


def _unpack_pw_expr(pw: al.Expr) -> tuple[al.Expr, al.Expr, al.Expr, al.Expr, al.Expr, al.Expr, al.Expr]:
  x_scale = pw[OFFSETS[0] : OFFSETS[1]].block()
  w0 = pw[OFFSETS[1] : OFFSETS[2]].reshape(W0_SHAPE).block()
  b0 = pw[OFFSETS[2] : OFFSETS[3]].block()
  w1 = pw[OFFSETS[3] : OFFSETS[4]].reshape(W1_SHAPE).block()
  b1 = pw[OFFSETS[4] : OFFSETS[5]].block()
  w2 = pw[OFFSETS[5] : OFFSETS[6]].reshape(W2_SHAPE).block()
  b2 = pw[OFFSETS[6] : OFFSETS[7]].block()
  return x_scale, w0, b0, w1, b1, w2, b2


@al.function("ctdt_ctfull_ode", {"state": NSTATE, "u": NCTRL, "pw": N_PW, "physics": N_PHYSICS})
def alloy_ctfull_ode_fn(state, u, pw, physics):  # type: ignore[no-untyped-def]
  lf, lr, max_delta, steering_time_constant = [physics[i] for i in range(4)]
  theta, vf, beta_f, beta_r, delta = state[2], state[3], state[4], state[5], state[6]
  omega = vf * (beta_f - beta_r).sin() / ((lf + lr) * beta_r.cos())
  vx_b = vf * beta_f.cos()
  vy_b = vf * beta_f.sin() - lf * omega
  x_dot = vx_b * theta.cos() - vy_b * theta.sin()
  y_dot = vx_b * theta.sin() + vy_b * theta.cos()
  delta_dot = (u[1] * max_delta - delta) / (steering_time_constant * 3.0)
  x_scale, w0, b0, w1, b1, w2, b2 = _unpack_pw_expr(pw)
  phi = al.concat([state[3:7] / x_scale, u]).block()
  h = _silu_expr((w0 @ phi + b0).block()).block()
  h = _silu_expr((w1 @ h + b1).block()).block()
  learned = (w2 @ h + b2).block()
  return {"xdot": al.stack([x_dot, y_dot, omega, learned[0], learned[1], learned[2], delta_dot])}


@al.function("ctdt_ctfull_rk4", {"state": NSTATE, "u": NCTRL, "pw": N_PW, "physics": N_PHYSICS, "dt": 1})
def alloy_ctfull_rk4_fn(state, u, pw, physics, dt):  # type: ignore[no-untyped-def]
  h = dt[0]
  k1 = alloy_ctfull_ode_fn.call([state, u, pw, physics])[0]
  k2 = alloy_ctfull_ode_fn.call([state + 0.5 * h * k1, u, pw, physics])[0]
  k3 = alloy_ctfull_ode_fn.call([state + 0.5 * h * k2, u, pw, physics])[0]
  k4 = alloy_ctfull_ode_fn.call([state + h * k3, u, pw, physics])[0]
  return {"next": (state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).block()}


def _single_output(fn: al.Function, output_name: str, new_name: str) -> al.Function:
  idx = fn.output_names.index(output_name)
  return al.Function(new_name, fn.inputs, [fn.outputs[idx]], fn.input_names, [fn.output_names[idx]], [fn.output_sparsities[idx]])


def build_alloy_oracle(loop_cfg: ClosedLoopConfig, filt_cfg: FilterConfig) -> al.Function:
  ncars = loop_cfg.ncars
  n_u = NCTRL * ncars
  z = al.sym("z", n_u + 1)
  bar_x = al.sym("bar_x", NSTATE * ncars, diff=False)
  u_des = al.sym("u_des", n_u, diff=False)
  pw = al.sym("pw", N_PW, diff=False)
  physics = al.sym("physics", N_PHYSICS, diff=False)
  dt = al.sym("dt", 1, diff=False)
  u = z[:n_u]
  slack = z[n_u : n_u + 1]
  states_next = al.map_(alloy_ctfull_rk4_fn, ncars, [(bar_x, 0, NSTATE), (u, 0, NCTRL), (pw, 0, 0), (physics, 0, 0), (dt, 0, 0)])

  def pair_h(pack: al.Expr, i: int, j: int) -> al.Expr:
    xi = pack[NSTATE * i : NSTATE * (i + 1)]
    xj = pack[NSTATE * j : NSTATE * (j + 1)]
    d = xi[0:2] - xj[0:2]
    return (al.dot(d, d) - loop_cfg.safety_radius**2).scalar()

  m = loop_cfg.wall_margin
  x_min, x_max, y_min, y_max = [physics[i] for i in range(4, 8)]

  def wall_h(pack: al.Expr, i: int) -> list[al.Expr]:
    xi = pack[NSTATE * i : NSTATE * (i + 1)]
    return [
      (xi[0] - (x_min + m)).scalar(),
      ((x_max - m) - xi[0]).scalar(),
      (xi[1] - (y_min + m)).scalar(),
      ((y_max - m) - xi[1]).scalar(),
    ]

  rows: list[al.Expr] = []
  for i in range(ncars):
    for j in range(i + 1, ncars):
      rows.append((pair_h(states_next, i, j) - (1.0 - loop_cfg.pair_gamma) * pair_h(bar_x, i, j) + slack[0]).scalar())
  if loop_cfg.arena_avoidance:
    for i in range(ncars):
      for hn, hc in zip(wall_h(states_next, i), wall_h(bar_x, i), strict=True):
        rows.append((hn - (1.0 - loop_cfg.wall_gamma) * hc + slack[0]).scalar())
  g = al.stack(rows).scalar() if rows else al.const(np.zeros((0,)))
  diff = u - u_des
  weights = al.const(np.tile(np.asarray(filt_cfg.R, dtype=np.float64), ncars))
  cost = (al.dot(diff, weights * diff) + filt_cfg.slack_weight * slack[0] * slack[0]).scalar()
  return al.Function(
    f"ctdt_alloy_oracle_N{ncars}_{'walls' if loop_cfg.arena_avoidance else 'pairs'}",
    [z, bar_x, u_des, pw, physics, dt],
    [cost, g],
    ["z", "bar_x", "u_des", "pw", "physics", "dt"],
    ["cost", "g"],
  )


class AlloyDTCBFSafetyFilter:
  name = "alloy_dt_pos_cbf"

  def __init__(self, loop_cfg: ClosedLoopConfig, filt_cfg: FilterConfig, weights: CTFullWeights):
    self.loop_cfg = loop_cfg
    self.filt_cfg = filt_cfg
    self.weights = weights
    self.ncars = loop_cfg.ncars
    self.n_u = NCTRL * self.ncars
    self.n_z = self.n_u + 1
    self.stats_history: list[FilterStats] = []
    self.last_z: np.ndarray | None = None
    self.last_mult_g: np.ndarray | None = None
    self.last_z_L: np.ndarray | None = None
    self.last_z_U: np.ndarray | None = None
    self._build_ms = 0.0
    self._compile_ms: dict[str, float] = {}
    t0 = time.perf_counter()
    self.base_fn = build_alloy_oracle(loop_cfg, filt_cfg)
    self.cost_fn = _single_output(self.base_fn, "cost", self.base_fn.name + "_cost")
    self.g_fn = _single_output(self.base_fn, "g", self.base_fn.name + "_g")
    inputs = ["z", "bar_x", "u_des", "pw", "physics", "dt"]
    self.grad_fn = self.base_fn.factory(self.base_fn.name + "_grad_cost_z", inputs, ["grad:cost:z"])
    self.jac_fn = self.base_fn.factory(self.base_fn.name + "_spjac_g_z", inputs, ["spjac:g:z"])
    self.jac_sparsity = self.jac_fn.output_sparsities[0]
    assert self.jac_sparsity is not None
    self._build_ms = (time.perf_counter() - t0) * 1000.0
    self._warm_compile()

  def _warm_compile(self) -> None:
    z = np.zeros(self.n_z)
    bar_x = np.zeros(NSTATE * self.ncars)
    u_des = np.zeros(self.n_u)
    pw = self.weights.packed
    physics = self.loop_cfg.physics.array()
    dt = np.array([self.loop_cfg.dt])
    for label, fn in (("cost", self.cost_fn), ("g", self.g_fn), ("grad_f", self.grad_fn), ("jac_g", self.jac_fn)):
      t0 = time.perf_counter()
      fn.eval_list(z, bar_x, u_des, pw, physics, dt)
      self._compile_ms[label] = (time.perf_counter() - t0) * 1000.0

  def dump_c(self, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, fn in (("cost", self.cost_fn), ("g", self.g_fn), ("grad_f", self.grad_fn), ("jac_g", self.jac_fn)):
      module = render_c_module(fn, header_name=f"{fn.name}.h", source_name=f"{fn.name}.c", typed_buffers=False)
      (out_dir / module.header_name).write_text(module.header)
      (out_dir / module.source_name).write_text(module.source)
      (out_dir / f"{label}.txt").write_text(f"{module.source_name}: {module.source.count(chr(10)) + 1} lines\n")

  def _timed(self, label: str, counts: dict[str, int], totals: dict[str, float], fn, *args) -> np.ndarray:
    t0 = time.perf_counter()
    out = np.asarray(fn.eval_list(*args)[0], dtype=np.float64).reshape(-1)
    totals[label] = totals.get(label, 0.0) + (time.perf_counter() - t0) * 1000.0
    counts[label] = counts.get(label, 0) + 1
    return out

  def compute_safe_input(self, states: np.ndarray, desired: np.ndarray, step: int = 0) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    desired = np.clip(np.asarray(desired, dtype=np.float64), -1.0, 1.0)
    bar_x = states.reshape(-1)
    u_des = desired.reshape(-1)
    pw = self.weights.packed
    physics = self.loop_cfg.physics.array()
    dt = np.array([self.loop_cfg.dt])
    if self.last_z is None:
      z0 = np.concatenate([u_des, np.zeros(1)])
    else:
      z0 = self.last_z.copy()
      z0[: self.n_u] = np.clip(z0[: self.n_u], -1.0, 1.0)
    n_g = self.g_fn.outputs[0].shape[0]
    jac_sp = self.jac_sparsity
    assert jac_sp is not None
    jac_rows = np.asarray(jac_sp.rows, dtype=np.int32)
    jac_cols = np.asarray(jac_sp.cols, dtype=np.int32)
    counts: dict[str, int] = {}
    totals: dict[str, float] = {}

    def eval_f(z: np.ndarray) -> float:
      return float(self._timed("f", counts, totals, self.cost_fn, z, bar_x, u_des, pw, physics, dt)[0])

    def eval_grad_f(z: np.ndarray) -> np.ndarray:
      return self._timed("grad_f", counts, totals, self.grad_fn, z, bar_x, u_des, pw, physics, dt)

    def eval_g(z: np.ndarray) -> np.ndarray:
      return self._timed("g", counts, totals, self.g_fn, z, bar_x, u_des, pw, physics, dt)

    def eval_jac_g(z: np.ndarray) -> np.ndarray:
      return self._timed("jac_g", counts, totals, self.jac_fn, z, bar_x, u_des, pw, physics, dt)

    def eval_h(_z: np.ndarray, _obj_factor: float, _lam: np.ndarray) -> np.ndarray:
      counts["hess_lag"] = counts.get("hess_lag", 0) + 1
      return np.zeros(0, dtype=np.float64)

    options: dict[str, str | int | float] = {
      "print_level": 0,
      "sb": "yes",
      "tol": self.filt_cfg.ipopt_tol,
      "max_iter": self.filt_cfg.ipopt_max_iter,
      "hessian_approximation": "limited-memory",
    }
    if self.last_mult_g is not None:
      options["warm_start_init_point"] = "yes"
    t0 = time.perf_counter()
    sol = solve_ipopt(
      n=self.n_z,
      m=n_g,
      x0=z0,
      x_L=np.concatenate([-np.ones(self.n_u), np.zeros(1)]),
      x_U=np.concatenate([np.ones(self.n_u), np.full(1, IPOPT_INF)]),
      g_L=np.zeros(n_g),
      g_U=np.full(n_g, IPOPT_INF),
      jac_rows=jac_rows,
      jac_cols=jac_cols,
      hess_rows=np.zeros(0, dtype=np.int32),
      hess_cols=np.zeros(0, dtype=np.int32),
      eval_f=eval_f,
      eval_grad_f=eval_grad_f,
      eval_g=eval_g,
      eval_jac_g=eval_jac_g,
      eval_h=eval_h,
      options=options,
      lam_g0=self.last_mult_g,
      z_L0=self.last_z_L,
      z_U0=self.last_z_U,
    )
    solver_ms = (time.perf_counter() - t0) * 1000.0
    raw_success = sol.status in (0, 1, 6)
    feasible = bool(np.all(np.isfinite(sol.x)) and (not sol.g.size or np.min(sol.g) >= -1e-6))
    success = raw_success or feasible
    if success:
      self.last_z = sol.x.copy()
      self.last_mult_g = sol.mult_g.copy()
      self.last_z_L = sol.mult_x_L.copy()
      self.last_z_U = sol.mult_x_U.copy()
      u_safe = np.clip(sol.x[: self.n_u], -1.0, 1.0).reshape(self.ncars, NCTRL)
    else:
      u_safe = desired.copy()
      u_safe[:, 0] = -1.0
      u_safe[:, 1] = 0.0
    du = u_safe.reshape(-1) - u_des
    tracking = float(du @ (np.tile(np.asarray(self.filt_cfg.R), self.ncars) * du))
    eval_avg_ms = {k: totals[k] / max(1, counts.get(k, 1)) for k in totals}
    self.stats_history.append(
      FilterStats(
        self.name,
        step,
        success,
        sol.status_name + (" (accepted feasible)" if success and not raw_success else ""),
        solver_ms,
        sol.iters,
        sol.obj,
        float(np.min(sol.g)) if sol.g.size else float("inf"),
        float(sol.x[-1]),
        tracking,
        counts,
        eval_avg_ms,
        {
          "build_ms": self._build_ms,
          "compile_ms": dict(self._compile_ms),
          "jac_nnz": int(jac_sp.nnz),
          "eval_total_ms": totals,
          "ipopt_eval_counts": sol.eval_counts,
          "raw_success": raw_success,
        },
      )
    )
    return u_safe
