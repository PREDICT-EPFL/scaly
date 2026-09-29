"""Neural Process MPC (Waibel, Mello Rella, Jones 2026) in Scaly.

The authors' conditional neural process, read from their checkpoint with `nn.load_torch_state_dict`
(this file never imports torch): the encoder, GELU with `erf`, turns a context trajectory into the
latent code z; the decoder's mean, sigmoid, is the one-step model. Their two MPCs on the Furuta
pendulum, the neural one and the equation-based baseline (the analytic model under implicit
midpoint), are each written in two forms, so that every upstream transcription has a Scaly twin
posing the same NLP to the same solver:

  opti    `MPCController._build_optimization` (CasADi Opti, Python and C++): every constraint a row,
          the initial state as a +-1e-3 band of rows, four slacks of which only the arm angle's is used
  laopt   `MultipleShootingXDiff` over `FurutaNPOCP`/`FurutaEqOCP` (laOPT): input, slack and initial-state
          bands as variable bounds, the three unused slacks fixed at zero, the softened arm-angle rows

Each problem has one generated solver per method, IPOPT (the upstream's CasADi and laOPT setting:
tol 1e-6, 50 iterations) or Scaly's SQP with PIQP subproblems (laOPT's SQP setting: objective-only
Hessian, filter line search, 15 iterations). The initial state, the model parameter (z or the plant
parameters) and the terminal weight P are runtime parameters, so one compiled solver serves every
latent code the encoder produces; CasADi's Opti bakes z and P into its graph and is rebuilt per code.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.linalg import solve_discrete_are

import scaly as sc
from scaly import nn
from protocol import COST, DT, HORIZON, NU, NX, PHI_LIMIT, SLACK_LIMIT, SLACK_WEIGHT, TORQUE_LIMIT, X0_BAND, midpoint_step

REPO = Path(__file__).resolve().parents[3]
CHECKPOINT = REPO / "bench" / "problems" / "npmpc" / "data" / "cnp_model.pth"  # the upstream's model/model.pth, byte for byte
CONTEXT_MAX = 100  # the encoder Function's context capacity; shorter contexts are masked
FORMS = ("opti", "laopt")
IPOPT_OPTIONS = {"max_iter": 50, "tol": 1e-6, "print_level": 0, "sb": "yes"}  # the upstream's configure_solver
SQP_OPTIONS = {"max_iter": 15, "hessian": "objective", "globalization": "filter", "tol": 1e-6, "dual_tol": 1e-4}  # laOPT's

Layers = list[tuple[np.ndarray, np.ndarray | None]]


@dataclass(frozen=True)
class CNP:
  """The trained conditional neural process: layers as `(W, b)` in float64, scalers as `(weight, bias)`."""

  encoder: Layers  # [x, y] (7) -> 64 -> 128 -> 128 -> 64 -> z (4), GELU
  mu: Layers  # [x, z] (9) -> 32 -> 32 -> 2, sigmoid
  x_scale: tuple[np.ndarray, np.ndarray]
  y_scale: tuple[np.ndarray, np.ndarray]
  y_inverse: tuple[np.ndarray, np.ndarray]


def load_cnp(path: str | Path = CHECKPOINT) -> CNP:
  """Their checkpoint, a pickled dict of state dicts; the activations are checked against its config."""
  ckpt = cast(dict[str, Any], nn.load_torch_state_dict(path))  # a whole checkpoint: nested state dicts and its config
  params = ckpt["params"]
  if (params["cnp_encoder"]["inner_activation"], params["cnp_decoder"]["inner_activation"], params["cnp_decoder"]["mu_activation"]) != (
    "GELU",
    "Sigmoid",
    "None",
  ):
    raise ValueError(f"unexpected activations in {path}: {params}")
  encoder = nn.layers_from_state_dict(ckpt["encoder"], [f"cnp_encoder.{i}" for i in range(0, 10, 2)])
  mu = nn.layers_from_state_dict(ckpt["decoder"], [f"mu_decoder.{i}" for i in range(0, 6, 2)])

  def scaler(name: str) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(ckpt[name]["weight"], np.float64), np.asarray(ckpt[name]["bias"], np.float64)

  return CNP(encoder, mu, scaler("x_direct_scaler"), scaler("y_direct_scaler"), scaler("y_inverse_scaler"))


def gelu(x: sc.Expr) -> sc.Expr:
  """PyTorch's exact GELU, `x Phi(x)`."""
  return x * 0.5 * (1.0 + (x / math.sqrt(2.0)).erf())


def decoder_mean(cnp: CNP, x_nn: sc.Expr, z: sc.Expr) -> sc.Expr:
  """The decoder's mean velocity change for features `x_nn = (sin theta, cos theta, theta_dot, phi_dot, u)`."""
  (xw, xb), (yw, yb) = cnp.x_scale, cnp.y_inverse
  h = sc.concat([x_nn * sc.const(xw) + sc.const(xb), z])
  return nn.mlp(h, cnp.mu, nn.sigmoid) * sc.const(yw) + sc.const(yb)


def np_step(cnp: CNP, x: sc.Expr, u: sc.Expr, z: sc.Expr, dt: float = DT) -> sc.Expr:
  """The learned one-step map (`FurutaNPIntegration`): velocities advance by the decoder's mean, angles
  by the trapezoid over the step."""
  y = decoder_mean(cnp, sc.stack([x[0].sin(), x[0].cos(), x[2], x[3], u[0]]), z)
  return x + sc.concat([dt * (x[2:4] + y / 2.0), y])


def furuta_ode(x: sc.Expr, u: sc.Expr, p: sc.Expr) -> sc.Expr:
  """`protocol.furuta_ode` with the plant parameters p = (l_p, m_p, l_r, m_r) symbolic."""
  theta_dot, phi_dot = x[2], x[3]
  lp, mp, lr, mr = p[0], p[1], p[2], p[3]
  j_r = mr * lr * lr / 3.0 + mp * lr * lr
  j_p = mp * lp * lp / 3.0
  s, c, s2 = x[0].sin(), x[0].cos(), (2.0 * x[0]).sin()
  b00, b01, b11 = j_p, -mp * lr * lp / 2.0 * c, j_r + j_p * s * s
  a0 = j_p * s2 / 2.0 * phi_dot * phi_dot + mp * lp * 9.81 / 2.0 * s
  a1 = -j_p * s2 * phi_dot * theta_dot - mp * lr * lp / 2.0 * s * theta_dot * theta_dot + u[0]
  det = b00 * b11 - b01 * b01
  return sc.stack([theta_dot, phi_dot, (b11 * a0 - b01 * a1) / det, (b00 * a1 - b01 * a0) / det])


class Model:
  """One of the two dynamics models with its Functions: the OCP's per-stage defect, the step the
  controller predicts with, and the linearization at upright that sets the terminal weight."""

  def __init__(self, kind: str, cnp: CNP | None = None) -> None:
    if kind not in ("neural", "equation"):
      raise ValueError(f"unknown model {kind!r}")
    self.kind = kind
    stage_inputs = sc.G(sc.L("x", NX), sc.L("u", NU), sc.L("xn", NX), sc.L("theta", 4))
    if kind == "neural":
      assert cnp is not None

      def step(x: sc.Expr, u: sc.Expr, z: sc.Expr) -> sc.Expr:
        return np_step(cnp, x, u, z)

      @sc.function(stage_inputs, name="np_defect")
      def defect(inputs: tuple[sc.Expr, ...]) -> sc.Expr:
        x, u, xn, z = inputs
        return step(x, u, z) - xn

    else:

      def step(x: sc.Expr, u: sc.Expr, p: sc.Expr) -> sc.Expr:
        k1 = furuta_ode(x, u, p)
        return x + DT * furuta_ode(x + DT / 2.0 * k1, u, p)  # the explicit midpoint FurutaMPC linearizes

      @sc.function(stage_inputs, name="eq_defect")
      def defect(inputs: tuple[sc.Expr, ...]) -> sc.Expr:
        x, u, xn, p = inputs  # implicit midpoint, laOPT's IRK2 and MidpointIntegration.casadi_implicit
        return x + DT * furuta_ode((x + xn) / 2.0, u, p) - xn

    @sc.function(sc.G(sc.L("x", NX), sc.L("u", NU), sc.L("theta", 4)), name=f"{kind}_step")
    def step_fn(inputs: tuple[sc.Expr, ...]) -> sc.Expr:
      return step(*inputs)

    @sc.function(sc.G(sc.L("x", NX), sc.L("u", NU), sc.L("theta", 4)), name=f"{kind}_linearization")
    def linearization(inputs: tuple[sc.Expr, ...]) -> sc.Expr:
      x, u, theta = inputs
      xn = step(x, u, theta)
      return sc.concat([sc.jacobian(xn, x).reshape((NX * NX,)), sc.jacobian(xn, u).reshape((NX * NU,))])

    self.defect, self.step, self.linearization = defect, step_fn, linearization

  def rollout(self, theta: np.ndarray, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
    """The controller's own prediction from x0 under the inputs u, `(len(u) + 1, 4)`."""
    states = [np.asarray(x0, dtype=np.float64)]
    for uk in u:
      if self.kind == "equation":
        states.append(midpoint_step(states[-1], float(uk), DT, theta))  # FurutaMPC.integrate
      else:
        states.append(np.asarray(self.step((states[-1], np.array([uk]), theta)), dtype=np.float64))
    return np.array(states)

  def terminal_weight(self, theta: np.ndarray) -> np.ndarray:
    """P from the discrete Riccati equation for the model linearized at upright, as
    `compute_terminal_P` does with torch's autograd."""
    jac = np.asarray(self.linearization((np.zeros(NX), np.zeros(NU), np.asarray(theta, np.float64))), np.float64)
    A, B = jac[: NX * NX].reshape(NX, NX), jac[NX * NX :].reshape(NX, NU)
    return np.asarray(solve_discrete_are(A, B, np.diag(COST["x_end"]), np.diag(COST["u"])), np.float64)


@sc.function(sc.G(sc.L("x", NX), sc.L("u", NU), sc.L("xn", NX)), name="stage_cost")
def stage_cost(inputs: tuple[sc.Expr, ...]) -> sc.Expr:
  """`furuta_cost`'s stage and inter-stage terms for one step; the pendulum angle enters through the
  2 pi-periodic lift 2 (1 - cos theta) = (2 sin(theta / 2))^2."""
  x, u, xn = inputs
  lifted = sc.stack([2.0 * (1.0 - x[0].cos()), x[1] * x[1], x[2] * x[2], x[3] * x[3]])
  dx = xn - x
  return sc.dot(sc.const(COST["x"]), lifted) + COST["u"][0] * u[0] * u[0] + sc.dot(sc.const(COST["x_diff"]), dx * dx)


def problem(model: Model, form: str, horizon: int = HORIZON) -> Any:
  """The MPC's NLP. Variables: the states `(N+1) x 4` row by row, the torques, the four slacks.
  Parameters: the initial state, the model parameter (z or p) and the terminal weight, row-major."""
  if form not in FORMS:
    raise ValueError(f"unknown form {form!r}")
  n = horizon
  inf = np.inf

  @sc.opt.problem(
    vars=sc.G(sc.L("x", (n + 1) * NX), sc.L("u", n * NU), sc.L("slack", NX)),
    params=sc.G(sc.L("x0", NX), sc.L("theta", 4), sc.L("P", NX * NX)),
    name=f"npmpc_{model.kind}_{form}_N{n}",
  )
  def ocp(variables: tuple[sc.Expr, ...], params: tuple[sc.Expr, ...]) -> sc.opt.ProblemSpec:
    x, u, s = variables
    x0, theta, P = params
    defects = sc.vmap(model.defect, n, [(x, 0, NX), (u, 0, NU), (x, NX, NX), (theta, 0, 0)])
    stages = sc.vmap(stage_cost, n, [(x, 0, NX), (u, 0, NU), (x, NX, NX)])
    xn = x[n * NX :]
    lifted = sc.stack([2.0 * (xn[0] / 2.0).sin(), xn[1], xn[2], xn[3]])
    cost = stages.sum() + sc.dot(lifted, P.reshape((NX, NX)) @ lifted) + SLACK_WEIGHT * 0.5 * (s[1] * s[1] + s[1])
    phi = x[1 : (n + 1) * NX : NX]
    if form == "opti":
      # Opti's canonical rows, offsets included: a side free of variables becomes the row's bound
      # (x_0 <= x0 + 1e-3 is the row x_0 with upper bound x0 + 1e-3), otherwise `a <= b` is the row
      # a - b <= 0. IPOPT pushes its slacks off a bound by 1e-2 max(1, |bound|) and relaxes each bound
      # by 1e-8 max(1, |bound|), so where the constant sits changes its iterates.
      ineq = (
        sc.opt.bounded(x[:NX], hi=x0 + X0_BAND, name="x0_upper"),
        sc.opt.bounded(x[:NX], lo=x0 - X0_BAND, name="x0_lower"),
        sc.opt.bounded(-PHI_LIMIT - s[1] - phi, hi=0.0, name="phi_lower"),
        sc.opt.bounded(phi - (PHI_LIMIT + s[1]), hi=0.0, name="phi_upper"),
        sc.opt.bounded(u, lo=-TORQUE_LIMIT, name="u_lower"),
        sc.opt.bounded(u, hi=TORQUE_LIMIT, name="u_upper"),
        sc.opt.bounded(s[1:2], lo=0.0, name="slack_lower"),
        sc.opt.bounded(s[1:2], hi=SLACK_LIMIT, name="slack_upper"),
      )
      return sc.opt.ProblemSpec(minimize=cost, eq=(defects,), ineq=ineq)
    free = sc.const(np.full(n * NX, inf))
    return sc.opt.ProblemSpec(
      minimize=cost,
      eq=(defects,),
      ineq=(sc.opt.bounded(phi - s[1], hi=PHI_LIMIT, name="phi_upper"), sc.opt.bounded(phi + s[1], lo=-PHI_LIMIT, name="phi_lower")),
      lb=(sc.concat([x0 - X0_BAND, -free]), sc.const(np.full(n * NU, -TORQUE_LIMIT)), sc.const(np.zeros(NX))),
      ub=(sc.concat([x0 + X0_BAND, free]), sc.const(np.full(n * NU, TORQUE_LIMIT)), sc.const(np.array([0.0, SLACK_LIMIT, 0.0, 0.0]))),
    )

  return ocp


class Controller:
  """A generated MPC solver and the warm-start state around it: the dual guess carries over from one
  solve to the next, as laOPT's SQP object keeps its multipliers; IPOPT, cold in its duals as the
  upstream runs it, ignores it."""

  def __init__(self, model: Model, form: str, method: str, horizon: int = HORIZON, options: dict[str, Any] | None = None) -> None:
    self.model, self.form, self.method, self.horizon = model, form, method, horizon
    self.nlp = problem(model, form, horizon)
    if method == "ipopt":
      backend = sc.opt.IPOPT(options={**IPOPT_OPTIONS, **(options or {})})
    elif method == "sqp":
      backend = sc.opt.SQP(options={**SQP_OPTIONS, **(options or {})})
    else:
      raise ValueError(f"unknown method {method!r}")
    self.solver = sc.opt.solver(self.nlp, backend, name=f"npmpc_{model.kind}_{form}_{method}")
    self.reset()

  def reset(self) -> None:
    n = self.horizon
    self.lam_box = (np.zeros((n + 1) * NX), np.zeros(n * NU), np.zeros(NX))
    self.lam_eq, self.lam_ineq = np.zeros(self.nlp.n_eq), np.zeros(self.nlp.n_ineq)

  def compile(self, theta: np.ndarray, P: np.ndarray) -> float:
    """Generate and compile the solver (the first call does both); seconds taken."""
    t0 = time.perf_counter()
    self.solve(np.zeros(NX), np.zeros((self.horizon + 1, NX)), np.zeros(self.horizon), np.zeros(NX), theta, P)
    self.reset()
    return time.perf_counter() - t0

  def solve(self, x0: np.ndarray, x_guess: np.ndarray, u_guess: np.ndarray, s_guess: np.ndarray, theta: np.ndarray, P: np.ndarray) -> dict[str, Any]:
    """One solve from the given primal guess; `solve_s` is the Python-level wall time, `t_total` the
    generated C's own timer."""
    guess = (np.asarray(x_guess, np.float64).reshape(-1), np.asarray(u_guess, np.float64).reshape(-1), np.asarray(s_guess, np.float64))
    params = (np.asarray(x0, np.float64), np.asarray(theta, np.float64), np.asarray(P, np.float64).reshape(-1))
    t0 = time.perf_counter()
    (x, u, s), self.lam_box, self.lam_eq, self.lam_ineq, _ = self.solver(guess, self.lam_box, self.lam_eq, self.lam_ineq, params)
    wall = time.perf_counter() - t0
    stats = sc.opt.solver_stats(self.solver)
    status = stats.to_solver_status()
    return {
      "x": np.asarray(x).reshape(self.horizon + 1, NX),
      "u": np.asarray(u).reshape(-1),
      "slack": np.asarray(s),
      "converged": bool(status.ok),
      "status": status.name,
      "iter": int(stats.iter),
      "qp_iter": int(stats.qp_iter),
      "objective": float(stats.obj),
      "solve_s": wall,
      "t_total": float(stats.t_total),
      "t_fe": float(stats.t_fe),
    }


def encoder_function(cnp: CNP, capacity: int = CONTEXT_MAX) -> sc.Function:
  """z from up to `capacity` context pairs: the encoder MLP on each scaled pair, one `vmap`, then the
  masked mean `CNPEncoder.forward` takes, as a product with the normalized mask `w`."""
  (xw, xb), (yw, yb) = cnp.x_scale, cnp.y_scale

  @sc.function(sc.G(sc.L("x", 5), sc.L("y", 2)), name="cnp_encoder_point")
  def point(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, y = inputs
    return nn.mlp(sc.concat([x * sc.const(xw) + sc.const(xb), y * sc.const(yw) + sc.const(yb)]), cnp.encoder, gelu)

  @sc.function(sc.G(sc.L("x", capacity * 5), sc.L("y", capacity * 2), sc.L("w", capacity)), name=f"cnp_encode_{capacity}")
  def encode(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
    x, y, w = inputs
    r = sc.vmap(point, capacity, [(x, 0, 5), (y, 0, 2)]).reshape((capacity, 4))
    return r.T @ w

  return encode


def encode(fn: sc.Function, x_ctx: np.ndarray, y_ctx: np.ndarray, capacity: int = CONTEXT_MAX) -> np.ndarray:
  """z for a context of `len(x_ctx) <= capacity` pairs."""
  n = len(x_ctx)
  x, y, w = np.zeros((capacity, 5)), np.zeros((capacity, 2)), np.zeros(capacity)
  x[:n], y[:n], w[:n] = x_ctx, y_ctx, 1.0 / max(n, 1)
  return np.asarray(fn((x.reshape(-1), y.reshape(-1), w)), np.float64)
