"""Neural process MPC on the Furuta pendulum: the conditional-neural-process dynamics model, its horizon transcription, and the CasADi mirror."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import alloy as al
from alloy.utils import load_torch_state_dict

# State (theta, phi, theta_dot, phi_dot) with theta = 0 upright, input (torque,), decoder output (d theta_dot, d phi_dot).
NX, NU, NY = 4, 1, 2
# Decoder features are (sin theta, cos theta, theta_dot, phi_dot, torque), then the latent code is appended.
NFEAT, NLATENT = 5, 4
HORIZON = 12
DT = 0.02
DEFAULT_HIDDEN = (32, 32)
DATA = Path(__file__).parent / "data"
DEFAULT_MODEL_PATH = DATA / "cnp_model.pth"
DEFAULT_EPISODE_PATH = DATA / "reference_episode.npz"
DEFAULT_CONFIG_PATH = DATA / "reference_config.json"

# Operating limits, from the reference implementation's `furuta_mpc.json`. The arm angle is the only
# bounded state and the only one the slack softens; the torque bound is hard.
PHI_LIMIT = 2.0
SLACK_LIMIT = 5.0
TORQUE_LIMIT = 0.05
# Their controller pins the first horizon node to the measured state with a band of inequalities
# rather than an equality. The band is reproduced because the reference episode we gate against was
# produced with it, and it is what their reported solve times correspond to.
X0_BAND = 1e-3
# Their plant integrates at a nominal 4 kHz, so one 20 ms control interval is 80 midpoint substeps.
PLANT_SUBSTEPS = 80

# The deployed latent code for the paper's system 3, the pendulum its closed-loop figure uses.
#
# The reference implementation computes this at startup by encoding a short open-loop context
# trajectory and never saves it, and the trajectory itself cannot be reproduced: its plant runs in
# a separate process at 4 kHz driven by wall-clock time steps. The code is only four numbers,
# though, and the released episode `datasets/furuta_mpc/experiment_np_m3.npz` carries a full
# neural-process open-loop rollout, so the four unknowns can be fitted to that rollout. Multistart
# Levenberg-Marquardt over the ~1200 one-step residuals converges here from many starts, with a
# largest residual of 1.7e-5 against velocity changes of magnitude up to 11 -- agreement at the
# precision of their float32 checkpoint, which also confirms `decoder_mu_np` below is faithful.
LATENT_SYS3 = np.array([-2.34141442, 0.16122490, -10.00758909, -1.65464341], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Decoder:
  """The mean decoder's architecture, and the layout of the parameter tail that carries it.

  `hidden` is the tuple of hidden widths. Following the reference implementation's `create_mlp`,
  every layer but the last is bias-free and followed by a sigmoid; the last carries a bias and no
  activation. The shipped checkpoint is `(32, 32)` -- the size the paper shrank to in order to
  meet its 20 ms budget (paper section 5.3).
  """

  hidden: tuple[int, ...] = DEFAULT_HIDDEN

  @property
  def widths(self) -> tuple[int, ...]:
    return (NFEAT + NLATENT, *self.hidden, NY)

  @property
  def weight_shapes(self) -> tuple[tuple[int, int], ...]:
    w = self.widths
    return tuple((w[i + 1], w[i]) for i in range(len(w) - 1))

  @property
  def blocks(self) -> tuple[tuple[str, int], ...]:
    """The parameter tail, in order: input scaler, output scaler, weight matrices, final bias, latent code."""
    return (
      ("x_scale_w", NFEAT),
      ("x_scale_b", NFEAT),
      ("y_inv_w", NY),
      ("y_inv_b", NY),
      *((f"w{i}", int(np.prod(shape))) for i, shape in enumerate(self.weight_shapes)),
      ("bias", NY),
      ("latent", NLATENT),
    )

  @property
  def offsets(self) -> tuple[int, ...]:
    return tuple(int(x) for x in np.cumsum([0, *(size for _, size in self.blocks)]))

  @property
  def n_pw(self) -> int:
    return self.offsets[-1]

  def slice(self, name: str) -> slice:
    names = [n for n, _ in self.blocks]
    i = names.index(name)
    return slice(self.offsets[i], self.offsets[i + 1])


def n_dec(horizon: int) -> int:
  """Decision-vector length: the states blocked first, then the controls, then the arm-angle slack."""
  return NX * (horizon + 1) + NU * horizon + 1


def n_param(decoder: Decoder = Decoder()) -> int:
  """Parameter length for the transcribed NLP: the state the first node is pinned to, then the decoder tail."""
  return NX + decoder.n_pw


def constraint_counts(horizon: int) -> tuple[int, int]:
  """Row counts of the transcription: the dynamics equalities, then the inequality rows.

  One definition, because the episode runner, the gates and the CasADi mirror all need these and a
  disagreement between them would show up as a shape error a long way from its cause.
  """
  return NX * horizon, NX + 2 * (horizon + 1)


def load_decoder_weights(decoder: Decoder = Decoder(), path: str | Path = DEFAULT_MODEL_PATH) -> np.ndarray:
  """Read the trained mean decoder and its scalers out of the vendored checkpoint, in `blocks` order minus the latent code."""
  if decoder.hidden != DEFAULT_HIDDEN:
    raise ValueError(f"the vendored checkpoint holds a {DEFAULT_HIDDEN} decoder, not {decoder.hidden}; use random_decoder_weights for other widths")
  state = load_torch_state_dict(path)
  mu, x_scale, y_inv = state["decoder"], state["x_direct_scaler"], state["y_inverse_scaler"]
  # The reference `Sequential` interleaves activations, so linear layers sit at even indices.
  weights = [mu[f"mu_decoder.{2 * i}.weight"] for i in range(len(decoder.weight_shapes))]
  for weight, shape in zip(weights, decoder.weight_shapes, strict=True):
    if tuple(weight.shape) != shape:
      raise ValueError(f"checkpoint layer shape {tuple(weight.shape)} does not match the expected {shape}")
  packed = np.concatenate(
    [
      x_scale["weight"].reshape(-1),
      x_scale["bias"].reshape(-1),
      y_inv["weight"].reshape(-1),
      y_inv["bias"].reshape(-1),
      *(weight.reshape(-1) for weight in weights),
      mu[f"mu_decoder.{2 * (len(weights) - 1)}.bias"].reshape(-1),
    ]
  ).astype(np.float64)
  expected = decoder.n_pw - NLATENT
  if packed.size != expected:
    raise ValueError(f"packed checkpoint has {packed.size} entries, expected {expected}")
  return packed


def random_decoder_weights(decoder: Decoder, seed: int = 0) -> np.ndarray:
  """Untrained weights, for the decoder-width sweep only.

  Kernel timing and generated code size depend on the graph's shape rather than on the numbers in
  it, so an untrained decoder is a legitimate sweep cell. It is not a legitimate closed-loop or
  accuracy column, and any measurement taken with these has to say so.
  """
  rng = np.random.default_rng(seed)
  parts = [np.ones(NFEAT), np.zeros(NFEAT), np.ones(NY), np.zeros(NY)]
  # Kaiming-style scaling, so the sigmoids sit in their responsive range rather than saturating.
  parts += [rng.normal(scale=np.sqrt(2.0 / shape[1]), size=shape).reshape(-1) for shape in decoder.weight_shapes]
  parts.append(np.zeros(NY))
  return np.concatenate(parts).astype(np.float64)


def pack_params(decoder: Decoder, weights: np.ndarray, latent: np.ndarray = LATENT_SYS3) -> np.ndarray:
  """Append the latent code to the decoder weights, giving the full parameter tail."""
  packed = np.concatenate([np.asarray(weights, dtype=np.float64).reshape(-1), np.asarray(latent, dtype=np.float64).reshape(-1)])
  if packed.size != decoder.n_pw:
    raise ValueError(f"parameter tail has {packed.size} entries, expected {decoder.n_pw}")
  return packed


def decoder_mu_np(decoder: Decoder, pw: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
  """The mean decoder in NumPy: the velocity change it predicts for one state and control."""
  feat = np.array([np.sin(x[0]), np.cos(x[0]), x[2], x[3], np.reshape(u, -1)[0]], dtype=np.float64)
  h = np.concatenate([feat * pw[decoder.slice("x_scale_w")] + pw[decoder.slice("x_scale_b")], pw[decoder.slice("latent")]])
  for i, shape in enumerate(decoder.weight_shapes[:-1]):
    h = 1.0 / (1.0 + np.exp(-(pw[decoder.slice(f"w{i}")].reshape(shape) @ h)))
  last = len(decoder.weight_shapes) - 1
  y = pw[decoder.slice(f"w{last}")].reshape(decoder.weight_shapes[last]) @ h + pw[decoder.slice("bias")]
  return y * pw[decoder.slice("y_inv_w")] + pw[decoder.slice("y_inv_b")]


def step_np(decoder: Decoder, pw: np.ndarray, x: np.ndarray, u: np.ndarray, dt: float = DT) -> np.ndarray:
  """One step of the learned model: the decoder gives the velocity rows, explicit trapezoidal integration the position rows."""
  y = decoder_mu_np(decoder, pw, x, u)
  return np.asarray(x, dtype=np.float64) + np.concatenate([dt * (np.asarray(x, dtype=np.float64)[2:4] + y / 2.0), y])


@dataclass(frozen=True, slots=True)
class FurutaParams:
  """The analytic Furuta pendulum the learned model approximates -- the true plant of the closed loop.

  The defaults are the paper's system 3, the pendulum its closed-loop figure and the vendored
  reference episode both belong to, and the one `LATENT_SYS3` was recovered for.
  """

  l_p: float = 0.1377502795417797
  m_p: float = 0.00880773464519363
  l_r: float = 0.08949330135206067
  m_r: float = 0.027303962257072
  gravity: float = 9.81


def furuta_ode_np(x: np.ndarray, u: np.ndarray, params: FurutaParams = FurutaParams()) -> np.ndarray:
  """The analytic rigid-body dynamics: `B(theta) [theta_ddot, phi_ddot] = a(x, torque)`."""
  theta, _, theta_dot, phi_dot = (float(value) for value in np.asarray(x, dtype=np.float64))
  torque = float(np.reshape(u, -1)[0])
  j_r = params.m_r * params.l_r**2 / 3.0 + params.m_p * params.l_r**2
  j_p = params.m_p * params.l_p**2 / 3.0
  coupling = params.m_p * params.l_r * params.l_p / 2.0
  B = np.array([[j_p, -coupling * np.cos(theta)], [-coupling * np.cos(theta), j_r + j_p * np.sin(theta) ** 2]])
  a = np.array(
    [
      j_p * np.sin(2.0 * theta) / 2.0 * phi_dot**2 + params.m_p * params.l_p * params.gravity / 2.0 * np.sin(theta),
      -j_p * np.sin(2.0 * theta) * phi_dot * theta_dot - coupling * np.sin(theta) * theta_dot**2 + torque,
    ]
  )
  return np.concatenate([[theta_dot, phi_dot], np.linalg.solve(B, a)])


def furuta_midpoint_np(x: np.ndarray, u: np.ndarray, dt: float, params: FurutaParams = FurutaParams()) -> np.ndarray:
  """One explicit midpoint step of the analytic plant -- the reference implementation's integrator."""
  x = np.asarray(x, dtype=np.float64)
  return x + dt * furuta_ode_np(x + 0.5 * dt * furuta_ode_np(x, u, params), u, params)


def plant_step(x: np.ndarray, u: np.ndarray, dt: float = DT, params: FurutaParams = FurutaParams(), substeps: int = PLANT_SUBSTEPS) -> np.ndarray:
  """Advance the plant across one control interval, holding the torque, in `substeps` midpoint steps.

  The reference implementation's plant runs in a separate process at a nominal 4 kHz driven by
  wall-clock time steps, so no two of its runs agree. Ours takes the same number of substeps per
  20 ms interval at a fixed size, which makes the episode reproducible; see the problem README.
  """
  if substeps < 1:
    raise ValueError(f"substeps must be positive, got {substeps}")
  h = dt / substeps
  for _ in range(substeps):
    x = furuta_midpoint_np(x, u, h, params)
  return x


def load_reference_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
  """The reference implementation's controller configuration, vendored verbatim.

  Their `model/furuta_mpc.json`, unmodified. It is the authoritative source for the cost weights,
  the state and input bounds, the sample time, the horizon length and system 3's geometry, so
  `checks.py` reads those numbers out of this file rather than trusting a transcription of them.
  """
  with open(path) as handle:
    return json.load(handle)


def load_reference_episode(path: str | Path = DEFAULT_EPISODE_PATH) -> dict[str, np.ndarray]:
  """The reference implementation's released closed-loop episode for system 3, as vendored.

  Trimmed from their `datasets/furuta_mpc/experiment_np_m3.npz`: the 4 MB `x_mc` Monte Carlo
  rollouts and the pickled metrics are dropped, and every step-indexed array is truncated to the
  `k + 1 = 100` steps their run actually completed, so there is no trailing all-zero row to mistake
  for data. Keys: `x0` measured states, `u0` the control applied one step earlier, `x`/`u` the
  solved horizon, `x_np` the neural-process open-loop rollout behind it, `x_oracle` the same
  rollout under the analytic plant, and `mpc_t` their measured per-step loop times, plus the scalars
  `steps`, `dt` and `horizon_steps` the gates read back.
  """
  with np.load(path) as stored:
    return {key: np.asarray(stored[key]) for key in stored.files}


@functools.cache
def stage_function(decoder: Decoder = Decoder(), dt: float = DT) -> al.Function:
  """The dynamics residual of one horizon stage: `x + f(x, u) - xnext`, with the weights read out of the parameter tail."""
  shapes = decoder.weight_shapes
  name = "npmpc_stage_h" + "x".join(str(h) for h in decoder.hidden)

  @al.function(name, {"x": NX, "xnext": NX, "u": NU, "pw": decoder.n_pw})
  def stage(x, xnext, u, pw):
    feat = al.stack([x[0].sin(), x[0].cos(), x[2], x[3], u[0]]) * pw[decoder.slice("x_scale_w")] + pw[decoder.slice("x_scale_b")]
    h = al.concat([feat, pw[decoder.slice("latent")]])
    for i, shape in enumerate(shapes[:-1]):
      h = 1.0 / (1.0 + (-(pw[decoder.slice(f"w{i}")].reshape(shape) @ h)).exp())
    last = len(shapes) - 1
    y = (pw[decoder.slice(f"w{last}")].reshape(shapes[last]) @ h + pw[decoder.slice("bias")]) * pw[decoder.slice("y_inv_w")] + pw[
      decoder.slice("y_inv_b")
    ]
    return {"eq": x + al.concat([dt * (x[2:4] + y / 2.0), y]) - xnext}

  return stage


@functools.cache
def npmpc_eq_function(horizon: int, decoder: Decoder = Decoder(), dt: float = DT) -> al.Function:
  """The `horizon` dynamics equalities, as one `al.scan` over the stage function.

  States are blocked ahead of controls in `z`, so both windows into it stride cleanly and there is
  no dead trailing control the way an interleaved layout would leave.
  """
  z = al.sym("z", n_dec(horizon))
  p = al.sym("p", decoder.n_pw, diff=False)
  eq = al.scan(
    stage_function(decoder, dt),
    length=horizon,
    inputs={"x": (z, 0, NX), "xnext": (z, NX, NX), "u": (z, NX * (horizon + 1), NU), "pw": (p, 0, 0)},
  )
  return al.Function(f"npmpc_eq_N{horizon}", [z, p], [eq], ["z", "p"], ["eq"])


def npmpc_eq_function_unrolled(horizon: int, decoder: Decoder = Decoder(), dt: float = DT) -> al.Function:
  """Same equalities built stage by stage, so the scanned form can be differentially tested against it."""
  z = al.sym("z", n_dec(horizon))
  p = al.sym("p", decoder.n_pw, diff=False)
  stage = stage_function(decoder, dt)
  parts = [
    stage.call([z[NX * i : NX * (i + 1)], z[NX * (i + 1) : NX * (i + 2)], z[NX * (horizon + 1) + NU * i : NX * (horizon + 1) + NU * (i + 1)], p])[0]
    for i in range(horizon)
  ]
  return al.Function(f"npmpc_eq_unrolled_N{horizon}", [z, p], [al.concat(parts)], ["z", "p"], ["eq"])


def sample_inputs(horizon: int, decoder: Decoder = Decoder(), weights: np.ndarray | None = None, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
  """A synthetic decision vector and parameter tail for kernel cells, spread over a swing-up's range of angles and speeds."""
  rng = np.random.default_rng(seed)
  states = np.stack(
    [
      rng.uniform(-np.pi, np.pi, horizon + 1),
      rng.uniform(-2.0, 2.0, horizon + 1),
      rng.normal(scale=4.0, size=horizon + 1),
      rng.normal(scale=4.0, size=horizon + 1),
    ],
    axis=1,
  )
  z = np.concatenate([states.reshape(-1), rng.uniform(-0.05, 0.05, horizon * NU), np.zeros(1)])
  if weights is None:
    weights = load_decoder_weights(decoder) if decoder.hidden == DEFAULT_HIDDEN else random_decoder_weights(decoder)
  return z, pack_params(decoder, weights)


@functools.cache
def _stage_jac_function(decoder: Decoder, dt: float) -> al.Function:
  stage = stage_function(decoder, dt)
  return stage.factory(
    f"npmpc_stage_jac_h{'x'.join(str(h) for h in decoder.hidden)}",
    ["x", "xnext", "u", "pw"],
    [al.jac("eq", "x"), al.jac("eq", "u"), al.jac("eq", "xnext")],
  )


def npmpc_eq_jac_dense_reference(horizon: int, z: np.ndarray, p: np.ndarray, decoder: Decoder = Decoder(), dt: float = DT) -> np.ndarray:
  """Dense Jacobian of the equalities, assembled from per-stage dense blocks.

  Each stage's residual touches only its own state, control, and successor state, so the matrix is
  block-banded and the reference costs one small dense Jacobian per stage rather than one enormous
  one over the whole horizon. Differentiating the *stage* function and scattering the blocks is also
  the independence the check needs: it never builds the scanned graph the sparse kernel comes from.
  """
  jac = _stage_jac_function(decoder, dt)
  offset = NX * (horizon + 1)
  dense = np.zeros((NX * horizon, n_dec(horizon)), dtype=np.float64)
  for i in range(horizon):
    x, xnext = z[NX * i : NX * (i + 1)], z[NX * (i + 1) : NX * (i + 2)]
    u = z[offset + NU * i : offset + NU * (i + 1)]
    d_x, d_u, d_xnext = (np.asarray(block, dtype=np.float64).reshape(NX, -1) for block in jac(x, xnext, u, p))
    rows = slice(NX * i, NX * (i + 1))
    dense[rows, NX * i : NX * (i + 1)] = d_x
    dense[rows, NX * (i + 1) : NX * (i + 2)] = d_xnext
    dense[rows, offset + NU * i : offset + NU * (i + 1)] = d_u
  return dense.reshape(-1)


@dataclass(frozen=True, slots=True)
class CostWeights:
  """The reference implementation's cost weights (paper Table 2).

  `x` is the stage weight on the lifted state, `x_diff` the inter-stage weight on state
  differences, `x_end` and `u_end` the weights the terminal Riccati solve uses, `u` the input
  weight, and `slack` the penalty on the arm-angle softening.
  """

  x: tuple[float, float, float, float] = (2.5, 1.5, 0.005, 0.01)
  x_diff: tuple[float, float, float, float] = (0.0, 0.0, 0.05, 0.125)
  x_end: tuple[float, float, float, float] = (1.0, 10.0, 0.1, 0.1)
  u: float = 1.0
  u_end: float = 1.0
  slack: float = 1000.0


def linearize(decoder: Decoder, pw: np.ndarray, dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
  """`A`, `B` of the learned one-step map at the upright equilibrium.

  The stage residual is `x + f(x, u) - xnext`, so its derivatives with respect to `x` and `u` at
  the equilibrium *are* `A` and `B`. Taking them from `al.jac` keeps the reference implementation's
  torch dependency out and exercises Alloy's differentiation in the problem's own setup.
  """
  jac = _stage_jac_function(decoder, dt)
  d_x, d_u, _ = (np.asarray(block, dtype=np.float64).reshape(NX, -1) for block in jac(np.zeros(NX), np.zeros(NX), np.zeros(NU), pw))
  return d_x, d_u


def terminal_P(decoder: Decoder, pw: np.ndarray, weights: CostWeights = CostWeights(), dt: float = DT) -> np.ndarray:
  """Terminal weight from the discrete algebraic Riccati equation for the linearized learned model."""
  from scipy.linalg import solve_discrete_are

  A, B = linearize(decoder, pw, dt)
  return np.asarray(solve_discrete_are(A, B, np.diag(weights.x_end), np.diag([weights.u_end])), dtype=np.float64)


def riccati_residual(P: np.ndarray, A: np.ndarray, B: np.ndarray, weights: CostWeights = CostWeights()) -> float:
  """How far `P` is from solving the Riccati equation it claims to come from -- the gate on a pinned terminal weight."""
  Q, R = np.diag(weights.x_end), np.diag([weights.u_end])
  rhs = A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) + Q
  return float(np.abs(P - rhs).max())


@functools.cache
def stage_cost_function(weights: CostWeights = CostWeights()) -> al.Function:
  """One horizon stage of the objective: state, input, and inter-stage terms for stage `i`.

  Scanned over the horizon by `npmpc_cost_expr`, so the objective's generated source stays constant
  in the horizon exactly as the dynamics' does. Building it as a Python loop instead unrolls it,
  which grows the source linearly and, past roughly 75 stages, exceeds the Program IR passes'
  recursion depth during lowering (the limitation recorded in `BENCHMARKS.md`).
  """

  @al.function("npmpc_stage_cost", {"x": NX, "xnext": NX, "u": NU})
  def stage_cost(x, xnext, u):
    dx = xnext - x
    # The pendulum angle uses the 2*pi-periodic half-angle lift, so every upright pose costs the
    # same: (2 sin(theta/2))^2 = 2 (1 - cos theta).
    total = weights.x[0] * 2.0 * (1.0 - x[0].cos()) + weights.u * u[0] * u[0]
    for k in (1, 2, 3):
      total = total + weights.x[k] * x[k] * x[k]
    for k, weight in enumerate(weights.x_diff):
      if weight:
        total = total + weight * dx[k] * dx[k]
    return {"cost": total.scalar()}

  return stage_cost


def npmpc_cost_expr(z: al.Expr, horizon: int, P: np.ndarray, weights: CostWeights = CostWeights()) -> al.Expr:
  """Total objective: stage, inter-stage, LQR terminal, and slack penalty (paper eq. 15).

  The per-stage part is scanned; only the terminal and slack terms, which exist once, sit outside
  the loop.
  """
  offset = NX * (horizon + 1)
  stages = al.scan(
    stage_cost_function(weights),
    length=horizon,
    inputs={"x": (z, 0, NX), "xnext": (z, NX, NX), "u": (z, offset, NU)},
  )
  xN = z[NX * horizon : NX * (horizon + 1)]
  e_end = al.stack([2.0 * (xN[0] / 2.0).sin(), xN[1], xN[2], xN[3]])
  terminal = al.dot(e_end, al.const(np.asarray(P, dtype=np.float64)) @ e_end)
  slack = z[offset + NU * horizon]
  return (stages.sum() + terminal + weights.slack * 0.5 * (slack * slack + slack)).scalar()


def npmpc_ineq_bounds(horizon: int) -> tuple[np.ndarray, np.ndarray]:
  """Bounds on the inequality rows: the initial-state band, then the two softened arm-angle families.

  Solver bounds have to be constants, so the slack cannot appear in them; `npmpc_constraint_exprs`
  moves it onto the rows instead, turning `-PHI_LIMIT - s <= phi_i <= PHI_LIMIT + s` into one
  one-sided row per side.
  """
  band, free = np.full(NX, X0_BAND), np.full(horizon + 1, np.inf)
  lower = np.concatenate([-band, np.full(horizon + 1, -PHI_LIMIT), -free])
  upper = np.concatenate([band, free, np.full(horizon + 1, PHI_LIMIT)])
  return lower, upper


def npmpc_constraint_exprs(z: al.Expr, xstart: al.Expr, horizon: int) -> tuple[al.Expr, np.ndarray, np.ndarray]:
  """The inequality rows and their bounds: the initial-state band, then the softened arm-angle limits.

  Both arm-angle families are one strided gather plus the scalar slack, so the rendered source does
  not grow with the horizon any more than the scanned dynamics do.
  """
  slack = z[NX * (horizon + 1) + NU * horizon]
  phi = z[1 : NX * (horizon + 1) : NX]
  return al.concat([z[:NX] - xstart, phi + slack, phi - slack]), *npmpc_ineq_bounds(horizon)


def npmpc_bounds(horizon: int) -> tuple[np.ndarray, np.ndarray]:
  """Decision-vector box bounds: the states are free, the torques hard-bounded, the slack non-negative."""
  offset = NX * (horizon + 1)
  lower, upper = np.full(n_dec(horizon), -np.inf), np.full(n_dec(horizon), np.inf)
  lower[offset : offset + NU * horizon], upper[offset : offset + NU * horizon] = -TORQUE_LIMIT, TORQUE_LIMIT
  lower[-1], upper[-1] = 0.0, SLACK_LIMIT
  return lower, upper


def npmpc_nlp(
  P: np.ndarray,
  horizon: int = HORIZON,
  decoder: Decoder = Decoder(),
  *,
  weights: CostWeights = CostWeights(),
  dt: float = DT,
  solver: str = "ipopt",
  options: dict[str, str | int | float] | None = None,
) -> al.SolverFunction:
  """The neural-process MPC as one `al.nlp`: scanned dynamics, scanned cost, one arm-angle slack.

  `P` is the terminal weight from `terminal_P`. `p` carries the state the first horizon node is
  pinned to and then the decoder tail, in the order `n_param` describes -- the state first, as
  `race_cars` and `chain` also do it. The weights and the latent code travelling as parameters is
  what lets one compiled artifact serve every pendulum in the paper's family.
  """
  z = al.sym("z", n_dec(horizon))
  p = al.sym("p", n_param(decoder), diff=False)
  eq = al.scan(
    stage_function(decoder, dt),
    length=horizon,
    inputs={"x": (z, 0, NX), "xnext": (z, NX, NX), "u": (z, NX * (horizon + 1), NU), "pw": (p, NX, 0)},
  )
  rows, l_ineq, u_ineq = npmpc_constraint_exprs(z, p[:NX], horizon)
  lower, upper = npmpc_bounds(horizon)
  return al.nlp(
    x=z,
    p=p,
    f=npmpc_cost_expr(z, horizon, P, weights),
    h_eq=eq,
    g_ineq=rows,
    l_ineq=l_ineq,
    u_ineq=u_ineq,
    x_lb=lower,
    x_ub=upper,
    solver=solver,
    name=f"npmpc_N{horizon}_{solver}",
    options=options,
  )


def npmpc_lag_function(
  horizon: int, decoder: Decoder = Decoder(), P: np.ndarray | None = None, weights: CostWeights = CostWeights(), dt: float = DT
) -> al.Function:
  """Objective and dynamics equalities together, the pair an exact Lagrangian Hessian needs."""
  z = al.sym("z", n_dec(horizon))
  p = al.sym("p", decoder.n_pw, diff=False)
  eq = al.scan(
    stage_function(decoder, dt),
    length=horizon,
    inputs={"x": (z, 0, NX), "xnext": (z, NX, NX), "u": (z, NX * (horizon + 1), NU), "pw": (p, 0, 0)},
  )
  terminal = np.diag(weights.x_end) if P is None else P
  cost = npmpc_cost_expr(z, horizon, terminal, weights)
  return al.Function(f"npmpc_lag_N{horizon}", [z, p], [cost, eq], ["z", "p"], ["cost", "eq"])


def ca_npmpc_pieces(
  horizon: int,
  decoder: Decoder = Decoder(),
  sym_t=None,
  *,
  P: np.ndarray | None = None,
  weights: CostWeights = CostWeights(),
  dt: float = DT,
  cost: bool = True,
) -> dict:
  """The CasADi mirror's symbolic pieces, in Alloy's own row and column order.

  One builder behind all four CasADi consumers -- the sweep's equality-Jacobian and Lagrangian-
  Hessian kernels and the closed loop's IPOPT and SQP columns -- so the mirror cannot drift from
  itself. `z`, `xstart` and `pw` mean exactly what they mean on the Alloy side, every constraint row
  and decision column sits in the same place, and the bounds come from the same two functions. That
  identity is what makes the two-oracle comparison controlled: the only difference left is which
  tool differentiates and evaluates.
  """
  import casadi as ca

  sym_t = ca.SX if sym_t is None else sym_t
  z = sym_t.sym("z", n_dec(horizon))
  xstart = sym_t.sym("xstart", NX)
  pw = sym_t.sym("pw", decoder.n_pw)
  xw, xb = pw[decoder.slice("x_scale_w")], pw[decoder.slice("x_scale_b")]
  yw, yb = pw[decoder.slice("y_inv_w")], pw[decoder.slice("y_inv_b")]
  latent, bias = pw[decoder.slice("latent")], pw[decoder.slice("bias")]
  # CasADi reshapes in column-major order, so build the transpose and flip it.
  mats = [ca.reshape(pw[decoder.slice(f"w{i}")], shape[1], shape[0]).T for i, shape in enumerate(decoder.weight_shapes)]
  offset = NX * (horizon + 1)
  rows, cost_terms = [], []
  for i in range(horizon):
    x, xnext = z[NX * i : NX * (i + 1)], z[NX * (i + 1) : NX * (i + 2)]
    u = z[offset + NU * i : offset + NU * (i + 1)]
    h = ca.vertcat(ca.vertcat(ca.sin(x[0]), ca.cos(x[0]), x[2], x[3], u[0]) * xw + xb, latent)
    for mat in mats[:-1]:
      h = 1.0 / (1.0 + ca.exp(-(mat @ h)))
    y = (mats[-1] @ h + bias) * yw + yb
    rows.append(x + ca.vertcat(dt * (x[2:4] + y / 2.0), y) - xnext)
    if cost:
      dx = xnext - x
      cost_terms.append(
        weights.x[0] * 2.0 * (1.0 - ca.cos(x[0]))
        + sum(weights.x[k] * x[k] ** 2 for k in (1, 2, 3))
        + weights.u * u[0] ** 2
        + sum(weight * dx[k] ** 2 for k, weight in enumerate(weights.x_diff) if weight)
      )
  slack = z[offset + NU * horizon]
  objective = None
  if cost:
    xN = z[NX * horizon : offset]
    e_end = ca.vertcat(2.0 * ca.sin(xN[0] / 2.0), xN[1], xN[2], xN[3])
    terminal = np.diag(weights.x_end) if P is None else np.asarray(P, dtype=np.float64)
    objective = sum(cost_terms) + e_end.T @ ca.DM(terminal) @ e_end + weights.slack * 0.5 * (slack**2 + slack)
  phi = ca.vertcat(*[z[NX * i + 1] for i in range(horizon + 1)])
  l_ineq, u_ineq = npmpc_ineq_bounds(horizon)
  x_lb, x_ub = npmpc_bounds(horizon)
  n_eq, n_ineq = constraint_counts(horizon)
  return {
    "z": z,
    "xstart": xstart,
    "pw": pw,
    "f": objective,
    "h_eq": ca.vertcat(*rows),
    "g_ineq": ca.vertcat(z[:NX] - xstart, phi + slack, phi - slack),
    "n_eq": n_eq,
    "n_ineq": n_ineq,
    "l_ineq": l_ineq,
    "u_ineq": u_ineq,
    "x_lb": x_lb,
    "x_ub": x_ub,
  }


def ca_npmpc_eq_jac(horizon: int, decoder: Decoder = Decoder(), name: str = "npmpc_eq_jac", sym_t=None, dt: float = DT):
  """The CasADi mirror of the equality-Jacobian kernel: the same stage residuals in the same order."""
  import casadi as ca

  pieces = ca_npmpc_pieces(horizon, decoder, sym_t, dt=dt, cost=False)
  return ca.Function(name, [pieces["z"], pieces["pw"]], [ca.jacobian(pieces["h_eq"], pieces["z"])])


def ca_npmpc_lag_hess(
  horizon: int,
  decoder: Decoder = Decoder(),
  name: str = "npmpc_lag_hess",
  sym_t=None,
  P: np.ndarray | None = None,
  weights: CostWeights = CostWeights(),
  dt: float = DT,
):
  """The CasADi mirror of the exact Lagrangian Hessian: same cost, same rows, same parameter tail."""
  import casadi as ca

  sym_t = ca.SX if sym_t is None else sym_t
  pieces = ca_npmpc_pieces(horizon, decoder, sym_t, P=P, weights=weights, dt=dt)
  lam_f, lam_g = sym_t.sym("lam_f"), sym_t.sym("lam_g", pieces["n_eq"])
  lag = lam_f * pieces["f"] + ca.dot(lam_g, pieces["h_eq"])
  return ca.Function(name, [pieces["z"], lam_f, lam_g, pieces["pw"]], [ca.hessian(lag, pieces["z"])[0]])
