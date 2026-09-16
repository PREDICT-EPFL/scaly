"""Neural process MPC on the Furuta pendulum: the conditional-neural-process dynamics model, its horizon transcription, and the CasADi mirror."""

from __future__ import annotations

import builtins
import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, assert_type

import numpy as np

import scaly as sc
from scaly.utils import load_torch_state_dict

type StageFunction = sc.Function[
  tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr],
  tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
  sc.Expr,
  np.ndarray,
]
type StageJacFunction = sc.Function[
  tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr],
  tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
  tuple[sc.Expr, sc.Expr, sc.Expr],
  tuple[np.ndarray, np.ndarray, np.ndarray],
]
type StageCostFunction = sc.Function[
  tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr],
  tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
  sc.Expr,
  np.ndarray,
]
type NpmpcFunction = sc.Function[
  tuple[sc.Expr, sc.Expr],
  tuple[np.ndarray, np.ndarray],
  sc.Expr,
  np.ndarray,
]
type NpmpcLagFunction = sc.Function[
  tuple[sc.Expr, sc.Expr],
  tuple[np.ndarray, np.ndarray],
  tuple[sc.Expr, sc.Expr],
  tuple[np.ndarray, np.ndarray],
]
type NpmpcSolver = sc.Function[
  tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr],
  tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
  tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr],
  tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]

# State (theta, phi, theta_dot, phi_dot) with theta = 0 upright, input (torque,), decoder output (d theta_dot, d phi_dot).
NX, NU, NY = 4, 1, 2
# Decoder features are (sin theta, cos theta, theta_dot, phi_dot, torque), then the latent code is appended.
NFEAT, NLATENT = 5, 4
N_COST_WEIGHTS = 10  # x (4), x_diff (4), u, slack
N_RUNTIME_CONFIG = 1 + N_COST_WEIGHTS + NX * NX  # dt, cost weights, terminal P
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

  def slice(self, name: str) -> builtins.slice:
    names = [n for n, _ in self.blocks]
    i = names.index(name)
    return slice(self.offsets[i], self.offsets[i + 1])


def n_dec(horizon: int) -> int:
  """Decision-vector length: the states blocked first, then the controls, then the arm-angle slack."""
  return NX * (horizon + 1) + NU * horizon + 1


def n_param(decoder: Decoder = Decoder()) -> int:
  """Parameter length: initial state, decoder tail, time step, cost weights, then terminal weight."""
  return NX + decoder.n_pw + N_RUNTIME_CONFIG


def _param_slices(decoder: Decoder) -> tuple[slice, slice, slice, slice]:
  pw_stop = NX + decoder.n_pw
  return (
    slice(NX, pw_stop),
    slice(pw_stop, pw_stop + 1),
    slice(pw_stop + 1, pw_stop + 1 + N_COST_WEIGHTS),
    slice(pw_stop + 1 + N_COST_WEIGHTS, pw_stop + N_RUNTIME_CONFIG),
  )


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
  feat = np.array([np.sin(x[0]), np.cos(x[0]), x[2], x[3], np.reshape(u, -1)[0]])
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


# The decoder architecture changes the packed parameter-vector shape, so this remains a builder
# until FunctionTemplate can own its shape-specialized instances. For this benchmark, the vector
# length identifies the architecture even though that is not true for arbitrary layer layouts.
@functools.cache
def stage_function(decoder: Decoder = Decoder()) -> StageFunction:
  """The dynamics residual of one horizon stage: `x + f(x, u) - xnext`, with the weights read out of the parameter tail."""
  shapes = decoder.weight_shapes
  name = "npmpc_stage_h" + "x".join(str(h) for h in decoder.hidden)

  @sc.function(
    sc.G(sc.L("x", NX), sc.L("xnext", NX), sc.L("u", NU), sc.L("pw", decoder.n_pw), sc.L("dt", ())),
    sc.L("eq", ...),
    name=name,
  )
  def stage(inputs: tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
    x, xnext, u, pw, dt = inputs
    feat = sc.stack([x[0].sin(), x[0].cos(), x[2], x[3], u[0]]) * pw[decoder.slice("x_scale_w")] + pw[decoder.slice("x_scale_b")]
    h = sc.concat([feat, pw[decoder.slice("latent")]])
    for i, shape in enumerate(shapes[:-1]):
      h = 1.0 / (1.0 + (-(pw[decoder.slice(f"w{i}")].reshape(shape) @ h)).exp())
    last = len(shapes) - 1
    y = (pw[decoder.slice(f"w{last}")].reshape(shapes[last]) @ h + pw[decoder.slice("bias")]) * pw[decoder.slice("y_inv_w")] + pw[
      decoder.slice("y_inv_b")
    ]
    return x + sc.concat([dt * (x[2:4] + y / 2.0), y]) - xnext

  return stage


@functools.cache
def npmpc_eq_function(horizon: int, decoder: Decoder = Decoder()) -> NpmpcFunction:
  """The `horizon` dynamics equalities, as one `sc.vmap` over the stage function.

  States are blocked ahead of controls in `z`, so both windows into it stride cleanly and there is
  no dead trailing control the way an interleaved layout would leave.
  """
  pw_slice, dt_slice, _, _ = _param_slices(decoder)

  @sc.function(sc.G(sc.L("z", n_dec(horizon)), sc.L("p", n_param(decoder))), sc.L("eq", ...), name=f"npmpc_eq_N{horizon}")
  def equality(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = inputs
    return sc.vmap(
      stage_function(decoder),
      length=horizon,
      inputs={
        "x": (z, 0, NX),
        "xnext": (z, NX, NX),
        "u": (z, NX * (horizon + 1), NU),
        "pw": (p[pw_slice], 0, 0),
        "dt": (p[dt_slice], 0, 0),
      },
    )

  return equality


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
def _stage_jac_function(decoder: Decoder) -> StageJacFunction:
  stage = stage_function(decoder)
  return stage.factory(
    f"npmpc_stage_jac_h{'x'.join(str(h) for h in decoder.hidden)}",
    list(stage.input_names),
    [sc.factory.Jac("eq", "x"), sc.factory.Jac("eq", "u"), sc.factory.Jac("eq", "xnext")],
  )


def npmpc_eq_jac_dense_reference(horizon: int, z: np.ndarray, p: np.ndarray, decoder: Decoder = Decoder()) -> np.ndarray:
  """Dense Jacobian of the equalities, assembled from per-stage dense blocks.

  Each stage's residual touches only its own state, control, and successor state, so the matrix is
  block-banded and the reference costs one small dense Jacobian per stage rather than one enormous
  one over the whole horizon. Differentiating the *stage* function and scattering the blocks is also
  the independence the check needs: it never builds the VMAP graph the sparse kernel comes from.
  """
  _, pw, dt, _, _ = _unpack_nlp_params(decoder, p)
  jac = _stage_jac_function(decoder)
  offset = NX * (horizon + 1)
  dense = np.zeros((NX * horizon, n_dec(horizon)), dtype=np.float64)
  for i in range(horizon):
    x, xnext = z[NX * i : NX * (i + 1)], z[NX * (i + 1) : NX * (i + 2)]
    u = z[offset + NU * i : offset + NU * (i + 1)]
    args = (x, xnext, u, pw, np.array(dt))
    d_x, d_u, d_xnext = (np.asarray(block, dtype=np.float64).reshape(NX, -1) for block in jac(args))
    rows = slice(NX * i, NX * (i + 1))
    dense[rows, NX * i : NX * (i + 1)] = d_x
    dense[rows, NX * (i + 1) : NX * (i + 2)] = d_xnext
    dense[rows, offset + NU * i : offset + NU * (i + 1)] = d_u
  return dense.reshape(-1)


def npmpc_constraint_jac_dense_reference(horizon: int, z: np.ndarray, p: np.ndarray, decoder: Decoder = Decoder()) -> np.ndarray:
  """Dense reference for every equality and inequality row in the solver descriptor."""
  z, p = np.asarray(z, dtype=np.float64), np.asarray(p, dtype=np.float64)
  if z.shape != (n_dec(horizon),) or p.shape != (n_param(decoder),):
    raise ValueError(f"invalid z/p shapes {z.shape} / {p.shape}")
  n_eq, n_ineq = constraint_counts(horizon)
  dense = np.zeros((n_eq + n_ineq, z.size), dtype=np.float64)
  dense[:n_eq] = npmpc_eq_jac_dense_reference(horizon, z, p, decoder).reshape(n_eq, -1)
  dense[n_eq : n_eq + NX, :NX] = np.eye(NX)
  slack_col = z.size - 1
  for stage in range(horizon + 1):
    phi_col = NX * stage + 1
    dense[n_eq + NX + stage, [phi_col, slack_col]] = 1.0
    dense[n_eq + NX + horizon + 1 + stage, [phi_col, slack_col]] = [1.0, -1.0]
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


def pack_cost_weights(weights: CostWeights = CostWeights()) -> np.ndarray:
  """Pack the stage, input, and slack cost coefficients in their runtime ABI order."""
  return np.asarray((*weights.x, *weights.x_diff, weights.u, weights.slack), dtype=np.float64)


def pack_nlp_params(
  decoder: Decoder,
  xstart: np.ndarray,
  pw: np.ndarray,
  P: np.ndarray,
  *,
  weights: CostWeights = CostWeights(),
  dt: float = DT,
) -> np.ndarray:
  """Pack every runtime NLP parameter in the order shared by Scaly and CasADi."""
  xstart = np.asarray(xstart, dtype=np.float64).reshape(-1)
  pw = np.asarray(pw, dtype=np.float64).reshape(-1)
  P = np.asarray(P, dtype=np.float64)
  if xstart.shape != (NX,) or pw.shape != (decoder.n_pw,) or P.shape != (NX, NX):
    raise ValueError(f"invalid xstart/pw/P shapes {xstart.shape} / {pw.shape} / {P.shape}")
  return np.concatenate([xstart, pw, np.array([dt]), pack_cost_weights(weights), P.reshape(-1)])


def _unpack_nlp_params(decoder: Decoder, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
  p = np.asarray(p, dtype=np.float64).reshape(-1)
  if p.shape != (n_param(decoder),):
    raise ValueError(f"invalid parameter shape {p.shape}, expected {(n_param(decoder),)}")
  pw_slice, dt_slice, cost_slice, P_slice = _param_slices(decoder)
  return p[:NX], p[pw_slice], float(p[dt_slice][0]), p[cost_slice], p[P_slice].reshape(NX, NX)


def linearize(decoder: Decoder, pw: np.ndarray, dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
  """`A`, `B` of the learned one-step map at the upright equilibrium.

  The stage residual is `x + f(x, u) - xnext`, so its derivatives with respect to `x` and `u` at
  the equilibrium *are* `A` and `B`. Taking them from `sc.factory.Jac` keeps the reference implementation's
  torch dependency out and exercises Scaly's differentiation in the problem's own setup.
  """
  jac = _stage_jac_function(decoder)
  args = (np.zeros(NX), np.zeros(NX), np.zeros(NU), pw, np.array(dt))
  d_x, d_u, _ = (np.asarray(block, dtype=np.float64).reshape(NX, -1) for block in jac(args))
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


@sc.function(
  sc.G(sc.L("x", NX), sc.L("xnext", NX), sc.L("u", NU), sc.L("cost_weights", N_COST_WEIGHTS)),
  sc.L("cost", ...),
  name="npmpc_stage_cost",
)
def stage_cost_function(inputs: tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
  """One horizon stage of the objective with runtime cost coefficients.

  Scanned over the horizon by `npmpc_cost_expr`, so the objective's generated source stays constant
  in the horizon exactly as the dynamics' does. Building it as a Python loop instead unrolls it,
  which grows the source linearly and, past roughly 75 stages, exceeds the Program IR passes'
  recursion depth during lowering (the limitation recorded in `internal/todo.md`).
  """
  x, xnext, u, weights = inputs
  dx = xnext - x
  # The pendulum angle uses the 2*pi-periodic half-angle lift, so every upright pose costs the
  # same: (2 sin(theta/2))^2 = 2 (1 - cos theta).
  total = weights[0] * 2.0 * (1.0 - x[0].cos()) + weights[8] * u[0] * u[0]
  for k in (1, 2, 3):
    total = total + weights[k] * x[k] * x[k]
  for k in range(NX):
    total = total + weights[NX + k] * dx[k] * dx[k]
  return total


def npmpc_cost_expr(z: sc.Expr, horizon: int, P: sc.Expr, weights: sc.Expr) -> sc.Expr:
  """Total objective: stage, inter-stage, LQR terminal, and slack penalty (paper eq. 15).

  The per-stage part uses VMAP; only the terminal and slack terms, which exist once, sit outside
  the loop.
  """
  offset = NX * (horizon + 1)
  stages = sc.vmap(
    stage_cost_function,
    length=horizon,
    inputs={"x": (z, 0, NX), "xnext": (z, NX, NX), "u": (z, offset, NU), "cost_weights": (weights, 0, 0)},
  )
  xN = z[NX * horizon : NX * (horizon + 1)]
  e_end = sc.stack([2.0 * (xN[0] / 2.0).sin(), xN[1], xN[2], xN[3]])
  terminal = sc.dot(e_end, P.reshape((NX, NX)) @ e_end)
  slack = z[offset + NU * horizon]
  return stages.sum() + terminal + weights[9] * 0.5 * (slack * slack + slack)


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


def npmpc_constraint_exprs(z: sc.Expr, xstart: sc.Expr, horizon: int) -> tuple[sc.Expr, np.ndarray, np.ndarray]:
  """The inequality rows and their bounds: the initial-state band, then the softened arm-angle limits.

  Both arm-angle families are one strided gather plus the scalar slack, so the rendered source does
  not grow with the horizon any more than the VMAP dynamics do.
  """
  slack = z[NX * (horizon + 1) + NU * horizon]
  phi = z[1 : NX * (horizon + 1) : NX]
  return sc.concat([z[:NX] - xstart, phi + slack, phi - slack]), *npmpc_ineq_bounds(horizon)


def npmpc_bounds(horizon: int) -> tuple[np.ndarray, np.ndarray]:
  """Decision-vector box bounds: the states are free, the torques hard-bounded, the slack non-negative."""
  offset = NX * (horizon + 1)
  lower, upper = np.full(n_dec(horizon), -np.inf), np.full(n_dec(horizon), np.inf)
  lower[offset : offset + NU * horizon], upper[offset : offset + NU * horizon] = -TORQUE_LIMIT, TORQUE_LIMIT
  lower[-1], upper[-1] = 0.0, SLACK_LIMIT
  return lower, upper


def npmpc_lag_hess_dense_reference(
  horizon: int,
  z: np.ndarray,
  p: np.ndarray,
  lam_f: float,
  lam_g: np.ndarray,
  decoder: Decoder = Decoder(),
) -> np.ndarray:
  """Dense NumPy Hessian of ``lam_f * f + dot(lam_g, [h_eq; g_ineq])`` for ``npmpc_nlp``.

  Stage ``i`` touches ``x_i``, ``x_{i+1}`` and ``u_i``: the stage cost through the inter-stage
  difference, the learned step through ``x_i`` and ``u_i``. The inequality rows are linear. The
  terminal term and the slack penalty are added on their own blocks. Each block is one exact
  hyper-dual evaluation per column.
  """
  from benchmarks.harness.hyperdual import lagrangian_hessian_np

  z, p, lam_g = np.asarray(z, dtype=np.float64), np.asarray(p, dtype=np.float64), np.asarray(lam_g, dtype=np.float64)
  if z.shape != (n_dec(horizon),) or p.shape != (n_param(decoder),) or lam_g.shape != (sum(constraint_counts(horizon)),):
    raise ValueError(f"invalid z/p/lam_g shapes {z.shape} / {p.shape} / {lam_g.shape}")
  _, pw, dt, weights, terminal_weight = _unpack_nlp_params(decoder, p)
  offset = NX * (horizon + 1)
  dense = np.zeros((z.size, z.size), dtype=np.float64)

  def stage(v):
    x, xnext, u = v[:NX], v[NX : 2 * NX], v[2 * NX :]
    dx = xnext - x
    cost = weights[0] * 2.0 * (1.0 - np.cos(x[0])) + weights[8] * u[0] * u[0]
    cost = cost + sum(weights[k] * x[k] * x[k] for k in (1, 2, 3))
    cost = cost + sum(weights[NX + k] * dx[k] * dx[k] for k in range(NX))
    y = decoder_mu_np(decoder, pw, x, u)
    return [cost, *(x + np.concatenate([dt * (x[2:4] + y / 2.0), y]) - xnext)]

  def terminal(xn):
    e_end = [2.0 * np.sin(xn[0] / 2.0), xn[1], xn[2], xn[3]]
    return [sum(e_end[a] * float(terminal_weight[a, b]) * e_end[b] for a in range(NX) for b in range(NX))]

  for i in range(horizon):
    index = np.r_[NX * i : NX * (i + 2), offset + NU * i : offset + NU * (i + 1)]
    dense[np.ix_(index, index)] += lagrangian_hessian_np(stage, z[index], np.concatenate([[lam_f], lam_g[NX * i : NX * (i + 1)]]))
  block = slice(NX * horizon, offset)
  dense[block, block] += lagrangian_hessian_np(terminal, z[block], np.array([lam_f]))
  dense[-1, -1] += lam_f * weights[9]
  return dense


def npmpc_nlp(
  horizon: int = HORIZON,
  decoder: Decoder = Decoder(),
  *,
  solver: str = "ipopt",
  options: dict[str, str | int | float] | None = None,
) -> NpmpcSolver:
  """The neural-process MPC as one typed `sc.Problem`: VMAP dynamics, VMAP cost, one arm-angle slack.

  `p` carries the initial state, decoder tail, time step, cost coefficients, and terminal weight in
  the order `pack_nlp_params` defines. All numerical configuration remains available to generated-C
  callers. Only the horizon and decoder architecture specialize the compiled function.
  """
  lower, upper = npmpc_bounds(horizon)
  problem_name = f"npmpc_N{horizon}"
  pw_slice, dt_slice, cost_slice, P_slice = _param_slices(decoder)

  @sc.problem(vars=sc.L("z", n_dec(horizon)), params=sc.L("p", n_param(decoder)), name=problem_name)
  def problem(z: sc.Expr, p: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    eq = sc.vmap(
      stage_function(decoder),
      length=horizon,
      inputs={
        "x": (z, 0, NX),
        "xnext": (z, NX, NX),
        "u": (z, NX * (horizon + 1), NU),
        "pw": (p[pw_slice], 0, 0),
        "dt": (p[dt_slice], 0, 0),
      },
    )
    rows, l_ineq, u_ineq = npmpc_constraint_exprs(z, p[:NX], horizon)
    return sc.ProblemSpec(
      minimize=npmpc_cost_expr(z, horizon, p[P_slice], p[cost_slice]),
      eq=(eq,),
      ineq=(
        sc.bounded(
          rows,
          lo=sc.const(l_ineq),
          hi=sc.const(u_ineq),
          name="soft_bounds",
        ),
      ),
      lb=sc.const(lower),
      ub=sc.const(upper),
    )

  return sc.solver(problem, solver, name=f"{problem_name}_{solver}", options=options)


@functools.cache
def npmpc_lag_function(horizon: int, decoder: Decoder = Decoder()) -> NpmpcLagFunction:
  """Objective and dynamics equalities together, the pair an exact Lagrangian Hessian needs."""
  pw_slice, dt_slice, cost_slice, P_slice = _param_slices(decoder)

  @sc.function(
    sc.G(sc.L("z", n_dec(horizon)), sc.L("p", n_param(decoder))),
    sc.G(sc.L("cost", ...), sc.L("eq", ...)),
    name=f"npmpc_lag_N{horizon}",
  )
  def lagrangian_inputs(inputs: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr]:
    z, p = inputs
    eq = sc.vmap(
      stage_function(decoder),
      length=horizon,
      inputs={
        "x": (z, 0, NX),
        "xnext": (z, NX, NX),
        "u": (z, NX * (horizon + 1), NU),
        "pw": (p[pw_slice], 0, 0),
        "dt": (p[dt_slice], 0, 0),
      },
    )
    return npmpc_cost_expr(z, horizon, p[P_slice], p[cost_slice]), eq

  return lagrangian_inputs


def ca_npmpc_pieces(
  horizon: int,
  decoder: Decoder = Decoder(),
  sym_t=None,
  *,
  cost: bool = True,
  dynamics: bool = True,
  batched: bool = False,
  mtimes: str | None = None,
) -> dict:
  """The CasADi mirror's symbolic pieces, in Scaly's own row and column order.

  One builder behind all four CasADi consumers -- the sweep's constraint-Jacobian and Lagrangian-
  Hessian kernels and the closed loop's IPOPT and SQP columns -- so the mirror cannot drift from
  itself. Its runtime symbols match Scaly's parameter vector, so neither provider specializes
  numerical tuning data into generated code. Every constraint row and decision column sits in the
  same place. That identity makes the comparison controlled: only the tool that differentiates and
  evaluates the oracles changes.

  Set ``dynamics=False`` only when the caller replaces ``h_eq`` with an equivalent encoding.
  ``batched`` decodes every stage's features as one matrix-matrix product per layer, and ``mtimes``
  selects CasADi 3.8's dense kernel (``"reference"``, ``"classic"`` BLAS or ``"blasfeo"``).
  """
  import casadi as ca

  sym_t = ca.SX if sym_t is None else sym_t
  mm = (lambda a, b: ca.mtimes(a, b)) if mtimes is None else (lambda a, b: ca.mtimes(a, b, mtimes))
  z = sym_t.sym("z", n_dec(horizon))
  xstart = sym_t.sym("xstart", NX)
  pw = sym_t.sym("pw", decoder.n_pw)
  dt = sym_t.sym("dt")
  cost_weights = sym_t.sym("cost_weights", N_COST_WEIGHTS)
  terminal_weight = sym_t.sym("P", NX, NX)
  if dynamics:
    xw, xb = pw[decoder.slice("x_scale_w")], pw[decoder.slice("x_scale_b")]
    yw, yb = pw[decoder.slice("y_inv_w")], pw[decoder.slice("y_inv_b")]
    latent, bias = pw[decoder.slice("latent")], pw[decoder.slice("bias")]
    # CasADi reshapes in column-major order, so build the transpose and flip it.
    mats = [ca.reshape(pw[decoder.slice(f"w{i}")], shape[1], shape[0]).T for i, shape in enumerate(decoder.weight_shapes)]

    def decode(h, k: int):
      for mat in mats[:-1]:
        h = 1.0 / (1.0 + ca.exp(-mm(mat, h)))
      return (mm(mats[-1], h) + ca.repmat(bias, 1, k)) * ca.repmat(yw, 1, k) + ca.repmat(yb, 1, k)

  offset = NX * (horizon + 1)
  stages = [(z[NX * i : NX * (i + 1)], z[NX * (i + 1) : NX * (i + 2)], z[offset + NU * i : offset + NU * (i + 1)]) for i in range(horizon)]
  ys: list = []
  if dynamics:
    feats = [ca.vertcat(ca.vertcat(ca.sin(x[0]), ca.cos(x[0]), x[2], x[3], u[0]) * xw + xb, latent) for x, _, u in stages]
    if batched:
      y_all = decode(ca.horzcat(*feats), horizon)
      ys = [y_all[:, i] for i in range(horizon)]
    else:
      ys = [decode(h, 1) for h in feats]
  rows, cost_terms = [], []
  for i, (x, xnext, u) in enumerate(stages):
    if dynamics:
      y = ys[i]
      rows.append(x + ca.vertcat(dt * (x[2:4] + y / 2.0), y) - xnext)
    if cost:
      dx = xnext - x
      cost_terms.append(
        cost_weights[0] * 2.0 * (1.0 - ca.cos(x[0]))
        + sum(cost_weights[k] * x[k] ** 2 for k in (1, 2, 3))
        + cost_weights[8] * u[0] ** 2
        + sum(cost_weights[NX + k] * dx[k] ** 2 for k in range(NX))
      )
  slack = z[offset + NU * horizon]
  objective = None
  if cost:
    xN = z[NX * horizon : offset]
    e_end = ca.vertcat(2.0 * ca.sin(xN[0] / 2.0), xN[1], xN[2], xN[3])
    objective = sum(cost_terms) + e_end.T @ terminal_weight @ e_end + cost_weights[9] * 0.5 * (slack**2 + slack)
  phi = ca.vertcat(*[z[NX * i + 1] for i in range(horizon + 1)])
  l_ineq, u_ineq = npmpc_ineq_bounds(horizon)
  x_lb, x_ub = npmpc_bounds(horizon)
  n_eq, n_ineq = constraint_counts(horizon)
  return {
    "z": z,
    "xstart": xstart,
    "pw": pw,
    "dt": dt,
    "cost_weights": cost_weights,
    "P": terminal_weight,
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


def _ca_npmpc_joint_parameter_pieces(
  horizon: int,
  decoder: Decoder = Decoder(),
  sym_t=None,
  *,
  dynamics: bool = True,
  batched: bool = False,
  mtimes: str | None = None,
) -> dict:
  """Fold the CasADi graph's two parameter symbols into the solver's single vector."""
  import casadi as ca

  sym_t = ca.MX if sym_t is None else sym_t
  pieces = ca_npmpc_pieces(horizon, decoder, sym_t, dynamics=dynamics, batched=batched, mtimes=mtimes)
  p = sym_t.sym("p", n_param(decoder))
  pw_slice, dt_slice, cost_slice, P_slice = _param_slices(decoder)
  f, h_eq, g_ineq = ca.substitute(
    [pieces["f"], pieces["h_eq"], pieces["g_ineq"]],
    [pieces["xstart"], pieces["pw"], pieces["dt"], pieces["cost_weights"], pieces["P"]],
    [p[:NX], p[pw_slice], p[dt_slice], p[cost_slice], ca.reshape(p[P_slice], NX, NX).T],
  )
  return {**pieces, "p": p, "f": f, "h_eq": h_eq, "g_ineq": g_ineq}


if TYPE_CHECKING:
  assert_type(stage_function(), StageFunction)
  assert_type(stage_cost_function, StageCostFunction)
  assert_type(npmpc_eq_function(HORIZON), NpmpcFunction)
  assert_type(npmpc_lag_function(HORIZON), NpmpcLagFunction)
  assert_type(npmpc_nlp(), NpmpcSolver)
