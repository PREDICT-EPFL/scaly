from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from alloy.utils import load_torch_state_dict

NSTATE = 7
NCTRL = 2
INTERNAL_DIM = 4
LEARNED_DIM = 3
HIDDEN = (64, 64)
X_SCALE_SIZE = 4
W0_SHAPE = (64, 6)
B0_SHAPE = (64,)
W1_SHAPE = (64, 64)
B1_SHAPE = (64,)
W2_SHAPE = (3, 64)
B2_SHAPE = (3,)
WEIGHT_SHAPES = (W0_SHAPE, B0_SHAPE, W1_SHAPE, B1_SHAPE, W2_SHAPE, B2_SHAPE)
WEIGHT_SIZES = tuple(int(np.prod(s)) for s in WEIGHT_SHAPES)
OFFSETS = tuple(int(x) for x in np.cumsum([0, X_SCALE_SIZE, *WEIGHT_SIZES]))
N_PW = OFFSETS[-1]
N_PHYSICS = 8
DEFAULT_CT_MODEL_PATH = Path(__file__).parent / "data" / "ct_full_xlarge.pt"
DEFAULT_DT_MODEL_PATH = Path(__file__).parent / "data" / "dt_kinematic_mlp.pt"
# The discrete model's feature scaling is a code constant upstream, not a checkpoint entry.
DT_X_SCALE = np.array([1.9, 0.6, 0.1, 2.012], dtype=np.float64)
DT_VF_DEADZONE = 0.03
DT_W0_SHAPE = (256, INTERNAL_DIM + NCTRL)
DT_W1_SHAPE = (128, 256)
DT_W2_SHAPE = (LEARNED_DIM, 128)
DT_WEIGHT_SHAPES = (DT_W0_SHAPE, (256,), DT_W1_SHAPE, (128,), DT_W2_SHAPE, (LEARNED_DIM,))
DT_WEIGHT_SIZES = tuple(int(np.prod(s)) for s in DT_WEIGHT_SHAPES)
DT_OFFSETS = tuple(int(x) for x in np.cumsum([0, X_SCALE_SIZE, *DT_WEIGHT_SIZES]))
N_PW_DT = DT_OFFSETS[-1]
# Filter-side smoothing of the plant's ReLU: relu(x) ≈ (x + sqrt(x^2 + eps^2)) / 2, the same
# smooth-|x| idiom the barrier uses for |d|. IPOPT wants C² and the plant's ReLU is not, but the
# usual softplus costs an exp and a log per activation where this costs one sqrt, and there are
# 384 of them per car per step in a graph whose evaluation is ~90% of solve time. At this eps the
# worst error in the predicted next `vf` is 0.005 m/s, a quarter of softplus at `beta = 50`, and
# unlike softplus it cannot overflow. The plant keeps the exact ReLU; only the filter smooths.
DT_RELU_EPS = 0.01


@dataclass(slots=True)
class CarPhysics:
  lf: float = 0.54
  lr: float = 0.33
  max_delta: float = 2.012
  steering_time_constant: float = 0.1550
  x_min: float = 0.0
  x_max: float = 15.0
  y_min: float = 0.0
  y_max: float = 15.0

  def array(self) -> np.ndarray:
    return np.array([self.lf, self.lr, self.max_delta, self.steering_time_constant, self.x_min, self.x_max, self.y_min, self.y_max], dtype=np.float64)


@dataclass(slots=True)
class HCBFConfig:
  """Order-1 hyperbolic barrier for a car pair (`gradient_HCBF` in the colleague's
  `bumper_car_simulator`), which constrains the *closing speed* of the pair rather
  than its distance:

      b = v_x + smooth_sign(d) * (V(|d|)^4 + (c(|d|) v_y)^4)^(1/4)

  with `d = ‖p_j - p_i‖ - safety_radius`, `v_x` the separation rate and `v_y` the
  transverse rate. `V` is the braking envelope — the largest closing speed from
  which the pair can still stop within the clearance it has — and the `v_y` term
  credits the pair for passing wide instead of braking.

  `V(d) = envelope_c * d^envelope_q` replaces the reference implementation's tabulated
  inversion of the full-brake speed map, which has no symbolic counterpart — the reason
  that implementation refuses HCBF in its discrete-time mode. A power law is the form that
  can describe *either* candidate model: fitting the pair envelope `V_pair(d) = 2 V_single(d/2)`
  to the exact discrete stopping distance gives `q ≈ 0.5` for the continuous-time model — the
  square root a constant deceleration implies — and `q ≈ 0.84` for the natively discrete MLP,
  whose braking is drag-like instead.

  **The shipped constants are one conservative fit covering both models**: the tightest power
  law that never over-predicts either model's exact discrete pair stopping envelope on
  `d ∈ [0.1, 3] m` (the discrete MLP binds below ~2.56 m, the continuous-time model above).
  It sits snug under the discrete MLP's envelope (10.6% mean shortfall) and deliberately loose
  under the continuous-time model's at short range (25% mean) — this benchmark is a
  representative reproduction of the reference filter, not its development center, so one
  stable fit that behaves under every plant/filter-model pairing beats two per-model ones.
  Both matched-model closed loops run collision- and failure-free at the canonical point with
  these constants. The per-model fits, and why an "honest" refit under a *mismatched* filter
  model made the closed loop less safe, are recorded in the problem README's envelope sections.

  The argument is always `d_eps >= eps > 0`, so the power never reaches its infinite-slope
  point at zero.
  """

  envelope_c: float = 1.00994
  envelope_q: float = 0.8355
  eps: float = 0.05

  @property
  def single_envelope_c(self) -> float:
    """Recover ``V_single(d)`` from ``V_pair(d) = 2 V_single(d / 2)``."""
    return self.envelope_c / 2.0 ** (1.0 - self.envelope_q)


@dataclass(slots=True)
class ClosedLoopConfig:
  ncars: int = 8
  steps: int = 200
  dt: float = 0.1
  seed: int = 42
  # Which model the plant steps with: "dt" is the natively discrete MLPModel, "ct" the RK4
  # map of the continuous-time model. The filter's own choice is FilterConfig.model.
  plant: str = "dt"
  # collision_radius is the centre-to-centre distance at which the two body discs
  # touch — the only thing a collision is counted against. safety_radius is what the
  # filter enforces, a margin above it (the reference implementation's safety_factor
  # of 1.2), so the barrier has room to act before contact.
  collision_radius: float = 1.9
  safety_radius: float = 2.28
  wall_margin: float = 1.0
  wall_eps: float = 1e-4
  pair_gamma: float = 0.35
  wall_gamma: float = 0.35
  arena_avoidance: bool = True
  target_center: bool = True
  nominal_speed: float = 0.55
  hcbf: HCBFConfig = field(default_factory=HCBFConfig)
  physics: CarPhysics = field(default_factory=CarPhysics)

  @property
  def n_pairs(self) -> int:
    return self.ncars * (self.ncars - 1) // 2

  @property
  def n_slack(self) -> int:
    """One slack per constraint row: the pair rows, then four wall rows per car."""
    return self.n_pairs + (4 * self.ncars if self.arena_avoidance else 0)


@dataclass(slots=True)
class FilterConfig:
  # Which model the filter predicts one step ahead with: "dt" is the natively discrete MLP
  # (smoothed; see dt_mlp_step_smooth_np), "ct" the RK4 map of the continuous-time model. "dt" is
  # the canonical configuration -- it matches the plant, and the mismatch is what caused the
  # collisions and the chaotic sensitivity the problem README documents. The weights handed to the
  # filter have to match the choice, so `run_one`'s caller loads accordingly. Historical gates and
  # measurements that were written against the continuous-time model ask for it explicitly; the
  # current scalability sweep uses this discrete model and its exact Lagrangian Hessian.
  model: str = "dt"
  R: tuple[float, float] = (10.0, 1.0)
  # L1 weight on the per-row slacks. Exactness needs it above the largest constraint
  # multiplier, not the 1e5 the squared shared slack used to need to stay near zero.
  slack_weight: float = 1_000.0
  ipopt_tol: float = 1e-6
  ipopt_max_iter: int = 300
  eval_repeats: int = 1
  # Exact Lagrangian Hessians from both oracle providers by default: Alloy's sphess-through-VMAP path
  # is what this problem exists to exercise, and it is gated against CasADi's.
  limited_memory_hessian: bool = False


@dataclass(slots=True)
class CTFullWeights:
  path: Path
  x_scale: np.ndarray
  w0: np.ndarray
  b0: np.ndarray
  w1: np.ndarray
  b1: np.ndarray
  w2: np.ndarray
  b2: np.ndarray

  @property
  def packed(self) -> np.ndarray:
    return np.concatenate(
      [
        self.x_scale.reshape(-1),
        self.w0.reshape(-1),
        self.b0.reshape(-1),
        self.w1.reshape(-1),
        self.b1.reshape(-1),
        self.w2.reshape(-1),
        self.b2.reshape(-1),
      ]
    ).astype(np.float64)


@dataclass(slots=True)
class DTMLPWeights:
  """`MLPModel`'s kinematic network: the natively discrete one-step map for the velocity block.

  Its featurization differs from the continuous-time model's in ways that all matter: the
  state block is ``[vf, alpha_f, alpha_r, delta]`` with ``alpha_f = beta_f - delta`` rather
  than the raw ``beta``s, **the two controls come in the opposite order** (``u_s`` before
  ``u_m``, confirmed by the model's author), the activations are ReLU rather than SiLU, and
  the three outputs are absolute next-state values rather than derivatives.
  """

  path: Path
  x_scale: np.ndarray
  w0: np.ndarray
  b0: np.ndarray
  w1: np.ndarray
  b1: np.ndarray
  w2: np.ndarray
  b2: np.ndarray

  @property
  def packed(self) -> np.ndarray:
    """Same tail layout as :class:`CTFullWeights`: x_scale, then each layer's weight and bias."""
    return np.concatenate([a.reshape(-1) for a in (self.x_scale, self.w0, self.b0, self.w1, self.b1, self.w2, self.b2)]).astype(np.float64)


def load_ct_full_weights(path: str | Path = DEFAULT_CT_MODEL_PATH) -> CTFullWeights:
  p = Path(path).expanduser()
  if not p.exists():
    raise FileNotFoundError(f"CTFull checkpoint not found at {p}")
  ckpt = load_torch_state_dict(p)
  state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
  if not isinstance(state, dict):
    raise TypeError(f"expected checkpoint state_dict to be a dict, got {type(state).__name__}")
  state_dict = cast(dict[str, Any], state)
  x_scale_src = state_dict.get("x_scale")
  if x_scale_src is None and isinstance(ckpt, dict):
    x_scale_src = cast(dict[str, Any], ckpt).get("x_scale")
  if x_scale_src is None:
    raise KeyError("checkpoint does not contain x_scale")
  x_scale = np.asarray(x_scale_src, dtype=np.float64)
  weights = CTFullWeights(
    path=p,
    x_scale=x_scale,
    w0=np.asarray(state_dict["net.0.weight"], dtype=np.float64),
    b0=np.asarray(state_dict["net.0.bias"], dtype=np.float64),
    w1=np.asarray(state_dict["net.2.weight"], dtype=np.float64),
    b1=np.asarray(state_dict["net.2.bias"], dtype=np.float64),
    w2=np.asarray(state_dict["net.4.weight"], dtype=np.float64),
    b2=np.asarray(state_dict["net.4.bias"], dtype=np.float64),
  )
  if weights.packed.size != N_PW:
    raise ValueError(f"packed CTFull weights have size {weights.packed.size}, expected {N_PW}")
  return weights


def load_dt_mlp_weights(path: str | Path = DEFAULT_DT_MODEL_PATH) -> DTMLPWeights:
  p = Path(path).expanduser()
  if not p.exists():
    raise FileNotFoundError(f"DT MLP checkpoint not found at {p}")
  state = load_torch_state_dict(p)
  weights = DTMLPWeights(
    path=p,
    x_scale=DT_X_SCALE,
    w0=np.asarray(state["model.0.weight"], dtype=np.float64),
    b0=np.asarray(state["model.0.bias"], dtype=np.float64),
    w1=np.asarray(state["model.2.weight"], dtype=np.float64),
    b1=np.asarray(state["model.2.bias"], dtype=np.float64),
    w2=np.asarray(state["model.4.weight"], dtype=np.float64),
    b2=np.asarray(state["model.4.bias"], dtype=np.float64),
  )
  if weights.w0.shape[1] != INTERNAL_DIM + NCTRL or weights.w2.shape[0] != LEARNED_DIM:
    raise ValueError(f"DT MLP checkpoint at {p} maps {weights.w0.shape[1]} features to {weights.w2.shape[0]} outputs, expected 6 to 3")
  return weights


def unpack_packed_weights(pw: np.ndarray) -> CTFullWeights:
  pw = np.asarray(pw, dtype=np.float64).reshape(-1)
  if pw.size != N_PW:
    raise ValueError(f"expected packed weights of size {N_PW}, got {pw.size}")
  parts = []
  for i, shape in enumerate(((X_SCALE_SIZE,), *WEIGHT_SHAPES)):
    parts.append(pw[OFFSETS[i] : OFFSETS[i + 1]].reshape(shape))
  return CTFullWeights(Path("<packed>"), *parts)


def normalize_angle(angle: np.ndarray | float) -> np.ndarray | float:
  return (angle + np.pi) % (2.0 * np.pi) - np.pi


def silu_np(x: np.ndarray) -> np.ndarray:
  return x / (1.0 + np.exp(-x))


def world_velocity_np(state: np.ndarray, physics: CarPhysics) -> tuple[float, float, float]:
  """World-frame velocity and yaw rate of a car, i.e. the pose part of the ODE.

  The pair barrier is built on this rather than on a separate transcription, so the
  velocity it constrains is by construction the one the model propagates.
  """
  theta, vf, beta_f, beta_r = state[2], state[3], state[4], state[5]
  omega = vf * np.sin(beta_f - beta_r) / ((physics.lf + physics.lr) * np.cos(beta_r))
  vx_b = vf * np.cos(beta_f)
  vy_b = vf * np.sin(beta_f) - physics.lf * omega
  return vx_b * np.cos(theta) - vy_b * np.sin(theta), vx_b * np.sin(theta) + vy_b * np.cos(theta), omega


def ct_full_ode_np(state: np.ndarray, u: np.ndarray, weights: CTFullWeights, physics: CarPhysics) -> np.ndarray:
  state = np.asarray(state, dtype=np.float64)
  u = np.asarray(u, dtype=np.float64)
  delta = state[6]
  x_dot, y_dot, omega = world_velocity_np(state, physics)
  delta_ref = u[1] * physics.max_delta
  delta_dot = (delta_ref - delta) / (physics.steering_time_constant * 3.0)

  phi = np.concatenate([state[3:7] / weights.x_scale, u])
  h = silu_np(weights.w0 @ phi + weights.b0)
  h = silu_np(weights.w1 @ h + weights.b1)
  learned = weights.w2 @ h + weights.b2
  return np.array([x_dot, y_dot, omega, learned[0], learned[1], learned[2], delta_dot], dtype=np.float64)


def rk4_step_np(state: np.ndarray, u: np.ndarray, dt: float, weights: CTFullWeights, physics: CarPhysics) -> np.ndarray:
  state = np.asarray(state, dtype=np.float64)
  u = np.asarray(u, dtype=np.float64)
  k1 = ct_full_ode_np(state, u, weights, physics)
  k2 = ct_full_ode_np(state + 0.5 * dt * k1, u, weights, physics)
  k3 = ct_full_ode_np(state + 0.5 * dt * k2, u, weights, physics)
  k4 = ct_full_ode_np(state + dt * k3, u, weights, physics)
  nxt = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
  nxt[2] = normalize_angle(nxt[2])
  return nxt


def pose_rk4_np(state: np.ndarray, dt: float, physics: CarPhysics, wrap: bool = True) -> np.ndarray:
  """RK4 on the pose rows with the velocity block held over the step (`MLPModel.pose_forward`).

  Only theta feeds back into the stages, since the velocity states the kinematics read are
  frozen. The plant wraps the result to keep the angle bounded over a long rollout; the filter's
  one-step prediction does not, since the barriers read theta only through sin/cos and the
  symbolic mirrors would otherwise disagree by 2 pi on a crossing.
  """

  def pose_dot(s: np.ndarray) -> np.ndarray:
    out = np.zeros(NSTATE, dtype=np.float64)
    out[0], out[1], out[2] = world_velocity_np(s, physics)
    return out

  k1 = pose_dot(state)
  k2 = pose_dot(state + 0.5 * dt * k1)
  k3 = pose_dot(state + 0.5 * dt * k2)
  k4 = pose_dot(state + dt * k3)
  pose = (state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))[:3]
  if wrap:
    pose[2] = normalize_angle(pose[2])
  return pose


def dt_mlp_step_np(state: np.ndarray, u: np.ndarray, dt: float, weights: DTMLPWeights, physics: CarPhysics) -> np.ndarray:
  """One step of the natively discrete `MLPModel`, three pieces of which only the middle is learned.

  The steering actuator runs at the checkpoint's own time constant rather than the
  continuous-time model's ``3 tau``: that is what the network was trained against, so
  adopting the network means adopting its actuator.
  """
  state = np.asarray(state, dtype=np.float64)
  u = np.asarray(u, dtype=np.float64)
  vf, beta_f, beta_r, delta = state[3], state[4], state[5], state[6]
  phi = np.concatenate([np.array([vf, beta_f - delta, beta_r, delta]) / weights.x_scale, np.array([u[1], u[0]])])
  h = np.maximum(0.0, weights.w0 @ phi + weights.b0)
  h = np.maximum(0.0, weights.w1 @ h + weights.b1)
  learned = weights.w2 @ h + weights.b2
  # At dt / tau < 1 this is a convex combination of delta and delta_ref, so delta stays
  # inside [-max_delta, max_delta] and the rate peaks at 2 max_delta / tau = 26 rad/s —
  # under the reference's 28.36 rad/s clip, which is therefore left out.
  delta_next = delta + dt * (u[1] * physics.max_delta - delta) / physics.steering_time_constant
  # Below the deadzone the network's own vf output is dropped to a standstill.
  vf_next = learned[0] if learned[0] >= DT_VF_DEADZONE else 0.0
  velocity = np.array([vf_next, learned[1] + delta_next, learned[2], delta_next], dtype=np.float64)
  return np.concatenate([pose_rk4_np(state, dt, physics), velocity])


def smooth_relu_np(x: np.ndarray) -> np.ndarray:
  return 0.5 * (x + np.sqrt(x * x + DT_RELU_EPS**2))


def dt_mlp_step_smooth_np(state: np.ndarray, u: np.ndarray, dt: float, weights: DTMLPWeights, physics: CarPhysics) -> np.ndarray:
  """What the *filter* predicts with when its model is the discrete MLP.

  Two deliberate departures from `dt_mlp_step_np`, the plant: the ReLUs are smoothed (IPOPT
  wants C²) and the `vf` deadzone is dropped, since it never fires in these episodes and is a
  jump discontinuity. Everything else — featurization, control order, the steering actuator at
  the checkpoint's own tau, the RK4 pose with the velocity held over the step — is the plant's.
  """
  state = np.asarray(state, dtype=np.float64)
  u = np.asarray(u, dtype=np.float64)
  vf, beta_f, beta_r, delta = state[3], state[4], state[5], state[6]
  phi = np.concatenate([np.array([vf, beta_f - delta, beta_r, delta]) / weights.x_scale, np.array([u[1], u[0]])])
  h = smooth_relu_np(weights.w0 @ phi + weights.b0)
  h = smooth_relu_np(weights.w1 @ h + weights.b1)
  learned = weights.w2 @ h + weights.b2
  delta_next = delta + dt * (u[1] * physics.max_delta - delta) / physics.steering_time_constant
  velocity = np.array([learned[0], learned[1] + delta_next, learned[2], delta_next], dtype=np.float64)
  return np.concatenate([pose_rk4_np(state, dt, physics, wrap=False), velocity])


def pair_b_np(state_i: np.ndarray, state_j: np.ndarray, cfg: ClosedLoopConfig) -> float:
  """NumPy reference for the order-1 pair barrier; see :class:`HCBFConfig`."""
  hcbf, R = cfg.hcbf, cfg.safety_radius
  p = np.asarray(state_j[:2], dtype=np.float64) - np.asarray(state_i[:2], dtype=np.float64)
  vxi, vyi, _ = world_velocity_np(state_i, cfg.physics)
  vxj, vyj, _ = world_velocity_np(state_j, cfg.physics)
  v = np.array([vxj - vxi, vyj - vyi])
  r = np.sqrt(p @ p)
  v_x = (p[0] * v[0] + p[1] * v[1]) / r
  v_y = (p[0] * v[1] - p[1] * v[0]) / r
  d_eps = np.sqrt((r - R) ** 2 + hcbf.eps**2)
  s = (r - R) / d_eps
  a_env = hcbf.envelope_c * d_eps**hcbf.envelope_q
  q = np.sqrt(d_eps * (r + R)) / R * v_y
  return float(v_x + s * np.sqrt(np.sqrt(a_env**4 + q**4)))


def wall_h_np(state: np.ndarray, cfg: ClosedLoopConfig) -> list[float]:
  """Signed distances to the four arena walls, inset by ``wall_margin``."""
  m, p = cfg.wall_margin, cfg.physics
  return [state[0] - (p.x_min + m), (p.x_max - m) - state[0], state[1] - (p.y_min + m), (p.y_max - m) - state[1]]


def wall_b_np(state: np.ndarray, cfg: ClosedLoopConfig) -> list[float]:
  """Order-1 wall barriers: braking envelope minus outward wall-closing speed."""
  vx, vy, _ = world_velocity_np(state, cfg.physics)
  out = []
  for clearance, v_closing in zip(wall_h_np(state, cfg), (-vx, vx, -vy, vy), strict=True):
    d_eps = np.sqrt(clearance * clearance + cfg.wall_eps**2)
    s = clearance / d_eps
    v_max = cfg.hcbf.single_envelope_c * d_eps**cfg.hcbf.envelope_q
    out.append(float(s * v_max - v_closing))
  return out


def min_pair_distance(states: np.ndarray) -> float:
  states = np.asarray(states, dtype=np.float64)
  if states.shape[0] < 2:
    return float("inf")
  best = float("inf")
  for i in range(states.shape[0]):
    for j in range(i + 1, states.shape[0]):
      best = min(best, float(np.linalg.norm(states[i, :2] - states[j, :2])))
  return best


def sample_initial_states(cfg: ClosedLoopConfig, *, collision_free: bool = True) -> np.ndarray:
  """Sample states in the arena, optionally allowing collisions for kernel measurements."""
  rng = np.random.default_rng(cfg.seed)
  margin = max(cfg.safety_radius + 0.4, cfg.wall_margin + 0.4)
  lo = np.array([cfg.physics.x_min + margin, cfg.physics.y_min + margin])
  hi = np.array([cfg.physics.x_max - margin, cfg.physics.y_max - margin])
  if np.any(lo >= hi):
    raise ValueError("arena is too small for the requested safety margins")
  states = np.zeros((cfg.ncars, NSTATE), dtype=np.float64)
  placed: list[np.ndarray] = []
  for i in range(cfg.ncars):
    for _ in range(5000):
      xy = rng.uniform(lo, hi)
      if not collision_free or all(np.linalg.norm(xy - p) >= cfg.safety_radius + 0.35 for p in placed):
        states[i, 0:2] = xy
        states[i, 2] = rng.uniform(-np.pi, np.pi)
        states[i, 3] = 0.05
        placed.append(xy)
        break
    else:
      raise RuntimeError(f"could not place car {i}; reduce --ncars or --safety-radius")
  return states


def write_json(path: Path, payload: dict[str, Any]) -> None:
  def default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, Path):
      return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
      return asdict(obj)
    raise TypeError(type(obj).__name__)

  path.write_text(json.dumps(payload, indent=2, default=default))
