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
DEFAULT_MODEL_PATH = Path(__file__).parent / "data" / "ct_full_xlarge.pt"


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


@dataclass(slots=True)
class ClosedLoopConfig:
  ncars: int = 4
  horizon: int = 80
  dt: float = 0.1
  seed: int = 42
  safety_radius: float = 1.9
  wall_margin: float = 1.0
  pair_gamma: float = 0.35
  wall_gamma: float = 0.35
  arena_avoidance: bool = True
  target_center: bool = True
  nominal_speed: float = 0.55
  physics: CarPhysics = field(default_factory=CarPhysics)


@dataclass(slots=True)
class FilterConfig:
  R: tuple[float, float] = (10.0, 1.0)
  slack_weight: float = 100_000.0
  ipopt_tol: float = 1e-6
  ipopt_max_iter: int = 300
  eval_repeats: int = 1
  limited_memory_hessian: bool = True


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


def load_ct_full_weights(path: str | Path = DEFAULT_MODEL_PATH) -> CTFullWeights:
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


def ct_full_ode_np(state: np.ndarray, u: np.ndarray, weights: CTFullWeights, physics: CarPhysics) -> np.ndarray:
  state = np.asarray(state, dtype=np.float64)
  u = np.asarray(u, dtype=np.float64)
  theta, vf, beta_f, beta_r, delta = state[2], state[3], state[4], state[5], state[6]
  omega = vf * np.sin(beta_f - beta_r) / ((physics.lf + physics.lr) * np.cos(beta_r))
  vx_b = vf * np.cos(beta_f)
  vy_b = vf * np.sin(beta_f) - physics.lf * omega
  x_dot = vx_b * np.cos(theta) - vy_b * np.sin(theta)
  y_dot = vx_b * np.sin(theta) + vy_b * np.cos(theta)
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


def pair_h(state_i: np.ndarray, state_j: np.ndarray, safety_radius: float) -> float:
  d = np.asarray(state_i[:2], dtype=np.float64) - np.asarray(state_j[:2], dtype=np.float64)
  return float(d @ d - safety_radius**2)


def min_pair_distance(states: np.ndarray) -> float:
  states = np.asarray(states, dtype=np.float64)
  if states.shape[0] < 2:
    return float("inf")
  best = float("inf")
  for i in range(states.shape[0]):
    for j in range(i + 1, states.shape[0]):
      best = min(best, float(np.linalg.norm(states[i, :2] - states[j, :2])))
  return best


def sample_initial_states(cfg: ClosedLoopConfig) -> np.ndarray:
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
      if all(np.linalg.norm(xy - p) >= cfg.safety_radius + 0.35 for p in placed):
        states[i, 0:2] = xy
        states[i, 2] = rng.uniform(-np.pi, np.pi)
        states[i, 3] = 0.05
        placed.append(xy)
        break
    else:
      raise RuntimeError(f"could not place car {i}; reduce --ncars or --safety-radius")
  return states


def desired_inputs_to_center(states: np.ndarray, cfg: ClosedLoopConfig) -> np.ndarray:
  out = np.zeros((cfg.ncars, NCTRL), dtype=np.float64)
  center = np.array([(cfg.physics.x_min + cfg.physics.x_max) / 2.0, (cfg.physics.y_min + cfg.physics.y_max) / 2.0])
  for i, s in enumerate(states):
    out[i, 0] = cfg.nominal_speed
    if cfg.target_center:
      angle_to_center = np.arctan2(center[1] - s[1], center[0] - s[0])
      alpha = float(normalize_angle(angle_to_center - s[2]))
      out[i, 1] = np.clip(alpha / cfg.physics.max_delta, -1.0, 1.0)
  return np.clip(out, -1.0, 1.0)


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
