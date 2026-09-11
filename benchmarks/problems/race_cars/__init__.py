"""Race-car NMPC workload: a full-size Formula Student car on an FSDS track.

The kinematic bicycle model is discretised with RK4 and transcribed stage-wise
into one equality residual per horizon. The seven physical parameters travel
symbolically in the tail of ``p`` so the sweep can vary them without rebuilding
the graph; the body dimensions and actuator limits are plain constants because
nothing sweeps them.

Constants come from the Formula Student car in ``minimal_tracking_nmpc``: the
throttle ``T`` is a physical quantity in ``[-T_MAX, T_MAX]``, not a normalized
command.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import alloy as al

NX = 4
NU = 2
NZ = NX + NU
WHEELBASE = 1.5706
DT = 0.05
M = 230.0
C_M0 = 4.950
C_R0 = 297.030
C_R1 = 16.665
C_R2 = 0.6784
N_PARAMS = 7
T_MAX = 500.0
DELTA_MAX = 0.5
# body footprint: 1.5706 m wheelbase plus ~0.6 m of overhang per end, ~1.5 m across the tyres
CAR_LENGTH = 2.8
CAR_WIDTH = 1.5
CAR_HEIGHT = 0.55


@dataclass(frozen=True)
class RaceCarParams:
  wheelbase: float = WHEELBASE
  dt: float = DT
  mass: float = M
  c_m0: float = C_M0
  c_r0: float = C_R0
  c_r1: float = C_R1
  c_r2: float = C_R2

  def array(self) -> np.ndarray:
    return np.array([self.wheelbase, self.dt, self.mass, self.c_m0, self.c_r0, self.c_r1, self.c_r2], dtype=np.float64)


def continuous_dynamics_np(x: np.ndarray, u: np.ndarray, params: RaceCarParams = RaceCarParams()) -> np.ndarray:
  x, u = np.asarray(x), np.asarray(u)
  if x.shape != (NX,) or u.shape != (NU,):
    raise ValueError(f"expected x/u shapes {(NX,)} / {(NU,)}, got {x.shape} / {u.shape}")
  beta = 0.5 * u[1]
  vx = x[3] * np.cos(beta)
  resistance = (params.c_r0 + params.c_r1 * vx + params.c_r2 * vx * vx) * np.tanh(10.0 * vx)
  return np.array(
    [
      x[3] * np.cos(x[2] + beta),
      x[3] * np.sin(x[2] + beta),
      x[3] * np.sin(beta) / (0.5 * params.wheelbase),
      (params.c_m0 * u[0] - resistance) / params.mass,
    ]
  )


def rk4_step_np(x: np.ndarray, u: np.ndarray, params: RaceCarParams = RaceCarParams()) -> np.ndarray:
  """Advance the NumPy plant by the physical parameter's fixed sample time."""
  x, u, h = np.asarray(x), np.asarray(u), params.dt
  k1 = continuous_dynamics_np(x, u, params)
  k2 = continuous_dynamics_np(x + 0.5 * h * k1, u, params)
  k3 = continuous_dynamics_np(x + 0.5 * h * k2, u, params)
  k4 = continuous_dynamics_np(x + h * k3, u, params)
  return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def n_param(horizon: int) -> int:
  return NX * (horizon + 1) + N_PARAMS


def _continuous_dynamics(x, u, params):
  wheelbase, _, mass, c_m0, c_r0, c_r1, c_r2 = [params[i] for i in range(N_PARAMS)]
  phi, v = x[2], x[3]
  throttle, delta = u[0], u[1]
  beta = 0.5 * delta
  vx = v * beta.cos()
  lr = 0.5 * wheelbase
  return al.stack(
    [
      v * (phi + beta).cos(),
      v * (phi + beta).sin(),
      v * beta.sin() / lr,
      (c_m0 * throttle - (c_r0 + c_r1 * vx + c_r2 * vx * vx) * (10 * vx).tanh()) / mass,
    ]
  )


def _rk4(x, u, params):
  dt = params[1]
  k1 = _continuous_dynamics(x, u, params)
  k2 = _continuous_dynamics(x + dt / 2 * k1, u, params)
  k3 = _continuous_dynamics(x + dt / 2 * k2, u, params)
  k4 = _continuous_dynamics(x + dt * k3, u, params)
  return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@al.function(al.G(al.L("z", NZ), al.L("p", NX)), al.L("eq", ...), name="race_car_eq_initial")
def eq_initial(inputs: tuple[al.Expr, al.Expr]) -> al.Expr:
  z, p = inputs
  return z[:NX] - p[:NX]


@al.function(al.G(al.L("z", NZ), al.L("znext", NZ), al.L("params", N_PARAMS)), al.L("eq", ...), name="race_car_eq_interstage")
def eq_interstage(inputs: tuple[al.Expr, al.Expr, al.Expr]) -> al.Expr:
  z, znext, params = inputs
  return _rk4(z[:NX], z[NX : NX + NU], params) - znext[:NX]


# TODO(API-1): Replace this horizon-specialized builder with an ``@al.function`` template.
def race_car_eq_function(horizon: int) -> al.Function:
  """Build the multiple-shooting equality residual for one prediction horizon."""

  @al.function(
    al.G(al.L("z", NZ * (horizon + 1)), al.L("p", al.TensorType((n_param(horizon),), diff=False))),
    al.L("eq", ...),
    name=f"race_car_eq_N{horizon}",
  )
  def equality(inputs: tuple[al.Expr, al.Expr]) -> al.Expr:
    z, p = inputs
    params = p[NX * (horizon + 1) :]
    parts = [eq_initial((z[:NZ], p[:NX]))]
    for i in range(horizon):
      zi = z[i * NZ : (i + 1) * NZ]
      znext = z[(i + 1) * NZ : (i + 2) * NZ]
      parts.append(eq_interstage((zi, znext, params)))
    return al.concat(parts)

  return equality


def _race_car_eq_vmap_expr(z: al.Expr, p: al.Expr, horizon: int) -> al.Expr:
  initial = eq_initial((z[:NZ], p[:NX]))
  mapped = al.vmap(
    eq_interstage,
    length=horizon,
    inputs={"z": (z, 0, NZ), "znext": (z, NZ, NZ), "params": (p, NX * (horizon + 1), 0)},
  )
  return al.concat([initial, mapped])


def race_car_constraint_jac_dense_reference(horizon: int, z: np.ndarray, p: np.ndarray) -> np.ndarray:
  """Dense NumPy reference for the equality and corridor Jacobian handed to the solver."""
  z, p = np.asarray(z, dtype=np.float64), np.asarray(p, dtype=np.float64)
  if z.shape != (NZ * (horizon + 1),) or p.shape != (n_param(horizon),):
    raise ValueError(f"invalid z/p shapes {z.shape} / {p.shape}")
  params = RaceCarParams(*p[-N_PARAMS:])

  def step(stage: np.ndarray) -> np.ndarray:
    return rk4_step_np(stage[:NX], stage[NX:], params)

  dense = np.zeros((NX * (horizon + 1) + 2 * horizon, z.size), dtype=np.float64)
  dense[:NX, :NX] = np.eye(NX)
  complex_step = 1e-30
  for stage in range(horizon):
    row, col = NX * (stage + 1), NZ * stage
    value = z[col : col + NZ]
    for j in range(NZ):
      perturbed = value.astype(np.complex128)
      perturbed[j] += complex_step * 1j
      dense[row : row + NX, col + j] = np.imag(step(perturbed)) / complex_step
    dense[row : row + NX, col + NZ : col + NZ + NX] = -np.eye(NX)

  for stage in range(horizon):
    row, col = NX * (horizon + 1) + 2 * stage, NZ * (stage + 1)
    ref = p[NX * (stage + 1) : NX * (stage + 2)]
    sin_ref, cos_ref = np.sin(ref[2]), np.cos(ref[2])
    d_phi = z[col + 2] - ref[2]
    dense[row : row + 2, col] = -sin_ref
    dense[row : row + 2, col + 1] = cos_ref
    dense[row : row + 2, col + 2] = 0.5 * CAR_LENGTH * np.cos(d_phi) + np.array([-0.5, 0.5]) * CAR_WIDTH * np.sin(d_phi)
  return dense
