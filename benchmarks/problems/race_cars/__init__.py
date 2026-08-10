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

import time
from dataclasses import dataclass

import numpy as np

import alloy as al
from alloy.codegen.c import render_c_source
from alloy.expr import topo

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


@al.function("race_car_eq_initial", {"z": NZ, "p": NX})
def eq_initial(z, p):
  return {"eq": z[:NX] - p[:NX]}


@al.function("race_car_eq_interstage", {"z": NZ, "znext": NZ, "params": N_PARAMS})
def eq_interstage(z, znext, params):
  return {"eq": _rk4(z[:NX], z[NX : NX + NU], params) - znext[:NX]}


def race_car_eq_function(horizon: int) -> al.Function:
  z = al.sym("z", NZ * (horizon + 1))
  p = al.sym("p", n_param(horizon), diff=False)
  params = p[NX * (horizon + 1) :]
  parts = [eq_initial.call([z[:NZ], p[:NX]])[0]]
  for i in range(horizon):
    zi = z[i * NZ : (i + 1) * NZ]
    znext = z[(i + 1) * NZ : (i + 2) * NZ]
    parts.append(eq_interstage.call([zi, znext, params])[0])
  return al.Function(f"race_car_eq_N{horizon}", [z, p], [al.concat(parts)], ["z", "p"], ["eq"])


def race_car_eq_function_map(horizon: int) -> al.Function:
  """Same semantics as ``race_car_eq_function`` but using ``al.scan`` for the interstage residuals.

  This lets benchmarks measure the impact of loop-preserving lowering directly against the
  per-stage unrolled construction.
  """
  z = al.sym("z", NZ * (horizon + 1))
  p = al.sym("p", n_param(horizon), diff=False)
  initial = eq_initial.call([z[:NZ], p[:NX]])[0]
  mapped = al.scan(
    eq_interstage,
    length=horizon,
    inputs={"z": (z, 0, NZ), "znext": (z, NZ, NZ), "params": (p, NX * (horizon + 1), 0)},
  )
  return al.Function(f"race_car_eq_map_N{horizon}", [z, p], [al.concat([initial, mapped])], ["z", "p"], ["eq"])


def ca_race_car_eq_jac(horizon: int, name: str = "race_car_eq_jac", sym_t=None):
  import casadi

  sym_t = casadi.SX if sym_t is None else sym_t
  z = sym_t.sym("z", NZ * (horizon + 1))
  p = sym_t.sym("p", n_param(horizon))
  params = p[NX * (horizon + 1) :]

  def ca_cont(x, u):
    wheelbase, _, mass, c_m0, c_r0, c_r1, c_r2 = [params[i] for i in range(N_PARAMS)]
    phi, v = x[2], x[3]
    throttle, delta = u[0], u[1]
    beta = 0.5 * delta
    vx = v * casadi.cos(beta)
    lr = 0.5 * wheelbase
    return casadi.vertcat(
      v * casadi.cos(phi + beta),
      v * casadi.sin(phi + beta),
      v * casadi.sin(beta) / lr,
      (c_m0 * throttle - (c_r0 + c_r1 * vx + c_r2 * vx * vx) * casadi.tanh(10 * vx)) / mass,
    )

  def ca_rk4(x, u):
    dt = params[1]
    k1 = ca_cont(x, u)
    k2 = ca_cont(x + dt / 2 * k1, u)
    k3 = ca_cont(x + dt / 2 * k2, u)
    k4 = ca_cont(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  parts = [z[:NX] - p[:NX]]
  for i in range(horizon):
    zi = z[i * NZ : (i + 1) * NZ]
    znext = z[(i + 1) * NZ : (i + 2) * NZ]
    parts.append(ca_rk4(zi[:NX], zi[NX : NX + NU]) - znext[:NX])
  eq = casadi.vertcat(*parts)
  return casadi.Function(name, [z, p], [casadi.jacobian(eq, z)])


def race_car_eq_sparse_metrics(horizon: int, *, render_source: bool = False) -> dict[str, float | int]:
  t0 = time.perf_counter()
  fn = race_car_eq_function(horizon)
  build_ms = (time.perf_counter() - t0) * 1000.0

  base_nodes = len(topo(fn.outputs))
  sparsity = al.jacobian_sparsity(fn.outputs[0], fn.inputs[0])
  colors = al.column_coloring(sparsity)

  t0 = time.perf_counter()
  sj = al.sparse_jacobian_colored(fn.outputs[0], fn.inputs[0])
  ad_ms = (time.perf_counter() - t0) * 1000.0
  spjf = al.Function(f"race_car_eq_N{horizon}_spjac_colored", fn.inputs, [sj.values], fn.input_names, ["spjac_eq_z"], [sj.sparsity])
  source_bytes = len(render_c_source(spjf)) if render_source else 0

  return {
    "horizon": horizon,
    "expr_nodes": base_nodes,
    "sparse_expr_nodes": len(topo(spjf.outputs)),
    "sparsity_nnz": sparsity.nnz,
    "colors": max(colors) + 1 if colors else 0,
    "function_build_ms": build_ms,
    "colored_ad_ms": ad_ms,
    "source_bytes": source_bytes,
  }
