from __future__ import annotations

import time

import alloy as al
from alloy.codegen.c import render_c_source
from alloy.expr import topo

NX = 4
NU = 2
NZ = NX + NU
WHEELBASE = 0.3
DT = 0.05
M = 3.47
C_M0 = 11.0
C_R0 = 0.1
C_R1 = 0.01
C_R2 = 0.001


def _continuous_dynamics(x, u):
  phi, v = x[2], x[3]
  throttle, delta = u[0], u[1]
  beta = 0.5 * delta
  vx = v * beta.cos()
  lr = 0.5 * WHEELBASE
  return al.stack(
    [
      v * (phi + beta).cos(),
      v * (phi + beta).sin(),
      v * beta.sin() / lr,
      (C_M0 * throttle - (C_R0 + C_R1 * vx + C_R2 * vx * vx) * (10 * vx).tanh()) / M,
    ]
  )


def _rk4(x, u):
  k1 = _continuous_dynamics(x, u)
  k2 = _continuous_dynamics(x + DT / 2 * k1, u)
  k3 = _continuous_dynamics(x + DT / 2 * k2, u)
  k4 = _continuous_dynamics(x + DT * k3, u)
  return x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@al.function("tracking_eq_initial", {"z": NZ, "p": NX})
def eq_initial(z, p):
  return {"eq": z[:NX] - p[:NX]}


@al.function("tracking_eq_interstage", {"z": NZ, "znext": NZ, "p": NX})
def eq_interstage(z, znext, p):
  return {"eq": _rk4(z[:NX], z[NX : NX + NU]) - znext[:NX]}


def tracking_eq_function(horizon: int) -> al.Function:
  z = al.sym("z", NZ * (horizon + 1))
  p = al.sym("p", NX * (horizon + 1), diff=False)
  parts = [eq_initial.call([z[:NZ], p[:NX]])[0]]
  for i in range(horizon):
    zi = z[i * NZ : (i + 1) * NZ]
    znext = z[(i + 1) * NZ : (i + 2) * NZ]
    pi = p[(i + 1) * NX : (i + 2) * NX]
    parts.append(eq_interstage.call([zi, znext, pi])[0])
  return al.Function(f"tracking_eq_N{horizon}", [z, p], [al.concat(parts)], ["z", "p"], ["eq"])


def tracking_eq_function_map(horizon: int) -> al.Function:
  """Same semantics as ``tracking_eq_function`` but using ``al.scan`` for the interstage residuals.

  This lets benchmarks measure the impact of loop-preserving lowering directly against the
  per-stage unrolled construction.
  """
  z = al.sym("z", NZ * (horizon + 1))
  p = al.sym("p", NX * (horizon + 1), diff=False)
  initial = eq_initial.call([z[:NZ], p[:NX]])[0]
  mapped = al.scan(eq_interstage, length=horizon, inputs={"z": (z, 0, NZ), "znext": (z, NZ, NZ), "p": (p, NX, NX)})
  return al.Function(f"tracking_eq_map_N{horizon}", [z, p], [al.concat([initial, mapped])], ["z", "p"], ["eq"])


def ca_tracking_eq_jac(horizon: int, name: str = "tracking_eq_jac", sym_t=None):
  import casadi

  sym_t = casadi.SX if sym_t is None else sym_t
  z = sym_t.sym("z", NZ * (horizon + 1))
  p = sym_t.sym("p", NX * (horizon + 1))

  def ca_cont(x, u):
    phi, v = x[2], x[3]
    throttle, delta = u[0], u[1]
    beta = 0.5 * delta
    vx = v * casadi.cos(beta)
    lr = 0.5 * WHEELBASE
    return casadi.vertcat(
      v * casadi.cos(phi + beta),
      v * casadi.sin(phi + beta),
      v * casadi.sin(beta) / lr,
      (C_M0 * throttle - (C_R0 + C_R1 * vx + C_R2 * vx * vx) * casadi.tanh(10 * vx)) / M,
    )

  def ca_rk4(x, u):
    k1 = ca_cont(x, u)
    k2 = ca_cont(x + DT / 2 * k1, u)
    k3 = ca_cont(x + DT / 2 * k2, u)
    k4 = ca_cont(x + DT * k3, u)
    return x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  parts = [z[:NX] - p[:NX]]
  for i in range(horizon):
    zi = z[i * NZ : (i + 1) * NZ]
    znext = z[(i + 1) * NZ : (i + 2) * NZ]
    parts.append(ca_rk4(zi[:NX], zi[NX : NX + NU]) - znext[:NX])
  eq = casadi.vertcat(*parts)
  return casadi.Function(name, [z, p], [casadi.jacobian(eq, z)])


def tracking_eq_sparse_metrics(horizon: int, *, render_source: bool = False) -> dict[str, float | int]:
  t0 = time.perf_counter()
  fn = tracking_eq_function(horizon)
  build_ms = (time.perf_counter() - t0) * 1000.0

  base_nodes = len(topo(fn.outputs))
  sparsity = al.jacobian_sparsity(fn.outputs[0], fn.inputs[0])
  colors = al.column_coloring(sparsity)

  t0 = time.perf_counter()
  sj = al.sparse_jacobian_colored(fn.outputs[0], fn.inputs[0])
  ad_ms = (time.perf_counter() - t0) * 1000.0
  spjf = al.Function(f"tracking_eq_N{horizon}_spjac_colored", [fn.inputs[0]], [sj.values], ["z"], ["spjac_eq_z"], [sj.sparsity])
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
