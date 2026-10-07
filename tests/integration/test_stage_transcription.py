"""Core AD/codegen coverage for a stage-transcribed OCP constraint, self-contained.

A 4-state / 2-control kinematic bicycle discretised with RK4 and transcribed into one
equality residual per horizon. The shape matters for Scaly: scoped stage functions
composed through ``Function.call``, ``sc.concat`` of the per-stage pieces, a symbolic
parameter tail read by every stage, and a `cos`/`sin`/`tanh`/divide mix inside the
integrator. It is where structured sparse Jacobians, column coloring, and derivative
factories all meet.

This is a *minimal reproduction* of a form the race-car benchmark surfaced, deliberately
copied rather than imported: per `AGENTS.md` the pytest suite covers Scaly's core and must
not depend on a benchmark problem, so retiring or reshaping that problem cannot silently
drop this coverage. The benchmark keeps its own formulation gates in
``benchmarks/problems/race_cars/checks.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from scaly.function.model import as_concrete
import scaly as sc
from typing import Any, cast
from scaly.ir.expr import topo

NX, NU, NZ = 4, 2, 6
N_PARAMS = 7
# [wheelbase, dt, mass, c_m0, c_r0, c_r1, c_r2] — one distinct value each, so a permuted
# tail cannot pass by coincidence
PARAMS = np.array([1.5706, 0.05, 230.0, 4.95, 297.03, 16.665, 0.6784])


def n_param(horizon: int) -> int:
  return NX * (horizon + 1) + N_PARAMS


def _ode(x: sc.Expr, u: sc.Expr, params: sc.Expr) -> sc.Expr:
  wheelbase, _, mass, c_m0, c_r0, c_r1, c_r2 = [params[i] for i in range(N_PARAMS)]
  phi, v = x[2], x[3]
  beta = 0.5 * u[1]
  vx = v * beta.cos()
  return sc.stack(
    [
      v * (phi + beta).cos(),
      v * (phi + beta).sin(),
      v * beta.sin() / (0.5 * wheelbase),
      (c_m0 * u[0] - (c_r0 + c_r1 * vx + c_r2 * vx * vx) * (10 * vx).tanh()) / mass,
    ]
  )


def _rk4(x: sc.Expr, u: sc.Expr, params: sc.Expr) -> sc.Expr:
  dt = params[1]
  k1 = _ode(x, u, params)
  k2 = _ode(x + dt / 2 * k1, u, params)
  k3 = _ode(x + dt / 2 * k2, u, params)
  k4 = _ode(x + dt * k3, u, params)
  return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@sc.function(sc.group(sc.arg("z", NZ), sc.arg("p", NX)), outputs=sc.arg("eq"), name="bicycle_stage_initial")
def stage_initial(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
  z, p = inputs
  return z[:NX] - p[:NX]


@sc.function(sc.group(sc.arg("z", NZ), sc.arg("znext", NZ), sc.arg("params", N_PARAMS)), outputs=sc.arg("eq"), name="bicycle_stage_interstage")
def stage_interstage(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
  z, znext, params = inputs
  return _rk4(z[:NX], z[NX : NX + NU], params) - znext[:NX]


def bicycle_eq_function(horizon: int) -> sc.Function:
  """Per-stage unrolled transcription, built from `Function.call` on the stage functions."""

  @sc.function(
    sc.group(sc.arg("z", NZ * (horizon + 1)), sc.arg("p", sc.TensorType((n_param(horizon),), diff=False))),
    outputs=sc.arg("eq"),
    name=f"bicycle_eq_N{horizon}",
  )
  def fn(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = inputs
    params = p[NX * (horizon + 1) :]
    parts = [stage_initial((z[:NZ], p[:NX]))]
    for i in range(horizon):
      parts.append(stage_interstage((z[i * NZ : (i + 1) * NZ], z[(i + 1) * NZ : (i + 2) * NZ], params)))
    return sc.concat(parts)

  return fn


def bicycle_eq_function_vmap(horizon: int) -> sc.Function:
  """Same semantics through `sc.vmap`, so the loop survives into the rendered C."""

  @sc.function(
    sc.group(sc.arg("z", NZ * (horizon + 1)), sc.arg("p", sc.TensorType((n_param(horizon),), diff=False))),
    outputs=sc.arg("eq"),
    name=f"bicycle_eq_vmap_N{horizon}",
  )
  def fn(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = inputs
    initial = stage_initial((z[:NZ], p[:NX]))
    mapped = sc.vmap(stage_interstage, horizon)((sc.window(z, 0, NZ), sc.window(z, NZ, NZ), sc.window(p, NX * (horizon + 1), 0))).vec()
    return sc.concat([initial, mapped])

  return fn


def ca_bicycle_eq_jac(horizon: int, name: str = "ca_bicycle_eq_jac", sym_t=None):
  import casadi

  sym_t = casadi.SX if sym_t is None else sym_t
  z = sym_t.sym("z", NZ * (horizon + 1))
  p = sym_t.sym("p", n_param(horizon))
  params = p[NX * (horizon + 1) :]

  def ode(x, u):
    wheelbase, _, mass, c_m0, c_r0, c_r1, c_r2 = [params[i] for i in range(N_PARAMS)]
    phi, v = x[2], x[3]
    beta = 0.5 * u[1]
    vx = v * casadi.cos(beta)
    return casadi.vertcat(
      v * casadi.cos(phi + beta),
      v * casadi.sin(phi + beta),
      v * casadi.sin(beta) / (0.5 * wheelbase),
      (c_m0 * u[0] - (c_r0 + c_r1 * vx + c_r2 * vx * vx) * casadi.tanh(10 * vx)) / mass,
    )

  def rk4(x, u):
    dt = params[1]
    k1 = ode(x, u)
    k2 = ode(x + dt / 2 * k1, u)
    k3 = ode(x + dt / 2 * k2, u)
    k4 = ode(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  parts = [z[:NX] - p[:NX]]
  for i in range(horizon):
    zi = z[i * NZ : (i + 1) * NZ]
    parts.append(rk4(zi[:NX], zi[NX : NX + NU]) - z[(i + 1) * NZ : (i + 1) * NZ + NX])
  return casadi.Function(name, [z, p], [casadi.jacobian(casadi.vertcat(*parts), z)])


def _sample(horizon: int, seed: int, params: np.ndarray = PARAMS) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  return rng.normal(size=NZ * (horizon + 1)), np.concatenate([rng.normal(size=NX * (horizon + 1)), params])


def test_forward_residual_matches_numpy_at_asymmetric_parameters() -> None:
  """Pins forward semantics and the parameter-tail read: a distinct value per entry means any
  permutation of the tail changes the answer."""
  horizon = 2
  params = np.array([2.9, 0.13, 0.77, 13.0, 0.21, 0.033, 0.0047])
  zv, pv = _sample(horizon, 4, params)
  wheelbase, dt, mass, c_m0, c_r0, c_r1, c_r2 = params

  def ode(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    beta = 0.5 * u[1]
    vx = x[3] * np.cos(beta)
    resist = (c_r0 + c_r1 * vx + c_r2 * vx * vx) * np.tanh(10 * vx)
    return np.array(
      [
        x[3] * np.cos(x[2] + beta),
        x[3] * np.sin(x[2] + beta),
        x[3] * np.sin(beta) / (0.5 * wheelbase),
        (c_m0 * u[0] - resist) / mass,
      ]
    )

  def rk4(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    k1 = ode(x, u)
    k2 = ode(x + dt / 2 * k1, u)
    k3 = ode(x + dt / 2 * k2, u)
    k4 = ode(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  parts = [zv[:NX] - pv[:NX]]
  for i in range(horizon):
    zi, znext = zv[i * NZ : (i + 1) * NZ], zv[(i + 1) * NZ : (i + 2) * NZ]
    parts.append(rk4(zi[:NX], zi[NX:NZ]) - znext[:NX])
  got = np.asarray(bicycle_eq_function(horizon)((zv, pv))).reshape(-1)
  np.testing.assert_allclose(got, np.concatenate(parts), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("horizon", [1, 2])
def test_dense_jacobian_matches_casadi(horizon: int) -> None:
  fn = bicycle_eq_function(horizon)
  jf = fn.factory(f"bicycle_eq_jac_N{horizon}", ["z", "p"], [sc.factory.Jac("eq", "z")])
  zv, pv = _sample(horizon, 0)
  np.testing.assert_allclose(jf(*(zv, pv)), np.array(ca_bicycle_eq_jac(horizon)(zv, pv)), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("horizon", [1, 2])
def test_sparse_jacobian_metadata_matches_dense(horizon: int) -> None:
  fn = bicycle_eq_function(horizon)
  spjf = fn.factory(f"bicycle_eq_spjac_N{horizon}", ["z", "p"], [sc.factory.SpJac("eq", "z")])
  jf = fn.factory(f"bicycle_eq_jac_dense_N{horizon}", ["z", "p"], [sc.factory.Jac("eq", "z")])
  zv, pv = _sample(horizon, 1)
  sparsity = as_concrete(spjf).output_sparsities[0]
  assert sparsity is not None
  dense, compact = jf(*(zv, pv)), spjf(*(zv, pv))
  assert isinstance(dense, np.ndarray) and isinstance(compact, np.ndarray)
  flat = np.asarray(sparsity.rows) * dense.shape[1] + np.asarray(sparsity.cols)
  np.testing.assert_allclose(compact, np.ravel(dense)[flat])
  assert sparsity.nnz < dense.size


@pytest.mark.parametrize("horizon", [1, 2])
def test_colored_sparse_jacobian_matches_the_reference_path(horizon: int) -> None:
  fn = bicycle_eq_function(horizon)
  colored = sc.sparse_jacobian_colored(as_concrete(fn).outputs[0], as_concrete(fn).inputs[0])
  reference = sc.sparse_jacobian_reference(as_concrete(fn).outputs[0], as_concrete(fn).inputs[0])
  from scaly.ir.expr import substitute

  def rebound(expr: sc.Expr, inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    return substitute(expr, dict(zip(as_concrete(fn).inputs, inputs, strict=True)))

  @sc.function(cast(Any, as_concrete(fn).input_tree).parts[0], outputs=sc.arg("colored"), name="bicycle_spjac_colored")
  def colored_fn(inputs):
    return rebound(colored.values, inputs)

  @sc.function(cast(Any, as_concrete(fn).input_tree).parts[0], outputs=sc.arg("reference"), name="bicycle_spjac_reference")
  def reference_fn(inputs):
    return rebound(reference.values, inputs)

  @sc.function(cast(Any, as_concrete(fn).input_tree).parts[0], outputs=sc.group(sc.arg("colored"), sc.arg("reference")), name="bicycle_spjac_compare")
  def compare(inputs):
    return rebound(colored.values, inputs), rebound(reference.values, inputs)

  zv, pv = _sample(horizon, 2)
  assert colored.sparsity == reference.sparsity
  assert len(topo(as_concrete(colored_fn).outputs)) < len(topo(as_concrete(reference_fn).outputs))
  colored_values, reference_values = compare((zv, pv))
  np.testing.assert_allclose(colored_values, reference_values, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("horizon", [2, 3])
def test_vmap_transcription_matches_the_unrolled_one(horizon: int) -> None:
  """`sc.vmap` and the unrolled `Function.call` chain must agree on residual and Jacobian."""
  mapped, unrolled = bicycle_eq_function_vmap(horizon), bicycle_eq_function(horizon)
  zv, pv = _sample(horizon, 5)
  np.testing.assert_allclose(np.asarray(mapped((zv, pv))).reshape(-1), np.asarray(unrolled((zv, pv))).reshape(-1), rtol=1e-12, atol=1e-12)
  mapped_jac = mapped.factory(f"bicycle_vmap_jac_N{horizon}", ["z", "p"], [sc.factory.Jac("eq", "z")])
  unrolled_jac = unrolled.factory(f"bicycle_unrolled_jac_N{horizon}", ["z", "p"], [sc.factory.Jac("eq", "z")])
  np.testing.assert_allclose(mapped_jac(*(zv, pv)), unrolled_jac(*(zv, pv)), rtol=1e-10, atol=1e-10)
