from __future__ import annotations

import json
import math
import os
import tempfile
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

import alloy as al
from alloy.codegen.c import render_c_source

casadi = pytest.importorskip("casadi")
if TYPE_CHECKING:
  import casadi

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


def _ca_tracking_eq_jac(horizon: int, name: str = "tracking_eq_jac", sym_t=casadi.SX):
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

  base_nodes = len(fn.tape().instructions)
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
    "sparse_tape_nodes": len(spjf.tape().instructions),
    "sparsity_nnz": sparsity.nnz,
    "colors": max(colors) + 1 if colors else 0,
    "function_build_ms": build_ms,
    "colored_ad_ms": ad_ms,
    "source_bytes": source_bytes,
  }


@pytest.mark.parametrize("horizon", [1, 2])
def test_tracking_eq_jacobian_matches_casadi(horizon: int) -> None:
  fn = tracking_eq_function(horizon)
  jf = al.jacobian(fn, "z", "eq")
  ca_jf = _ca_tracking_eq_jac(horizon)

  rng = np.random.default_rng(0)
  zv = rng.normal(size=NZ * (horizon + 1))
  pv = rng.normal(size=NX * (horizon + 1))

  np.testing.assert_allclose(jf(zv), np.array(ca_jf(zv, pv)), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("horizon", [1, 2])
def test_tracking_eq_sparse_jacobian_metadata_matches_dense(horizon: int) -> None:
  fn = tracking_eq_function(horizon)
  spjf = al.spjacobian(fn, "z", "eq")
  jf = al.jacobian(fn, "z", "eq")

  rng = np.random.default_rng(1)
  zv = rng.normal(size=NZ * (horizon + 1))

  sparsity = spjf.output_sparsities[0]
  assert sparsity is not None
  dense = jf(zv)
  compact = spjf(zv)
  assert isinstance(dense, np.ndarray)
  assert isinstance(compact, np.ndarray)
  flat = np.asarray(sparsity.rows) * dense.shape[1] + np.asarray(sparsity.cols)
  np.testing.assert_allclose(compact, np.ravel(dense)[flat])
  assert sparsity.nnz < dense.size


@pytest.mark.parametrize("horizon", [1, 2])
def test_tracking_eq_colored_sparse_jacobian_matches_reference_path(horizon: int) -> None:
  fn = tracking_eq_function(horizon)
  colored = al.sparse_jacobian_colored(fn.outputs[0], fn.inputs[0])
  reference = al.sparse_jacobian_reference(fn.outputs[0], fn.inputs[0])
  colored_fn = al.Function("tracking_spjac_colored", [fn.inputs[0]], [colored.values], ["z"], ["colored"])
  reference_fn = al.Function("tracking_spjac_reference", [fn.inputs[0]], [reference.values], ["z"], ["reference"])
  compare = al.Function("tracking_spjac_compare", [fn.inputs[0]], [colored.values, reference.values], ["z"], ["colored", "reference"])

  rng = np.random.default_rng(2)
  zv = rng.normal(size=NZ * (horizon + 1))

  assert colored.sparsity == reference.sparsity
  assert len(colored_fn.tape().instructions) < len(reference_fn.tape().instructions)
  colored_values, reference_values = compare(zv)
  np.testing.assert_allclose(colored_values, reference_values, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(
  os.environ.get("ALLOY_TRACKING_SWEEP") != "1", reason="set ALLOY_TRACKING_SWEEP=1 to run the opt-in tracking sparse-Jacobian sweep"
)
@pytest.mark.parametrize("horizon", [1, 2, 5, 10])
def test_tracking_eq_opt_in_sparse_jacobian_sweep(horizon: int, record_property) -> None:
  metrics = tracking_eq_sparse_metrics(horizon, render_source=True)
  for name, value in metrics.items():
    record_property(name, value)

  fn = tracking_eq_function(horizon)
  spjf = al.spjacobian(fn, "z", "eq")
  ca_jf = _ca_tracking_eq_jac(horizon)

  rng = np.random.default_rng(3)
  zv = rng.normal(size=NZ * (horizon + 1))
  pv = rng.normal(size=NX * (horizon + 1))
  ca_dense = np.asarray(ca_jf(zv, pv))
  sparsity = spjf.output_sparsities[0]
  assert sparsity is not None
  flat = np.asarray(sparsity.rows) * ca_dense.shape[1] + np.asarray(sparsity.cols)
  np.testing.assert_allclose(spjf(zv), ca_dense.reshape(-1)[flat], rtol=1e-10, atol=1e-10)
  assert metrics["sparsity_nnz"] == sparsity.nnz
  assert 0 < metrics["colors"] <= NZ * (horizon + 1)
  assert metrics["source_bytes"] > 0


@pytest.mark.skipif(os.environ.get("ALLOY_GBENCH") != "1", reason="set ALLOY_GBENCH=1 to run the Google Benchmark Python-dispatch microbenchmark")
def test_tracking_eq_jac_python_gbench(record_property) -> None:
  """Reproduce the C++ ``BM_AlloyTrackingEqJacN`` cell with the google-benchmark Python bindings.

  The scalability sweep (`benchmarks/scalability_sweep.py`) calls the generated C kernel from a
  tight C++ loop, so it measures *codegen quality* with zero Python in the timed region. This test
  instead times ``Function.__call__`` — the realistic end-to-end Python dispatch (ctypes FFI +
  workspace/output allocation + the kernel). The two answer different questions; see
  docs/scalability.md ("Python-driven benchmarking") for the gap and why the CasADi comparison must
  stay on the C++ harness. The gbench loop itself adds ~14 ns/iter, so the number is faithfully
  the dispatch cost.
  """
  gb = pytest.importorskip("google_benchmark")
  from google_benchmark import _benchmark

  horizon = 10
  fn = tracking_eq_function_map(horizon)
  spjf = al.spjacobian(fn, "z", "eq")
  sparsity = spjf.output_sparsities[0]
  assert sparsity is not None

  rng = np.random.default_rng(7)
  zv = rng.normal(scale=0.4, size=NZ * (horizon + 1))
  pv = rng.normal(scale=0.4, size=NX * (horizon + 1))

  # Correctness against the CasADi dense reference, outside the timed loop.
  ca_dense = np.asarray(_ca_tracking_eq_jac(horizon)(zv, pv))
  flat = np.asarray(sparsity.rows) * ca_dense.shape[1] + np.asarray(sparsity.cols)
  np.testing.assert_allclose(spjf(zv), ca_dense.reshape(-1)[flat], rtol=1e-10, atol=1e-10)

  bench_name = f"BM_AlloyTrackingEqJacN{horizon}_pydispatch"

  @gb.register(name=bench_name)
  def _bench(state: gb.State) -> None:
    z = zv  # captured; setup before the loop is not timed
    while state:
      spjf(z)
    state.items_processed = state.iterations * sparsity.nnz

  try:
    with tempfile.TemporaryDirectory() as tmp:
      out = os.path.join(tmp, "result.json")
      _benchmark.Initialize(
        ["alloy", "--benchmark_min_time=0.1s", f"--benchmark_filter={bench_name}", f"--benchmark_out={out}", "--benchmark_out_format=json"]
      )
      _benchmark.RunSpecifiedBenchmarks()
      with open(out) as fp:
        result = json.load(fp)
  finally:
    _benchmark.ClearRegisteredBenchmarks()

  entry = next(b for b in result["benchmarks"] if b["name"] == bench_name)
  cpu_time = float(entry["cpu_time"])
  record_property("python_dispatch_cpu_time", cpu_time)
  record_property("time_unit", entry["time_unit"])
  record_property("iterations", entry["iterations"])
  record_property("nnz", sparsity.nnz)
  assert cpu_time > 0 and math.isfinite(cpu_time)
