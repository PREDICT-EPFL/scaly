import numpy as np
import pytest

import alloy as al
from alloy.expr import topo
from benchmarks.problems.tracking_nmpc import NX, NZ, TrackingParams, ca_tracking_eq_jac, n_param, tracking_eq_function

pytest.importorskip("casadi")


@pytest.mark.parametrize("horizon", [1, 2])
def test_tracking_eq_jacobian_matches_casadi(horizon: int) -> None:
  fn = tracking_eq_function(horizon)
  jf = fn.factory(f"tracking_eq_jac_N{horizon}", ["z", "p"], ["jac:eq:z"])
  ca_jf = ca_tracking_eq_jac(horizon)

  rng = np.random.default_rng(0)
  zv = rng.normal(size=NZ * (horizon + 1))
  pv = np.concatenate([rng.normal(size=NX * (horizon + 1)), TrackingParams().array()])

  np.testing.assert_allclose(jf(zv, pv), np.array(ca_jf(zv, pv)), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("horizon", [1, 2])
def test_tracking_eq_sparse_jacobian_metadata_matches_dense(horizon: int) -> None:
  fn = tracking_eq_function(horizon)
  spjf = fn.factory(f"tracking_eq_spjac_N{horizon}", ["z", "p"], ["spjac:eq:z"])
  jf = fn.factory(f"tracking_eq_jac_N{horizon}", ["z", "p"], ["jac:eq:z"])

  rng = np.random.default_rng(1)
  zv = rng.normal(size=NZ * (horizon + 1))
  pv = np.concatenate([rng.normal(size=NX * (horizon + 1)), TrackingParams().array()])

  sparsity = spjf.output_sparsities[0]
  assert sparsity is not None
  dense = jf(zv, pv)
  compact = spjf(zv, pv)
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
  colored_fn = al.Function("tracking_spjac_colored", fn.inputs, [colored.values], fn.input_names, ["colored"])
  reference_fn = al.Function("tracking_spjac_reference", fn.inputs, [reference.values], fn.input_names, ["reference"])
  compare = al.Function("tracking_spjac_compare", fn.inputs, [colored.values, reference.values], fn.input_names, ["colored", "reference"])

  rng = np.random.default_rng(2)
  zv = rng.normal(size=NZ * (horizon + 1))
  pv = np.concatenate([rng.normal(size=NX * (horizon + 1)), TrackingParams().array()])

  assert colored.sparsity == reference.sparsity
  assert len(topo(colored_fn.outputs)) < len(topo(reference_fn.outputs))
  colored_values, reference_values = compare(zv, pv)
  np.testing.assert_allclose(colored_values, reference_values, rtol=1e-10, atol=1e-10)


def test_tracking_eq_matches_numpy_at_asymmetric_params() -> None:
  # independent NumPy reference with a distinct value per parameter entry pins the tail ordering [wheelbase, dt, mass, c_m0, c_r0, c_r1, c_r2]
  horizon = 2
  params = TrackingParams(wheelbase=2.9, dt=0.13, mass=0.77, c_m0=13.0, c_r0=0.21, c_r1=0.033, c_r2=0.0047)
  rng = np.random.default_rng(4)
  zv = rng.normal(size=NZ * (horizon + 1))
  pv = np.concatenate([rng.normal(size=NX * (horizon + 1)), params.array()])

  def ode(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    beta = 0.5 * u[1]
    vx = x[3] * np.cos(beta)
    resist = (params.c_r0 + params.c_r1 * vx + params.c_r2 * vx * vx) * np.tanh(10 * vx)
    return np.array(
      [
        x[3] * np.cos(x[2] + beta),
        x[3] * np.sin(x[2] + beta),
        x[3] * np.sin(beta) / (0.5 * params.wheelbase),
        (params.c_m0 * u[0] - resist) / params.mass,
      ]
    )

  def rk4(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    k1, h = ode(x, u), params.dt
    k2 = ode(x + h / 2 * k1, u)
    k3 = ode(x + h / 2 * k2, u)
    k4 = ode(x + h * k3, u)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  parts = [zv[:NX] - pv[:NX]]
  for i in range(horizon):
    zi, znext = zv[i * NZ : (i + 1) * NZ], zv[(i + 1) * NZ : (i + 2) * NZ]
    parts.append(rk4(zi[:NX], zi[NX:NZ]) - znext[:NX])
  np.testing.assert_allclose(np.asarray(tracking_eq_function(horizon)(zv, pv)).reshape(-1), np.concatenate(parts), rtol=1e-12, atol=1e-12)


def test_tracking_default_parameters_preserve_legacy_jacobian() -> None:
  horizon = 1
  fn = tracking_eq_function(horizon).factory("tracking_default_params_jac", ["z", "p"], ["jac:eq:z"])
  rng = np.random.default_rng(0)
  zv = rng.normal(size=NZ * (horizon + 1))
  pv = np.zeros(n_param(horizon))
  pv[-TrackingParams().array().size :] = TrackingParams().array()
  expected = np.zeros((8, 12))
  expected[:4, :4] = np.eye(4)
  expected[4:, :10] = np.array(
    [
      [1.0, 0.0, -0.002272942123027472, 0.033783111130420075, 0.0026799332174869705, -0.0011470634374039701, -1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0021078026196618242, 0.036566092555961556, 0.0029007177282505143, 0.001065637187688752, 0.0, -1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.059678307139590103, 0.0047341484385286705, 0.010166136544295384, 0.0, 0.0, -1.0, 0.0],
      [0.0, 0.0, 0.0, 0.99004032583145973, 0.15760709298302963, 0.000051244048497743486, 0.0, 0.0, 0.0, -1.0],
    ]
  )
  np.testing.assert_allclose(fn(zv, pv), expected, rtol=1e-12, atol=1e-12)
