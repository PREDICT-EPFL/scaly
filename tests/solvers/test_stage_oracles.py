"""Dense derivatives of mapped stage Functions against whole-problem derivatives."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scaly as sc
from scaly.function.model import as_concrete
from scaly.solvers import registry
from scaly.solvers.nlp import build_nlp


def _ocp(N=3, *, globals_=True):
  @sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("T", ()), sc.arg("gain", ()))
  def step(x, u, T, gain):
    return x + 0.1 * T * sc.stack([x[1].sin(), gain * u[0] - x[0] ** 3])

  @sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("ref", 2))
  def cost(x, u, ref):
    return sc.sumsqr(x - ref) + 0.2 * u[0] ** 4 + 0.1 * u[0] ** 2

  @sc.function(sc.arg("x", 2), sc.arg("T", ()), sc.arg("ref", 2))
  def end_cost(x, T, ref):
    return sc.sumsqr(x - ref) + (T - 1.0) ** 2

  @sc.function(sc.arg("x", 2), sc.arg("u", 1))
  def path(x, u):
    return sc.stack([x[0] * u[0], x[1].cos()])

  @sc.function(sc.arg("x", 2), sc.arg("T", ()))
  def terminal(x, T):
    return T + x[0] ** 2

  return sc.DiscreteOCP(
    N=N,
    dynamics=step,
    stage_cost=cost,
    terminal_cost=end_cost,
    path=(sc.StageConstraint(path, lo=[-0.5, 0.0], hi=[0.5, 1.0]),),
    terminal=(sc.StageConstraint(terminal, lo=0.5, hi=2.0),),
    x_bounds=(-3.0, 3.0),
    u_bounds=(-0.7, 0.7),
    stage_params=("ref",),
    global_vars=("T",) if globals_ else (),
    name="stage_test",
  )


@pytest.mark.parametrize("declared", [None, False])
def test_backend_without_stage_oracles_keeps_whole_problem_oracles(monkeypatch, declared):
  backend = SimpleNamespace(
    name="fatrop",
    kind="nlp",
    hess_triangle="lower",
    prepare_options=lambda options: ({}, options),
    header="stage.h",
    lib_stem="stage",
    link_flags=(),
    include_dir=lambda: Path("/nonexistent"),
    lib_dir=lambda: Path("/nonexistent"),
  )
  if declared is not None:
    backend.stage_oracles = declared
  monkeypatch.setattr(registry, "get_backend", lambda name: backend)
  problem = _ocp().stage_problem()
  desc = as_concrete(build_nlp(problem, registry.require_backend("fatrop", "nlp"), name="legacy", options=None)).descriptor
  assert desc.stage_oracles is None
  assert desc.jac is problem._nlp.jac
  assert desc.hess is problem._nlp.hess("lower")


def test_backend_can_request_stage_oracles_under_any_name(monkeypatch):
  backend = SimpleNamespace(
    name="mapped_backend",
    kind="nlp",
    hess_triangle="lower",
    stage_oracles=True,
    prepare_options=lambda options: ({}, options),
    header="stage.h",
    lib_stem="stage",
    link_flags=(),
    include_dir=lambda: Path("/nonexistent"),
    lib_dir=lambda: Path("/nonexistent"),
  )
  monkeypatch.setattr(registry, "get_backend", lambda name: backend)
  problem = _ocp().stage_problem()
  desc = as_concrete(build_nlp(problem, registry.require_backend("mapped_backend", "nlp"), name="mapped", options=None)).descriptor
  assert desc.stage_oracles is not None
  assert desc.grad is desc.stage_oracles.grad
  assert desc.jac is desc.hess is None
  assert "jac" not in problem._nlp.__dict__ and "hess_full" not in problem._nlp.__dict__


@pytest.mark.parametrize(
  "nx, nu",
  [
    ((3, 4, 3, 3), (1, 1, 1, 0)),
    ((4, 4, 4, 4), (1, 1, 1, 0)),
    ((3, 3, 3, 3), (1, 2, 1, 0)),
    ((3, 3, 3, 3), (1, 1, 1, 1)),
  ],
)
def test_attached_stage_functions_require_matching_layout_sizes(nx, nu):
  problem = _ocp().stage_problem()
  assert problem.stage_layout is not None
  layout = replace(problem.stage_layout, nx=nx, nu=nu)
  with pytest.raises(ValueError, match="stage Functions require"):
    replace(problem, stage_layout=layout)


def test_attached_stage_functions_require_a_layout():
  problem = _ocp().stage_problem()
  with pytest.raises(ValueError, match="stage Functions require"):
    replace(problem, stage_layout=None)


def test_dense_stage_derivatives_match_whole_problem():
  problem = _ocp().stage_problem()
  desc = as_concrete(sc.solver(problem, "fatrop").function).descriptor
  stage = desc.stage_oracles
  assert stage is not None and desc.jac is desc.hess is None
  assert "jac" not in problem._nlp.__dict__ and "hess_full" not in problem._nlp.__dict__
  reference = as_concrete(sc.solver(problem, "ipopt").function).descriptor
  layout = desc.stage_layout
  rng = np.random.default_rng(42)
  x = rng.normal(size=desc.n)
  params = (np.array([0.2, -0.1]), np.array(0.8), rng.normal(size=(4, 2)))
  lam, scale = rng.normal(size=desc.n_eq + desc.n_ineq), np.array(0.7)
  actual_jac = stage.jac(x, *params)
  expected_jac = reference.jac(x, params).toarray()
  dyn, initial, path, terminal = actual_jac
  np.testing.assert_allclose(initial, expected_jac[list(layout.eq[0]), :3], atol=1e-12)
  for k in range(3):
    np.testing.assert_allclose(dyn[k], expected_jac[list(layout.dynamics[k]), 4 * k : 4 * k + 4], atol=1e-12)
    np.testing.assert_allclose(path[k], expected_jac[list(layout.ineq[k]), 4 * k : 4 * k + 4], atol=1e-12)
  np.testing.assert_allclose(terminal, expected_jac[list(layout.ineq[-1]), -3:], atol=1e-12)
  np.testing.assert_allclose(desc.grad(x, *params), reference.grad(x, params), atol=1e-12)
  blocks, last = stage.hess(x, *params, scale, lam)
  expected_hess = reference.hess(x, params, (scale, lam)).toarray()
  expected_hess += np.tril(expected_hess, -1).T
  expected_grad = scale * reference.grad(x, params) + expected_jac.T @ lam
  assembled = np.zeros_like(expected_hess)
  for k, block in enumerate((*blocks, last)):
    start, width = 4 * k, block.shape[1]
    assembled[start : start + width, start : start + width] = block[:width]
    local_grad = expected_grad[start : start + width].copy()
    if k:
      local_grad[:3] += lam[list(layout.dynamics[k - 1])]
    np.testing.assert_allclose(block[-1], local_grad, atol=1e-11)
  np.testing.assert_allclose(assembled, expected_hess, atol=1e-11)

  def numpy_problem(v):
    z, last = v[:12].reshape(3, 4), v[12:]
    states = np.vstack([z[:, :3], last])
    initial, gain, ref = params
    cost = np.sum((z[:, :2] - ref[:3]) ** 2) + np.sum(0.2 * z[:, 3] ** 4 + 0.1 * z[:, 3] ** 2)
    cost += np.sum((last[:2] - ref[3]) ** 2) + (last[2] - 1.0) ** 2
    dynamics = []
    path = []
    for k in range(3):
      x0, x1, T, u = z[k]
      dynamics.extend([x0 + 0.1 * T * np.sin(x1), x1 + 0.1 * T * (gain * u - x0**3), T] - states[k + 1])
      path.extend([x0 * u, np.cos(x1)])
    g = np.concatenate([states[0, :2] - initial, dynamics, path, [last[2] + last[0] ** 2]])
    return cost, g

  steps = np.eye(x.size) * 1e-5
  numpy_jac = np.stack([(numpy_problem(x + step)[1] - numpy_problem(x - step)[1]) / 2e-5 for step in steps], axis=1)
  np.testing.assert_allclose(expected_jac, numpy_jac, atol=1e-8)

  def numpy_lag(v):
    f, g = numpy_problem(v)
    return scale * f + lam @ g

  steps *= 10
  numpy_hess = np.array(
    [[(numpy_lag(x + a + b) - numpy_lag(x + a - b) - numpy_lag(x - a + b) + numpy_lag(x - a - b)) / 4e-8 for b in steps] for a in steps]
  )
  np.testing.assert_allclose(assembled, numpy_hess, atol=1e-6)
