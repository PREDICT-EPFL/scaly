"""Implicit Runge-Kutta maps: order, stiff stability, Newton variants, and derivatives by the
implicit function theorem against finite differences and against AD through converged iterations."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import scaly as sc
from scaly import integrators as si
from scaly import linalg
from scaly.integrators.implicit import eigen_split
from scaly.ir.expr import ExprOp, topo
from scaly.linalg.ops.dense import LU

A = np.array([[0.0, 1.0], [-4.0, -0.3]])
B = np.array([[0.0], [1.0]])
MU, LAM = 3.0, -1e6


@sc.function(2, 1, output="xdot")
def damped(x, u):
  return sc.const(A) @ x + sc.const(B) @ u


@sc.function(2, 1, output="xdot")
def van_der_pol(x, u):
  return sc.stack([x[1], MU * (1 - x[0] * x[0]) * x[1] - x[0] + u[0]])


@sc.function(2, output="xdot")
def prothero_robinson(x):
  """``y' = lambda (y - cos t) - sin t`` with time as the second state: ``y = cos t`` is the smooth
  solution, and every other one is drawn to it at rate ``lambda``."""
  return sc.stack([LAM * (x[0] - x[1].cos()) - x[1].sin(), sc.const(1.0) + 0.0 * x[0]])


def vdp_np(t, x, u=0.4):
  return [x[1], MU * (1 - x[0] ** 2) * x[1] - x[0] + u]


METHODS = [
  ("gauss_legendre", 1),
  ("gauss_legendre", 2),
  ("gauss_legendre", 3),
  ("radau_iia", 1),
  ("radau_iia", 2),
  ("radau_iia", 3),
  ("lobatto_iiia", 2),
  ("lobatto_iiia", 3),
  ("lobatto_iiic", 2),
  ("lobatto_iiic", 3),
  ("sdirk2", None),
  ("sdirk3", None),
]


@pytest.mark.parametrize(("method", "stages"), METHODS)
def test_linear_system_converges_at_the_method_order(method: str, stages: int | None) -> None:
  tab = si.tableau(method, stages)
  step = si.implicit(damped, method, stages=stages, dt=None, newton_iters=1)  # linear: one Newton step is exact
  x0, u, T = np.array([1.0, 0.5]), np.array([0.7]), 1.0
  big = expm(np.block([[A, B], [np.zeros((1, 3))]]) * T)
  exact = big[:2, :2] @ x0 + big[:2, 2:] @ u
  errors = []
  sizes = (32, 64) if tab.order <= 2 else (8, 16) if tab.order <= 4 else (2, 4)
  for n in sizes:
    x = x0
    for _ in range(n):
      x = step(x, u, np.array(T / n))
    errors.append(float(np.abs(x - exact).max()))
  assert abs(np.log2(errors[0] / errors[1]) - tab.order) < 0.3, errors


def test_radau_iia3_agrees_with_scipy_radau_on_van_der_pol() -> None:
  T, n, x0 = 1.0, 50, np.array([2.0, 0.0])
  exact = solve_ivp(vdp_np, (0, T), x0, method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1]
  scipy_radau = solve_ivp(vdp_np, (0, T), x0, method="Radau", rtol=1e-10, atol=1e-12).y[:, -1]
  ours = si.implicit(van_der_pol, "radau_iia", stages=3, dt=T, steps=n, tol=1e-13)(x0, np.array([0.4]))
  assert np.abs(ours - exact).max() < 1e-8 and np.abs(scipy_radau - exact).max() < 1e-8


@pytest.mark.parametrize(("method", "stages"), [("radau_iia", 3), ("radau_iia", 1), ("lobatto_iiic", 2), ("sdirk2", None), ("sdirk3", None)])
def test_l_stable_methods_follow_a_stiff_problem_at_huge_steps(method: str, stages: int | None) -> None:
  h = 0.02  # h * lambda = -2e4
  step = si.implicit(prothero_robinson, method, stages=stages, dt=h, newton_iters=2)
  x = np.array([1.0, 0.0])
  for _ in range(50):
    x = step(x)
  assert abs(x[0] - np.cos(x[1])) < 1e-7
  explicit = si.rk4(prothero_robinson, dt=h)
  x = np.array([1.0, 0.0])
  for _ in range(3):
    x = explicit(x)
  assert not abs(x[0]) < 1e10


VARIANTS = [
  {"method": "radau_iia", "stages": 3},
  {"method": "gauss_legendre", "stages": 2, "newton": "full"},
  {"method": "lobatto_iiia", "stages": 3},
  {"method": "sdirk3"},
  {"method": "sdirk2", "newton": "full"},
  {"method": "radau_iia", "stages": 2, "tol": 1e-12},
  {"method": "sdirk3", "tol": 1e-12, "newton": "full"},
  {"method": "radau_iia", "stages": 2, "steps": si.UNROLL_STEPS + 1},
]


def _fd(fn, args: list[np.ndarray], k: int, eps: float = 1e-6) -> np.ndarray:
  cols = []
  for i in range(args[k].size):
    plus, minus = [a.copy() for a in args], [a.copy() for a in args]
    plus[k].flat[i] += eps
    minus[k].flat[i] -= eps
    cols.append((np.asarray(fn(*plus)) - np.asarray(fn(*minus))) / (2 * eps))
  return np.stack(cols, axis=-1).reshape(-1, args[k].size)


@pytest.mark.parametrize("kw", VARIANTS, ids=lambda kw: "-".join(str(v) for v in kw.values()))
def test_derivatives_in_every_mode_match_finite_differences(kw: dict) -> None:
  converged = {"newton_iters": 8} if "tol" not in kw else {}
  step = si.implicit(van_der_pol, dt=None, **converged, **kw)
  args = [np.array([1.5, -0.3]), np.array([0.4]), np.array(0.1)]
  for k, wrt in enumerate(("x", "u", "dt")):
    np.testing.assert_allclose(sc.jacobian(step, wrt)(*args), _fd(step, args, k), rtol=1e-6, atol=1e-8)

  @sc.function(2, 1, (), output="c", name=f"cost_{step.name}")
  def cost(x, u, h):
    y = step(x, u, h)
    return (y * y).sum() + y[0] * u[0] * h

  for k, wrt in enumerate(("x", "u", "h")):
    grad = sc.gradient(cost, wrt)
    np.testing.assert_allclose(grad(*args), _fd(lambda *a: np.array([cost(*a)]), args, k)[0], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sc.hessian(cost, wrt)(*args), _fd(grad, args, k), rtol=1e-5, atol=1e-7)
  # Forward over forward differentiates the forward rule itself, which the level-1 rules serve.
  jac_x = sc.jacobian(step, "x")
  np.testing.assert_allclose(sc.jacobian(jac_x, "x")(*args), _fd(jac_x, args, 0), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("method", ["radau_iia", "sdirk3"])
def test_a_folded_interval_with_substeps_differentiates_at_the_substep(method: str) -> None:
  """The rules are built once per map with the substep's length folded in: ``dt / steps``."""
  step = si.implicit(van_der_pol, method, stages=3 if method == "radau_iia" else None, dt=0.2, steps=3, newton_iters=8)
  args = [np.array([1.5, -0.3]), np.array([0.4])]
  for k, wrt in enumerate(("x", "u")):
    np.testing.assert_allclose(sc.jacobian(step, wrt)(*args), _fd(step, args, k), rtol=1e-6, atol=1e-8)


def test_the_implicit_derivative_is_the_limit_of_differentiating_newton() -> None:
  """Backward Euler written out by hand: full Newton unrolled eight times from ``f(x)``, each solve
  differentiable, so AD goes through every iteration; at convergence its derivative is the one the
  implicit function theorem gives without the iterations."""
  h = 0.1

  @sc.function(2, 1, output="xnext", name="by_hand")
  def by_hand(x, u):
    k = van_der_pol(x, u)
    for _ in range(8):
      y = x + h * k
      jac = sc.jacobian(van_der_pol(y, u), y)
      k = k - linalg.solve(sc.const(np.eye(2)) - h * jac, k - van_der_pol(y, u), assume="gen")
    return x + h * k

  step = si.implicit(van_der_pol, "backward_euler", dt=h, newton_iters=8, newton="full")
  x, u = np.array([1.5, -0.3]), np.array([0.4])
  np.testing.assert_allclose(step(x, u), by_hand(x, u), rtol=1e-14, atol=1e-15)
  for wrt in ("x", "u"):
    np.testing.assert_allclose(sc.jacobian(step, wrt)(x, u), sc.jacobian(by_hand, wrt)(x, u), rtol=1e-10, atol=1e-12)


def test_fixed_iterations_and_a_tolerance_find_the_same_stages() -> None:
  x, u = np.array([1.5, -0.3]), np.array([0.4])
  for method, stages in (("radau_iia", 3), ("sdirk3", None)):
    fixed = si.implicit(van_der_pol, method, stages=stages, dt=0.1, newton_iters=10, name=f"fixed_{method}")
    tolerance = si.implicit(van_der_pol, method, stages=stages, dt=0.1, tol=1e-14, max_iter=40, name=f"tol_{method}")
    np.testing.assert_allclose(fixed(x, u), tolerance(x, u), rtol=1e-13, atol=1e-14)
    few = si.implicit(van_der_pol, method, stages=stages, dt=0.1, newton_iters=1, name=f"few_{method}")
    assert np.abs(few(x, u) - tolerance(x, u)).max() > 1e-8  # one simplified step has not converged


def _lu_orders(fn) -> list[int]:
  return sorted({e.shape[1] for e in topo(fn.outputs) if e.op == LU})


def test_structure_of_the_generated_graph() -> None:
  radau = si.implicit(van_der_pol, "radau_iia", stages=3, dt=0.1)
  sdirk = si.implicit(van_der_pol, "sdirk3", dt=0.1)
  assert _lu_orders(radau) == [2, 4] and _lu_orders(sdirk) == [2]  # split by the eigenvalues of A, or a system per stage
  assert sum(e.op == LU for e in topo(sdirk.outputs)) == 1  # simplified: every stage shares I - h g J(x)
  assert _lu_orders(si.implicit(van_der_pol, "radau_iia", stages=3, dt=0.1, newton="full")) == [6]
  assert ExprOp.WHILE not in {e.op for e in topo(radau.outputs)}
  looped = si.implicit(van_der_pol, "radau_iia", stages=2, dt=0.1, tol=1e-10)
  assert ExprOp.WHILE in {e.op for e in topo(looped.outputs)}
  assert radau.name == "van_der_pol_radau_iia3" and sdirk.name == "van_der_pol_sdirk3"


def test_a_template_model_gives_a_template_map() -> None:
  @sc.function(output="xdot")
  def decay(x, k):
    return -k * x * x

  step = si.implicit(decay, "radau_iia", stages=2, dt=0.1, steps=10, tol=1e-14)
  x, k = np.array([1.0, 2.0, 0.5]), np.array(1.5)
  y = step(x, k)
  np.testing.assert_allclose(y, x / (1 + k * 0.1 * x), rtol=1e-6)  # the exact solution, to the method's accuracy
  assert list(step.instances) == ["decay__3_s_radau_iia2"]


def test_refusals() -> None:
  with pytest.raises(ValueError, match="rk4 is an explicit method"):
    si.implicit(damped, "rk4", dt=0.1)
  with pytest.raises(ValueError, match="newton must be"):
    si.implicit(damped, "sdirk2", dt=0.1, newton="quasi")  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match="newton_iters must be a positive integer"):
    si.implicit(damped, "sdirk2", dt=0.1, newton_iters=0)
  with pytest.raises(ValueError, match="tol must be positive"):
    si.implicit(damped, "sdirk2", dt=0.1, tol=0.0)
  with pytest.raises(ValueError, match="is a family"):
    si.implicit(damped, "gauss_legendre", dt=0.1)


def _block_diagonal(blocks: list[tuple[float, ...]]) -> np.ndarray:
  size = sum(len(b) for b in blocks)
  lam, row = np.zeros((size, size)), 0
  for b in blocks:
    if len(b) == 1:
      lam[row, row] = b[0]
    else:
      lam[row : row + 2, row : row + 2] = [[b[0], b[1]], [-b[1], b[0]]]
    row += len(b)
  return lam


@pytest.mark.parametrize(
  ("method", "stages", "kinds"),
  [
    ("gauss_legendre", 2, [2]),
    ("gauss_legendre", 3, [1, 2]),
    ("radau_iia", 3, [1, 2]),
    ("radau_iia", 5, [1, 2, 2]),
    ("lobatto_iiia", 3, [1, 2]),
    ("lobatto_iiic", 4, [2, 2]),
  ],
)
def test_the_stage_coefficients_split_into_real_blocks(method: str, stages: int, kinds: list[int]) -> None:
  a = si.tableau(method, stages).a
  split = eigen_split(a)
  assert split is not None
  t, t_inv, blocks = split
  assert sorted(len(b) for b in blocks) == kinds
  np.testing.assert_allclose(t @ _block_diagonal(blocks) @ t_inv, a, atol=1e-13)
  np.testing.assert_allclose(t @ t_inv, np.eye(stages), atol=1e-12)
  if method == "lobatto_iiia":
    assert any(len(b) == 1 and abs(b[0]) < 1e-14 for b in blocks)  # the explicit first stage


def _jordan_tableau() -> si.Tableau:
  """A first-order implicit method whose ``A`` is similar to a Jordan block, so no eigenvector basis exists."""
  t = np.array([[1.0, 1.0], [1.0, -1.0]])
  a = t @ np.array([[0.5, 1.0], [0.0, 0.5]]) @ np.linalg.inv(t)
  return si.Tableau(a, np.array([0.5, 0.5]), a.sum(axis=1), 1, "jordan")


def test_a_defective_matrix_is_not_split_and_newton_factors_the_whole_system() -> None:
  tab = _jordan_tableau()
  assert eigen_split(tab.a) is None and not tab.diagonally_implicit and not tab.explicit
  simplified = si.implicit(van_der_pol, tab, dt=0.05, newton_iters=12, name="jordan_simplified")
  assert _lu_orders(simplified) == [4]
  converged = si.implicit(van_der_pol, tab, dt=0.05, tol=1e-14, newton="full", name="jordan_full")
  x, u = np.array([1.5, -0.3]), np.array([0.4])
  np.testing.assert_allclose(simplified(x, u), converged(x, u), rtol=1e-12, atol=1e-13)


def test_the_split_and_the_whole_system_iterate_alike(monkeypatch: pytest.MonkeyPatch) -> None:
  """Simplified Newton through the eigenvalue split and through one factorization of the whole
  matrix take the same iterates, up to rounding: one unconverged iteration shows it."""
  x, u = np.array([1.5, -0.3]), np.array([0.4])
  split = si.implicit(van_der_pol, "radau_iia", stages=3, dt=0.1, newton_iters=1, name="split_once")
  monkeypatch.setattr(importlib.import_module("scaly.integrators.implicit"), "eigen_split", lambda a: None)
  whole = si.implicit(van_der_pol, "radau_iia", stages=3, dt=0.1, newton_iters=1, name="whole_once")
  assert _lu_orders(whole) == [6] and _lu_orders(split) == [2, 4]
  np.testing.assert_allclose(split(x, u), whole(x, u), rtol=1e-13, atol=1e-14)
