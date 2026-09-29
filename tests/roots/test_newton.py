"""``Newton`` and ``NewtonBisection`` against SciPy's solvers: every variant of the step, the statuses,
and the solution's derivatives by the implicit function theorem against finite differences."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize

import scaly as sc

N = 6


def broyden(z: sc.Expr, p: sc.Expr) -> sc.Expr:
  """Broyden's tridiagonal function, ``(3 - 2 z_i) z_i - z_{i-1} - 2 z_{i+1} + p_i``, shifted."""
  padded = sc.concat([sc.const(np.zeros(1)), z, sc.const(np.zeros(1))])
  return (3.0 - 2.0 * z) * z - padded[:-2] - 2.0 * padded[2:] + p


def broyden_np(z: np.ndarray, p: np.ndarray) -> np.ndarray:
  padded = np.concatenate([[0.0], z, [0.0]])
  return (3.0 - 2.0 * z) * z - padded[:-2] - 2.0 * padded[2:] + p


P = np.linspace(0.8, 1.2, N)
Z0 = -np.ones(N)


@sc.roots.root(vars=sc.L("z", N), params=sc.L("p", N), name="broyden")
def BROYDEN(z: sc.Expr, p: sc.Expr) -> sc.Expr:
  return broyden(z, p)


def reference(p: np.ndarray = P) -> np.ndarray:
  z = optimize.root(broyden_np, Z0, args=(p,), tol=1e-13).x
  assert np.abs(broyden_np(z, p)).max() < 1e-12
  return z


VARIANTS = {
  "tol": sc.roots.Newton(),
  "fixed": sc.roots.Newton(tol=None, max_iter=8),
  "simplified": sc.roots.Newton(simplified=True, max_iter=100),
  "simplified_fixed": sc.roots.Newton(simplified=True, tol=None, max_iter=40),
  "damped": sc.roots.Newton(max_step=0.1),
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_newton_finds_scipys_root(variant: str) -> None:
  method = VARIANTS[variant]
  solve = sc.roots.solver(BROYDEN, method, name=f"broyden_{variant}")
  z, info = solve(Z0, P)
  np.testing.assert_allclose(z, reference(), atol=1e-10)
  assert sc.Status(int(info.status)) == sc.Status.OK
  assert float(info.residual) < 1e-9
  assert int(info.iter) == method.max_iter if method.tol is None else 1 <= int(info.iter) < method.max_iter


def test_a_gradient_system_by_cholesky_and_a_sparse_one_by_sparse_ldl() -> None:
  # The stationarity of a strictly convex energy: its Jacobian, the Hessian, is symmetric positive definite.
  @sc.roots.root(vars=sc.L("z", N), params=sc.L("p", N), name="convex")
  def convex(z: sc.Expr, p: sc.Expr) -> sc.Expr:
    return sc.gradient(0.5 * sc.sumsqr(z) + (z.exp()).sum() + (z[1:] * z[:-1]).sum() * 0.25 - (p * z).sum(), z)

  # Cholesky and SparseLDL also give the implicit derivative its solves, since the Jacobian is symmetric.
  p = sc.sym("p", N)
  results = []
  for linear in sc.roots.LINEAR:
    z, info = sc.roots.solver(convex, sc.roots.Newton(linear=linear), name=f"convex_{linear}")(sc.const(np.zeros(N)), p)
    fn = sc.Function.from_exprs(
      f"convex_{linear}_derivs", [p], [z, info.residual, sc.jacobian(z, p), sc.gradient((z * z).sum(), p)], ["p"], ["z", "r", "J", "g"]
    )
    results.append(fn(P))
  for z, residual, jac, grad in results:
    assert float(residual) < 1e-10
    np.testing.assert_allclose(z, results[0][0], atol=1e-12)
    np.testing.assert_allclose(jac, results[0][2], atol=1e-12)
    np.testing.assert_allclose(grad, results[0][3], atol=1e-12)
  np.testing.assert_allclose(results[0][2], results[0][2].T, atol=1e-12)  # the inverse of a symmetric Hessian


def test_a_capped_step_reaches_a_root_plain_newton_overshoots() -> None:
  # Newton on atan(z) = p diverges from |z0 - root| beyond about 1.39; steps capped at 0.5 get there.
  @sc.roots.root(vars=sc.L("z", 1), params=sc.L("p", 1), name="atan")
  def atan(z: sc.Expr, p: sc.Expr) -> sc.Expr:
    return z.atan() - p

  plain = sc.roots.solver(atan, sc.roots.Newton(max_iter=20), name="atan_plain")(np.array([3.0]), np.array([0.0]))
  damped = sc.roots.solver(atan, sc.roots.Newton(max_iter=40, max_step=0.5), name="atan_damped")(np.array([3.0]), np.array([0.0]))
  assert sc.Status(int(plain[1].status)) != sc.Status.OK
  assert sc.Status(int(damped[1].status)) == sc.Status.OK and abs(float(damped[0][0])) < 1e-12


def test_the_statuses_of_an_unfinished_and_a_broken_iteration() -> None:
  z, info = sc.roots.solver(BROYDEN, sc.roots.Newton(tol=1e-14, max_iter=2), name="broyden_short")(Z0, P)
  assert sc.Status(int(info.status)) == sc.Status.MAX_ITER and int(info.iter) == 2

  @sc.roots.root(vars=sc.L("z", 1), params=sc.L("p", 1), name="sqrt")
  def sqrt(z: sc.Expr, p: sc.Expr) -> sc.Expr:
    return z.sqrt() - p

  _, info = sc.roots.solver(sqrt, name="sqrt_negative")(np.array([-1.0]), np.array([2.0]))
  assert sc.Status(int(info.status)) == sc.Status.NUMERICS


def test_the_solution_is_differentiable_in_the_parameters_and_not_in_the_warm_start() -> None:
  solve = sc.roots.solver(BROYDEN, name="broyden_diff")
  p, z0 = sc.sym("p", N), sc.sym("z0", N)
  z, _ = solve(z0, p)
  w = sc.const(np.arange(1.0, N + 1))
  fn = sc.Function.from_exprs(
    "broyden_derivs",
    [z0, p],
    [sc.jacobian(z, p), sc.gradient((w * z).sum(), p), sc.hessian((z * z).sum(), p), sc.jacobian(z, z0)],
    ["z0", "p"],
    ["J", "g", "H", "J0"],
  )
  jac, grad, hess, jac0 = fn((Z0, P))
  eps = 1e-6
  fd = np.stack([(reference(P + eps * e) - reference(P - eps * e)) / (2 * eps) for e in np.eye(N)], axis=1)
  np.testing.assert_allclose(jac, fd, atol=1e-7)
  np.testing.assert_allclose(grad, np.arange(1.0, N + 1) @ fd, atol=1e-7)
  np.testing.assert_allclose(jac0, 0.0)

  def grad_sq(p: np.ndarray) -> np.ndarray:
    return 2.0 * reference(p) @ np.stack([(reference(p + eps * e) - reference(p - eps * e)) / (2 * eps) for e in np.eye(N)], axis=1)

  fd2 = np.stack([(grad_sq(P + 1e-4 * e) - grad_sq(P - 1e-4 * e)) / 2e-4 for e in np.eye(N)], axis=1)
  np.testing.assert_allclose(hess, fd2, atol=1e-5)
  np.testing.assert_allclose(hess, hess.T, atol=1e-12)


def cubic_np(x: float, p: float) -> float:
  return x**3 + x - p


@pytest.mark.parametrize("p", [-30.0, -1.0, 0.0, 0.5, 7.0, 900.0])
def test_newton_bisection_finds_brentqs_root_in_a_bracket_from_the_parameters(p: float) -> None:
  @sc.roots.root(vars=sc.L("x", ()), params=sc.G(sc.L("p", ()), sc.L("width", ())), name="cubic")
  def cubic(x: sc.Expr, params: tuple[sc.Expr, sc.Expr]) -> sc.roots.RootSpec:
    target, width = params
    return sc.roots.RootSpec(x * x * x + x - target, lb=-width, ub=width)

  solve = sc.roots.solver(cubic, name="cubic_bracketed")
  assert solve.name == "cubic_bracketed" and isinstance(sc.roots.REGISTRY.auto(cubic), sc.roots.NewtonBisection)
  x, info = solve(np.array(0.9 * 10.0), (np.array(p), np.array(10.0)))
  want = optimize.brentq(cubic_np, -10.0, 10.0, args=(p,), xtol=1e-15)
  assert abs(float(x) - want) <= 1e-14 * max(abs(want), 20.0)  # tol times the larger of |x| and the bracket's width
  assert sc.Status(int(info.status)) == sc.Status.OK and int(info.iter) < 60


def test_newton_bisection_on_a_decreasing_residual_and_its_derivative() -> None:
  @sc.roots.root(vars=sc.L("x", ()), params=sc.L("p", ()), name="decay")
  def decay(x: sc.Expr, p: sc.Expr) -> sc.roots.RootSpec:
    return sc.roots.RootSpec((-x).exp() - p, lb=sc.const(-5.0), ub=sc.const(40.0))

  solve = sc.roots.solver(decay, sc.roots.NewtonBisection(increasing=False), name="decay_root")
  p = sc.sym("p")
  x, _ = solve(sc.const(20.0), p)
  fn = sc.Function.from_exprs("decay_derivs", [p], [x, sc.gradient(x, p)], ["p"], ["x", "dx"])
  for target in (1e-12, 0.3, 50.0):
    value, slope = fn(np.array(target))
    np.testing.assert_allclose(value, -np.log(target), rtol=0, atol=1e-14 * 45.0)
    np.testing.assert_allclose(slope, -1.0 / target, rtol=1e-12)


def test_which_method_solves_what() -> None:
  @sc.roots.root(vars=sc.L("x", ()), params=sc.L("p", ()), name="bounded")
  def bounded(x: sc.Expr, p: sc.Expr) -> sc.roots.RootSpec:
    return sc.roots.RootSpec(x - p, lb=sc.const(0.0), ub=sc.const(1.0))

  assert isinstance(sc.roots.REGISTRY.auto(BROYDEN), sc.roots.Newton)
  with pytest.raises(ValueError, match="NewtonBisection brackets one unknown"):
    sc.roots.solver(bounded, sc.roots.Newton())
  with pytest.raises(ValueError, match="more than one unknown, no lower bound to bracket it"):
    sc.roots.solver(BROYDEN, "newton_bisection")
  with pytest.raises(ValueError, match="linear must be one of"):
    sc.roots.Newton(linear="qr")
  with pytest.raises(ValueError, match="SparseLDL"):
    sc.roots.Newton(linear="sparse_ldl", simplified=True)


def test_a_newton_step_too_small_to_move_the_root_ends_the_bracketing_iteration() -> None:
  # Newton on x^3 = p from above: at the last iterate the rounding in x^3 - p, divided by the slope,
  # is below half an ulp of x, so the step rounds to x itself, which a positive residual has just
  # made the bracket's upper end. Bisecting there would throw the root away for the bracket's far
  # lower end and start over.
  @sc.roots.root(vars=sc.L("x", ()), params=sc.L("p", ()), name="cube_root")
  def cube_root(x: sc.Expr, p: sc.Expr) -> sc.roots.RootSpec:
    return sc.roots.RootSpec(x * x * x - p, lb=sc.const(0.5), ub=sc.const(3.0))

  solve = sc.roots.solver(cube_root, sc.roots.NewtonBisection(tol=1e-15), name="cube_root_bracketed")
  targets = np.linspace(2.0, 4.0, 2000)
  p = sc.sym("p", targets.size)

  @sc.function((), output=sc.G("x", "iterations"))
  def one(q: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    x, info = solve(sc.const(2.9), q)
    return x, info.iter

  fn = sc.Function.from_exprs("cube_root_batch", [p], [sc.vmap(one, targets.size, [(p, 0, 1)], output=k) for k in range(2)], ["p"], ["x", "n"])
  x, iterations = fn(targets)
  np.testing.assert_allclose(x, np.cbrt(targets), rtol=0, atol=3e-15 * 2.5)
  assert iterations.max() <= 12


def test_a_relative_tolerance_for_a_residual_in_the_unknowns_units() -> None:
  # Near z = 1.4e8 an ulp of z is 3e-8, and no double brings this residual below 2e-9: an absolute
  # 1e-10 is out of reach, and a tolerance relative to |z| is what the residual's units call for.
  c = 1e8 * np.sqrt(2.0)

  @sc.roots.root(vars=sc.L("z", 1), params=sc.L("p", 1), name="large")
  def large(z: sc.Expr, p: sc.Expr) -> sc.Expr:
    return z - p + 1e-3 * (z * 1e-8).sin()

  args = (np.array([1e8]), np.array([c]))
  _, absolute = sc.roots.solver(large, sc.roots.Newton(tol=1e-10, max_iter=20), name="large_absolute")(*args)
  z, relative = sc.roots.solver(large, sc.roots.Newton(tol=1e-10, rtol=1e-15, max_iter=20), name="large_relative")(*args)
  assert sc.Status(int(absolute.status)) == sc.Status.MAX_ITER
  assert sc.Status(int(relative.status)) == sc.Status.OK and int(relative.iter) < 10
  assert abs(float(z[0]) - c + 1e-3 * np.sin(float(z[0]) * 1e-8)) < 3e-8
