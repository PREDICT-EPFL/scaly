"""``sc.custom_derivative``: rules set at construction, honoured wherever a body is differentiated.

Several tests give ``square`` a deliberately wrong rule, the derivative of ``1.5 x^2`` instead of
``x^2``, so a result shows which derivative ran. Values are checked against NumPy, against the body
differentiated as usual, or against finite differences.
"""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference
from scaly.codegen import render_c_source
from scaly.function.model import as_concrete
from scaly.ir.expr import ExprOp, topo


def _reached(fn) -> list:
  """Every ConcreteFunction ``fn`` reaches through calls and maps, ``fn`` first."""
  found, todo = [], [as_concrete(fn)]
  while todo:
    current = todo.pop()
    if current not in found:
      found.append(current)
      todo += [node.attrs["callee"] for node in topo(current.results) if node.op in (ExprOp.CALL, ExprOp.VMAP)]
  return found


@sc.function(sc.arg("x", 3))
def square(x):
  return x * x


@sc.function()
def square_jvp(x, dx):
  return 3.0 * x * dx


@sc.function()
def square_fwd(x):
  return x * x, x


@sc.function()
def square_bwd(x, ybar):
  return 3.0 * x * ybar


ruled = sc.custom_derivative(square, jvp=square_jvp, fwd=square_fwd, bwd=square_bwd)
POINT = np.array([0.3, -1.1, 0.7])


def test_a_call_honours_both_rules() -> None:
  @sc.function(sc.arg("q", 3))
  def outer(q):
    return ruled(q.sin()).sum()

  @sc.function(sc.arg("q", 3))
  def mapped(q):
    return ruled(q.sin())

  expected = 3.0 * np.sin(POINT) * np.cos(POINT)
  np.testing.assert_allclose(sc.gradient(outer)(POINT), expected, rtol=1e-14)
  np.testing.assert_allclose(sc.jacobian(mapped)(POINT), np.diag(expected), rtol=1e-14)
  np.testing.assert_allclose(ruled(POINT), POINT**2, rtol=1e-15)


def test_a_map_and_a_library_function_honour_the_rules() -> None:
  batched = sc.vmap(ruled, 2)

  @sc.function(sc.arg("q", (2, 3)))
  def library(q):
    return batched(q * 2.0)

  @sc.function(sc.arg("q", (2, 3)))
  def user(q):
    return (library(q) * library(q)).sum()

  qv = np.stack([POINT, POINT[::-1]])
  # y = (2q)^2 under the rule y' = 3 x x', so d sum(y^2) / dq = 2 y * 3 (2q) * 2.
  expected = 2 * (2 * qv) ** 2 * 3 * (2 * qv) * 2
  np.testing.assert_allclose(sc.gradient(user)(qv), expected, rtol=1e-14)
  jac = sc.jacobian(library)(qv)
  np.testing.assert_allclose(jac, np.diag((3 * (2 * qv) * 2).reshape(-1)), rtol=1e-14)


def test_without_a_rule_a_direction_differentiates_the_body() -> None:
  forward_only = sc.custom_derivative(square, jvp=square_jvp)
  reverse_only = sc.custom_derivative(square, fwd=square_fwd, bwd=square_bwd)

  def both(fn):
    @sc.function(sc.arg("q", 3))
    def both_fn(q):
      y = fn(q)
      return sc.jacobian(y, q), sc.gradient(y.sum(), q)

    return both_fn

  for fn, slope_jvp, slope_vjp in ((forward_only, 3.0, 2.0), (reverse_only, 2.0, 3.0)):
    jac, grad = both(fn)(POINT)
    np.testing.assert_allclose(jac, np.diag(slope_jvp * POINT), rtol=1e-15)
    np.testing.assert_allclose(grad, slope_vjp * POINT, rtol=1e-15)


def test_several_seeds_map_one_single_seed_rule() -> None:
  """Forward mode over four seeds calls the single-seed rule through one map of length four."""

  @sc.function(sc.arg("q", 3))
  def outer(q):
    return ruled(q.sin()) * 2.0

  jac = sc.jacobian(outer)
  rules = as_concrete(ruled).rules
  assert rules is not None
  rule = rules.jvp
  maps = [node for fn in _reached(jac) for node in topo(fn.results) if node.op == ExprOp.VMAP and node.attrs["callee"] is rule]
  assert [node.attrs["length"] for node in maps] == [3]
  np.testing.assert_allclose(jac(POINT), np.diag(6.0 * np.sin(POINT) * np.cos(POINT)), rtol=1e-14)


def test_tangents_a_rule_never_reads_are_not_built() -> None:
  """An implicit rule for ``x^3 + x = p`` ignores its initial guess, so the guess, here the output
  of another Function, is never differentiated."""

  @sc.function(sc.arg("p", 3), sc.arg("x0", 3))
  def newton(p, x0):
    x = x0
    for _ in range(30):
      x = x - (x**3 + x - p) / (3.0 * x * x + 1.0)
    return x

  @sc.function()
  def implicit_jvp(p, x0, dp, dx0):
    x = newton(p, x0)
    return dp / (3.0 * x * x + 1.0)

  @sc.function(sc.arg("q", 3))
  def guess(q):
    return q.tanh()

  solve = sc.custom_derivative(newton, jvp=implicit_jvp)

  @sc.function(sc.arg("q", 3))
  def outer(q):
    return solve(q * 2.0, guess(q))

  jac = sc.jacobian(outer)
  assert not any(fn.role == "forward" and fn.name.startswith("guess") for fn in _reached(jac))
  x = outer(POINT)
  np.testing.assert_allclose(x**3 + x, 2 * POINT, rtol=1e-14)
  np.testing.assert_allclose(jac(POINT), np.diag(2.0 / (3 * x * x + 1)), rtol=1e-12)


# --- smooth rules that are the true derivative --------------------------------------------------


@sc.function(sc.arg("x", 3), sc.arg("p", 2))
def wave(x, p):
  u = (p[0] * x).sin()
  return u * p[1], (x * x).sum()


@sc.function()
def wave_jvp(x, p, dx, dp):
  c = (p[0] * x).cos()
  return c * (dp[0] * x + p[0] * dx) * p[1] + (p[0] * x).sin() * dp[1], 2.0 * (x * dx).sum()


@sc.function()
def wave_fwd(x, p):
  c, u = (p[0] * x).cos(), (p[0] * x).sin()
  return (u * p[1], (x * x).sum()), (x, p, c, u)


@sc.function()
def wave_bwd(res, cot):
  x, p, c, u = res
  ybar, sbar = cot
  g = ybar * p[1] * c
  return g * p[0] + 2.0 * sbar * x, sc.stack([(g * x).sum(), (ybar * u).sum()])


smooth = sc.custom_derivative(wave, jvp=wave_jvp, fwd=wave_fwd, bwd=wave_bwd)


def _wave_point(seed: int) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  return rng.normal(size=3), rng.normal(size=2)


@pytest.mark.parametrize("seed", range(4))
def test_rules_are_linear_dual_and_match_the_body(seed: int) -> None:
  """Random seeds: the JVP rule is linear in its tangents, ``<J v, w> = <v, J^T w>`` between the
  JVP rule and the reverse pair, and both agree with differentiating the body."""

  def probe(fn):
    @sc.function(sc.arg("x", 3), sc.arg("p", 2), sc.arg("v", (3, 5)), sc.arg("w", 3), sc.arg("t"))
    def probe_fn(x, p, v, w, t):
      y, s = fn(x, p)
      jv = sc.jvp_many(y, x, v.T)
      xbar, pbar = sc.vjp((y, s), (x, p), (w, t))
      return jv, xbar, pbar

    return probe_fn

  x, p = _wave_point(seed)
  rng = np.random.default_rng(100 + seed)
  v, w, t = rng.normal(size=(3, 5)), rng.normal(size=3), rng.normal()
  jv, xbar, pbar = probe(smooth)(x, p, v, w, np.array(t))
  ref = probe(wave)(x, p, v, w, np.array(t))
  for got, want in zip((jv, xbar, pbar), ref, strict=True):
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-14)
  a, b = rng.normal(size=2)
  # Linearity: the seed columns combine as the tangents do.
  combined = probe(smooth)(x, p, np.column_stack([v[:, :3], a * v[:, 0] + b * v[:, 1], v[:, 4]]), w, np.array(t))[0]
  np.testing.assert_allclose(combined[3], a * jv[0] + b * jv[1], rtol=1e-12, atol=1e-14)
  # Duality against the y cotangent alone.
  _, xbar_y, _ = probe(smooth)(x, p, v, w, np.array(0.0))
  np.testing.assert_allclose(jv @ w, v.T @ xbar_y, rtol=1e-12)


def test_forward_over_reverse_differentiates_the_rule_and_its_residuals() -> None:
  """The Hessian of a loss through the reverse pair: forward mode differentiates ``bwd`` and the
  residuals it reads, which keep their dependence on the inputs."""

  def hessian(fn):
    @sc.function(sc.arg("x", 3), sc.arg("p", 2))
    def loss(x, p):
      y, s = fn(x, p)
      return (y * y * y).sum() + s * s

    return sc.hessian(loss, wrt="x")

  x, p = _wave_point(7)
  got, want = hessian(smooth)(x, p), hessian(wave)(x, p)
  assert np.abs(want).max() > 0.1
  np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


def test_forward_over_forward_differentiates_the_rule_graph() -> None:
  def second(fn):
    @sc.function(sc.arg("x", 3), sc.arg("p", 2))
    def curvature(x, p):
      y, _ = fn(x, p)
      return sc.jacobian(sc.jacobian(y, x).reshape((9,)), x)

    return curvature

  x, p = _wave_point(8)
  got, want = second(smooth)(x, p), second(wave)(x, p)
  assert np.abs(want).max() > 0.1
  np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)
  first = sc.jacobian(smooth, "out0", "x")
  np.testing.assert_allclose(first(x, p), finite_difference(lambda z: smooth(z, p)[0], x), rtol=1e-7, atol=1e-8)


# --- declared sparsity --------------------------------------------------------------------------


@sc.function(sc.arg("x", 3))
def coupled(x):
  return x * x + 0.0 * x.sum()


def test_declared_sparsity_replaces_the_body_pattern() -> None:
  diagonal = sc.custom_derivative(coupled, jvp=square_jvp, sparsity=lambda of, wrt: np.eye(3, dtype=bool))
  undeclared = sc.custom_derivative(coupled, jvp=square_jvp)
  q = sc.sym("q", 3)
  assert sc.jacobian_sparsity(coupled(q), q) == sc.SparsityPattern.dense((3, 3))
  assert sc.jacobian_sparsity(diagonal(q), q) == sc.SparsityPattern.from_mask(np.eye(3, dtype=bool))
  assert sc.jacobian_sparsity(sc.custom_derivative(square, jvp=square_jvp)(q), q) == sc.SparsityPattern.dense((3, 3))
  assert sc.jacobian_sparsity(undeclared(q), q) == sc.SparsityPattern.dense((3, 3))
  qs = sc.sym("qs", (4, 3))
  assert sc.jacobian_sparsity(sc.vmap(diagonal, 4)(qs), qs) == sc.SparsityPattern.from_mask(np.eye(12, dtype=bool))

  @sc.function(sc.arg("qs", (4, 3)))
  def compact(qs):
    jac = sc.sparse_jacobian(sc.vmap(diagonal, 4)(qs.sin()), qs)
    return jac.values, jac.to_dense()

  values, dense = compact(np.arange(12.0).reshape(4, 3) / 10)
  assert values.shape == (12,)
  s = np.sin(np.arange(12.0) / 10)
  np.testing.assert_allclose(dense, np.diag(3 * s * np.cos(np.arange(12.0) / 10)), rtol=1e-14)


def test_declared_sparsity_takes_masks_and_patterns_and_checks_their_shape() -> None:
  pattern = sc.SparsityPattern.from_mask(np.eye(3, dtype=bool))
  q = sc.sym("q", 3)
  assert sc.jacobian_sparsity(sc.custom_derivative(coupled, sparsity=lambda of, wrt: pattern)(q), q) == pattern
  with pytest.raises(ValueError, match="pattern of shape"):
    sc.custom_derivative(coupled, sparsity=lambda of, wrt: np.eye(2, dtype=bool))


# --- a solve whose derivative reuses its factorization ------------------------------------------

N = 4
LOWER = [(i, j) for i in range(N) for j in range(i + 1)]


def _cholesky(a):
  lower: dict[tuple[int, int], sc.Expr] = {}
  for j in range(N):
    d = a[j, j]
    for k in range(j):
      d = d - lower[j, k] * lower[j, k]
    lower[j, j] = d.sqrt()
    for i in range(j + 1, N):
      e = a[i, j]
      for k in range(j):
        e = e - lower[i, k] * lower[j, k]
      lower[i, j] = e / lower[j, j]
  return lower


def _cho_solve(lower, b):
  y: list[sc.Expr] = []
  for i in range(N):
    e = b[i]
    for k in range(i):
      e = e - lower[i, k] * y[k]
    y.append(e / lower[i, i])
  x: list[sc.Expr | None] = [None] * N
  for i in reversed(range(N)):
    e = y[i]
    for k in range(i + 1, N):
      e = e - lower[k, i] * x[k]
    x[i] = e / lower[i, i]
  return sc.stack(x)


@sc.function(sc.arg("A", (N, N)), sc.arg("b", N))
def chol_solve(A, b):
  return _cho_solve(_cholesky(0.5 * (A + A.T)), b)


@sc.function()
def chol_fwd(A, b):
  lower = _cholesky(0.5 * (A + A.T))
  x = _cho_solve(lower, b)
  return x, (sc.stack([lower[ij] for ij in LOWER]), x)


@sc.function()
def chol_bwd(res, xbar):
  packed, x = res
  lower = {ij: packed[k] for k, ij in enumerate(LOWER)}
  bbar = _cho_solve(lower, xbar)
  outer = bbar.reshape((N, 1)) * x.reshape((1, N))
  return -0.5 * (outer + outer.T), bbar


reusing = sc.custom_derivative(chol_solve, fwd=chol_fwd, bwd=chol_bwd)


def _value_and_gradient(solve):
  @sc.function(sc.arg("A", (N, N)), sc.arg("b", N))
  def value_and_gradient(A, b):
    x = solve(A, b)
    value = (x * x * x).sum()
    abar, bbar = sc.vjp((value,), (A, b), (sc.const(1.0),))
    return value, abar, bbar

  return value_and_gradient


def _spd(seed: int) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  m = rng.normal(size=(N, N))
  return m @ m.T + N * np.eye(N), rng.normal(size=N)


def test_a_solve_derivative_reuses_its_factorization() -> None:
  """The reverse rule reads the Cholesky factor as a residual. The value, the factor and the
  solution are results of one invocation, so the generated C takes the ``N`` square roots of one
  factorization; differentiating the steps would differentiate them instead."""
  fn = _value_and_gradient(reusing)
  a, b = _spd(3)
  value, abar, bbar = fn(a, b)
  ref = _value_and_gradient(chol_solve)(a, b)
  np.testing.assert_allclose(value, ref[0], rtol=1e-13)
  np.testing.assert_allclose(abar, ref[1], rtol=1e-11, atol=1e-13)
  np.testing.assert_allclose(bbar, ref[2], rtol=1e-11, atol=1e-13)
  x = np.linalg.solve(a, b)
  np.testing.assert_allclose(bbar, np.linalg.solve(a, 3 * x * x), rtol=1e-11)

  callee = as_concrete(reusing)
  invocations = {node.args for node in topo(as_concrete(fn).results) if node.op == ExprOp.CALL and node.attrs["callee"] is callee}
  assert len(invocations) == 1
  sqrts = [node for g in _reached(fn) for node in topo(g.results) if node.op == ExprOp.SQRT]
  assert len(sqrts) == N
  assert render_c_source(fn).count("sqrt(") == N


# --- templates and errors -----------------------------------------------------------------------


def test_template_functions_and_rules_bind_per_shape() -> None:
  @sc.function()
  def softplus(x):
    return (1.0 + x.exp()).log()

  @sc.function()
  def softplus_jvp(x, dx):
    return dx / (1.0 + (-x).exp())

  @sc.function()
  def softplus_fwd(x):
    return (1.0 + x.exp()).log(), 1.0 / (1.0 + (-x).exp())

  @sc.function()
  def softplus_bwd(slope, ybar):
    return slope * ybar

  smooth_plus = sc.custom_derivative(softplus, jvp=softplus_jvp, fwd=softplus_fwd, bwd=softplus_bwd)
  assert isinstance(smooth_plus, sc.Function)
  for n in (2, 3):

    @sc.function(sc.arg("x", n))
    def total(x):
      return smooth_plus(x).sum()

    @sc.function(sc.arg("x", n))
    def image(x):
      return smooth_plus(x)

    x = np.linspace(-1.0, 2.0, n)
    np.testing.assert_allclose(sc.gradient(total)(x), 1.0 / (1.0 + np.exp(-x)), rtol=1e-14)
    np.testing.assert_allclose(sc.jacobian(image)(x), np.diag(1.0 / (1.0 + np.exp(-x))), rtol=1e-14)


def test_rules_are_fields_and_name_their_helpers() -> None:
  concrete = as_concrete(ruled)
  assert concrete.rules is not None and concrete.rules.jvp is not None and concrete.rules.bwd is not None
  assert len(concrete.results) == 2 and concrete.output_names == ("square",)
  with pytest.raises(AttributeError):
    concrete.rules = None  # ty: ignore[invalid-assignment]

  @sc.function(sc.arg("q", 3))
  def both(q):
    return square(q) + ruled(q)

  helpers = [fn.name for fn in _reached(sc.jacobian(both)) if fn.role == "forward"]
  assert len(helpers) == len(set(helpers)) == 2


@pytest.mark.parametrize(
  ("kwargs", "match"),
  [
    ({}, "at least one"),
    ({"fwd": square_fwd}, "together"),
    ({"bwd": square_bwd}, "together"),
  ],
)
def test_incomplete_rules_raise(kwargs, match) -> None:
  with pytest.raises(TypeError, match=match):
    sc.custom_derivative(square, **kwargs)


def test_rules_of_the_wrong_type_raise() -> None:
  @sc.function()
  def wrong_jvp(x, dx):
    return (x * dx).sum()

  @sc.function()
  def unpaired_fwd(x):
    return x * x

  with pytest.raises(TypeError, match="jvp"):
    sc.custom_derivative(square, jvp=wrong_jvp)
  with pytest.raises(TypeError, match="fwd"):
    sc.custom_derivative(square, fwd=unpaired_fwd, bwd=square_bwd)
