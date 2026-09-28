"""Derivatives with respect to an expression that is not an input, such as a slice of a loop carry.

The ``wrt`` is an independent variable: every other path to the inputs is held fixed. Each form is
checked against the same derivative taken with respect to an input symbol standing in its place.
"""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.ad import jvp_many, vjp, vjp_many
from scaly.ad.sparse import sparse_jacobian_colored, sparse_jacobian_reference
from scaly.ir.expr import substitute

C = sc.sym("c", 6)
P = C[1:4]  # the wrt: a slice, not an input
X = np.array([0.3, 1.0, 2.0, -1.5, 0.7, 5.0])


def _energy(p: sc.Expr) -> sc.Expr:
  # p[0] * p[1] * c[5] and c[0] * p[2]: other paths to c that the derivative holds fixed; the
  # slices of the slice are what a simplifier folds into slices of c.
  return (p * p * p).sum() + p[0] * p[1] * C[5] + C[0] * p[2].sin() + sc.sumsqr(C[2:5])


def _vector(p: sc.Expr) -> sc.Expr:
  return sc.concat([p * p * p, (p[0] * p[2] * C[5]).reshape((1,)), C[4:6]])


def _compare(tag: str, build) -> None:
  """``build(expr_fn, wrt)`` at the slice against the same at an input ``q`` evaluated at ``q = c[1:4]``."""
  q = sc.sym("q", 3)
  at_slice = build(P)
  at_input = [substitute(e, {q: P}) for e in build(q)]
  f = sc.Function.from_exprs(f"wrt_{tag}", [C], [*at_slice, *at_input], ["c"], [f"o{i}" for i in range(2 * len(at_slice))])
  out = f(X)
  n = len(at_slice)
  for got, want in zip(out[:n], out[n:], strict=True):
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-13)
  assert any(np.abs(np.asarray(w)).max() > 0 for w in out[n:])


def test_dense_forms() -> None:
  _compare("dense", lambda w: [sc.gradient(_energy(w), w), sc.hessian(_energy(w), w), sc.jacobian(_vector(w), w)])


def test_forward_and_reverse_primitives() -> None:
  seed = sc.const(np.array([0.5, -1.0, 2.0]))
  seeds = sc.const(np.arange(9.0).reshape(3, 3))
  cot = sc.const(np.linspace(1.0, 2.0, 6))
  cots = sc.const(np.arange(12.0).reshape(2, 6))
  _compare(
    "primitives",
    lambda w: [
      sc.jvp(_vector(w), w, seed),
      jvp_many(_vector(w), w, seeds),
      vjp((_vector(w),), (w,), (cot,))[0],
      vjp_many((_vector(w),), (w,), (cots,))[0],
    ],
  )


def test_a_slice_and_an_input_together() -> None:
  """Adjoints stay in ``wrts`` order, and the input's adjoint excludes the path through the slice."""
  cot = sc.const(np.linspace(1.0, 2.0, 6))
  (g_slice, g_input) = vjp((_vector(P),), (P, C), (cot,))
  q = sc.sym("q", 3)
  (r_slice, r_input) = vjp((_vector(q),), (q, C), (cot,))
  f = sc.Function.from_exprs(
    "wrt_mixed", [C], [g_slice, g_input, substitute(r_slice, {q: P}), substitute(r_input, {q: P})], ["c"], ["a", "b", "c2", "d"]
  )
  a, b, ra, rb = f(X)
  np.testing.assert_allclose(a, ra, rtol=1e-13)
  np.testing.assert_allclose(b, rb, rtol=1e-13)
  assert np.all(b[1:4] == 0.0) and b[5] != 0.0


@pytest.mark.parametrize("triangle", ["full", "lower", "upper"])
def test_sparse_forms(triangle) -> None:
  q = sc.sym("q", 3)
  for tag, fn in (
    ("sj", sc.sparse_jacobian),
    ("sjc", sparse_jacobian_colored),
    ("sjr", sparse_jacobian_reference),
  ):
    assert fn(_vector(P), P).sparsity == fn(_vector(q), q).sparsity
    _compare(f"{tag}_{triangle}", lambda w, fn=fn: [fn(_vector(w), w).values])
  assert sc.sparse_hessian(_energy(P), P, triangle=triangle).sparsity == sc.sparse_hessian(_energy(q), q, triangle=triangle).sparsity
  _compare(f"sh_{triangle}", lambda w: [sc.sparse_hessian(_energy(w), w, triangle=triangle).values])
  assert sc.jacobian_sparsity(_vector(P), P) == sc.jacobian_sparsity(_vector(q), q)


def test_newton_on_a_slice_of_a_while_loop_carry() -> None:
  """The use that exposed the gap: a loop body sees only its carry, so its Newton step
  differentiates with respect to a slice of it. With zero Hessians the loop never converged."""
  carry = sc.sym("carry", 4)
  x, a = carry[:3], carry[3]
  energy = sc.sumsqr(x * x - a) + 0.1 * sc.sumsqr(x)
  step = -sc.linalg.solve(sc.hessian(energy, x), sc.gradient(energy, x))
  body = sc.Function.from_exprs("wrt_newton", [carry], [sc.concat([x + step, a.reshape((1,))])], ["c"], ["n"])
  cond = sc.Function.from_exprs("wrt_newton_go", [carry], [sc.greater(sc.norm_inf(sc.gradient(energy, x)), 1e-12)], ["c"], ["g"])
  start = sc.sym("start", 4)
  out, steps = sc.while_loop(cond, body, start, max_iter=50)
  fn = sc.Function.from_exprs("wrt_newton_run", [start], [out, steps], ["s"], ["x", "k"])
  final, k = fn(np.array([1.3, 1.5, 1.4, 2.0]))
  np.testing.assert_allclose(final[:3], np.full(3, np.sqrt(2.0 - 0.05)), rtol=1e-12)
  assert k < 10


def _mapped(p: sc.Expr) -> sc.Expr:
  body = sc.Function.from_exprs("wrt_pair", [a := sc.sym("a", 2)], [sc.stack([a[0] * a[1], a[0].sin() * a[1] * a[1]])], ["a"], ["o"])
  return sc.vmap(body, 2, [(sc.concat([p, C[4:5]]), 0, 2)])


def test_the_structured_sparse_jacobian_of_a_map_over_a_slice() -> None:
  q = sc.sym("q", 3)
  assert sc.sparse_jacobian(_mapped(P), P).sparsity == sc.sparse_jacobian(_mapped(q), q).sparsity
  _compare("mapped", lambda w: [sc.sparse_jacobian(_mapped(w), w).values])


def test_a_triangle_taken_later_reads_no_stand_in() -> None:
  """``triangle`` re-reads the compressed batch, so it has to be mapped back too."""
  _compare("later_triangle", lambda w: [sc.sparse_hessian(_energy(w), w).triangle("lower").values])
