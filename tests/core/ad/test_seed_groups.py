"""Mapped tangent bodies too large for the target's instruction cache are split into groups of seeds
(``ad.forward._seed_groups``), each its own mapped call: the same derivative, in more loops."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.ad import forward
from scaly.ir.expr import ExprOp, topo
from scaly.ir.target import Target

ROOMY = Target("roomy", vector_bytes=16, l1i_bytes=1 << 30, l1d_bytes=32768)
STAGES = 10
_RNG = np.random.default_rng(3)
_A, _B, _C = (sc.const(_RNG.standard_normal(shape)) for shape in ((6, 6), (6, 6), (4, 6)))


@sc.function(sc.L("z", 6), output=sc.L("y", ...), name="seed_group_stage")
def _stage(z):
  """Two layers of a network: its tangents are several times its primal, as a real stage's are."""
  return _C @ (_B @ (_A @ z).tanh()).tanh()


def _budget(ops: int) -> Target:
  """A target whose ``body_bytes`` is ``ops`` operations."""
  return Target(f"budget_{ops}", vector_bytes=16, l1i_bytes=2 * ops * forward.BYTES_PER_BODY_OP, l1d_bytes=32768)


def _mapped() -> tuple[sc.Expr, sc.Expr]:
  x = sc.sym("x", 6 * STAGES)
  return sc.vmap(_stage, STAGES, [(x, 0, 6)]), x


def _maps(*exprs: sc.Expr) -> tuple[int, int]:
  """The mapped calls under ``exprs`` and the largest of their bodies, in ``_body_ops``."""
  bodies = [forward._body_ops(e.attrs["callee"]) for e in topo(list(exprs)) if e.op == ExprOp.VMAP]
  return len(bodies), max(bodies)


def _values(name: str, inputs: list, outputs: list, point: np.ndarray, *extra: np.ndarray) -> list[np.ndarray]:
  fn = sc.Function.from_exprs(name, inputs, outputs, [str(e.name) for e in inputs], [f"o{k}" for k in range(len(outputs))])
  return [np.asarray(v) for v in fn._flat_numerical_call(point, *extra)]


POINT = np.random.default_rng(0).standard_normal(6 * STAGES)


def _derivative(kind: str) -> tuple[list[sc.Expr], sc.Expr, tuple[np.ndarray, ...]]:
  mapped, x = _mapped()
  if kind == "hessian":  # periodic constant seeds: the star colors repeat every stage
    return [x], sc.sparse_hessian(mapped.sum(), x).values, ()
  if kind == "jacobian":  # local colors: identity seeds, a different tile each stage
    return [x], sc.jacobian(mapped, x), ()
  if kind == "sparse_jacobian":  # the structured sparse Jacobian of a mapped call
    return [x], sc.sparse_jacobian(mapped, x).values, ()
  # Seeds that are values, not constants, fewer than the stage's local colors: the generic path.
  # Three seeds in two groups, of two and one, have different bodies, not packed into one again.
  seeds = sc.sym("S", (3, 6 * STAGES))
  return [x, seeds], sc.jvp_many(mapped, x, seeds), (np.random.default_rng(1).standard_normal((3, 6 * STAGES)),)


@pytest.mark.parametrize("kind", ["hessian", "jacobian", "sparse_jacobian", "runtime"])
def test_split_bodies_give_the_same_derivative(kind: str) -> None:
  with sc.target(ROOMY):
    inputs, out, extra = _derivative(kind)
  few, whole = _maps(out)
  one = _values(f"seed_groups_{kind}_one", inputs, [out], POINT, *extra)[0]
  with sc.target(_budget(7 * whole // 10)):  # two groups
    inputs, out, extra = _derivative(kind)
  many, part = _maps(out)
  split = _values(f"seed_groups_{kind}_split", inputs, [out], POINT, *extra)[0]
  assert many > few and part < whole  # more mapped calls, one per group, each a smaller body
  np.testing.assert_allclose(split, one, rtol=1e-13, atol=1e-13)


def test_groups_are_as_few_as_fit() -> None:
  """Each group recomputes the primal (``shared``); the tangents (``total - shared``) are divided."""
  fn, *_ = forward._call_jvp_many_function(_stage, 0, (0,), 6, (None,))
  shared, total = forward._body_ops(_stage), forward._body_ops(fn)
  tangents = total - shared
  assert 0 < shared < total // 4 and tangents % 12 == 0  # so each budget below makes exactly its count
  capped = 0
  for count in range(1, 7):
    with sc.target(_budget(shared + tangents // count)):
      groups = forward._seed_groups(_stage, fn, 6)
    if (count - 1) * shared > forward.MAX_RECOMPUTE * total:  # the recomputation would cost too much
      assert groups == [(0, 6)]
      capped += 1
      continue
    assert len(groups) == count and groups[0][0] == 0 and groups[-1][1] == 6
    assert all(a[1] == b[0] for a, b in zip(groups, groups[1:], strict=False))
    sizes = [hi - lo for lo, hi in groups]
    assert max(sizes) - min(sizes) <= 1  # as even as the seeds allow
  assert 0 < capped < 5
  # The body fits; it does not fit but the primal alone is at the budget; not even one seed fits.
  for budget, count in ((total, 1), (total - 1, 2), (shared, 1), (shared + 1, 1)):
    with sc.target(_budget(budget)):
      assert len(forward._seed_groups(_stage, fn, 6)) == count


def test_the_body_estimate_counts_arithmetic_not_moves() -> None:
  x, a, b = sc.sym("x", (3, 4)), sc.sym("a", (3, 4)), sc.sym("b", (4, 5))
  moves = sc.Function.from_exprs("seed_group_moves", [x], [sc.concat([x.T.reshape(12), x[0]], axis=0)], ["x"], ["y"])
  arithmetic = sc.Function.from_exprs("seed_group_arithmetic", [x], [x * x + 1.0], ["x"], ["y"])
  product = sc.Function.from_exprs("seed_group_product", [a, b], [a @ b], ["a", "b"], ["y"])
  assert forward._body_ops(moves) == 0
  assert forward._body_ops(arithmetic) == 24  # a product and a sum of twelve each; the constant is data
  assert forward._body_ops(product) == 60  # fifteen entries of four multiply-adds


def test_seeds_stay_together_when_one_each_would_not_fit() -> None:
  """A primal small beside its tangents makes the recomputation cheap, but groups of one seed each
  still past the budget would only add it: the body stays whole."""
  x, y = sc.sym("x", 12), sc.sym("y", 1200)
  primal = sc.Function.from_exprs("seed_group_primal", [x], [x * x], ["x"], ["y"])
  body = sc.Function.from_exprs("seed_group_body", [x, y], [x * x, y * y], ["x", "y"], ["a", "b"])
  for budget, count in ((12 + 200, 6), (12 + 199, 1)):
    with sc.target(_budget(budget)):
      assert len(forward._seed_groups(primal, body, 6)) == count
