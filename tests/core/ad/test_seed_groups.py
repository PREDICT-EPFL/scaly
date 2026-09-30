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
  """Two layers of a network: its tangents are several times its primal, as a real stage's are.
  Only a body expanded into scalar code grows with its operations, so it asks to be."""
  return (_C @ (_B @ (_A @ z).tanh()).tanh()).scalar()


_A2, _B2, _C2 = (sc.const(_RNG.standard_normal(shape)) for shape in ((6, 8), (6, 6), (3, 6)))


@sc.function(sc.L("a", 4), sc.L("b", 4), output=sc.L("y", ...), name="seed_group_pair")
def _pair(a, b):
  """Two formals mapped over overlapping windows, as a stage's ``z`` and ``znext``: three outputs of
  eight inputs, so the Hessian's star colours repeat every four stages and leave some inactive."""
  return (_C2 @ (_B2 @ (_A2 @ sc.concat([a, b])).tanh()).tanh()).scalar()


def _windows() -> tuple[sc.Expr, sc.Expr]:
  """The pair over eight windows of ``x``, each overlapping the next: a whole number of periods."""
  x = sc.sym("x", 4 * 9)
  return sc.vmap(_pair, 8, [(x, 0, 4), (x, 4, 4)]), x


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
  return [np.asarray(v) for v in fn._flat_numerical_call(point[: inputs[0].size], *extra)]


POINT = np.random.default_rng(0).standard_normal(6 * STAGES)
SEEDS = 5


def _derivative(kind: str) -> tuple[list[sc.Expr], sc.Expr, tuple[np.ndarray, ...]]:
  if kind.endswith("_pair"):
    mapped, x = _windows()
    if kind == "hessian_pair":  # periodic tiles with inactive rows, a period of four
      return [x], sc.sparse_hessian((mapped * mapped).sum(), x).values, ()
    if kind == "sparse_jacobian_pair":
      return [x], sc.sparse_jacobian(mapped, x).values, ()
    # Two runtime seeds, fewer than a formal's local colors: one joint body for both formals.
    seeds = sc.sym("S", (2, x.size))
    return [x, seeds], sc.jvp_many(mapped, x, seeds), (np.random.default_rng(2).standard_normal((2, x.size)),)
  mapped, x = _mapped()
  if kind == "hessian":  # periodic constant seeds: the star colors repeat every stage
    return [x], sc.sparse_hessian(mapped.sum(), x).values, ()
  if kind == "jacobian":  # local colors: identity seeds, a different tile each stage
    return [x], sc.jacobian(mapped, x), ()
  if kind == "sparse_jacobian":  # the structured sparse Jacobian of a mapped call
    return [x], sc.sparse_jacobian(mapped, x).values, ()
  # Seeds that are values, not constants, fewer than the stage's local colors: the generic path.
  # Five seeds in two groups, of three and two, have different bodies, not packed into one again.
  seeds = sc.sym("S", (SEEDS, 6 * STAGES))
  return [x, seeds], sc.jvp_many(mapped, x, seeds), (np.random.default_rng(1).standard_normal((SEEDS, 6 * STAGES)),)


@pytest.mark.parametrize("kind", ["hessian", "jacobian", "sparse_jacobian", "runtime", "runtime_pair"])
def test_split_bodies_give_the_same_derivative(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
  with sc.target(ROOMY):
    inputs, out, extra = _derivative(kind)
  few, whole = _maps(out)
  one = _values(f"seed_groups_{kind}_one", inputs, [out], POINT, *extra)[0]
  if kind == "runtime_pair":  # the pair's tangents are small beside its primal: lift the cap
    monkeypatch.setattr(forward, "MAX_RECOMPUTE", 10.0)
  with sc.target(_budget(7 * whole // 10)):  # two groups of the one body
    inputs, out, extra = _derivative(kind)
  many, part = _maps(out)
  split = _values(f"seed_groups_{kind}_split", inputs, [out], POINT, *extra)[0]
  assert many > few and part < whole  # more mapped calls, one per group, each a smaller body
  np.testing.assert_allclose(split, one, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("kind", ["hessian_pair", "sparse_jacobian_pair"])
def test_groups_keep_each_seed_row_where_colours_are_inactive(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
  """Across budgets that split one formal's bodies or the other's, into two to four groups: each
  group holds its seed rows by colour, not by position, where a tile leaves colours inactive, and
  the rows no group holds stay zero. Each formal's body is judged alone, a part of the one packed
  body the unsplit derivative maps, so the budgets are in operations."""
  monkeypatch.setattr(forward, "MAX_RECOMPUTE", 10.0)  # the pair's tangents are small beside its primal
  inactive: list[int] = []
  grouped = forward._grouped_const_tangents

  def spy(callee, output_idx, formal_idx, seed, active, groups, *rest):
    if len(active) < seed.shape[0]:
      inactive.append(len(groups))
    return grouped(callee, output_idx, formal_idx, seed, active, groups, *rest)

  monkeypatch.setattr(forward, "_grouped_const_tangents", spy)
  with sc.target(ROOMY):
    inputs, out, _ = _derivative(kind)
  few, _ = _maps(out)
  one = _values(f"seed_groups_{kind}_one", inputs, [out], POINT)[0]
  split = 0
  for budget in (183, 249, 306, 345, 399, 577):
    with sc.target(_budget(budget)):
      inputs, out, _ = _derivative(kind)
    if _maps(out)[0] == few:
      continue
    split += 1
    got = _values(f"seed_groups_{kind}_{budget}", inputs, [out], POINT)[0]
    np.testing.assert_allclose(got, one, rtol=1e-13, atol=1e-12)
  assert split >= 2
  if kind == "hessian_pair":
    assert any(count > 1 for count in inactive)  # a tile with an inactive colour was split


def test_groups_are_as_few_as_fit(monkeypatch: pytest.MonkeyPatch) -> None:
  """Each group recomputes the primal (``shared``); the tangents (``total - shared``) are divided, in
  groups as even as the seeds allow. The cap on the recomputation then falls back to one body."""
  fn, *_ = forward._call_jvp_many_function(_stage, 0, (0,), 6, (None,))
  shared, total = forward._body_ops(_stage), forward._body_ops(fn)
  tangents = total - shared
  assert 0 < shared < total // 4 and tangents % 12 == 0  # so each budget below makes exactly its count
  made = {}
  with monkeypatch.context() as patched:
    patched.setattr(forward, "MAX_RECOMPUTE", 10.0)
    for count in (1, 2, 3, 4, 6):
      with sc.target(_budget(shared + tangents // count)):
        groups = made[count] = forward._seed_groups(_stage, fn, 6)
      assert len(groups) == count and groups[0][0] == 0 and groups[-1][1] == 6
      assert all(a[1] == b[0] for a, b in zip(groups, groups[1:], strict=False))
      sizes = [hi - lo for lo, hi in groups]
      assert min(sizes) > 0 and max(sizes) - min(sizes) <= 1
  # A split into ``count`` groups adds ``count`` recomputations; past the cap, the body stays whole.
  capped = [count for count in made if count > 1 and count * shared > forward.MAX_RECOMPUTE * total]
  assert capped and len(capped) < len(made) - 1
  for count in made:
    with sc.target(_budget(shared + tangents // count)):
      assert forward._seed_groups(_stage, fn, 6) == ([(0, 6)] if count in capped else made[count])
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
  body = sc.Function.from_exprs("seed_group_body", [x, y], [(x * x).scalar(), (y * y).scalar()], ["x", "y"], ["a", "b"])
  for budget, count in ((12 + 200, 6), (12 + 199, 1)):
    with sc.target(_budget(budget)):
      assert len(forward._seed_groups(primal, body, 6)) == count


def test_only_bodies_expanded_into_scalar_code_are_split() -> None:
  """A body kept in loops stays compact at any size, so a budget that splits the expanded stage's
  derivative leaves a stage in loops (automatic or ``.block()``) one body; so does a scalar body
  with a run-time index, which expansion cannot follow."""
  with sc.target(ROOMY):
    _, out, _ = _derivative("jacobian")
  few, whole = _maps(out)
  tight = _budget(7 * whole // 10)
  with sc.target(tight):
    assert _maps(_derivative("jacobian")[1])[0] > few  # the expanded stage splits

  for lowering in ("auto", "block"):

    @sc.function(sc.L("z", 6), output=sc.L("y", ...), name=f"seed_group_{lowering}")
    def looped(z):
      y = _C @ (_B @ (_A @ z).tanh()).tanh()
      return y.block() if lowering == "block" else y

    x = sc.sym("x", 6 * STAGES)
    with sc.target(tight):
      assert _maps(sc.jacobian(sc.vmap(looped, STAGES, [(x, 0, 6)]), x))[0] == few
  x, i = sc.sym("x", 12), sc.sym("i", 12, dtype="int64")
  indexed = sc.Function.from_exprs("seed_group_indexed", [x, i], [(x * sc.take(x, i)).scalar()], ["x", "i"], ["y"])
  assert forward._expands(sc.Function.from_exprs("seed_group_plain", [x], [(x * x).scalar()], ["x"], ["y"]))
  assert not forward._expands(indexed)
