"""Sparse seed loops preserve mapped tangents and skip their known zeros."""

import numpy as np

import scaly as sc
import pytest
from scaly.passes.lowering import plans
from scaly.codegen import render_c_module
from scaly.ir.program import ProgramOp


def _mapped_tangents():
  mask = np.zeros((12, 4))
  mask[np.arange(12), np.arange(12) % 4] = 1

  @sc.function(sc.arg("x", 4), sc.arg("d", (12, 4)), outputs=sc.arg("y"), name="seed_pair_inner")
  def inner(x, d):
    return (d * x.cos() + d * x).sum(axis=1).block()

  @sc.function(sc.arg("x", 4), sc.arg("d", (12, 4)), outputs=sc.arg("y"), name="seed_pair_stage")
  def stage(x, d):
    return inner(x, d * sc.const(mask)).block()

  @sc.function(sc.arg("x", (3, 4)), sc.arg("d", (3, 12, 4)), outputs=sc.arg("y"), name="seed_pair_host")
  def host(x, d):
    return sc.vmap(stage, 3)(x, d).block()

  return host, mask


def test_mapped_partly_zero_tangents_match_numpy_and_skip_zero_pairs():
  fun, mask = _mapped_tangents()
  rng = np.random.default_rng(225)
  x, d = rng.normal(size=(3, 4)), rng.normal(size=(3, 12, 4))
  expected = (d * mask * (np.cos(x) + x)[:, None, :]).sum(axis=2)
  np.testing.assert_allclose(fun(x, d), expected)
  module = render_c_module(fun, lanes=1)
  inner = next(proc for proc in module.program.args if proc.attrs.get("name", "").startswith("seed_pair_inner"))

  def arithmetic(node):
    if node.op == ProgramOp.FOR:
      start, stop, step = (bound.attrs["value"] for bound in node.args[0].args)
      return len(range(start, stop, step)) * sum(arithmetic(child) for child in node.args[1:])
    return int(node.dtype.is_floating and node.op in (ProgramOp.ADD, ProgramOp.MUL, ProgramOp.COS)) + sum(arithmetic(child) for child in node.args)

  assert arithmetic(inner) <= 60


def test_perturbed_seed_list_changes_the_mapped_tangent(monkeypatch, tmp_path):
  monkeypatch.setenv("SCALY_CACHE_DIR", str(tmp_path))
  original = plans.seed_regions

  def perturbed(writes, reads, width):
    regions = original(writes, reads, width)
    return tuple((plans.MapRegion(np.roll(region.writes, 1), region.reads), count) for region, count in regions)

  monkeypatch.setattr(plans, "seed_regions", perturbed)
  fun, mask = _mapped_tangents()
  rng = np.random.default_rng(225)
  x, d = rng.normal(size=(3, 4)), rng.normal(size=(3, 12, 4))
  expected = (d * mask * (np.cos(x) + x)[:, None, :]).sum(axis=2)
  fun.recompile()
  with pytest.raises(AssertionError):
    np.testing.assert_allclose(fun(x, d), expected)


def test_supported_scatter_sum_keeps_its_absorbed_operands():
  mask = np.array([[1.0, 0.0], [0.0, 1.0]])

  @sc.function(sc.arg("x", (2, 2)), name="seed_scatter_sum_inner")
  def inner(x):
    left = sc.scatter(x.reshape((4,)), [0, 2, 4, 6], 8).reshape((2, 4))
    right = sc.scatter((2 * x).reshape((4,)), [1, 3, 5, 7], 8).reshape((2, 4))
    return (left + right).block()

  @sc.function(sc.arg("x", (2, 2)), name="seed_scatter_sum_host")
  def host(x):
    return inner(x * sc.const(mask)).block()

  x = np.array([[0.3, -1.2], [2.5, 0.7]])
  values = (x * mask).reshape(-1)
  np.testing.assert_allclose(host(x), np.stack((values, 2 * values), axis=1).reshape((2, 4)))


@pytest.mark.parametrize("left_vector", [False, True])
@pytest.mark.parametrize("right_vector", [False, True])
def test_supported_contractions_match_numpy(left_vector, right_vector):
  ashape = (5,) if left_vector else (7, 5)
  bshape = (5,) if right_vector else (5, 3)
  rng = np.random.default_rng(225)
  amask = (rng.random(ashape) < 0.3).astype(float)
  bmask = (rng.random(bshape) < 0.4).astype(float)

  @sc.function(sc.arg("a", ashape), sc.arg("b", bshape), outputs=sc.arg("y"), name=f"seed_contraction_{left_vector}_{right_vector}")
  def inner(a, b):
    return (a @ b).block()

  @sc.function(sc.arg("a", ashape), sc.arg("b", bshape), outputs=sc.arg("y"), name=f"seed_contraction_host_{left_vector}_{right_vector}")
  def host(a, b):
    return inner(a * sc.const(amask), b * sc.const(bmask)).block()

  a, b = rng.normal(size=ashape), rng.normal(size=bshape)
  np.testing.assert_allclose(np.asarray(host(a, b)), (a * amask) @ (b * bmask), atol=1e-14)


def test_mapped_jvp_with_distinct_seed_supports_matches_numpy():
  rng = np.random.default_rng(225)
  masks = (rng.random((3, 12, 4)) < 0.3).astype(float)
  masks[0] = 0

  @sc.function(sc.arg("x", 4), outputs=sc.arg("y"), name="seed_pair_primal")
  def primal(x):
    return sc.concat((x[:2].sin() * x[2:], sc.sumsqr(x).reshape((1,)))).block()

  @sc.function(sc.arg("x", 4), sc.arg("d", (12, 4)), outputs=sc.arg("y"), name="seed_pair_jvp")
  def tangent(x, d):
    return sc.jvp_many(primal(x), x, d).block()

  @sc.function(sc.arg("x", (3, 4)), sc.arg("d", (3, 12, 4)), outputs=sc.arg("y"), name="seed_pair_mapped_jvp")
  def mapped(x, d):
    return sc.vmap(tangent, 3)(x, d * sc.const(masks))

  x, d = rng.normal(size=(3, 4)), rng.normal(size=(3, 12, 4))
  seeds = d * masks
  head = seeds[:, :, :2] * (np.cos(x[:, :2]) * x[:, 2:])[:, None, :] + seeds[:, :, 2:] * np.sin(x[:, :2])[:, None, :]
  tail = (2 * seeds * x[:, None, :]).sum(axis=2, keepdims=True)
  np.testing.assert_allclose(np.asarray(mapped(x, d)), np.concatenate((head, tail), axis=2), atol=1e-14)


def test_callee_support_variants_are_bounded_and_fallback_remains_sound():
  masks = ((np.arange(1, 35)[:, None] >> np.arange(6)) & 1).astype(float)

  @sc.function(sc.arg("x", 6), outputs=sc.arg("y"), name="seed_pair_bounded_inner")
  def inner(x):
    return x.sin().block()

  @sc.function(sc.arg("x", (34, 6)), outputs=sc.arg("y"), name="seed_pair_bounded_host")
  def host(x):
    return sc.vmap(inner, 34)(x * sc.const(masks))

  x = np.random.default_rng(225).normal(size=(34, 6))
  np.testing.assert_allclose(np.asarray(host(x)), np.sin(x * masks))
  module = render_c_module(host, lanes=1)
  variants = [proc for proc in module.program.args if proc.attrs.get("name", "").startswith("seed_pair_bounded_inner_support")]
  assert len(variants) <= 32


@pytest.mark.parametrize("dtype", ["int64", "bool"])
def test_supported_nonfloating_elementwise_uses_typed_zeros(dtype):
  mask = np.zeros((3, 4))
  mask[0, 1] = mask[1, 0] = 1

  @sc.function(sc.arg("x", (3, 4)), name=f"seed_typed_inner_{dtype}")
  def inner(x):
    return (x * 3).cast(dtype).cast("float64").block()

  @sc.function(sc.arg("x", (3, 4)), name=f"seed_typed_host_{dtype}")
  def host(x):
    return inner(x * sc.const(mask)).block()

  x = np.random.default_rng(225).normal(size=mask.shape)
  np.testing.assert_array_equal(host(x), (3 * x * mask).astype(dtype).astype(float))


@pytest.mark.parametrize("density", [0.3, 0.5, 0.9])
def test_supported_matmul_source_scales_with_operand_entries(density):
  n = 64
  rng = np.random.default_rng(225)
  w1, w2 = rng.normal(size=(2, n, n))
  seeds = rng.normal(size=(n, n)) * (rng.random((n, n)) < density)

  @sc.function(sc.arg("x", n), sc.arg("d", (n, n)), name=f"seed_mlp_tangent_{int(density * 100)}")
  def tangent(x, d):
    h = (sc.const(w1) @ x).tanh()
    return sc.jvp_many((sc.const(w2) @ h).tanh(), x, d).block()

  @sc.function(sc.arg("x", n), name=f"seed_mlp_host_{int(density * 100)}")
  def host(x):
    return tangent(x, sc.const(seeds)).block()

  module = render_c_module(host, lanes=1)
  assert len(module.body.encode()) < 400_000
  x = rng.normal(size=n)
  h = np.tanh(w1 @ x)
  y = np.tanh(w2 @ h)
  jac = (1 - y**2)[:, None] * w2 @ ((1 - h**2)[:, None] * w1)
  np.testing.assert_allclose(host(x), seeds @ jac.T, rtol=1e-12, atol=1e-12)
