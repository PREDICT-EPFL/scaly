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
