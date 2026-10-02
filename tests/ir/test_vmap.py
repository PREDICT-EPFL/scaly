from __future__ import annotations

import numpy as np
import pytest

from scaly.function.sugar import _mapped_call
import scaly as sc


@sc.function(sc.group(sc.arg("x", 3), sc.arg("p", 3)), outputs=sc.arg("y"), name="scale_add")
def scale_add(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
  x, p = inputs
  return 2.0 * x + p


def test_vmap_eval_matches_unrolled_concat_of_call() -> None:
  N = 4
  inputs = sc.group(sc.arg("z", 3 * N), sc.arg("p", 3 * N))

  @sc.function(inputs, outputs=sc.arg("y"), name="scaled_vmap")
  def fn_vmap(zp: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = zp
    return _mapped_call(scale_add, N, [(z, 0, 3), (p, 0, 3)])

  @sc.function(inputs, outputs=sc.arg("y"), name="scaled_concat")
  def fn_concat(zp: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = zp
    return sc.concat([scale_add((z[i * 3 : (i + 1) * 3], p[i * 3 : (i + 1) * 3])) for i in range(N)])

  rng = np.random.default_rng(0)
  zv = rng.normal(size=3 * N)
  pv = rng.normal(size=3 * N)

  np.testing.assert_allclose(fn_vmap((zv, pv)), fn_concat((zv, pv)))


def test_vmap_overlapping_strided_slices_match_unrolled() -> None:
  NZ = 6
  NX = 4
  N = 3

  @sc.function(sc.group(sc.arg("z", NZ), sc.arg("znext", NZ), sc.arg("p", NX)), outputs=sc.arg("eq"), name="step")
  def step(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
    z, znext, p = inputs
    return (z[:NX] - znext[:NX]) + p

  inputs = sc.group(sc.arg("z", NZ * (N + 1)), sc.arg("p", NX * (N + 1)))

  @sc.function(inputs, outputs=sc.arg("eq"), name="step_vmap")
  def fn_vmap(zp: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = zp
    return _mapped_call(step, N, [(z, 0, NZ), (z, NZ, NZ), (p, NX, NX)])

  @sc.function(inputs, outputs=sc.arg("eq"), name="step_concat")
  def fn_concat(zp: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = zp
    parts = []
    for i in range(N):
      parts.append(step((z[i * NZ : (i + 1) * NZ], z[(i + 1) * NZ : (i + 2) * NZ], p[(i + 1) * NX : (i + 2) * NX])))
    return sc.concat(parts)

  rng = np.random.default_rng(1)
  zv = rng.normal(size=NZ * (N + 1))
  pv = rng.normal(size=NX * (N + 1))

  np.testing.assert_allclose(fn_vmap((zv, pv)), fn_concat((zv, pv)))


def test_vmap_zero_length_returns_empty() -> None:
  @sc.function(sc.group(sc.arg("z", 3), sc.arg("p", 3)), outputs=sc.arg("y"), name="empty_vmap")
  def fn(zp: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = zp
    return _mapped_call(scale_add, 0, [(z, 0, 0), (p, 0, 0)])

  out = fn((np.zeros(3), np.zeros(3)))
  assert isinstance(out, np.ndarray)
  assert out.shape == (0,)


def test_vmap_broadcast_stride_zero_repeats_same_slice() -> None:
  N = 3

  @sc.function(sc.group(sc.arg("z", 3), sc.arg("p", 3)), outputs=sc.arg("y"), name="broadcast_vmap")
  def fn(zp: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, p = zp
    return _mapped_call(scale_add, N, [(z, 0, 0), (p, 0, 0)])

  zv = np.array([1.0, 2.0, 3.0])
  pv = np.array([0.5, -1.0, 0.25])
  expected = np.tile(2.0 * zv + pv, N)
  np.testing.assert_allclose(fn((zv, pv)), expected)


def test_vmap_rejects_bad_input_specs() -> None:
  z = sc.sym("z", 6)
  p = sc.sym("p", 6)

  with pytest.raises(ValueError, match="vmap length"):
    _mapped_call(scale_add, -1, [(z, 0, 3), (p, 0, 3)])

  with pytest.raises(ValueError, match="reads past outer tensor"):
    _mapped_call(scale_add, 3, [(z, 0, 3), (p, 0, 3)])  # would need size 9 > 6

  with pytest.raises(ValueError, match=r"expects 2 input specs"):
    _mapped_call(scale_add, 2, [(z, 0, 3)])

  with pytest.raises(ValueError, match="stride must be non-negative"):
    _mapped_call(scale_add, 2, [(z, 0, -1), (p, 0, 3)])


def test_vmap_structural_key_matches_for_equal_constructions() -> None:
  # Construction-time interning collapses two structurally-identical ``sc.vmap`` calls to
  # the same Expr instance, so ``a is b`` and the structural-equality contract is preserved.
  z = sc.sym("z", 6)
  p = sc.sym("p", 6)
  a = _mapped_call(scale_add, 2, [(z, 0, 3), (p, 0, 3)])
  b = _mapped_call(scale_add, 2, [(z, 0, 3), (p, 0, 3)])
  assert a is b
  assert a.structurally_equal(b)


def test_vmap_accepts_input_dict_keyed_by_name() -> None:
  N = 4
  z = sc.sym("z", 3 * N)
  p = sc.sym("p", 3 * N)
  positional = _mapped_call(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  by_name = _mapped_call(scale_add, length=N, inputs={"x": (z, 0, 3), "p": (p, 0, 3)})
  assert positional.structurally_equal(by_name)

  with pytest.raises(ValueError, match="unknown callee input names"):
    _mapped_call(scale_add, length=N, inputs={"x": (z, 0, 3), "q": (p, 0, 3)})
  with pytest.raises(ValueError, match="missing entries for callee inputs"):
    _mapped_call(scale_add, length=N, inputs={"x": (z, 0, 3)})


def test_vmap_infers_chunked_and_broadcast_strides_from_sizes() -> None:
  N = 4
  z = sc.sym("z", 3 * N)
  p = sc.sym("p", 3)
  inferred = _mapped_call(scale_add, N, {"x": z, "p": p})
  explicit = _mapped_call(scale_add, N, [(z, 0, 3), (p, 0, 0)])
  assert inferred.structurally_equal(explicit)

  with pytest.raises(ValueError, match="input 'x' has size 10; expected 12 .* or 3"):
    _mapped_call(scale_add, N, [sc.sym("bad", 10), p])
