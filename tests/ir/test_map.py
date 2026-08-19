from __future__ import annotations

import numpy as np
import pytest

import alloy as al


@al.function("scale_add", {"x": 3, "p": 3})
def scale_add(x, p):
  return {"y": 2.0 * x + p}


def test_map_eval_matches_unrolled_concat_of_call() -> None:
  N = 4
  z = al.sym("z", 3 * N)
  p = al.sym("p", 3 * N)

  mapped = al.map_(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  unrolled = al.concat([scale_add.call([z[i * 3 : (i + 1) * 3], p[i * 3 : (i + 1) * 3]])[0] for i in range(N)])

  fn_map = al.Function("scaled_map", [z, p], [mapped], ["z", "p"], ["y"])
  fn_concat = al.Function("scaled_concat", [z, p], [unrolled], ["z", "p"], ["y"])

  rng = np.random.default_rng(0)
  zv = rng.normal(size=3 * N)
  pv = rng.normal(size=3 * N)

  np.testing.assert_allclose(fn_map(zv, pv), fn_concat(zv, pv))


def test_map_overlapping_strided_slices_match_unrolled() -> None:
  NZ = 6
  NX = 4
  N = 3

  @al.function("step", {"z": NZ, "znext": NZ, "p": NX})
  def step(z, znext, p):
    return {"eq": (z[:NX] - znext[:NX]) + p}

  z = al.sym("z", NZ * (N + 1))
  p = al.sym("p", NX * (N + 1))

  mapped = al.map_(step, N, [(z, 0, NZ), (z, NZ, NZ), (p, NX, NX)])
  parts = []
  for i in range(N):
    parts.append(step.call([z[i * NZ : (i + 1) * NZ], z[(i + 1) * NZ : (i + 2) * NZ], p[(i + 1) * NX : (i + 2) * NX]])[0])
  unrolled = al.concat(parts)

  fn_map = al.Function("step_map", [z, p], [mapped], ["z", "p"], ["eq"])
  fn_concat = al.Function("step_concat", [z, p], [unrolled], ["z", "p"], ["eq"])

  rng = np.random.default_rng(1)
  zv = rng.normal(size=NZ * (N + 1))
  pv = rng.normal(size=NX * (N + 1))

  np.testing.assert_allclose(fn_map(zv, pv), fn_concat(zv, pv))


def test_map_zero_length_returns_empty() -> None:
  z = al.sym("z", 3)
  p = al.sym("p", 3)
  empty = al.map_(scale_add, 0, [(z, 0, 0), (p, 0, 0)])
  fn = al.Function("empty_map", [z, p], [empty], ["z", "p"], ["y"])
  out = fn(np.zeros(3), np.zeros(3))
  assert isinstance(out, np.ndarray)
  assert out.shape == (0,)


def test_map_broadcast_stride_zero_repeats_same_slice() -> None:
  N = 3
  z = al.sym("z", 3)
  p = al.sym("p", 3)
  mapped = al.map_(scale_add, N, [(z, 0, 0), (p, 0, 0)])
  fn = al.Function("broadcast_map", [z, p], [mapped], ["z", "p"], ["y"])
  zv = np.array([1.0, 2.0, 3.0])
  pv = np.array([0.5, -1.0, 0.25])
  expected = np.tile(2.0 * zv + pv, N)
  np.testing.assert_allclose(fn(zv, pv), expected)


def test_map_rejects_bad_input_specs() -> None:
  z = al.sym("z", 6)
  p = al.sym("p", 6)

  with pytest.raises(ValueError, match="map length"):
    al.map_(scale_add, -1, [(z, 0, 3), (p, 0, 3)])

  with pytest.raises(ValueError, match="reads past outer tensor"):
    al.map_(scale_add, 3, [(z, 0, 3), (p, 0, 3)])  # would need size 9 > 6

  with pytest.raises(ValueError, match=r"expects 2 input specs"):
    al.map_(scale_add, 2, [(z, 0, 3)])

  with pytest.raises(ValueError, match="stride must be non-negative"):
    al.map_(scale_add, 2, [(z, 0, -1), (p, 0, 3)])


def test_map_structural_key_matches_for_equal_constructions() -> None:
  # Construction-time interning collapses two structurally-identical ``al.map_`` calls to
  # the same Expr instance, so ``a is b`` and the structural-equality contract is preserved.
  z = al.sym("z", 6)
  p = al.sym("p", 6)
  a = al.map_(scale_add, 2, [(z, 0, 3), (p, 0, 3)])
  b = al.map_(scale_add, 2, [(z, 0, 3), (p, 0, 3)])
  assert a is b
  assert a.structurally_equal(b)


def test_map_accepts_input_dict_keyed_by_name() -> None:
  N = 4
  z = al.sym("z", 3 * N)
  p = al.sym("p", 3 * N)
  positional = al.map_(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  by_name = al.scan(scale_add, length=N, inputs={"x": (z, 0, 3), "p": (p, 0, 3)})
  assert positional.structurally_equal(by_name)

  with pytest.raises(ValueError, match="unknown callee input names"):
    al.scan(scale_add, length=N, inputs={"x": (z, 0, 3), "q": (p, 0, 3)})
  with pytest.raises(ValueError, match="missing entries for callee inputs"):
    al.scan(scale_add, length=N, inputs={"x": (z, 0, 3)})
