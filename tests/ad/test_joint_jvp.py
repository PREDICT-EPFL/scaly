"""Joint call tangents and packed mapped derivatives share work without changing seed layout."""

from __future__ import annotations

import numpy as np
import pytest

from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
import scaly as sc
from scaly.ad.forward import _call_jvp_function, _call_jvp_many_function
from scaly.ad.reverse import _vmap_adj_function
from scaly.ir.expr import ExprOp, topo
from scaly.ir.program import ProgramOp, walk_program
from scaly.passes.lowering import lower_function


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("nseed", [1, 2, 3])
def test_joint_active_formals(mapped: bool, constant: bool, nseed: int) -> None:
  wx, wy = np.array([[1.0, 0.2], [-0.3, 0.7]]), np.array([[0.8, -0.1], [0.4, 0.9]])

  @sc.function(sc.group(sc.arg("x", 2), sc.arg("y", 2), sc.arg("unused", 2)), outputs=sc.arg("out"), name="joint_stage")
  def stage(inputs):
    x, y, _unused = inputs
    a, b = x @ sc.const(wx), y @ sc.const(wy)
    return (a * b).sin() + a / b

  length = 4 if mapped else 1
  sv = np.random.default_rng(3).normal(size=(nseed, 2 * length + 1))
  if nseed > 1:
    sv[1] = 0
  input_tree = sc.arg("z", 2 * length + 1) if constant else sc.group(sc.arg("z", 2 * length + 1), sc.arg("seed", sv.shape))

  @sc.function(input_tree, outputs=sc.arg("dy"), name="joint_actual")
  def fn(inputs):
    z = inputs if constant else inputs[0]
    value = _mapped_call(stage, length, [(z, 0, 2), (z, 1, 2), (z, 0, 2)]) if mapped else stage((z[:2], z[1:], z[:2]))
    seed = sc.const(sv) if constant else inputs[1]
    return sc.jvp(value, z, seed[0]) if nseed == 1 else sc.jvp_many(value, z, seed)

  z = as_concrete(fn).inputs[0]
  derivative = as_concrete(fn).outputs[0]
  zv = np.linspace(0.5, 1.5, z.size)
  expected = np.empty((nseed, 2 * length))
  for it in range(length):
    a, b = zv[2 * it : 2 * it + 2] @ wx, zv[2 * it + 1 : 2 * it + 3] @ wy
    da, db = sv[:, 2 * it : 2 * it + 2] @ wx, sv[:, 2 * it + 1 : 2 * it + 3] @ wy
    expected[:, 2 * it : 2 * it + 2] = np.cos(a * b) * (da * b + a * db) + da / b - a * db / b**2
  actual = fn(zv) if constant else fn((zv, sv))
  np.testing.assert_allclose(actual, expected[0] if nseed == 1 else expected, atol=1e-12, rtol=1e-12)
  calls = [node for node in topo([derivative]) if node.op in {ExprOp.CALL, ExprOp.VMAP}]
  assert len(calls) == 1


def test_packed_mapped_hessian_shares_primal_and_preserves_zero_seed_rows() -> None:
  weights = np.array([0.3, -0.7, 0.4])

  @sc.function(sc.group(sc.arg("x", 2), sc.arg("u", 1)), outputs=sc.arg("cost"), name="packed_stage")
  def stage(inputs):
    x, u = inputs
    return (x @ sc.const(weights[:2]) + weights[2] * u[0]).exp().scalar()

  length = 4
  seeds = np.tile(np.vstack([np.eye(3), np.zeros(3)]), (1, length))

  @sc.function(sc.arg("z", 3 * length), outputs=sc.arg("h"), name="packed_hess")
  def fn(z):
    mapped = _mapped_call(stage, length, [(z, 0, 3), (z, 2, 3)])
    grad = sc.vjp((mapped,), (z,), (sc.const(np.ones(length)),))[0]
    return sc.jvp_many(grad, z, sc.const(seeds))

  hess = as_concrete(fn).outputs[0]
  z = as_concrete(fn).inputs[0]
  zv = np.linspace(-0.8, 1.0, z.size)
  expected = np.zeros((4, z.size))
  for it in range(length):
    expected[:3, 3 * it : 3 * it + 3] = np.exp(zv[3 * it : 3 * it + 3] @ weights) * np.outer(weights, weights)
  np.testing.assert_allclose(fn(zv), expected, atol=1e-12, rtol=1e-12)
  calls = [node for node in topo([hess]) if node.op == ExprOp.VMAP]
  assert len(calls) == 1 and "_fwd_pack_" in calls[0].attrs["callee"].name
  prog = lower_function(fn)
  exp_nodes = {node for node in walk_program(prog) if node.op == ProgramOp.EXP}
  assert len(exp_nodes) == 1


def test_joint_helper_cache_distinguishes_formal_sets() -> None:
  @sc.function(sc.group(sc.arg("a_b", 2), sc.arg("a", 2), sc.arg("b", 2)), outputs=sc.arg("y"), name="joint_names")
  def stage(inputs):
    a, b, c = inputs
    return a * b + c.sin()

  first = _call_jvp_function(stage.instantiate(), 0, (0,))
  second = _call_jvp_function(stage.instantiate(), 0, (1, 2))
  assert first[0] is _call_jvp_function(stage.instantiate(), 0, (0,))[0]
  assert first[0].name != second[0].name
  assert first[0] is not second[0]


def test_call_helpers_are_cached_on_their_callee() -> None:
  from scaly.ad import forward, reverse

  @sc.function(sc.group(sc.arg("x", 2), sc.arg("y", 2)), outputs=sc.arg("z"), name="memo_stage")
  def stage(inputs):
    x, y = inputs
    return (x * y).sin()

  callee = stage.instantiate()
  helpers = {
    _call_jvp_function(callee, 0, (0,))[0],
    _call_jvp_many_function(callee, 0, (0,), 2, (None,))[0],
    _vmap_adj_function(callee, 0, (0, 1))[0],
  }
  cached = {entry[0] for cache in (callee._memo.jvp, callee._memo.jvp_many, callee._memo.vmap_adjoints) for entry in cache.values()}
  assert cached == helpers
  assert not [name for name in (*vars(forward), *vars(reverse)) if name.endswith("_CACHE")]
  copy = callee._replace(name="memo_stage_copy")
  assert _call_jvp_function(copy, 0, (0,))[0] not in helpers


@pytest.mark.parametrize("matrix", [False, True])
def test_self_products_keep_matrix_product_rule(matrix: bool) -> None:
  shape = (2, 2) if matrix else (4,)

  @sc.function(
    sc.group(sc.arg("x", shape), sc.arg("seed", (2, *shape))),
    outputs=sc.group(sc.arg("one"), sc.arg("many"), sc.arg("square")),
    name="self_products",
  )
  def fn(inputs):
    x, seed = inputs
    dot = x @ x
    return sc.jvp(dot, x, seed[0]), sc.jvp_many(dot, x, seed), sc.jvp_many(x * x, x, seed)

  xv = np.array([0.2, -0.7, 1.1, 0.9]).reshape(shape)
  sv = np.arange(8.0).reshape((2, *shape)) / 5 - 0.6
  one, many, square = fn((xv, sv))
  expected = np.stack([tangent @ xv + xv @ tangent for tangent in sv])
  np.testing.assert_allclose(one, expected[0], atol=1e-12, rtol=1e-12)
  np.testing.assert_allclose(many, expected, atol=1e-12, rtol=1e-12)
  np.testing.assert_allclose(square, 2 * xv * sv, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_division_and_sqrt_derivatives_across_finite_scales(scale: float) -> None:
  @sc.function(
    sc.group(sc.arg("x", ()), sc.arg("y", ())),
    outputs=sc.group(sc.arg("dx"), sc.arg("dy"), sc.arg("gx"), sc.arg("gy"), sc.arg("root")),
    name="finite_derivatives",
  )
  def fn(inputs):
    x, y = inputs
    quotient = x / y
    gx, gy = sc.vjp((quotient,), (x, y), (sc.const(1.0),))
    return (
      sc.jvp(quotient, x, sc.const(1.0)),
      sc.jvp_many(quotient, y, sc.const([1.0, -0.5])),
      gx,
      gy,
      sc.jvp_many(x.sqrt(), x, sc.const([1.0, -0.5])),
    )

  actual = fn((np.array(scale), np.array(scale)))
  expected = [1.0 / scale, np.array([-1.0, 0.5]) / scale, 1.0 / scale, -1.0 / scale, np.array([0.5, -0.25]) / np.sqrt(scale)]
  for value, reference in zip(actual, expected, strict=True):
    assert np.isfinite(value).all()
    np.testing.assert_allclose(value, reference, rtol=1e-12, atol=0)


def test_mapped_local_and_generic_seeds_share_one_callee() -> None:
  weights = np.array([[1.0, 0.2], [-0.3, 0.7]])

  @sc.function(sc.group(sc.arg("x", 2), sc.arg("u", 1)), outputs=sc.arg("out"), name="mixed_seed_stage")
  def stage(inputs):
    x, u = inputs
    return ((x @ sc.const(weights)) * u).sin()

  @sc.function(sc.group(sc.arg("z", 12), sc.arg("seeds", (2, 12))), outputs=sc.arg("dy"), name="mixed_seed_actual")
  def fn(inputs):
    z, seeds = inputs
    return sc.jvp_many(_mapped_call(stage, 4, [(z, 0, 3), (z, 2, 3)]), z, seeds)

  derivative = as_concrete(fn).outputs[0]
  zv = np.linspace(0.1, 1.2, 12)
  sv = np.random.default_rng(2).normal(size=(2, 12))
  expected = np.empty((2, 8))
  for it in range(4):
    xval, uval = zv[3 * it : 3 * it + 2] @ weights, zv[3 * it + 2]
    dx, du = sv[:, 3 * it : 3 * it + 2] @ weights, sv[:, 3 * it + 2 : 3 * it + 3]
    expected[:, 2 * it : 2 * it + 2] = np.cos(xval * uval) * (dx * uval + xval * du)
  np.testing.assert_allclose(fn((zv, sv)), expected, atol=1e-12, rtol=1e-12)
  calls = [node for node in topo([derivative]) if node.op == ExprOp.VMAP]
  assert len(calls) == 1 and "_fwd_pack_" in calls[0].attrs["callee"].name


def test_call_combines_constant_and_runtime_tangents() -> None:
  @sc.function(sc.group(sc.arg("x", 2), sc.arg("y", 2)), outputs=sc.arg("out"), name="mixed_call")
  def stage(inputs):
    x, y = inputs
    return (x * y).sin()

  seeds = np.vstack([np.eye(2), np.zeros(2)])

  @sc.function(sc.arg("z", 2), outputs=sc.arg("dy"), name="mixed_call_actual")
  def fn(z):
    return sc.jvp_many(stage((z, z.sin() + 2)), z, sc.const(seeds))

  derivative = as_concrete(fn).outputs[0]
  zv = np.array([0.2, -0.7])
  expected = seeds * (np.cos(zv * (np.sin(zv) + 2)) * (np.sin(zv) + 2 + zv * np.cos(zv)))
  np.testing.assert_allclose(fn(zv), expected, atol=1e-12, rtol=1e-12)
  assert sum(node.op == ExprOp.CALL for node in topo([derivative])) == 1


@pytest.mark.parametrize("nseed", [0, 1, 3])
def test_nested_calls_with_shared_formals_and_runtime_cotangents(nseed, monkeypatch):
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  tree = sc.group(sc.arg("shared_x", 2), sc.arg("shared_y", 2))

  @sc.function(tree, outputs=sc.group(sc.arg("a"), sc.arg("b")), name="nested_leaf")
  def leaf(inputs):
    x, y = inputs
    return (x * y).sin(), x / (y + 2)

  @sc.function(tree, outputs=sc.arg("out"), name="nested_middle")
  def middle(inputs):
    x, y = inputs
    a, b = leaf((y + 0.3, x * 0.7))
    return a + b * y

  @sc.function(
    sc.group(tree, sc.arg("nested_seeds", (nseed, 4)), sc.arg("nested_cot", 2)),
    outputs=sc.group(sc.arg("value"), sc.arg("single"), sc.arg("many"), sc.arg("gx"), sc.arg("gy")),
    name="nested_products",
  )
  def products(inputs):
    (x, y), seeds, cot = inputs
    assert x is as_concrete(leaf).inputs[0] is as_concrete(middle).inputs[0]
    assert y is as_concrete(leaf).inputs[1] is as_concrete(middle).inputs[1]
    value = middle((x, y)) + middle((y, x))
    rows = [sc.jvp(value, x, seeds[i, :2]) + sc.jvp(value, y, seeds[i, 2:]) for i in range(nseed)]
    single = sc.stack(rows) if rows else sc.const(np.empty((0, 2)))
    many = sc.jvp_many(value, x, seeds[:, :2]) + sc.jvp_many(value, y, seeds[:, 2:])
    gx, gy = sc.vjp((value,), (x, y), (cot,))
    return value, single, many, gx, gy

  def reference(z):
    x, y = z[:2], z[2:]

    def body(x, y):
      return np.sin((y + 0.3) * (x * 0.7)) + (y + 0.3) / (x * 0.7 + 2) * y

    return body(x, y) + body(y, x)

  z = np.array([0.4, -0.3, 0.8, 0.2])
  seeds = np.random.default_rng(8).normal(size=(nseed, 4))
  if nseed > 1:
    seeds[1] = 0
  cot = np.array([-0.7, 1.2])
  value, single, many, gx, gy = products(((z[:2], z[2:]), seeds, cot))
  step = 1e-5
  expected = np.stack([(reference(z + step * seed) - reference(z - step * seed)) / (2 * step) for seed in seeds]) if nseed else np.empty((0, 2))
  np.testing.assert_allclose(value, reference(z), atol=1e-12, rtol=1e-12)
  np.testing.assert_allclose(single, expected, atol=1e-9, rtol=1e-8)
  np.testing.assert_allclose(many, expected, atol=1e-9, rtol=1e-8)
  np.testing.assert_allclose(many @ cot, seeds @ np.concatenate([gx, gy]), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("nseed", [1, 3])
def test_call_mixed_constant_and_runtime_seed_formals(nseed, monkeypatch):
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")

  @sc.function(sc.group(sc.arg("mixed_x", 2), sc.arg("mixed_y", 2)), outputs=sc.arg("out"), name="mixed_formal_stage")
  def stage(inputs):
    x, y = inputs
    return (x * y).sin() + x / (y + 2)

  seeds = np.array([[1.0, -0.4], [0.0, 0.0], [-0.3, 0.7]])[:nseed]

  @sc.function(
    sc.group(sc.arg("mixed_z", 2), sc.arg("mixed_seed", (nseed, 2))),
    outputs=sc.group(sc.arg("baked"), sc.arg("runtime"), sc.arg("single")),
    name="mixed_formal_products",
  )
  def products(inputs):
    z, seed = inputs
    value = stage((z, z.sin() + 1))
    constant = sc.const(seeds)
    return sc.jvp_many(value, z, constant), sc.jvp_many(value, z, seed), sc.stack([sc.jvp(value, z, seed[i]) for i in range(nseed)])

  z = np.array([0.2, -0.7])
  baked, runtime, single = products((z, seeds))
  y = np.sin(z) + 1
  expected = seeds * (np.cos(z * y) * (y + z * np.cos(z)) + 1 / (y + 2) - z * np.cos(z) / (y + 2) ** 2)
  for result in (baked, runtime, single):
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)
  calls = [node for node in topo([as_concrete(products).outputs[0]]) if node.op == ExprOp.CALL]
  assert len(calls) == 1
  helper = calls[0].attrs["callee"]
  assert sum(name.startswith("fwd:") for name in helper.input_names) == 1


@pytest.mark.parametrize(
  "mode",
  [
    pytest.param("single", marks=pytest.mark.xfail(strict=True, reason="#161: power JVP divides by a zero base with an inactive runtime exponent")),
    pytest.param(
      "many", marks=pytest.mark.xfail(strict=True, reason="#161: structural power JVP divides by a zero base with an inactive runtime exponent")
    ),
    "reverse",
  ],
)
def test_power_zero_base_with_inactive_runtime_exponent(mode, monkeypatch):
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")

  @sc.function(
    sc.group(sc.arg("power_x", 2), sc.arg("power_p", 2), sc.arg("power_seeds", (3, 2))),
    outputs=sc.arg("derivative"),
    name="power_zero_base",
  )
  def products(inputs):
    x, p, seeds = inputs
    y = x**p
    if mode == "single":
      return sc.stack([sc.jvp(y, x, seeds[i]) for i in range(3)])
    if mode == "many":
      return sc.jvp_many(y, x, seeds)
    return sc.vjp((y,), (x,), (sc.const([1.0, 1.0]),))[0]

  x, p = np.array([0.0, -1.5]), np.array([2.0, 2.0])
  seeds = np.array([[1.0, -0.4], [0.0, 0.0], [-0.3, 0.7]])
  gradient = p * x ** (p - 1)
  expected = gradient if mode == "reverse" else seeds * gradient
  actual = products((x, p, seeds))
  assert np.isfinite(actual).all()
  np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
  if mode != "reverse":
    step = 1e-5
    differences = np.stack([((x + step * seed) ** p - (x - step * seed) ** p) / (2 * step) for seed in seeds])
    np.testing.assert_allclose(actual, differences, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize("layout", ["call_of_map", "map_of_call"])
@pytest.mark.parametrize("nseed", [1, 3, 5])
def test_nested_call_and_map_against_numpy(layout, nseed, monkeypatch):
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")

  @sc.function(sc.arg("nested_map_x", 2), outputs=sc.arg("y"), name="nested_map_leaf")
  def leaf(x):
    return sc.stack([x[0] * x[1], x[1].sin()])

  @sc.function(sc.arg("nested_map_z", 6), outputs=sc.arg("y"), name="nested_call_of_map")
  def call_of_map(z):
    return _mapped_call(leaf, 3, [(z, 0, 2)]) * z

  @sc.function(sc.arg("nested_map_w", 2), outputs=sc.arg("y"), name="nested_stage_with_call")
  def stage_with_call(w):
    return leaf(w) + w.exp()

  @sc.function(
    sc.group(sc.arg("outer_map_z", 6), sc.arg("outer_map_seeds", (nseed, 6)), sc.arg("outer_map_cot", 6)),
    outputs=sc.group(sc.arg("value"), sc.arg("single"), sc.arg("many"), sc.arg("adj")),
  )
  def products(inputs):
    z, seeds, cot = inputs
    y = call_of_map(z) if layout == "call_of_map" else _mapped_call(stage_with_call, 3, [(z, 0, 2)])
    return y, sc.stack([sc.jvp(y, z, seeds[i]) for i in range(nseed)]), sc.jvp_many(y, z, seeds), sc.vjp((y,), (z,), (cot,))[0]

  def reference(z):
    blocks = z.reshape(3, 2)
    value = np.stack([blocks[:, 0] * blocks[:, 1], np.sin(blocks[:, 1])], axis=1).ravel()
    return value * z if layout == "call_of_map" else value + np.exp(z)

  rng = np.random.default_rng(19)
  z, seeds, cot = rng.normal(size=6), rng.normal(size=(nseed, 6)), rng.normal(size=6)
  if nseed > 1:
    seeds[1] = 0
  value, single, many, adj = products((z, seeds, cot))
  expected = np.stack([(reference(z + 1e-5 * row) - reference(z - 1e-5 * row)) / 2e-5 for row in seeds])
  np.testing.assert_allclose(value, reference(z), atol=1e-12, rtol=1e-12)
  for tangent in (single, many):
    np.testing.assert_allclose(tangent, expected, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(tangent @ cot, seeds @ adj, atol=1e-12, rtol=1e-12)
