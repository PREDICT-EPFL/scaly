"""Constant ``jvp_many`` seeds over a VMAP bake their per-iteration tiles into const-seed callees when the tiles repeat."""

from __future__ import annotations

import numpy as np
import pytest

from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
import scaly as sc
from scaly.ir.expr import ExprOp, topo
from scaly.ir.program import ProgramNode
from scaly.passes.lowering import lower_function

TILES = [
  np.eye(3),
  np.eye(3)[[1, 2, 0]],
  np.eye(3)[[2, 0, 1]],
  np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
  *(np.eye(3) * scale for scale in range(2, 7)),
]


def _stage() -> sc.Function:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="bake_stage")
  def stage(x):
    return x.sin() * (x @ sc.const(np.ones(3)))

  return stage


def _jac_np(x: np.ndarray) -> np.ndarray:
  return np.diag(np.cos(x) * x.sum()) + np.sin(x)[:, None]


def _seeds(pattern: list[int]) -> np.ndarray:
  seeds = np.zeros((3, 3 * len(pattern)))
  for it, tile in enumerate(pattern):
    seeds[:, 3 * it : 3 * it + 3] = TILES[tile]
  return seeds


def _callee_params(prog: ProgramNode) -> list[str]:
  return [str(a.attrs["name"]) for proc in prog.args[:-1] for a in proc.args[: proc.attrs["param_count"]]]


@pytest.mark.parametrize(
  ("pattern", "baked"),
  [
    ([0] * 5, True),
    ([3] * 4, True),
    ([0, 1] * 3, True),
    ([0, 1, 2] * 3, True),
    (list(range(8)) * 2, True),
    (list(range(9)) * 2, False),
    ([0, 1] * 4 + [2], False),
    ([0, 1, 2], False),
  ],
  ids=["equal", "equal_with_zero_row", "period2", "period3", "period8", "period9", "aperiodic", "distinct_short"],
)
def test_constant_seed_tiles(pattern: list[int], baked: bool) -> None:
  stage = _stage()
  length = len(pattern)
  seeds = _seeds(pattern)

  @sc.function(sc.arg("z", 3 * length), outputs=sc.arg("dy"), name="bake")
  def fn(z):
    return sc.jvp_many(_mapped_call(stage, length, [(z, 0, 3)]), z, sc.const(seeds))

  @sc.function(sc.arg("z", 3 * length), outputs=sc.arg("dy"), name="bake_ref")
  def ref(z):
    mapped = _mapped_call(stage, length, [(z, 0, 3)])
    return sc.stack([sc.jvp(mapped, z, sc.const(seed)) for seed in seeds])

  @sc.function(sc.group(sc.arg("z", 3 * length), sc.arg("runtime_seed", seeds.shape)), outputs=sc.arg("dy"), name="bake_runtime")
  def runtime(inputs):
    z, seed = inputs
    return sc.jvp_many(_mapped_call(stage, length, [(z, 0, 3)]), z, seed)

  (structural,) = as_concrete(fn).outputs
  zv = np.random.default_rng(1).normal(size=3 * length)
  expected = np.zeros((3, 3 * length))
  for it in range(length):
    block = slice(3 * it, 3 * it + 3)
    expected[:, block] = seeds[:, block] @ _jac_np(zv[block]).T
  np.testing.assert_allclose(fn(zv), ref(zv), rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(fn(zv), expected, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(runtime((zv, seeds)), expected, rtol=1e-12, atol=1e-12)
  permutation = [2, 0, 1]
  np.testing.assert_allclose(runtime((zv, seeds[permutation])), expected[permutation], rtol=1e-12, atol=1e-12)

  observed = {}
  lower_function(fn, observe=lambda name, program: observed.__setitem__(name, program))
  params = _callee_params(observed["pass:scalarize"])
  has_seed_input = any(name.startswith("fwd_") and "_" not in name[4:] for name in params)
  has_gather = any(n.op == ExprOp.GATHER for n in topo([structural]))
  assert has_seed_input is (not baked)
  assert has_gather is (not baked)


def test_local_coloring_and_packed_contributions_keep_seed_order():

  @sc.function(sc.group(sc.arg("local_x", 2), sc.arg("dense_y", 2)), outputs=sc.arg("out"), name="local_dense_stage")
  def stage(inputs):
    x, y = inputs
    return x.sin() + sc.stack([y[0] * y[1], y[0] / (y[1] + 2)])

  @sc.function(sc.group(sc.arg("local_z", 12), sc.arg("local_seeds", (2, 12))), outputs=sc.arg("dy"), name="local_dense_products")
  def products(inputs):
    z, seeds = inputs
    return sc.jvp_many(_mapped_call(stage, 3, [(z, 0, 4), (z, 2, 4)]), z, seeds)

  z = np.linspace(-0.4, 0.7, 12)
  seeds = np.random.default_rng(14).normal(size=(2, 12))
  expected = np.empty((2, 6))
  for it in range(3):
    x, y = z[4 * it : 4 * it + 2], z[4 * it + 2 : 4 * it + 4]
    dx, dy = seeds[:, 4 * it : 4 * it + 2], seeds[:, 4 * it + 2 : 4 * it + 4]
    expected[:, 2 * it : 2 * it + 2] = dx * np.cos(x) + np.column_stack(
      [
        dy[:, 0] * y[1] + y[0] * dy[:, 1],
        dy[:, 0] / (y[1] + 2) - y[0] * dy[:, 1] / (y[1] + 2) ** 2,
      ]
    )
  np.testing.assert_allclose(products((z, seeds)), expected, atol=1e-12, rtol=1e-12)
  np.testing.assert_allclose(products((z, seeds[::-1])), expected[::-1], atol=1e-12, rtol=1e-12)
  maps = [node for node in topo(as_concrete(products).outputs) if node.op == ExprOp.VMAP]
  assert len(maps) == 1 and "_fwd_pack_" in maps[0].attrs["callee"].name
  helpers = [entry[0] for key, entry in stage.instantiate()._memo.helpers.items() if key.nseed]
  assert {helper.outputs[0].shape for helper in helpers} == {(1, 2), (2, 2)}
  assert sum(name.startswith("fwd:") for name in maps[0].attrs["callee"].input_names) == 1
