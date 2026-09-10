"""Constant ``jvp_many`` seeds over a VMAP bake their per-iteration tiles into const-seed callees when the tiles repeat."""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.ad.forward import _jvp_many_unrolled
from alloy.ir.expr import ExprOp, topo
from alloy.ir.program import ProgramNode
from alloy.passes.lowering import lower_function

TILES = [
  np.eye(3),
  np.eye(3)[[1, 2, 0]],
  np.eye(3)[[2, 0, 1]],
  np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
  *(np.eye(3) * scale for scale in range(2, 7)),
]


def _stage() -> al.Function:
  x = al.sym("x", 3)
  return al.Function._from_exprs("bake_stage", [x], [x.sin() * (x @ al.const(np.ones(3)))], ["x"], ["y"])


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
  z = al.sym("z", 3 * length)
  mapped = al.vmap(stage, length, [(z, 0, 3)])
  seeds = _seeds(pattern)
  structural = al.jvp_many(mapped, z, al.const(seeds))
  unrolled = _jvp_many_unrolled(mapped, z, al.const(seeds))
  fn = al.Function._from_exprs("bake", [z], [structural], ["z"], ["dy"])
  ref = al.Function._from_exprs("bake_ref", [z], [unrolled], ["z"], ["dy"])
  zv = np.random.default_rng(1).normal(size=3 * length)
  expected = np.zeros((3, 3 * length))
  for it in range(length):
    block = slice(3 * it, 3 * it + 3)
    expected[:, block] = seeds[:, block] @ _jac_np(zv[block]).T
  np.testing.assert_allclose(fn(zv), ref(zv), rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(fn(zv), expected, rtol=1e-12, atol=1e-12)

  params = _callee_params(lower_function(fn))
  has_seed_input = any(name.startswith("fwd:") and ":" not in name[4:] for name in params)
  has_gather = any(n.op == ExprOp.GATHER for n in topo([structural]))
  assert has_seed_input is (not baked)
  assert has_gather is (not baked)
